# sphinx/tasks/tiles/tiles_compose_decompose.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass

from sphinx.base import Task
from sphinx.registry import register_task
from .common import MIN_TILING_WH, _max_wh_for, _out_px_for_dims, _components
from sphinx.tilings import TilingSpec, create_tiling, build_dual_graph, Colorer
from sphinx.config import MAX_BUILD_RETRIES, OUT_CELL, COLORS_NAMES

# Compose + drawing helpers
from ...utils.drawing import (
    load_font, labels_default,
    compose_top_bottom, add_tile_border,
)

# Reuse geometry helpers (region grow) + missing-tiles helpers (signatures and crisp overlays)
from .tiles_geometry import _grow_connected_chunk
from .tiles_missing_tiles import (
    _render_shape_uniform_edges, _render_overlay_and_bbox, _center_option_tile_from_overlay_noshrink,
    _ensure_margin_no_shrink, _apply_random_card_transform, _shape_signature_lattice,
    _compose_options_row_centered, _center_top_to_width,
)

from ...utils.rng import choice_weighted

from PIL import Image, ImageDraw

COLORS = list(COLORS_NAMES)

# ----------------------------- config knobs -----------------------------

# Which tilings to use and their sampling weights (as requested).
TILING_WEIGHTS: Dict[str, float] = {
    "square": 1.0,
    "triangular": 0.5,
    "hexagonal": 0.0,
    "orthogonal_split": 0.25,
}

# How many pieces to split the big tile into.
SPLIT_COUNT_WEIGHTS: Dict[int, float] = {2: 0.50, 3: 0.35, 4: 0.15, 5: 0.00}

# Target big-shape coverage relative to the board (keeps shapes nicely visible).
TARGET_COVER_FRAC_MIN: float = 0.22
TARGET_COVER_FRAC_MAX: float = 0.48

# Piece size variation control: Dirichlet concentration (higher -> more even)
SPLIT_SIZE_DIRICHLET_ALPHA: float = 1.2

# Visual spacings
SEP_PX = 40
SAFE_OPT_FRAC = 0.80
OUTLINE_PX = 1

# ----------------------------- color modes -----------------------------
# New: two colorization variants. Tune weights to bias difficulty.
# - 'uniform'         : every cell uses the same board color (harder)
# - 'random_per_cell' : each cell independently draws a color from COLORS (easier; color is a clue)
COLOR_MODE_WEIGHTS: Dict[str, float] = {
    "uniform": 0.75,
    "random_per_cell": 0.25,
}

# Decompose: top = one big connected tile, bottom = 4 bags of pieces
PROMPTS_DECOMPOSE = [
    "Top shows a single connected tile. Which option contains exactly the set of pieces that can rebuild it? Rotations/reflections allowed.",
    "The upper tile was cut into several pieces. Pick the bag that matches those pieces exactly (use all pieces once; you may rotate or mirror).",
    "Select the unique bag of pieces that can tile the top shape with no gaps or overlaps. Rotations and reflections are permitted.",
    "Which option’s pieces, after rotating and/or mirroring, reassemble the upper tile exactly? Only one bag works.",
    "Choose the bag whose pieces are congruent to the top shape’s parts (orientation may change).",
    "Identify the option that contains precisely the same pieces as the top tile’s decomposition (reorder and reorient as needed).",
    "One bag matches the top shape’s exact cut set. Which is it? Rotations/mirrors allowed; use every piece once.",
    "Which bag could be rearranged (with rotations or flips) to cover the top shape exactly once—no extras, no omissions?",
    "Pick the option whose multiset of pieces equals that of the top tile; orientation is free (rotate/flip allowed).",
    "From the four bags, choose the one that can reconstruct the upper tile using all pieces exactly once (rotations/reflections allowed).",
]

# Compose: top = bag of pieces, bottom = 4 candidate big tiles
PROMPTS_COMPOSE = [
    "Top: a bag of pieces. Bottom: four candidate tiles. Which single connected tile can be assembled from all the pieces? Rotations/reflections allowed.",
    "Using every piece on top exactly once, which bottom option can you build? You may rotate or mirror the pieces.",
    "Which candidate equals the union of all top pieces (after rotation/flip), with no gaps or overlaps?",
    "Identify the unique bottom tile that can be formed by rearranging the top pieces. Orientation may change.",
    "Select the shape the top pieces can form exactly; rotations and reflections are permitted.",
    "From the four candidates, choose the one buildable from the given pieces using each piece once (reorient as needed).",
    "Find the exact connected tile obtainable by combining all top pieces (rotation/mirroring allowed).",
    "Which option matches the exact coverage of the top pieces when assembled? You may rotate or flip them.",
    "Choose the only candidate that can be tiled by the top pieces with no leftover tiles; orientation is free.",
    "Pick the bottom tile that results from assembling all top pieces (rotations/reflections allowed).",
]

# ----------------------------------------------------------------------
# Small helpers reused across tiles tasks (locally copied for independence)

def _build_transform_shared(patch, target_px: int, margin_frac: float = 0.06):
    polys = patch.cell_polygons()
    x0 = min(x for poly in polys for (x, _) in poly)
    y0 = min(y for poly in polys for (_, y) in poly)
    x1 = max(x for poly in polys for (x, _) in poly)
    y1 = max(y for poly in polys for (_, y) in poly)
    bw = x1 - x0
    bh = y1 - y0

    W = int(target_px); H = int(target_px)
    margin = margin_frac * max(bw, bh)
    sx = (W - 2 * margin - 1) / max(1e-9, bw)
    sy = (H - 2 * margin - 1) / max(1e-9, bh)
    s = min(sx, sy)

    def TX(pt):
        x, y = pt
        xi = int(round((x - x0) * s + margin))
        yi = int(round((y - y0) * s + margin))
        xi = 0 if xi < 0 else (W - 1 if xi > W - 1 else xi)
        yi = 0 if yi < 0 else (H - 1 if yi > H - 1 else yi)
        return xi, yi

    return TX, (W, H)

def _class_id(tiling, cell) -> int:
    """Parity/Wythoffian class index; mirrors tiles_missing_tiles._class_id."""
    if getattr(tiling, "supports_wythoffian", False):
        return int(tiling.wythoffian_class_id(cell))
    i, j = cell.coord[:2]
    return (int(i) + int(j)) & 1

def _count_classes(tiling_name: str) -> int:
    return 4 if tiling_name == "square" else 3

def _parity_vector(tiling, patch, ids: Sequence[int]) -> Tuple[int, ...]:
    C = _count_classes(tiling.name)
    v = [0]*C
    for cid in ids:
        c = _class_id(tiling, patch.cells[cid])
        v[c % C] += 1
    return tuple(v)

def _bag_signature_for_pieces(patch, tiling_name: str, pieces: List[Set[int]]) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
    """Canonical multiset signature of a bag of pieces (sorted tuple of shape signatures)."""
    sigs = []
    for p in pieces:
        sig = _shape_signature_lattice(patch, tiling_name, list(p))
        sigs.append(sig)
    return tuple(sorted(sigs))

def _connected_subset_bfs_from(g: Dict[int, Set[int]], allowed: Set[int], k: int, rng: random.Random) -> Optional[Set[int]]:
    """Pick a random seed in 'allowed' and grow a connected subset of size k (best effort)."""
    if k <= 0 or not allowed:
        return None
    seed = rng.choice(tuple(allowed))
    region = {seed}
    frontier = [seed]
    used = {seed}
    while frontier and len(region) < k:
        u = frontier.pop(rng.randrange(len(frontier)))
        cand = [v for v in g[u] if v in allowed and v not in used]
        rng.shuffle(cand)
        for v in cand:
            region.add(v)
            used.add(v)
            frontier.append(v)
            if len(region) >= k:
                break
    if len(region) < k:
        return None
    return region

def _pick_far_seeds(rng: random.Random, g: Dict[int, Set[int]], region: Set[int], k: int) -> List[int]:
    """Pick k seeds in 'region' that are roughly far apart (greedy farthest-point)."""
    nodes = list(region)
    s0 = rng.choice(nodes)
    seeds = [s0]
    # BFS distance utility
    from collections import deque
    def dists_from(src: int) -> Dict[int, int]:
        q = deque([src]); dist = {src: 0}
        while q:
            u = q.popleft()
            for v in g[u]:
                if v in region and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    for _ in range(max(0, k - 1)):
        best = None; best_min_d = -1
        for cand in nodes:
            mind = min(dists_from(s)[cand] for s in seeds if cand in region)
            if mind > best_min_d:
                best_min_d = mind; best = cand
        if best is None:
            best = rng.choice(nodes)
        seeds.append(best)
    return seeds

def _dirichlet(rng: random.Random, n: int, alpha: float) -> List[float]:
    import random as pyrandom
    # Use Python's random.gammavariate for a lightweight Dirichlet
    xs = [pyrandom.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(xs) or 1.0
    return [x / s for x in xs]

def _split_region_into_pieces(
    rng: random.Random, g: Dict[int, Set[int]], region: Set[int], num_pieces: int,
    *, min_piece_size: int = 2, alpha: float = SPLIT_SIZE_DIRICHLET_ALPHA
) -> Optional[List[Set[int]]]:
    """
    Partition 'region' into 'num_pieces' edge-connected pieces. Sizes follow a Dirichlet
    (rounded) and every piece has at least min_piece_size cells.
    """
    N = len(region)
    if N < num_pieces * min_piece_size:
        return None

    # target sizes
    p = _dirichlet(rng, num_pieces, alpha=alpha)
    sizes = [max(min_piece_size, int(round(pi * N))) for pi in p]

    # Adjust to match N exactly
    delta = sum(sizes) - N
    while delta != 0:
        i = rng.randrange(num_pieces)
        if delta > 0 and sizes[i] > min_piece_size:
            sizes[i] -= 1; delta -= 1
        elif delta < 0:
            sizes[i] += 1; delta += 1

    # greedy growth from far seeds; enforce connected remainder at each step
    remaining = set(region)
    pieces: List[Set[int]] = []

    for i in range(num_pieces):
        k = sizes[i]
        # available must leave enough cells for remaining pieces
        min_left = (num_pieces - i - 1) * min_piece_size
        allowed = {u for u in remaining if len(remaining) - 1 >= min_left}
        if len(allowed) < k:
            return None
        # grow connected subset up to k
        for _ in range(64):
            subset = _connected_subset_bfs_from(g, allowed, k, rng)
            if subset is None:
                continue
            rem_after = remaining - subset
            # keep remainder connected to avoid stranded islands early
            if (i == num_pieces - 1) or len(_components(g, rem_after)) == 1:
                pieces.append(subset)
                remaining = rem_after
                break
        else:
            return None

    # final sanity
    if remaining:
        # last piece was not consumed properly
        return None
    if len(pieces) != num_pieces or sum(len(p) for p in pieces) != N:
        return None
    return pieces

def _render_shape_boundary_only(patch, TX, canvas_wh, ids, fill_color: str, outline_px: int = OUTLINE_PX):
    """
    Render only the union silhouette of 'ids': fill interior with fill_color and
    stroke only the *outer* boundary (no internal grid).
    """
    W, H = canvas_wh
    rgba = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(rgba)

    polys = patch.cell_polygons()

    # Draw fills
    for cid in ids:
        pts = [TX(p) for p in polys[cid]]
        draw.polygon(pts, fill=fill_color)

    # Collect edge multiplicities (shared edges appear twice)
    def edge_key(a, b):
        return tuple(sorted((a, b)))
    edge_count = {}
    for cid in ids:
        P = [TX(p) for p in polys[cid]]
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            k = edge_key(a, b)
            edge_count[k] = edge_count.get(k, 0) + 1

    # Draw only boundary edges (count == 1)
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            draw.line([a, b], fill="black", width=outline_px)

    return rgba

def _compose_option_bag_tile(
    piece_tiles: List["Image.Image"],
    *, box_px: int, grid_cols: Optional[int] = None, pad_px: int = 10, sep_px: int = 8
) -> "Image.Image":
    """
    Pack a list of square piece tiles into a single square option card (like a "bag of pieces").
    IMPORTANT: we preserve the *original* pixel scale of pieces (no downscaling). The card
    expands as needed.
    """
    from PIL import Image
    k = len(piece_tiles)
    if k == 0:
        return Image.new("RGBA", (box_px, box_px), (255, 255, 255, 255))

    if grid_cols is None:
        grid_cols = 1 if k <= 2 else (2 if k <= 4 else 3)
    cols = max(1, int(grid_cols))
    rows = (k + cols - 1) // cols

    # Determine cell side from the largest piece
    cell_side = max(max(t.width, t.height) for t in piece_tiles)

    # Compute exact card size needed (box_px is a minimum)
    W_needed = 2 * pad_px + cols * cell_side + (cols - 1) * sep_px
    H_needed = 2 * pad_px + rows * cell_side + (rows - 1) * sep_px
    card_w = max(box_px, W_needed)
    card_h = max(box_px, H_needed)

    card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
    x = pad_px; y = pad_px
    for i, tile in enumerate(piece_tiles):
        px = x + (cell_side - tile.width) // 2
        py = y + (cell_side - tile.height) // 2
        card.alpha_composite(tile, (px, py))
        if (i + 1) % cols == 0:
            x = pad_px; y += cell_side + sep_px
        else:
            x += cell_side + sep_px
    return card

# ----------------------------- color helpers -----------------------------

def _build_random_color_map_for_patch(rng: random.Random, patch) -> Dict[int, str]:
    """
    Assign a random color to each cell id in the *entire* patch. This ensures
    distractors (drawn from anywhere on the board) get consistent coloring too.
    """
    n = len(patch.cells)
    return {cid: rng.choice(COLORS) for cid in range(n)}

def _fill_mapping_for_ids(ids: Sequence[int], *, board_color: str, color_by_cell: Optional[Dict[int, str]], mode: str) -> Dict[int, str]:
    """
    Produce a per-cell fill mapping for the provided cell ids according to color mode.
    - uniform: all ids map to board_color
    - random_per_cell: ids map to color_by_cell[cid]
    """
    if mode == "random_per_cell":
        assert color_by_cell is not None, "color_by_cell must be provided for random_per_cell mode"
        return {cid: color_by_cell[cid] for cid in ids}
    # default: uniform
    return {cid: board_color for cid in ids}

# ----------------------------------------------------------------------
# Task state container

@dataclass
class _GeneratedCase:
    tiling: Any
    spec: TilingSpec
    patch: Any
    g: Dict[int, Set[int]]
    region: Set[int]                 # the "big tile" as node ids
    pieces: List[Set[int]]           # decomposition of region into connected pieces
    parity_vector: Tuple[int, ...]   # class-count vector of the sum of the pieces (== region parity)
    region_sig: Tuple[Tuple[float, float], ...]  # shape signature of the region

def _sample_base_case(rng: random.Random) -> Optional[_GeneratedCase]:
    """Sample a tiling + one connected 'big tile' region and split it into the pieces."""
    names = list(TILING_WEIGHTS.keys())
    weights = [TILING_WEIGHTS[n] for n in names]
    tname = choice_weighted(rng, names, weights)
    tiling = create_tiling(tname)

    # square board for clean presentation
    hi = _max_wh_for(tiling.name)
    side = rng.randint(MIN_TILING_WH, hi)

    seed = rng.randint(0, 2 ** 31 - 1)
    spec = TilingSpec(tiling.name, seed, width=side, height=side, uniform={"scheme": "same"})
    patch = tiling.generate(spec)
    Colorer().apply(tiling, patch, spec)

    g = build_dual_graph(patch, connect_on_touch=False)
    n = len(patch.cells)

    # Target region size
    target_min = max(8, int(round(TARGET_COVER_FRAC_MIN * n)))
    target_max = max(target_min + 1, int(round(TARGET_COVER_FRAC_MAX * n)))

    # Find a connected region that isn't too tiny/huge
    region = None
    for _ in range(64):
        want = rng.randint(target_min, target_max)
        cand = _grow_connected_chunk(rng, g, set(range(n)), want)
        if cand and len(cand) >= max(6, int(0.15 * n)):
            region = cand
            break
    if region is None:
        return None

    # Choose how many pieces
    ks = sorted(SPLIT_COUNT_WEIGHTS.keys())
    w = [SPLIT_COUNT_WEIGHTS[k] for k in ks]
    num_pieces = choice_weighted(rng, ks, w)

    # Split region
    pieces = None
    for _ in range(128):
        out = _split_region_into_pieces(rng, g, region, num_pieces, min_piece_size=2, alpha=SPLIT_SIZE_DIRICHLET_ALPHA)
        if out is not None:
            pieces = out
            break
    if pieces is None:
        return None

    # Parity vector invariant
    pv = _parity_vector(tiling, patch, list(region))

    reg_sig = _shape_signature_lattice(patch, tiling.name, list(region))

    return _GeneratedCase(
        tiling=tiling, spec=spec, patch=patch, g=g, region=region,
        pieces=pieces, parity_vector=pv, region_sig=reg_sig
    )

# ----------------------- distractors -----------------------

def _sample_piece_like_but_different(
    rng: random.Random, g: Dict[int, Set[int]], patch, tiling_name: str, want_size: int,
    forbid: Set[Tuple[Tuple[float, float], ...]]
) -> Optional[Set[int]]:
    """Sample a connected shape of given size with a signature not in 'forbid'."""
    all_nodes = set(range(len(patch.cells)))
    for _ in range(200):
        cand = _connected_subset_bfs_from(g, all_nodes, want_size, rng)
        if cand is None:
            continue
        sig = _shape_signature_lattice(patch, tiling_name, list(cand))
        if sig not in forbid:
            forbid.add(sig)
            return cand
    return None

def _make_bag_options_for_decompose(
    rng: random.Random, case: _GeneratedCase, tile_px: int,
    *, color_mode: str, color_by_cell: Optional[Dict[int, str]]
) -> Tuple[List["Image.Image"], int, List[List[Set[int]]]]:
    """
    Build 4 option cards (bag of pieces). Return (cards, correct_index, option_piece_sets).
    Colorization is controlled by color_mode, and is consistent across correct/distractors.
    """
    tiling = case.tiling; patch = case.patch
    pieces = case.pieces
    g = case.g
    board_color = getattr(patch.cells[0], "color", "#808080")

    # Correct bag signature for uniqueness checks
    correct_bag_sig = _bag_signature_for_pieces(patch, tiling.name, pieces)

    option_sets: List[List[Set[int]]] = [pieces]
    seen_bag_sigs = {correct_bag_sig}

    # Try to generate 3 distractors with same piece count and same total area first
    for _ in range(256):
        if len(option_sets) >= 4:
            break
        # mutate: pick one piece, replace with same-area different-shape
        idx = rng.randrange(len(pieces))
        forbid = { _shape_signature_lattice(patch, tiling.name, list(pieces[idx])) }
        repl = _sample_piece_like_but_different(rng, g, patch, tiling.name, len(pieces[idx]), forbid)
        if repl is None:
            continue
        new_bag = [set(p) for p in pieces]
        new_bag[idx] = set(repl)

        bag_sig = _bag_signature_for_pieces(patch, tiling.name, new_bag)
        if bag_sig in seen_bag_sigs:
            continue
        option_sets.append(new_bag)
        seen_bag_sigs.add(bag_sig)

    # Fallback: allow wrong total area or different piece count to guarantee impossibility
    while len(option_sets) < 4:
        idx = rng.randrange(len(pieces))
        want = max(1, len(pieces[idx]) + rng.choice([-1, +1]))
        want = max(1, min(want, max(3, len(case.region) - 1)))
        forbid = set()
        repl = _sample_piece_like_but_different(rng, g, patch, tiling.name, want, forbid)
        if repl is None:
            continue
        new_bag = [set(p) for p in pieces]
        new_bag[idx] = set(repl)
        bag_sig = _bag_signature_for_pieces(patch, tiling.name, new_bag)
        if bag_sig in seen_bag_sigs:
            continue
        option_sets.append(new_bag)
        seen_bag_sigs.add(bag_sig)

    # Render each bag into a single option card (preserve scale)
    TX, canvas_wh = _build_transform_shared(patch, target_px=tile_px, margin_frac=0.06)
    safe_box = int(tile_px * SAFE_OPT_FRAC)
    box_px = max(96, safe_box - 6)

    option_tiles: List["Image.Image"] = []
    for bag in option_sets:
        piece_tiles: List["Image.Image"] = []
        for sids in bag:
            ids = list(sids)
            fill_by_cell = _fill_mapping_for_ids(ids, board_color=board_color, color_by_cell=color_by_cell, mode=color_mode)
            # render at model scale; crop to bbox; do NOT shrink
            overlay = _render_shape_uniform_edges(patch, TX, canvas_wh, ids, fill_by_cell)
            overlay, bbox = _render_overlay_and_bbox(overlay)
            opt = _center_option_tile_from_overlay_noshrink(
                overlay, bbox,
                min_px=max(72, OUT_CELL // 1),
                max_px=None,  # preserve scale
                pad_frac=0.10,
            )
            # crisp variety (avoid 60° raster rotations)
            opt = _apply_random_card_transform(opt)
            opt = _ensure_margin_no_shrink(opt, edge=2, step_px=6)
            piece_tiles.append(opt)

        card = _compose_option_bag_tile(piece_tiles, box_px=box_px, grid_cols=None, pad_px=12, sep_px=10)
        option_tiles.append(card)

    # Shuffle options
    order = list(range(4)); rng.shuffle(order)
    option_tiles = [option_tiles[i] for i in order]
    option_sets = [option_sets[i] for i in order]
    correct_index = order.index(0)

    return option_tiles, int(correct_index), option_sets

def _make_bigshape_options_for_compose(
    rng: random.Random, case: _GeneratedCase, tile_px: int,
    *, color_mode: str, color_by_cell: Optional[Dict[int, str]]
) -> Tuple[List["Image.Image"], int, List[Set[int]]]:
    """
    Build 4 option cards (single connected shapes). Return (tiles, correct_index, option_shapes).
    Colorization is controlled by color_mode, and is consistent across correct/distractors.
    """
    tiling = case.tiling; patch = case.patch
    g = case.g
    board_color = getattr(patch.cells[0], "color", "#808080")
    total_size = len(case.region)
    parity_need = case.parity_vector
    correct_sig = case.region_sig

    # Start with correct region + 3 distractors
    option_shapes: List[Set[int]] = [set(case.region)]
    seen_sigs = {correct_sig}

    all_nodes = set(range(len(patch.cells)))

    for _ in range(512):
        if len(option_shapes) >= 4:
            break
        want = total_size  # try same area first
        cand = _connected_subset_bfs_from(g, all_nodes, want, rng)
        if cand is None or cand == case.region:
            continue
        sig = _shape_signature_lattice(patch, tiling.name, list(cand))
        if sig in seen_sigs:
            continue
        # parity constraint: require a mismatch to guarantee impossibility
        v = _parity_vector(tiling, patch, list(cand))
        if v == parity_need:
            # too risky: pieces might tile this as well; skip
            continue
        option_shapes.append(cand)
        seen_sigs.add(sig)

    # Fallback: allow area mismatch (guaranteed impossible)
    while len(option_shapes) < 4:
        want = total_size + rng.choice([-2, -1, +1, +2])
        want = max(3, min(want, len(all_nodes)-1))
        cand = _connected_subset_bfs_from(g, all_nodes, want, rng)
        if cand is None:
            continue
        sig = _shape_signature_lattice(patch, tiling.name, list(cand))
        if sig in seen_sigs:
            continue
        option_shapes.append(cand)
        seen_sigs.add(sig)

    # Render each shape as a single tile (preserve scale)
    TX, canvas_wh = _build_transform_shared(patch, target_px=tile_px, margin_frac=0.06)
    option_tiles: List["Image.Image"] = []
    for sids in option_shapes:
        ids = list(sids)
        fill_by_cell = _fill_mapping_for_ids(ids, board_color=board_color, color_by_cell=color_by_cell, mode=color_mode)
        overlay = Image.new("RGBA", canvas_wh, (255, 255, 255, 0))
        ov = _render_shape_uniform_edges(patch, TX, canvas_wh, ids, fill_by_cell)
        overlay.alpha_composite(ov, (0, 0))
        overlay, bbox = _render_overlay_and_bbox(overlay)
        card = _center_option_tile_from_overlay_noshrink(
            overlay, bbox, min_px=max(96, OUT_CELL // 1), max_px=None, pad_frac=0.10
        )
        option_tiles.append(card)

    # Shuffle
    order = list(range(4)); rng.shuffle(order)
    option_tiles = [option_tiles[i] for i in order]
    option_shapes = [option_shapes[i] for i in order]
    correct_index = order.index(0)

    return option_tiles, int(correct_index), option_shapes

# ----------------------------- public task -----------------------------

@register_task
class TilesDecomposeComposeTask(Task):
    """
    Two puzzle variants around decomposing a connected tile into smaller connected pieces
    and composing pieces back into a single connected tile.

    Variants (sampled 50/50):
      - decompose: show big tile on top; bottom shows four 'bags of pieces' (only one matches exactly).
      - compose  : show bag of pieces on top; bottom shows four single connected shapes; only one matches.

    Colorization modes (sampled by COLOR_MODE_WEIGHTS):
      - uniform:         all cells share the board color (harder; no color clue)
      - random_per_cell: cells are independently colored (easier; color provides a clue)
    """
    name = "tiles_decompose_compose"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, region_size: int) -> Dict[str, Any]:
        """Normalize region size (tile count) to EASY/HARD."""
        min_region = 5
        max_region = 35
        span = max(1, max_region - min_region)
        normalized = (int(region_size) - min_region) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "tiles-compose-decompose-region-v1",
            "range": {"min_region": min_region, "max_region": max_region},
            "region_size": int(region_size),
        }

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # sample case
        for _ in range(self.max_retries):
            case = _sample_base_case(rng)
            if case is None:
                continue

            tiling = case.tiling; patch = case.patch
            spec = case.spec

            tile_px = _out_px_for_dims(spec.width, spec.height)
            if max(spec.width, spec.height) <= 6:
                tile_px = int(tile_px * 1.25)

            # variant: decompose or compose
            variant = rng.choice(["decompose", "compose"])

            # color mode: uniform vs random_per_cell
            cm_names = list(COLOR_MODE_WEIGHTS.keys())
            cm_weights = [COLOR_MODE_WEIGHTS[n] for n in cm_names]
            color_mode = choice_weighted(rng, cm_names, cm_weights)

            # Build common transform
            TX, canvas_wh = _build_transform_shared(patch, target_px=tile_px, margin_frac=0.06)
            board_color = getattr(patch.cells[0], "color", "#808080")

            # Build per-cell color map *for the whole patch* if needed
            color_by_cell = None
            if color_mode == "random_per_cell":
                color_by_cell = _build_random_color_map_for_patch(rng, patch)

            # Top image
            if variant == "decompose":
                # show ONLY the big region silhouette (no outside grid)
                W, H = canvas_wh
                top_bg = Image.new("RGB", (W, H), "white")
                fill_by_cell = _fill_mapping_for_ids(list(case.region), board_color=board_color, color_by_cell=color_by_cell, mode=color_mode)
                overlay = _render_shape_uniform_edges(patch, TX, canvas_wh, list(case.region), fill_by_cell)

                top_bg.paste(overlay.convert("RGB"), (0, 0), overlay)
                top_img = top_bg

                # Options: bags of pieces (same scale as the top)
                option_tiles, answer_index, _ = _make_bag_options_for_decompose(
                    rng, case, tile_px, color_mode=color_mode, color_by_cell=color_by_cell
                )

                # Layout into composite
                font = load_font()
                labels = labels_default()
                labeled_opts = []
                for im, lab in zip(option_tiles, labels):
                    im = add_tile_border(im)
                    Wc = im.width
                    x0, y0, x1, y1 = font.getbbox(lab)
                    text_w, text_h = (x1 - x0), (y1 - y0)
                    pad_y = max(6, OUTLINE_PX + 4)
                    Hc = im.height + pad_y + text_h
                    canvas = Image.new("RGBA", (Wc, Hc), (255, 255, 255, 255))
                    canvas.alpha_composite(im, (0, 0))
                    d = ImageDraw.Draw(canvas)
                    tx = (Wc - text_w) // 2
                    ty = im.height + pad_y - y0
                    d.text((tx, ty), lab, fill=(0, 0, 0), font=font)
                    labeled_opts.append(canvas)

                opts_row = _compose_options_row_centered(labeled_opts, sep=SEP_PX // 2, cell_pad=16, bg="white", sep_color=None)
                top_centered = _center_top_to_width(top_img, opts_row.width, bg="white")
                composite = compose_top_bottom(top_centered, opts_row, sep_px=SEP_PX)

                question = random.choice(PROMPTS_DECOMPOSE)

                meta_variant = {
                    "kind": "decompose",
                    "measure": "bag_of_pieces_matches_big_shape",
                    "transforms_allowed": True,
                }

            else:
                # variant == "compose": show the pieces on top
                piece_tiles: List["Image.Image"] = []
                for sids in case.pieces:
                    ids = list(sids)
                    fill_by_cell = _fill_mapping_for_ids(ids, board_color=board_color, color_by_cell=color_by_cell, mode=color_mode)
                    overlay = _render_shape_uniform_edges(patch, TX, canvas_wh, ids, fill_by_cell)
                    overlay, bbox = _render_overlay_and_bbox(overlay)
                    opt = _center_option_tile_from_overlay_noshrink(
                        overlay, bbox,
                        min_px=max(72, OUT_CELL // 1),
                        max_px=None,   # preserve scale
                        pad_frac=0.10,
                    )
                    opt = _apply_random_card_transform(opt)
                    opt = _ensure_margin_no_shrink(opt, edge=2, step_px=6)
                    piece_tiles.append(opt)

                # pack pieces into one top card (same scale)
                top_img = _compose_option_bag_tile(piece_tiles, box_px=int(tile_px * 0.90), grid_cols=None, pad_px=12, sep_px=10)

                # Options: big shapes (one correct, three impossible by parity or area mismatch)
                option_tiles, answer_index, _ = _make_bigshape_options_for_compose(
                    rng, case, tile_px, color_mode=color_mode, color_by_cell=color_by_cell
                )

                font = load_font()
                labels = labels_default()
                labeled_opts = []
                for im, lab in zip(option_tiles, labels):
                    im = add_tile_border(im)
                    Wc = im.width
                    x0, y0, x1, y1 = font.getbbox(lab)
                    text_w, text_h = (x1 - x0), (y1 - y0)
                    pad_y = max(6, OUTLINE_PX + 4)
                    Hc = im.height + pad_y + text_h
                    canvas = Image.new("RGBA", (Wc, Hc), (255, 255, 255, 255))
                    canvas.alpha_composite(im, (0, 0))
                    d = ImageDraw.Draw(canvas)
                    tx = (Wc - text_w) // 2
                    ty = im.height + pad_y - y0
                    d.text((tx, ty), lab, fill=(0, 0, 0), font=font)
                    labeled_opts.append(canvas)

                opts_row = _compose_options_row_centered(labeled_opts, sep=SEP_PX // 2, cell_pad=16, bg="white", sep_color=None)
                top_centered = _center_top_to_width(top_img.convert("RGB"), opts_row.width, bg="white")
                composite = compose_top_bottom(top_centered, opts_row, sep_px=SEP_PX)

                question = random.choice(PROMPTS_COMPOSE)

                meta_variant = {
                    "kind": "compose",
                    "measure": "big_shape_buildable_from_pieces",
                    "transforms_allowed": True,
                }

            # metadata
            answer = int(answer_index)
            meta = {
                "pattern_kind": "tiles",
                "pattern": self.name,
                "grid": (1, 4),
                "variant": meta_variant,
                "question": question,
                "answer_idx": answer,
                "answer": labels_default()[answer],
                "tiling_kind": tiling.name,
                "dims": (spec.width, spec.height),
                "out_px": tile_px,
                "color_mode": color_mode,  # NEW: surface colorization mode
                "composite_ready": True,
                "stats": {
                    "region_size": len(case.region),
                    "piece_count": len(case.pieces),
                    "piece_sizes": [len(s) for s in case.pieces],
                },
            }

            complexity = self._compute_complexity(len(case.region))
            meta["complexity"] = complexity
            meta["complexity_score"] = complexity["score"]
            meta["complexity_level"] = complexity["level"]
            meta["complexity_version"] = complexity["version"]

            return composite, [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
