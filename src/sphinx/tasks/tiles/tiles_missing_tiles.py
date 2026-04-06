# sphinx/tasks/tiles/tiles_missing_tiles.py
from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sphinx.base import Task
from .common import _max_wh_for, _out_px_for_dims
from sphinx.registry import register_task
from sphinx.config import OUT_CELL, MAX_BUILD_RETRIES
from ...utils.rng import choice_weighted
from PIL import Image, ImageDraw
import numpy as np

from sphinx.tilings import (
    TilingSpec, create_tiling, Colorer, build_dual_graph,
)

# Compose helpers
from ...utils.drawing import (
    load_font, labels_default,
    compose_top_bottom,
)


from ...utils.drawing import add_tile_border
# ----------------------------------------------------------------------
# Two variants
MISSING_VARIANT_WEIGHTS = {
    "color": 1.0,
    "shape": 1.0,
}

# Tilings
TILING_WEIGHTS_COLOR = {"square": 1.0, "triangular": 0.75, "hexagonal": 0.75, "rhombille": 0.25}
TILING_WEIGHTS_SHAPE = {"square": 1.0, "triangular": 0.75, "hexagonal": 0.75, "rhombille": 0.25}

# Cut size
CUT_FRAC_MIN = 0.08
CUT_FRAC_MAX = 0.14
CUT_MIN = 3
CUT_MAX = 24
OVERLAY_PAD_PX = 4  # guard band so strokes never touch overlay edge
MIN_TILING_WH = 4   # a little larger than the default for nicer shapes

PUZZLE_BG = "white"
OUTLINE_PX = 1

# Spacing
SEP_PX = 40
SAFE_OPT_FRAC = 0.80

# ----------------------------- prompts -----------------------------
PROMPTS_COLOR = [
    "The top panel shows a figure with some tiles left blank. Which option (a)–(d) fills the blanks with the correct colors?",
    "In the top image, a region of tiles is missing. Which option (a)–(d) restores the original colors of that region?",
    "Some tiles are blank in the top panel. Orientation is fixed and only colors vary. Which option (a)–(d) completes the figure correctly?",
    "The top row shows a shape with missing tiles. Which option (a)–(d) assigns the correct colors to the blank region?",
    "Look at the top image: one region is uncolored. Which option (a)–(d) provides the exact missing colors?",
    "The shape and position of the blank region are fixed in the top figure. Which option (a)–(d) restores its correct coloring?",
    "Only the colors differ across the options. In the top panel, which option (a)–(d) completes the blank area correctly?",
    "The top figure has a region with missing tiles. Which option (a)–(d) recovers the correct color pattern?",
    "Which option (a)–(d) in the choices matches the exact colors of the blank region in the top image?",
    "Find the option (a)–(d) whose colors correctly complete the missing tiles in the top figure."
]

# ----------------------------- prompts (shape) -----------------------------
PROMPTS_SHAPE = [
    "The top panel shows a figure with a region missing. Which option (a)–(d) is the correct piece to fill the blank? Rotations and reflections are allowed.",
    "In the top image, a connected piece has been removed. Which option (a)–(d) fills the hole exactly, possibly after rotation or flipping?",
    "The top figure contains a blank region. Which option (a)–(d) matches the missing shape? Orientation may change (rotation or reflection).",
    "Look at the top panel: some tiles are missing. Which option (a)–(d) fits the blank region exactly, allowing rotations and flips?",
    "A piece is missing from the top figure. Which option (a)–(d) completes the puzzle when rotated or mirrored if needed?",
    "The top image shows a hole where a piece is missing. Which option (a)–(d) is the correct piece, with rotation/reflection allowed?",
    "Which option (a)–(d) matches the exact shape of the missing region in the top panel, possibly after turning or mirroring?",
    "The top panel has a blank. Which option (a)–(d) is the missing piece, if it can be rotated or flipped to fit?",
    "In the top figure, a shape is missing. Which option (a)–(d) corresponds to the missing piece when orientation changes are permitted?",
    "Look at the blank region in the top image. Which option (a)–(d) is the correct piece to complete the shape, allowing rotations or reflections?"
]


# ----------------------------------------------------------------------
# Helpers

# ----------------------- exact tilability (square) -----------------------

def _square_D4_orientations(coords: Set[Tuple[int,int]]) -> List[Set[Tuple[int,int]]]:
    """All unique D4 orientations for a shape given as (x,y) integer coords."""
    def rot90(pt): x,y = pt; return (-y, x)
    def rot180(pt): x,y = pt; return (-x, -y)
    def rot270(pt): x,y = pt; return (y, -x)
    def reflx(pt): x,y = pt; return (-x, y)   # reflect across y-axis
    mats = [
        lambda p: p,                 # id
        lambda p: rot90(p),
        lambda p: rot180(p),
        lambda p: rot270(p),
        lambda p: reflx(p),
        lambda p: rot90(reflx(p)),
        lambda p: rot180(reflx(p)),
        lambda p: rot270(reflx(p)),
    ]
    seen, out = set(), []
    for f in mats:
        Q = [f(pt) for pt in coords]
        minx = min(x for x,_ in Q); miny = min(y for _,y in Q)
        N = {(x - minx, y - miny) for (x,y) in Q}
        key = tuple(sorted(N))
        if key not in seen:
            seen.add(key); out.append(N)
    return out

def _coords_of_ids_square(patch, ids: Sequence[int]) -> Set[Tuple[int,int]]:
    out = set()
    for cid in ids:
        i, j = patch.cells[cid].coord[:2]
        out.add((int(i), int(j)))
    return out

def _placements_of_piece_in_target_square(piece_ids: Set[int], target_ids: Set[int], patch) -> List[Set[int]]:
    """All placements of a piece (D4 + translations) that fit entirely inside the target shape."""
    piece_xy  = _coords_of_ids_square(patch, list(piece_ids))
    target_xy = _coords_of_ids_square(patch, list(target_ids))
    xy_to_id  = {(int(patch.cells[c].coord[0]), int(patch.cells[c].coord[1])): int(c) for c in target_ids}

    placements: List[Set[int]] = []
    if not piece_xy or not target_xy:
        return placements

    T = set(target_xy)
    for Q in _square_D4_orientations(piece_xy):
        ax, ay = min(Q)  # anchor in the piece
        for tx, ty in T:  # try anchoring to each target coord
            dx, dy = tx - ax, ty - ay
            placed = {(x + dx, y + dy) for (x, y) in Q}
            if placed.issubset(T):
                try:
                    placements.append({xy_to_id[p] for p in placed})
                except KeyError:
                    pass
    return placements

def _can_tile_target_with_pieces_square(patch, pieces: List[Set[int]], target: Set[int], *, limit_solutions: int = 1) -> bool:
    """
    Decide if 'pieces' can tile 'target' exactly on a square grid, allowing D4 transforms and translations.
    Solves an exact-cover instance via Algorithm X. Returns True if ≥1 solution exists.
    """
    target_cells = set(target)
    piece_cols = [f"piece_{k}" for k in range(len(pieces))]
    columns = list(target_cells) + piece_cols

    rows: List[Set[Any]] = []
    from collections import defaultdict
    col_to_rows: Dict[Any, List[int]] = defaultdict(list)

    # generate rows (placements)
    for k, pset in enumerate(pieces):
        plcs = _placements_of_piece_in_target_square(pset, target_cells, patch)
        if not plcs:
            return False
        for cells in plcs:
            row = set(cells)
            row.add(piece_cols[k])        # one-of constraint for this piece
            idx = len(rows)
            rows.append(row)
            for col in row:
                col_to_rows[col].append(idx)

    # Algorithm X
    used_cols: Set[Any] = set()
    def choose_col():
        best_c, best_len = None, 10**9
        for c in columns:
            if c in used_cols: continue
            cand = [r for r in col_to_rows[c] if rows[r].isdisjoint(used_cols)]
            if len(cand) < best_len:
                best_c, best_len = c, len(cand)
                if best_len <= 1: break
        return best_c

    def backtrack() -> bool:
        if len(used_cols) == len(columns):
            return True
        c = choose_col()
        if c is None:
            return all(pc in used_cols for pc in piece_cols)
        for r in list(col_to_rows[c]):
            row = rows[r]
            if not row.isdisjoint(used_cols):
                continue
            prev = set(used_cols)
            used_cols.update(row)
            if backtrack():
                return True
            used_cols.clear(); used_cols.update(prev)
        return False

    return backtrack()

def _class_id(tiling, cell) -> int:
    """Return parity/Wythoffian class index for a cell."""
    if getattr(tiling, "supports_wythoffian", False):
        return int(tiling.wythoffian_class_id(cell))
    i, j = cell.coord[:2]
    return (int(i) + int(j)) & 1

def _count_classes(tiling_name: str) -> int:
    """Number of coloring classes for the given tiling."""
    return 4 if tiling_name == "square" else 3

def _connected_bfs(rng: random.Random, g: Dict[int, Set[int]], k: int) -> List[int]:
    """Return up to k node ids forming a connected region via BFS."""
    if not g:
        return []
    start = rng.randrange(len(g))
    seen = {start}
    order = [start]
    q = [start]
    while q and len(order) < k:
        u = q.pop(0)
        for v in g[u]:
            if v not in seen:
                seen.add(v)
                order.append(v)
                if len(order) >= k:
                    break
                q.append(v)
    return order

def _pad_transform(TX, canvas_wh, pad_px=OVERLAY_PAD_PX):
    """Return a padded transform and a larger canvas (W+2p, H+2p)."""
    W, H = canvas_wh
    def TXp(pt):
        x, y = TX(pt)
        return x + pad_px, y + pad_px
    return TXp, (W + 2 * pad_px, H + 2 * pad_px)

# ---- labeling and centered row -------------------------------------------------

def _label_option_fixed_width(card_rgba: "Image.Image", label: str, font) -> "Image.Image":
    """Attach a centered text label below an option card."""
    W = card_rgba.width
    x0, y0, x1, y1 = font.getbbox(label)
    text_w, text_h = (x1 - x0), (y1 - y0)
    pad_y = max(6, OUTLINE_PX + 4)

    H = card_rgba.height + pad_y + text_h
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    canvas.alpha_composite(card_rgba, (0, 0))
    draw = ImageDraw.Draw(canvas)
    tx = (W - text_w) // 2
    ty = card_rgba.height + pad_y - y0
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return canvas

def _compose_options_row_centered(
    tiles: List["Image.Image"],
    *,
    sep: int = SEP_PX // 2,
    cell_pad: int = 16,
    bg: str = "white",
    sep_color: Optional[Tuple[int, int, int]] = (50, 50, 50),
) -> "Image.Image":
    """Arrange option tiles in a centered row with separators."""
    from PIL import Image, ImageDraw
    if not tiles:
        return Image.new("RGB", (1, 1), bg)

    max_w = max(t.width for t in tiles)
    max_h = max(t.height for t in tiles)
    cell_w = max_w + 2 * cell_pad
    cell_h = max_h + 2 * cell_pad

    W = len(tiles) * cell_w + (len(tiles) - 1) * sep
    H = cell_h
    row = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(row)

    x = 0
    for i, im in enumerate(tiles):
        cell = Image.new("RGBA", (cell_w, cell_h), (255, 255, 255, 255))
        px = (cell_w - im.width) // 2
        py = (cell_h - im.height) // 2
        cell.alpha_composite(im, (px, py))
        row.paste(cell.convert("RGB"), (x, 0))
        x += cell_w
        if i < len(tiles) - 1:
            # Draw a separator bar only if requested
            if sep_color is not None:
                draw.rectangle([x, 0, x + sep - 1, H - 1], fill=sep_color)
            x += sep
    return row

# --- renderers ------------------------------------------------------------------

def _render_shape_uniform_edges(
    patch, TX, canvas_wh, ids: Sequence[int], fill_by_cell: Dict[int, str]
) -> "Image.Image":
    W, H = canvas_wh
    rgba = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(rgba)

    polys = patch.cell_polygons()

    # Fills (no outline)
    for cid in ids:
        pts = [TX(p) for p in polys[cid]]
        draw.polygon(pts, fill=fill_by_cell.get(cid, "#CCCCCC"))

    # Stroke each unique edge exactly once
    def edge_key(a, b):
        return tuple(sorted((a, b)))
    edges: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
    for cid in ids:
        P = [TX(p) for p in polys[cid]]
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            edges.add(edge_key(a, b))
    for (a, b) in edges:
        draw.line([a, b], fill="black", width=OUTLINE_PX)

    return rgba

def _touches_edge_nonwhite(img_rgba: "Image.Image", edge: int = 2) -> bool:
    arr = np.asarray(img_rgba.convert("RGBA"))
    H, W, _ = arr.shape
    e = max(1, int(edge))
    rgb = arr[..., :3]
    nonwhite = (rgb != 255).any(axis=-1)
    return (
        nonwhite[:e, :].any()
        or nonwhite[H - e :, :].any()
        or nonwhite[:, :e].any()
        or nonwhite[:, W - e :].any()
    )

def _fit_into_box(
    img_rgba: "Image.Image",
    box_px: int,
    *,
    min_margin_px: int = 4,
    shrink_step: float = 0.92,
    edge_guard_px: int = 2,
) -> "Image.Image":
    from PIL import Image
    box_px = int(box_px)
    scale = 1.0
    for _ in range(40):
        max_w = max(1, box_px - 2 * min_margin_px)
        max_h = max_w
        w = int(round(img_rgba.width * scale))
        h = int(round(img_rgba.height * scale))
        if w > max_w or h > max_h:
            s = min(max_w / max(1, w), max_h / max(1, h))
            scale *= s
            w = max(1, int(round(img_rgba.width * scale)))
            h = max(1, int(round(img_rgba.height * scale)))

        scaled = img_rgba if scale == 1.0 else img_rgba.resize((w, h), Image.NEAREST)
        card = Image.new("RGBA", (box_px, box_px), (255, 255, 255, 255))
        card.alpha_composite(scaled, ((box_px - w) // 2, (box_px - h) // 2))

        if not _touches_edge_nonwhite(card, edge=edge_guard_px):
            return card
        scale *= shrink_step

    # Fallback
    w = max(1, int(round(img_rgba.width * 0.6)))
    h = max(1, int(round(img_rgba.height * 0.6)))
    tiny = img_rgba.resize((w, h), Image.NEAREST)
    card = Image.new("RGBA", (box_px, box_px), (255, 255, 255, 255))
    card.alpha_composite(tiny, ((box_px - w) // 2, (box_px - h) // 2))
    return card

def _render_puzzle_with_blank(tiling, patch, TX, canvas_wh, cut_ids: Sequence[int]) -> "Image.Image":
    W, H = canvas_wh
    im = Image.new("RGB", (W, H), PUZZLE_BG)
    draw = ImageDraw.Draw(im)
    polys = patch.cell_polygons()
    missing = set(cut_ids)
    for cid, poly in enumerate(polys):
        pts = [TX(p) for p in poly]
        fill = "white" if cid in missing else getattr(patch.cells[cid], "color", "#CCCCCC")
        draw.polygon(pts, fill=fill, outline="black", width=OUTLINE_PX)
    return im

def _render_overlay_and_bbox(overlay_rgba: "Image.Image") -> Tuple["Image.Image", Tuple[int,int,int,int]]:
    arr = np.asarray(overlay_rgba)
    alpha = arr[..., 3]
    ys, xs = (alpha > 0).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return overlay_rgba, (0, 0, 1, 1)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    return overlay_rgba, (bx0, by0, bx1, by1)

def _center_option_tile_from_overlay(
    overlay_rgba: "Image.Image",
    bbox: Tuple[int,int,int,int],
    *,
    min_px: int = 96,
    max_px: Optional[int] = None,
    pad_frac: float = 0.10,
) -> "Image.Image":
    from PIL import Image
    bx0, by0, bx1, by1 = bbox
    bx0 = max(0, bx0); by0 = max(0, by0)
    bx1 = min(overlay_rgba.width, bx1); by1 = min(overlay_rgba.height, by1)
    cw, ch = max(1, bx1 - bx0), max(1, by1 - by0)

    pad = max(2 + OUTLINE_PX, int(round(pad_frac * max(cw, ch))))
    target = max(min_px, int(max(cw, ch) + 2 * pad))
    if max_px is not None:
        target = min(target, int(max_px))

    crop = overlay_rgba.crop((bx0, by0, bx1, by1))
    inner = max(1, target - 2 * pad)
    scale = min(1.0, inner / max(cw, ch))  # downscale only
    if scale < 1.0:
        nw, nh = int(round(cw * scale)), int(round(ch * scale))
        crop = crop.resize((max(1, nw), max(1, nh)), Image.NEAREST)

    tile = Image.new("RGBA", (target, target), (255, 255, 255, 255))
    x = (target - crop.width) // 2
    y = (target - crop.height) // 2
    tile.alpha_composite(crop, (x, y))
    return tile

def _center_option_tile_from_overlay_noshrink(
    overlay_rgba: "Image.Image",
    bbox: Tuple[int,int,int,int],
    *,
    min_px: int = 96,
    max_px: Optional[int] = None,
    pad_frac: float = 0.10,
) -> "Image.Image":
    from PIL import Image
    bx0, by0, bx1, by1 = bbox
    bx0 = max(0, bx0); by0 = max(0, by0)
    bx1 = min(overlay_rgba.width,  bx1)
    by1 = min(overlay_rgba.height, by1)

    cw, ch = max(1, bx1 - bx0), max(1, by1 - by0)
    crop = overlay_rgba.crop((bx0, by0, bx1, by1))

    base = max(cw, ch)
    pad = max(int(round(pad_frac * base)), 2 + OUTLINE_PX)
    target = max(min_px, base + 2 * pad)

    if max_px is not None and target > max_px:
        pad = max(2 + OUTLINE_PX, (int(max_px) - base) // 2)
        target = max(base + 2 * pad, min_px)

    tile = Image.new("RGBA", (int(target), int(target)), (255, 255, 255, 255))
    x = (tile.width  - crop.width)  // 2
    y = (tile.height - crop.height) // 2
    tile.alpha_composite(crop, (x, y))
    return tile

def _ensure_margin_no_shrink(card_rgba: "Image.Image", edge: int = 2, step_px: int = 6, max_steps: int = 6) -> "Image.Image":
    if not _touches_edge_nonwhite(card_rgba, edge=edge):
        return card_rgba
    from PIL import Image
    w, h = card_rgba.size
    for _ in range(max_steps):
        w += 2 * step_px
        h += 2 * step_px
        bigger = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        bigger.alpha_composite(card_rgba, ((w - card_rgba.width)//2, (h - card_rgba.height)//2))
        card_rgba = bigger
        if not _touches_edge_nonwhite(card_rgba, edge=edge):
            break
    return card_rgba

def _center_top_to_width(top_img: "Image.Image", target_width: int, bg="white") -> "Image.Image":
    from PIL import Image
    W = max(target_width, top_img.width)
    H = top_img.height
    canvas = Image.new("RGB", (W, H), bg)
    x = (W - top_img.width) // 2
    canvas.paste(top_img, (x, 0))
    return canvas

def _recenter_square_card(img_rgba: "Image.Image", pad_px: int = 2) -> "Image.Image":
    from PIL import Image
    w, h = img_rgba.size
    side = max(w, h) + 2 * max(pad_px, OUTLINE_PX + 1)
    card = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    card.alpha_composite(img_rgba, ((side - w)//2, (side - h)//2))
    return card

# ------- symmetry groups and shape signature -----------------------------------

def _transforms_D4() -> List[Tuple[float,float,float,float]]:
    # rotations R0..R270 and one reflection F=(x,-y) composed with all R
    mats = []
    rots = [
        (1, 0, 0, 1),          # 0°
        (0, -1, 1, 0),         # 90°
        (-1, 0, 0, -1),        # 180°
        (0, 1, -1, 0),         # 270°
    ]
    mats.extend(rots)
    # reflections
    F = (1, 0, 0, -1)  # reflect across x-axis
    for (a,b,c,d) in rots:
        mats.append((F[0]*a + F[1]*c, F[0]*b + F[1]*d,
                     F[2]*a + F[3]*c, F[2]*b + F[3]*d))
    return mats  # 8 total

def _transforms_D6() -> List[Tuple[float,float,float,float]]:
    mats = []
    # rotations by 60° steps
    for k in range(6):
        th = (math.pi/3.0) * k
        c, s = math.cos(th), math.sin(th)
        mats.append((c, -s, s, c))
    # reflections: reflect across x-axis, then rotate
    F = (1, 0, 0, -1)
    rots = mats[:6]
    for (a,b,c,d) in rots:
        mats.append((F[0]*a + F[1]*c, F[0]*b + F[1]*d,
                     F[2]*a + F[3]*c, F[2]*b + F[3]*d))
    return mats  # 12 total

def _dihedral_mats_for_tiling(tiling_name: str) -> List[Tuple[float,float,float,float]]:
    if tiling_name == "square":
        return _transforms_D4()
    if tiling_name in ("triangular", "hexagonal", "rhombille"):
        return _transforms_D6()
    # default conservative
    return _transforms_D4()

def _shape_signature_lattice(patch, tiling_name: str, ids: Sequence[int]) -> Tuple[Tuple[float,float], ...]:
    """
    Canonical signature of a connected set of cells, invariant under the
    dihedral group appropriate for the tiling (D4 or D6). We use cell
    centroids from the tiling geometry (not the integer coords) so the
    method works across different lattices.
    """
    polys = patch.cell_polygons()
    pts: List[Tuple[float,float]] = []
    for i in ids:
        poly = polys[i]
        cx = sum(x for (x, _) in poly) / len(poly)
        cy = sum(y for (_, y) in poly) / len(poly)
        pts.append((cx, cy))

    # center at centroid for rotation invariance
    mx = sum(x for x,_ in pts) / len(pts)
    my = sum(y for _,y in pts) / len(pts)
    P0 = [(x - mx, y - my) for (x,y) in pts]

    mats = _dihedral_mats_for_tiling(tiling_name)
    cands = []
    for (a,b,c,d) in mats:
        X = [(a*x + b*y, c*x + d*y) for (x,y) in P0]
        minx = min(x for (x,_) in X)
        miny = min(y for (_,y) in X)
        N = [(x - minx, y - miny) for (x,y) in X]
        # round to a stable precision (tiling coordinates are exact up to sqrt(3))
        key = tuple(sorted((round(x, 6), round(y, 6)) for (x,y) in N))
        cands.append(key)
    return min(cands)

# Pixel‑crisp card transforms (avoid 60° on raster)
def _apply_random_card_transform(img: "Image.Image") -> "Image.Image":
    # Only crisp operations: identity / 180° / horizontal / vertical flips
    ops = ["id", "r180", "fh", "fv"]
    t = random.choice(ops)
    if t == "id":
        return img
    if t == "r180":
        return img.rotate(180, expand=True, resample=Image.NEAREST)
    if t == "fh":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if t == "fv":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def _shape_prompts_for(tiling_name: str) -> List[str]:
    return PROMPTS_SHAPE

def _allowed_transforms_str(tiling_name: str) -> str:
    if tiling_name == "square":
        return "D4 (rot 0/90/180/270, mirror)"
    if tiling_name in ("triangular", "hexagonal", "rhombille"):
        return "D6 (rot 0/60/120/180/240/300, mirror)"
    return "rotations/reflections"

# ----------------------------------------------------------------------
@register_task
class TilesMissingTilesTask(Task):
    name = "tiles_missing_tiles"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, cell_count: int) -> Dict[str, Any]:
        """Normalize board cell count identically to tiles_geometry."""
        min_cells = MIN_TILING_WH * MIN_TILING_WH
        # We do not have enough large tile sizes; cap at 64 cells (matching tiles_geometry).
        max_cells = 64
        span = max(1, max_cells - min_cells)
        normalized = (int(cell_count) - min_cells) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "tiles-missing-tiles-board-cells-v1",
            "range": {"min_cells": min_cells, "max_cells": max_cells},
            "cell_count": int(cell_count),
        }

    def _sample_tiling(self, rng: random.Random, variant: str):
        """Sample a tiling and color it according to the chosen variant."""
        if variant == "shape":
            names = list(TILING_WEIGHTS_SHAPE.keys()); weights = [TILING_WEIGHTS_SHAPE[n] for n in names]
        else:
            names = list(TILING_WEIGHTS_COLOR.keys()); weights = [TILING_WEIGHTS_COLOR[n] for n in names]
        tname = choice_weighted(rng, names, weights)
        tiling = create_tiling(tname)

        # square canvas (width == height)
        hi = _max_wh_for(tiling.name)
        side = rng.randint(MIN_TILING_WH, hi)

        seed = rng.randint(0, 2 ** 31 - 1)
        if variant == "shape":
            spec = TilingSpec(tiling.name, seed, width=side, height=side, uniform={"scheme": "same"})
        else:
            spec = TilingSpec(tiling.name, seed, width=side, height=side, uniform={"scheme": "wythoffian"})

        patch = tiling.generate(spec)
        Colorer().apply(tiling, patch, spec)

        g = build_dual_graph(patch, connect_on_touch=False)
        return tiling, spec, patch, g

    # ----------------------------- public API -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # choose variant
        kinds = list(MISSING_VARIANT_WEIGHTS.keys())
        w = [MISSING_VARIANT_WEIGHTS[k] for k in kinds]
        variant = choice_weighted(rng, kinds, w)

        for _ in range(self.max_retries):
            tiling, spec, patch, g = self._sample_tiling(rng, variant)
            n = len(patch.cells)
            if n < CUT_MIN + 4:
                continue

            # Choose cut size
            kmin = max(CUT_MIN, int(round(CUT_FRAC_MIN * n)))
            kmax = min(CUT_MAX, int(round(CUT_FRAC_MAX * n)))
            if kmin > kmax:
                kmin, kmax = max(3, min(6, n // 6)), max(3, min(8, n // 5))
            k = rng.randint(kmin, kmax)

            # Connected removal; for "color" require at least two classes present
            ok_cut: Optional[List[int]] = None
            for _inner in range(self.max_retries * 3):
                cand = _connected_bfs(rng, g, k)
                if not cand or len(cand) < CUT_MIN:
                    continue
                if variant == "color":
                    classes = {_class_id(tiling, patch.cells[c]) for c in cand}
                    if len(classes) < 2:
                        continue
                ok_cut = cand
                break
            if ok_cut is None:
                continue
            cut_ids = ok_cut

            tile_px = _out_px_for_dims(spec.width, spec.height)
            if max(spec.width, spec.height) <= 6:
                tile_px = int(tile_px * 1.25)
            TX, canvas_wh = self._build_transform_shared(patch, target_px=tile_px, margin_frac=0.06)
            puzzle_img = _render_puzzle_with_blank(tiling, patch, TX, canvas_wh, cut_ids)

            # ---------------- COLOR variant ----------------
            if variant == "color":
                base_map = self._class_to_color_map(patch, tiling)
                C = _count_classes(tiling.name)

                # Generate swap maps, but enforce uniqueness restricted to classes used in the cut
                used_classes = sorted({_class_id(tiling, patch.cells[c]) for c in cut_ids})
                def proj_key(cmap: Dict[int, str]) -> Tuple[str, ...]:
                    return tuple(cmap[i] for i in used_classes)

                swap_maps = self._pair_swap_maps(base_map)
                rng.shuffle(swap_maps)

                maps_all = [base_map]
                seen = {proj_key(base_map)}
                for m in swap_maps:
                    pk = proj_key(m)
                    if pk not in seen:
                        maps_all.append(m)
                        seen.add(pk)
                    if len(maps_all) >= 4:
                        break

                if len(maps_all) < 4:
                    continue

                order = list(range(4)); rng.shuffle(order)
                maps_shuf = [maps_all[i] for i in order]
                answer_index = order.index(0)
                correct_label = labels_default()[answer_index]

                raw_option_tiles: List["Image.Image"] = []
                safe_box = int(tile_px * SAFE_OPT_FRAC)
                for cmap in maps_shuf:
                    fill_by_cell = {cid: cmap[_class_id(tiling, patch.cells[cid])] for cid in cut_ids}
                    TXp, canvas_wh_p = _pad_transform(TX, canvas_wh)
                    overlay = _render_shape_uniform_edges(patch, TXp, canvas_wh_p, cut_ids, fill_by_cell)
                    overlay, bbox = _render_overlay_and_bbox(overlay)

                    opt_rgba = _center_option_tile_from_overlay_noshrink(
                        overlay, bbox,
                        min_px=max(96, OUT_CELL // 1),
                        max_px=max(96, safe_box - 6),
                        pad_frac=0.10,
                    )
                    opt_rgba = _ensure_margin_no_shrink(opt_rgba, edge=2, step_px=8)
                    # NEW: add crisp border like sequence tasks
                    opt_rgba = add_tile_border(opt_rgba)
                    raw_option_tiles.append(opt_rgba)


                font = load_font()
                labels = labels_default()
                labeled_opts = [_label_option_fixed_width(im.convert("RGBA"), lab, font)
                                for im, lab in zip(raw_option_tiles, labels)]
                opts_row = _compose_options_row_centered(labeled_opts, sep=SEP_PX // 2, cell_pad=16, bg="white", sep_color=None)

                top_centered = _center_top_to_width(puzzle_img, opts_row.width, bg="white")
                composite = compose_top_bottom(top_centered, opts_row, sep_px=SEP_PX)

                question = rng.choice(PROMPTS_COLOR)
                complexity = self._compute_complexity(len(patch.cells))

                meta = {
                    "pattern_kind": "tiles",
                    "pattern": self.name,
                    "grid": (1, 4),
                    "variant": {
                        "kind": "color",
                        "measure": "identify_missing_colors",
                        "scope": "missing_region_color_only",
                        "transforms_allowed": False,
                    },
                    "question": question,
                    "answer": correct_label,
                    "answer_index": int(answer_index),
                    "tiling_kind": tiling.name,
                    "dims": (spec.width, spec.height),
                    "out_px": tile_px,
                    "composite_ready": True,
                    "indices": {"cut_ids": list(cut_ids)},
                    "class_count": C,
                    "complexity": complexity,
                    "complexity_score": complexity["score"],
                    "complexity_level": complexity["level"],
                    "complexity_version": complexity["version"],
                }
                return composite, [spec], meta

            # ---------------- SHAPE variant ----------------
            board_color = getattr(patch.cells[0], "color", "#CCCCCC")

            # Canonical signature under the correct dihedral group
            target_sig = _shape_signature_lattice(patch, tiling.name, cut_ids)

            def sample_unique_same_size(sig_forbid: Set[Tuple[Tuple[float,float], ...]], want_size: int) -> Optional[Set[int]]:
                for _ in range(200):
                    cand = _connected_bfs(rng, g, want_size)
                    if not cand or len(cand) != want_size:
                        continue
                    sig = _shape_signature_lattice(patch, tiling.name, cand)
                    if sig not in sig_forbid:
                        sig_forbid.add(sig)
                        return set(cand)
                return None

            forbid = {target_sig}
            distractor_sets: List[Set[int]] = []

            sizes = []
            if k > CUT_MIN: sizes.append(k - 1)
            if k + 1 <= min(CUT_MAX, n - 1): sizes.append(k + 1)
            rng.shuffle(sizes)
            got = None
            for s in sizes:
                got = sample_unique_same_size(forbid, s)
                if got is not None:
                    distractor_sets.append(got); break

            while len(distractor_sets) < 3:
                cand = sample_unique_same_size(forbid, k)
                if cand is None:
                    break
                distractor_sets.append(cand)

            if len(distractor_sets) < 3:
                continue

            option_sets = [set(cut_ids)] + distractor_sets[:3]
            order = list(range(4)); rng.shuffle(order)
            sets_shuf = [option_sets[i] for i in order]
            answer_index = order.index(0)
            correct_label = labels_default()[answer_index]

            raw_option_tiles: List["Image.Image"] = []
            safe_box = int(tile_px * SAFE_OPT_FRAC)
            for sids in sets_shuf:
                fill_by_cell = {cid: board_color for cid in sids}
                TXp, canvas_wh_p = _pad_transform(TX, canvas_wh)
                overlay = _render_shape_uniform_edges(patch, TXp, canvas_wh_p, list(sids), fill_by_cell)
                overlay, bbox = _render_overlay_and_bbox(overlay)

                opt_rgba = _center_option_tile_from_overlay_noshrink(
                    overlay, bbox,
                    min_px=max(96, OUT_CELL // 1),
                    max_px=max(96, safe_box - 6),
                    pad_frac=0.10,
                )
                # crisp visual variety, but avoid 60°/30° raster rotations
                opt_rgba = _apply_random_card_transform(opt_rgba)
                opt_rgba = _ensure_margin_no_shrink(opt_rgba, edge=2, step_px=8)
                # NEW: add crisp border like sequence tasks
                opt_rgba = add_tile_border(opt_rgba)
                raw_option_tiles.append(opt_rgba)

            font = load_font()
            labels = labels_default()
            labeled_opts = [_label_option_fixed_width(im.convert("RGBA"), lab, font)
                            for im, lab in zip(raw_option_tiles, labels)]
            opts_row = _compose_options_row_centered(labeled_opts, sep=SEP_PX // 2, cell_pad=16, bg="white", sep_color=None)

            top_centered = _center_top_to_width(puzzle_img, opts_row.width, bg="white")
            composite = compose_top_bottom(top_centered, opts_row, sep_px=SEP_PX)

            question = rng.choice(_shape_prompts_for(tiling.name))
            complexity = self._compute_complexity(len(patch.cells))

            meta = {
                "pattern_kind": "tiles",
                "pattern": self.name,
                "grid": (1, 4),
                "variant": {"kind": "shape", "measure": "identify_missing_shape",
                            "scope": "missing_region_shape_only",
                            "transforms_allowed": True, "allowed_transforms": _allowed_transforms_str(tiling.name)},
                "question": question,
                "answer": correct_label,
                "answer_index": int(answer_index),
                "tiling_kind": tiling.name,
                "dims": (spec.width, spec.height),
                "out_px": tile_px,
                "composite_ready": True,
                "indices": {"cut_ids": list(cut_ids)},
                "shape_k": int(k),
                "complexity": complexity,
                "complexity_score": complexity["score"],
                "complexity_level": complexity["level"],
                "complexity_version": complexity["version"],
            }
            return composite, [spec], meta

        raise RuntimeError(f"{self.name}: failed to sample a valid instance after {self.max_retries} attempts.")

    # ------------ small internal helpers moved onto the class -------------
    def _build_transform_shared(self, patch, target_px: int, margin_frac: float = 0.06):
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

    def _class_to_color_map(self, patch, tiling) -> Dict[int, str]:
        m: Dict[int, str] = {}
        for c in patch.cells:
            cid = _class_id(tiling, c)
            if cid not in m:
                m[cid] = getattr(c, "color", "#CCCCCC")
        return m

    def _pair_swap_maps(self, base: Dict[int, str]) -> List[Dict[int, str]]:
        keys = sorted(base.keys())
        out: List[Dict[int, str]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                m = dict(base)
                ki, kj = keys[i], keys[j]
                m[ki], m[kj] = m[kj], m[ki]
                if m != base and m not in out:
                    out.append(m)
        return out
