# sphinx/tasks/tiles/tiles_line_intersections.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from .common import MIN_TILING_WH, _max_wh_for, _out_px_for_dims
from sphinx.tilings import TilingSpec, create_tiling
from sphinx.config import MAX_BUILD_RETRIES
from ...utils.rng import choice_weighted

# --- use the exact 5 RectVenn colors (names + hex) -----------------------------
#   COLOR_POOL = [("blue","#1f77b4"), ("red","#d62728"), ("green","#2ca02c"),
#                 ("purple","#9467bd"), ("orange","#ff7f0e")]
COLOR_POOL = [
    ("blue",   "#1f77b4"),
    ("red",    "#d62728"),
    ("green",  "#2ca02c"),
    ("purple", "#9467bd"),
    ("orange", "#ff7f0e"),
]


# Build a name->hex and a flat name list for labels
_COLOR_NAME_TO_HEX = {nm: hx for (nm, hx) in COLOR_POOL}
_COLOR_NAMES = [nm for (nm, _) in COLOR_POOL]

# -----------------------------------------------------------------------------
# Config knobs
# -----------------------------------------------------------------------------

# Tiling choices: polygonal lattices with unambiguous shared edges.
TILING_WEIGHTS: Dict[str, float] = {
    "square": 1.0,
    "triangular": 1.0
}

# How many colored lines to draw (K). 2..5 (uniform by default).
LINE_COUNT_WEIGHTS: Dict[int, float] = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}

# Visuals
OUTLINE_PX = 1
LINE_PX = 5

# Length shaping (in edge steps) for each line
MIN_LINE_LEN = 3
MAX_LEN_SCALE = 2.0  # target up to ≈ 2× max(width,height)

# Make the rendered tile **larger** than the suite default
TILE_PX_SCALE = 1.35  # 35% bigger than _out_px_for_dims(...) baseline

# Desired distribution for the *correct* shared-corner count
ANSWER_MIN = 1
ANSWER_MAX = 8

# How many times to try new line sets on the *same* tiling before resampling the tiling
TARGET_MATCH_TRIES_PER_TILING = 24


# Prompts (corner/vertex-based wording; edges are never shared)
PROMPTS = [
    "How many grid corners are touched by both the {C1} and {C2} lines? Count every corner (vertex) where they meet or pass through the same point.",
    "Count the number of grid vertices that lie on both the {C1} and {C2} lines. Shared edges do not occur.",
    "At how many corners do the {C1} and {C2} lines meet (i.e., pass through the same vertex)?",
    "How many distinct grid points (tile corners) are common to the {C1} and {C2} lines?",
    "Counting shared vertices only (not edges), how many corners are on both the {C1} line and the {C2} line?",
    "How many corner points do the {C1} and {C2} lines have in common? (A corner counts if both lines pass through it.)",
    "What is the number of grid vertices where the {C1} and {C2} lines touch or cross?",
    "How many tile-corner intersections are shared by the {C1} and {C2} lines?",
    "Count the corners (grid vertices) that belong to both the {C1} and {C2} polylines.",
    "How many common corner points do {C1} and {C2} share on the grid?",
]

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

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

def _canon_pt(p: Tuple[float, float], prec: int = 6) -> Tuple[float, float]:
    return (round(float(p[0]), prec), round(float(p[1]), prec))

def _edge_key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (a, b) if a <= b else (b, a)

def _build_vertex_graph(patch) -> Tuple[
    Dict[Tuple[float, float], Set[Tuple[float, float]]],
    List[Tuple[Tuple[float, float], Tuple[float, float]]]
]:
    """Undirected graph over tile vertices; edges are shared tile edges."""
    adj: Dict[Tuple[float, float], Set[Tuple[float, float]]] = {}
    edges_set: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()

    polys = patch.cell_polygons()
    for poly in polys:
        m = len(poly)
        for i in range(m):
            a = _canon_pt(poly[i]); b = _canon_pt(poly[(i + 1) % m])
            if a == b:
                continue
            k = _edge_key(a, b)
            edges_set.add(k)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj, list(edges_set)

def _sample_simple_path(
    rng: random.Random,
    adj: Dict[Tuple[float, float], Set[Tuple[float, float]]],
    target_len: int,
    used_edges_global: Set[Tuple[Tuple[float, float], Tuple[float, float]]],
    max_tries: int = 256,
) -> Optional[List[Tuple[float, float]]]:
    """
    Sample a vertex-simple path (no repeated vertices), avoiding edges in 'used_edges_global'.
    Returns a list of vertices [v0, v1, ..., vL] with L == target_len.
    """
    verts = list(adj.keys())
    if not verts or target_len < 1:
        return None

    for _ in range(max_tries):
        start = rng.choice(verts)
        path: List[Tuple[float, float]] = [start]
        used_local_edges: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
        used_local_verts: Set[Tuple[float, float]] = {start}
        prev: Optional[Tuple[float, float]] = None

        while len(path) - 1 < target_len:
            u = path[-1]
            nbrs = list(adj.get(u, ()))
            rng.shuffle(nbrs)

            moved = False
            for v in nbrs:
                if prev is not None and v == prev:
                    continue  # avoid immediate backtrack
                if v in used_local_verts:
                    continue  # keep vertex-simple
                ek = _edge_key(u, v)
                if ek in used_local_edges or ek in used_edges_global:
                    continue

                # take the step
                used_local_edges.add(ek)
                used_local_verts.add(v)
                path.append(v)
                prev = u
                moved = True
                break

            if not moved:
                break

        if len(path) - 1 == target_len:
            used_edges_global.update(used_local_edges)
            return path

    return None

def _render_uniform_board_with_outlines(patch, TX, canvas_wh, board_color: str = "white") -> Image.Image:
    """Draw a uniformly colored board with black grid outlines."""
    W, H = canvas_wh
    img = Image.new("RGB", (W, H), board_color)
    draw = ImageDraw.Draw(img)
    polys = patch.cell_polygons()
    for poly in polys:
        pts = [TX(p) for p in poly]
        draw.polygon(pts, fill=board_color, outline="black", width=OUTLINE_PX)
    return img

def _overlay_colored_lines(img: Image.Image, TX, lines: List[List[Tuple[float, float]]], color_hexes: List[str]) -> None:
    draw = ImageDraw.Draw(img)
    for path, hx in zip(lines, color_hexes):
        pts = [TX(p) for p in path]
        draw.line(pts, fill=hx, width=LINE_PX)

# ----- intersections (shared vertices, including endpoints) --------------------

def _count_shared_vertices(p1: List[Tuple[float, float]], p2: List[Tuple[float, float]]) -> int:
    """Number of distinct vertices common to both polylines (includes endpoints)."""
    s1 = { _canon_pt(p) for p in p1 }
    s2 = { _canon_pt(p) for p in p2 }
    return len(s1 & s2)

# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------

@register_task
class TilesLineIntersectionsTask(Task):
    """
    Draw K colored polylines along shared tile edges (uniform background with grid outlines).
    Ask: how many *grid corners* (vertices) are common to two chosen lines?

    Conventions:
      • Lines are vertex-to-vertex along edges; paths are vertex-simple.
      • No two lines share an edge (global constraint).
      • Intersections are counted at vertices only (any shared corner counts, including endpoints).
      • Answer is guaranteed to be ≥ 1 (resamples otherwise).
    """
    name = "tiles_line_intersections"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, answer: int) -> Dict[str, Any]:
        """Normalize the number of shared corners using ANSWER_MIN/ANSWER_MAX."""
        min_ans = int(ANSWER_MIN)
        max_ans = min(7, int(ANSWER_MAX))
        span = max(1, max_ans - min_ans)
        normalized = (int(answer) - min_ans) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "tiles-line-intersections-answer-v1",
            "range": {"min_answer": min_ans, "max_answer": max_ans},
            "answer": int(answer),
        }

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # number of lines K (2..5)
        ks = sorted(LINE_COUNT_WEIGHTS.keys())
        kw = [LINE_COUNT_WEIGHTS[k] for k in ks]
        K = int(choice_weighted(rng, ks, kw))

        # ---- NEW: sample the *answer* uniformly first, then realize it ----
        target_shared = rng.randint(int(ANSWER_MIN), int(ANSWER_MAX))

        for _ in range(self.max_retries):
            # --- sample tiling geometry
            tnames = list(TILING_WEIGHTS.keys())
            tw = [TILING_WEIGHTS[n] for n in tnames]
            tname = choice_weighted(rng, tnames, tw)
            tiling = create_tiling(tname)

            hi = _max_wh_for(tiling.name)
            w = rng.randint(MIN_TILING_WH, hi)
            h = rng.randint(MIN_TILING_WH, hi)

            seed = rng.randint(0, 2 ** 31 - 1)
            spec = TilingSpec(tiling.name, seed, width=w, height=h, uniform={"scheme": "same"})
            patch = tiling.generate(spec)

            if len(patch.cells) < 8:
                continue

            # Build vertex graph
            adj, all_edges = _build_vertex_graph(patch)
            if not adj or len(all_edges) < 4:
                continue

            # Larger-than-default output size
            base_px = _out_px_for_dims(spec.width, spec.height)
            tile_px = int(round(base_px * TILE_PX_SCALE))
            TX, canvas_wh = _build_transform_shared(patch, target_px=tile_px, margin_frac=0.06)

            # Choose distinct colors from RectVenn palette (names for prompt; hex for drawing)
            if K <= len(_COLOR_NAMES):
                names = rng.sample(_COLOR_NAMES, K)
            else:
                names = [_COLOR_NAMES[i % len(_COLOR_NAMES)] for i in range(K)]
            hexes = [_COLOR_NAME_TO_HEX[nm] for nm in names]

            # Length shaping bounds (more length -> more chance of multiple shared corners)
            scale = max(spec.width, spec.height)
            L_max = max(MIN_LINE_LEN + 1, int(round(MAX_LEN_SCALE * float(scale))))

            # ---- Try multiple line sets on this same tiling to hit 'target_shared' exactly ----
            for _inner in range(int(TARGET_MATCH_TRIES_PER_TILING)):
                used_edges_global: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
                lines: List[List[Tuple[float, float]]] = []

                # Sample K vertex-simple paths with no shared edges
                for _k in range(K):
                    want = rng.randint(MIN_LINE_LEN, L_max)
                    path = None
                    for L in range(want, MIN_LINE_LEN - 1, -1):
                        path = _sample_simple_path(rng, adj, L, used_edges_global, max_tries=256)
                        if path is not None:
                            break
                    if path is None:
                        lines = []
                        break
                    lines.append(path)

                if len(lines) != K:
                    # try another set of lines on the same tiling
                    continue

                # Pairwise shared-corner counts
                pairs = []
                exact: List[Tuple[int, int, int]] = []
                for i in range(K):
                    for j in range(i + 1, K):
                        c = _count_shared_vertices(lines[i], lines[j])
                        pairs.append({
                            "i": i, "j": j,
                            "color_i_name": names[i],
                            "color_j_name": names[j],
                            "color_i_hex": hexes[i],
                            "color_j_hex": hexes[j],
                            "shared_corners": int(c),
                        })
                        if c == target_shared:
                            exact.append((i, j, c))

                if not exact:
                    # no pair meets the sampled target yet; try re-sampling lines again on this tiling
                    continue

                # We met the target—pick a matching pair uniformly
                qi, qj, ans = rng.choice(exact)

                # Render and return
                base = _render_uniform_board_with_outlines(patch, TX, canvas_wh, board_color="white")
                _overlay_colored_lines(base, TX, lines, hexes)

                C1 = names[qi]
                C2 = names[qj]
                question = rng.choice(PROMPTS).format(C1=C1, C2=C2)
                answer = int(ans)  # equals target_shared by construction

                complexity = self._compute_complexity(answer)
                meta = {
                    "pattern_kind": "tiles",
                    "pattern": self.name,
                    "grid": (1, 1),
                    "variant": {
                        "measure": "shared_vertex_count",
                        "scope": "two_colored_lines",
                        "definition": "count grid vertices common to both polylines (corners they touch or pass through); shared edges are not present.",
                    },
                    "question": question,
                    "answer": answer,
                    "tiling_kind": tiling.name,
                    "dims": (spec.width, spec.height),
                    "out_px": tile_px,
                    "composite_ready": True,
                    "lines": [
                        {"name": nm, "hex": hx, "length": int(len(p) - 1)}
                        for nm, hx, p in zip(names, hexes, lines)
                    ],
                    "query": {
                        "color_a_name": names[qi],
                        "color_b_name": names[qj],
                        "color_a_hex": hexes[qi],
                        "color_b_hex": hexes[qj],
                    },
                    "pairs": pairs,  # provenance for all pairwise shared-corner counts
                    "target_shared": int(target_shared),  # NEW: sampled target for audit
                    "target_range": [int(ANSWER_MIN), int(ANSWER_MAX)],
                    "complexity": complexity,
                    "complexity_score": complexity["score"],
                    "complexity_level": complexity["level"],
                    "complexity_version": complexity["version"],
                }
                return base, [spec], meta

            # If we couldn't meet the target on this tiling after several tries, resample the tiling.
            continue

        raise RuntimeError(
            f"{self.name}: failed to meet target shared-corner count={target_shared} after {self.max_retries} tiling attempts.")
