# sphinx/tasks/tiles/tiles_line_length.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from sphinx.base import Task
from sphinx.registry import register_task
from .common import MIN_TILING_WH, _max_wh_for, _out_px_for_dims  # sizing + bounds
from sphinx.tilings import TilingSpec, create_tiling
from sphinx.config import MAX_BUILD_RETRIES, COLORS_NAMES
from ...utils.rng import choice_weighted
from ...utils.colors import color_label
from PIL import Image, ImageDraw

COLORS = list(COLORS_NAMES)

# -----------------------------------------------------------------------------
# Config knobs
# -----------------------------------------------------------------------------

# Which polygonal tilings to use (lines follow *shared edges* of polygon cells).
# (We exclude circles/voronoi/orthogonal_split to avoid ambiguous edge geometry.)
TILING_WEIGHTS: Dict[str, float] = {
    "square": 1.0,
    "triangular": 1.0,
    "hexagonal": 1.0,
    "rhombille": 0.5,
}

# How many colored lines to draw; K sampled by these weights.
LINE_COUNT_WEIGHTS: Dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0, 2: 3.0, 5: 2.0}

# Visuals
OUTLINE_PX = 1
LINE_PX = 5  # stroke width for colored lines

# Length shaping (target in "edge steps")
MIN_LINE_LEN = 3  # at least 3 edges long
MAX_LEN_SCALE = 2.0  # max ≈ 2× max(width,height) edges (best-effort)

# Prompts
PROMPTS = [
    "Colored lines run along the edges of the tiling, from vertex to vertex. What is the total length of the {C} line, measured in edge steps?",
    "Each colored path follows shared tile edges and ends at corner intersections. How many edge steps long is the {C} line?",
    "Lines trace only along tile edges. What is the length of the {C} line, in number of edge steps?",
    "Count each shared-edge segment as 1. What is the total length of the {C} line?",
    "On this uniform tiling, lines follow edges between tile corners. What is the edge-step length of the {C} line?",
    "Measure the {C} line strictly along tile edges. How many tile-edge units long is it?",
    "The {C} line is a continuous path along grid edges. What is its total length in edge steps?",
    "How many edge segments make up the {C} line? Give your answer as an integer.",
    "Following shared edges only, what is the total length of the {C} line, measured in steps?",
    "Lines always start and end at tile vertices. How many edge steps does the {C} line contain?",
]


# -----------------------------------------------------------------------------
# Geometry helpers (shared-style transform; same spirit as other Tiles tasks)
# -----------------------------------------------------------------------------

def _build_transform_shared(patch, target_px: int, margin_frac: float = 0.06):
    """
    Scale the tiling's continuous coordinates to a square canvas of 'target_px'.
    Mirrors the transform pattern used in other tiles tasks so overlays align crisply.
    """
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
        # clamp
        xi = 0 if xi < 0 else (W - 1 if xi > W - 1 else xi)
        yi = 0 if yi < 0 else (H - 1 if yi > H - 1 else yi)
        return xi, yi

    return TX, (W, H)

def _canon_pt(p: Tuple[float, float], prec: int = 6) -> Tuple[float, float]:
    return (round(float(p[0]), prec), round(float(p[1]), prec))

def _edge_key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (a, b) if a <= b else (b, a)

def _build_vertex_graph(patch) -> Tuple[Dict[Tuple[float, float], Set[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    Build an undirected graph whose nodes are tile *vertices* (polygon corners),
    edges are *shared tile edges*. Uses geometric coordinates from the tiling.
    """
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
    Sample a simple path (no repeated edges) of 'target_len' edges on the vertex graph,
    avoiding edges in 'used_edges_global'. Returns a list of vertices [v0, v1, ..., vL].
    """
    verts = list(adj.keys())
    if not verts or target_len < 1:
        return None

    for _ in range(max_tries):
        path: List[Tuple[float, float]] = [rng.choice(verts)]
        used_local: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
        prev: Optional[Tuple[float, float]] = None

        while len(path) - 1 < target_len:
            u = path[-1]
            nbrs = list(adj.get(u, ()))
            rng.shuffle(nbrs)
            moved = False
            for v in nbrs:
                if prev is not None and v == prev:
                    continue  # avoid immediate backtrack
                ek = _edge_key(u, v)
                if ek in used_local or ek in used_edges_global:
                    continue
                # take this step
                used_local.add(ek)
                path.append(v)
                prev = u
                moved = True
                break
            if not moved:
                break  # dead end: restart from scratch

        if len(path) - 1 == target_len:
            # success
            used_edges_global.update(used_local)
            return path

    return None

def _render_uniform_board_with_outlines(patch, TX, canvas_wh, board_color: str = "white") -> Image.Image:
    """
    Draw a uniformly colored board with black grid outlines.
    """
    W, H = canvas_wh
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    polys = patch.cell_polygons()
    for poly in polys:
        pts = [TX(p) for p in poly]
        draw.polygon(pts, fill=board_color, outline="black", width=OUTLINE_PX)
    return img

def _overlay_colored_lines(img: Image.Image, TX, lines: List[List[Tuple[float, float]]], colors: List[str]) -> None:
    draw = ImageDraw.Draw(img)
    for path, col in zip(lines, colors):
        pts = [TX(p) for p in path]
        # Draw the polyline; endpoints land on vertices by construction
        draw.line(pts, fill=col, width=LINE_PX)

# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------

@register_task
class TilesLineLengthTask(Task):
    """
    Draw K colored polylines along *tile edges* on a uniformly colored tiling.
    Each line is a simple path between tile vertices; segments lie on shared edges.
    Ask for the length (edge-step count) of one specified colored line.

    Returns a single image (1×1 grid), integer answer.
    """
    name = "tiles_line_length"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, width: int, height: int, tiling_name: str) -> Dict[str, Any]:
        """Normalize board size (cell count) to EASY/HARD complexity."""
        min_cells = MIN_TILING_WH * MIN_TILING_WH
        max_cells = 64
        span = max(1, max_cells - min_cells)
        cell_count = max(1, int(width) * int(height))
        normalized = (cell_count - min_cells) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "tiles-line-length-board-cells-v1",
            "range": {"min_cells": min_cells, "max_cells": max_cells},
            "cell_count": int(cell_count),
            "width": int(width),
            "height": int(height),
        }

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # Sample K = number of lines
        ks = sorted(LINE_COUNT_WEIGHTS.keys())
        kw = [LINE_COUNT_WEIGHTS[k] for k in ks]
        K = int(choice_weighted(rng, ks, kw))

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

            n_cells = len(patch.cells)
            if n_cells < 8:
                continue

            # Build vertex graph (primal edges) from polygons
            adj, all_edges = _build_vertex_graph(patch)
            if not adj or len(all_edges) < 4:
                continue

            # Image size
            tile_px = _out_px_for_dims(spec.width, spec.height)
            TX, canvas_wh = _build_transform_shared(patch, target_px=tile_px, margin_frac=0.06)

            # Choose distinct colors for K lines
            kmax = len(COLORS)
            if kmax >= K:
                idxs = rng.sample(range(kmax), K)
                line_colors = [COLORS[i] for i in idxs]
            else:
                # Fallback if palette small
                line_colors = [COLORS[i % kmax] for i in range(K)]

            # Target lengths
            scale = max(spec.width, spec.height)
            L_max = max(MIN_LINE_LEN + 1, int(round(MAX_LEN_SCALE * float(scale))))
            # Sample K targets, then realize simple paths avoiding edge reuse
            used_edges_global: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
            lines: List[List[Tuple[float, float]]] = []
            lengths: List[int] = []

            for _k in range(K):
                # draw target from [MIN_LINE_LEN, L_max]
                target = rng.randint(MIN_LINE_LEN, L_max)
                path = None
                # try to realize 'target', then progressively shrink if needed
                for L in range(target, MIN_LINE_LEN - 1, -1):
                    path = _sample_simple_path(rng, adj, L, used_edges_global, max_tries=256)
                    if path is not None:
                        break
                if path is None:
                    break  # fail this tiling; resample outside
                lines.append(path)
                lengths.append(len(path) - 1)

            if len(lines) != K:
                continue  # resample tiling

            # Render board + overlay lines
            # Background board color: keep uniform, slightly off-white for contrast
            board_color = "white"
            base = _render_uniform_board_with_outlines(patch, TX, canvas_wh, board_color=board_color)
            _overlay_colored_lines(base, TX, lines, line_colors)

            # Build question
            q_idx = rng.randrange(K)
            q_color = line_colors[q_idx]
            C = color_label(q_color)
            question = rng.choice(PROMPTS).format(C=C)
            answer = int(lengths[q_idx])
            complexity = self._compute_complexity(spec.width, spec.height, tiling.name)
            meta = {
                "pattern_kind": "tiles",
                "pattern": self.name,
                "grid": (1, 1),
                "variant": {
                    "measure": "polyline_length_edge_steps",
                    "scope": "single_colored_line",
                    "line_count": int(K),
                },
                "question": question,
                "answer": answer,
                "tiling_kind": tiling.name,
                "dims": (spec.width, spec.height),
                "out_px": tile_px,
                "composite_ready": True,
                "lines": [
                    {"color": c, "color_label": color_label(c), "length": int(L)}
                    for c, L in zip(line_colors, lengths)
                ],
                "query_color": q_color,
                "complexity": complexity,
                "complexity_score": complexity["score"],
                "complexity_level": complexity["level"],
                "complexity_version": complexity["version"],
            }

            return base, [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")



