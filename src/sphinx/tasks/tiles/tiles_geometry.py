# sphinx/tasks/tiles/tiles_geometry.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from sphinx.base import Task
from .common import MIN_TILING_WH, MAX_TILING_WH_DEFAULT, _max_wh_for, _components, _out_px_for_dims
from sphinx.registry import register_task
from sphinx.utils.drawing import render_patch_crisp
from sphinx.config import MAX_BUILD_RETRIES, COLORS_NAMES
from ...utils.rng import choice_weighted
from ...utils.colors import color_label

COLORS = list(COLORS_NAMES)

from sphinx.tilings import (
    TilingSpec, create_tiling, build_dual_graph
)

# ----------------------------------------------------------------------
# Which tilings to use (polygonal; edge-adjacency well-defined)
TILING_WEIGHTS: Dict[str, float] = {
    "square": 1.0,
    "triangular": 1.0,
    "hexagonal": 1.0,
}

# Degree (shared-edge neighbors) for perimeter math
DEGREE_BY_TILING = {
    "square": 4,
    "triangular": 3,
    "hexagonal": 6,
}

# Hole fill colors (dark shades; distinct from white background)
HOLE_SHADE_CANDIDATES = ["#0f0f0f", "#151515", "#1a1a1a", "#202020", "#262626", "#2c2c2c"]

# ----------------------------------------------------------------------
# Query types + weights (explicit 1.0 each, as requested)
GEOM_TYPE_WEIGHTS: Dict[str, float] = {
    "area_single": 1.0,
    "perimeter_single": 1.0,
    "holes_single": 0.75,
    "area_diff_two": 0.5,
    "union_perimeter_two": 1.0,
}

# ----------------------------- prompts (10 each) -----------------------------
# Conventions used in ALL prompts:
#   - Edge adjacency ONLY: two tiles are neighbors iff they share an EDGE (corner/vertex touch does NOT connect).
#   - Area = number of tiles.
#   - Perimeter = number of shared-edge boundary steps.

# Area prompts
PROMPTS_AREA = [
    "The {C} shape is a single region. What is its area in tiles?",
    "How many tiles are in the {C} shape?",
    "What is the area (in tiles) of the {C} region?",
    "What is the tile area of the {C} shape?",
    "What is the total number of {C} tiles?",
    "What is the area of the {C} region, measured in tiles?",
    "What is the size of the {C} shape in tiles?",
    "What is the tile count of the {C} region?",
    "How many tiles belong to the {C} shape?",
    "What is the area (tiles) of the {C} component?"
]


# Perimeter prompts
PROMPTS_PERIM = [
    "What is the perimeter of the {C} shape, counted in boundary edges?",
    "How many boundary edges does the {C} region have?",
    "What is the outline length of the {C} shape in edges?",
    "What is the boundary length of the {C} component?",
    "How many edges form the perimeter of the {C} shape?",
    "What is the perimeter of the {C} region in terms of edge count?",
    "How many boundary edges surround the {C} shape?",
    "What is the total perimeter of the {C} component?",
    "What is the edge length of the {C} region’s boundary?",
    "What is the perimeter of the {C} shape?"
]

# Holes definition (clear + edge-only)
#   A hole is a dark region that is:
#     (i) completely enclosed by the {C}-colored tiles, and
#     (ii) itself edge-connected (dark tiles touching only at a vertex are separate holes),
#     (iii) not touching the board’s outer boundary.
PROMPTS_HOLES = [
    "Using edge connectivity, how many dark holes are fully enclosed by the {C} shape?",
    "How many enclosed dark holes lie inside the {C} region under edge adjacency?",
    "With edge adjacency (not corners), how many holes are surrounded by the {C} shape?",
    "How many interior dark holes does the {C} region contain, counting only edge-connected tiles?",
    "Under edge-only connectivity, how many distinct dark holes are inside the {C} shape?",
    "How many dark regions are fully enclosed by the {C} shape, when adjacency is defined by shared edges?",
    "Using shared-edge connectivity, how many interior holes are present in the {C} region?",
    "How many edge-connected dark holes are completely surrounded by the {C} shape?",
    "With adjacency defined only by shared edges, how many holes are inside the {C} region?"
    "How many enclosed dark holes are contained in the {C} shape under edge adjacency?"
]


PROMPTS_AREA_DIFF_TWO = [
    "What is the absolute difference in area, measured in tiles, between the {C1} and {C2} shapes?",
    "How many tiles apart are the areas of the {C1} region and the {C2} region?",
    "What is the tile-count difference between the {C1} shape and the {C2} shape?",
    "By how many tiles do the areas of the {C1} and {C2} regions differ?",
    "What is the absolute area difference between the {C1} region and the {C2} region?",
    "What is the difference in tile counts between the {C1} region and the {C2} region?"
    "What is the magnitude of the difference in tile counts between {C1} and {C2}?",
    "What is the absolute difference in tile count between the {C1} region and the {C2} region?"
    "What is the absolute difference in tile area between the {C1} shape and the {C2} shape?",
    "What is the difference in area, in tiles, between the {C1} region and the {C2} region?"
]

# For union perimeter we GUARANTEE the two colors are edge-adjacent, so union is one continuous shape.
PROMPTS_UNION_PERIM_TWO = [
    "Treat the {C1} and {C2} regions together as one shape. What is its perimeter?",
    "Consider the union of the {C1} and {C2} regions. What is the perimeter of the combined shape?",
    "What is the outline length, in edges, of the shape formed by {C1} and {C2} together?",
    "If the {C1} and {C2} regions are merged into one, what is the perimeter of the result?",
    "When {C1} and {C2} are joined, how many boundary edges does the combined region have?",
    "What is the perimeter of the single region formed by combining {C1} and {C2}?",
    "When {C1} and {C2} are taken together, what is the perimeter of the resulting shape?",
    "What is the boundary length, in edges, of the shape formed by {C1} and {C2}?",
    "How many boundary edges surround the combined {C1} and {C2} region?",
    "What is the perimeter of the merged shape consisting of {C1} and {C2}?"
]

# ----------------------------------------------------------------------
# Helpers (regions, perimeter, holes, rendering)

def _grow_connected_chunk(rng: random.Random, g: Dict[int, Set[int]],
                          allowed: Set[int], target_size: int) -> Optional[Set[int]]:
    """BFS grow a region from a random seed in 'allowed' up to target_size (best effort)."""
    if not allowed:
        return None
    seed = rng.choice(tuple(allowed))
    region = {seed}
    frontier = [seed]
    used = {seed}
    while frontier and len(region) < target_size:
        u = frontier.pop(rng.randrange(len(frontier)))
        cand = [v for v in g[u] if v in allowed and v not in used]
        rng.shuffle(cand)
        for v in cand:
            region.add(v)
            used.add(v)
            frontier.append(v)
            if len(region) >= target_size:
                break
    if len(region) < min(2, target_size):
        return None
    return region

def _perimeter_of_region(tiling_name: str, g: Dict[int, Set[int]], region: Set[int]) -> int:
    """Edge-perimeter in shared-edge steps: sum over cells of (K - #neighbors_in_region)."""
    K = DEGREE_BY_TILING.get(tiling_name, 4)
    perim = 0
    for u in region:
        in_neighbors = sum(1 for v in g[u] if v in region)
        perim += (K - in_neighbors)
    return int(perim)

def _holes_inside_region(tiling_name: str, g: Dict[int, Set[int]], region: Set[int]) -> int:
    """
    Count enclosed complement components (holes) under edge adjacency.
    A complement component is NOT a hole if any of its nodes touches the board boundary
    (i.e., has degree < K in the dual graph).
    """
    K = DEGREE_BY_TILING.get(tiling_name, 4)
    n_all = set(g.keys())
    comp_nodes = n_all - region
    comps = _components(g, comp_nodes)

    def touches_board_boundary(comp: Set[int]) -> bool:
        return any(len(g[u]) < K for u in comp)

    holes = [c for c in comps if not touches_board_boundary(c)]
    return len(holes)

def _sample_tiling(rng: random.Random):
    """Sample a tiling and its dual graph for geometry questions."""
    names = list(TILING_WEIGHTS.keys())
    weights = [TILING_WEIGHTS[n] for n in names]
    tname = choice_weighted(rng, names, weights)
    tiling = create_tiling(tname)

    hi = _max_wh_for(tiling.name)
    w = rng.randint(MIN_TILING_WH, hi)
    h = rng.randint(MIN_TILING_WH, hi)

    seed = rng.randint(0, 2 ** 31 - 1)
    # We paint explicitly; spec used for reproducibility/meta.
    spec = TilingSpec(tiling.name, seed, width=w, height=h, uniform={"scheme": "same", "colors_idx": [0]})
    patch = tiling.generate(spec)

    # Edge adjacency over polygonal tilings
    g = build_dual_graph(patch, connect_on_touch=False)
    return tiling, spec, patch, g

def _choose_shape_colors(rng: random.Random, k: int) -> List[str]:
    """Select up to k distinct colors for painting shapes."""
    kmax = len(COLORS)
    if k >= kmax:
        idxs = list(range(kmax))
        rng.shuffle(idxs)
    else:
        idxs = rng.sample(range(kmax), k)
    return [COLORS[i] for i in idxs]

def _paint_disjoint_shapes(rng: random.Random, patch, g: Dict[int, Set[int]],
                           tiling_name: str, k_shapes: int) -> Tuple[List[str], Dict[str, Set[int]]]:
    """
    Build k disjoint connected shapes (one component per color) on white background.
    Returns (cell_colors, color->node_set).
    """
    n = len(patch.cells)
    cell_colors = ["white"] * n
    avail: Set[int] = set(range(n))
    color_to_nodes: Dict[str, Set[int]] = {}

    shape_colors = _choose_shape_colors(rng, k_shapes)

    # Target sizes: diversify, keep total coverage <= ~65%
    min_sz = max(3, n // 30)
    max_sz = max(min_sz + 1, n // 7)
    total_cap = int(0.65 * n)

    cover = 0
    for col in shape_colors:
        hi = max(min_sz, min(max_sz, total_cap - cover))
        if hi < min_sz:
            break
        target = rng.randint(min_sz, hi)
        best = None
        for _ in range(32):
            region = _grow_connected_chunk(rng, g, avail, target)
            if region and len(region) >= min_sz:
                best = region
                break
        if best is None:
            continue
        for u in best:
            cell_colors[u] = col
        color_to_nodes[col] = set(best)
        avail -= best
        cover += len(best)

    # Ensure at least two shapes for the “two-color” query types
    if len(color_to_nodes) < 2:
        for _ in range(64):
            target = rng.randint(min_sz, max_sz)
            region = _grow_connected_chunk(rng, g, avail, target)
            if region:
                col = _choose_shape_colors(rng, 1)[0]
                for u in region:
                    cell_colors[u] = col
                color_to_nodes[col] = set(region)
                break

    return cell_colors, color_to_nodes

def _paint_region_with_holes(rng: random.Random, patch, g: Dict[int, Set[int]],
                             tiling_name: str) -> Tuple[List[str], str, List[Set[int]]]:
    """
    Create one large connected region (color REG) and cut 1..H holes inside (dark shades),
    guaranteeing holes do NOT touch the region boundary (i.e., are fully enclosed).
    Returns (cell_colors, region_color, list_of_hole_node_sets).
    """
    n = len(patch.cells)
    cell_colors = ["white"] * n
    all_nodes = set(range(n))

    # Build a large connected region
    target = rng.randint(max(10, n // 3), max(12, int(0.75 * n)))
    best = None
    for _ in range(64):
        region = _grow_connected_chunk(rng, g, all_nodes, target)
        if region and len(region) >= max(10, n // 4):
            best = region
            break
    if best is None:
        best = set(all_nodes)  # fallback: whole board

    region_color = _choose_shape_colors(rng, 1)[0]
    for u in best:
        cell_colors[u] = region_color

    # Boundary set of the region (neighbors not in region)
    boundary: Set[int] = set()
    for u in best:
        if any(v not in best for v in g[u]):
            boundary.add(u)

    # Candidates for hole seeds: strictly interior (no neighbor outside region)
    interior: Set[int] = {u for u in best if all(v in best for v in g[u])}
    if not interior:
        return cell_colors, region_color, []

    hole_sets: List[Set[int]] = []
    used_for_holes: Set[int] = set()

    Hmax = max(1, min(4, len(interior) // max(3, len(best) // 20)))
    H = rng.randint(1, Hmax)

    for _ in range(64):
        if len(hole_sets) >= H:
            break
        inner_free = list(interior - used_for_holes)
        if not inner_free:
            break
        rng.choice(inner_free)
        hsize = rng.randint(1, max(2, len(best) // 25))
        allowed = set(best) - used_for_holes
        hole = _grow_connected_chunk(rng, g, allowed, hsize)
        if hole is None:
            continue
        # ensure hole doesn't touch region boundary
        if any(u in boundary for u in hole):
            continue
        if any(u in used_for_holes for u in hole):
            continue
        hole_sets.append(set(hole))
        used_for_holes |= set(hole)

    if hole_sets:
        hole_col = rng.choice(HOLE_SHADE_CANDIDATES)
        for hs in hole_sets:
            for u in hs:
                cell_colors[u] = hole_col

    return cell_colors, region_color, hole_sets

def _sets_edge_touch(g: Dict[int, Set[int]], A: Set[int], B: Set[int]) -> bool:
    """Return True if any edge in g connects A to B (edge adjacency only)."""
    if len(A) > len(B):
        A, B = B, A
    Bset = set(B)
    for u in A:
        for v in g[u]:
            if v in Bset:
                return True
    return False

# ----------------------------------------------------------------------
# Task

@register_task
class TilesGeometryTask(Task):
    """
    A small suite of geometry questions on colored shapes drawn over a white background.

    Types (sampled by weights):
      - area_single           : area (tile count) of a particular color (single edge-connected component per color)
      - perimeter_single      : shared-edge perimeter length of a particular color
      - holes_single          : number of enclosed holes inside one large region
      - area_diff_two         : |area(A) - area(B)|
      - union_perimeter_two   : perimeter of the union of two colors that are edge-adjacent to each other (union is one continuous region)

    Conventions:
      - Edge adjacency ONLY for connectivity (share an edge; vertex-only contact does NOT connect).
      - Area = number of tiles. Perimeter = number of shared-edge boundary steps.

    Returns: (composite, [spec], meta) like other tiles tasks.
    """
    name = "tiles_geometry"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, cell_count: int) -> Dict[str, Any]:
        """Normalize board cell count to EASY/HARD difficulty."""
        min_cells = MIN_TILING_WH * MIN_TILING_WH
        # we do not have enough large tile sizes
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
            "version": "tiles-geometry-board-cells-v1",
            "range": {
                "min_cells": min_cells,
                "max_cells": max_cells,
            },
            "cell_count": int(cell_count),
        }

    # ----------------------------- public API -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # choose type
        types = list(GEOM_TYPE_WEIGHTS.keys())
        w = [GEOM_TYPE_WEIGHTS[t] for t in types]
        qtype = choice_weighted(rng, types, w)

        for _ in range(self.max_retries):
            tiling, spec, patch, g = _sample_tiling(rng)
            tile_px = _out_px_for_dims(spec.width, spec.height)
            cell_count = len(patch.cells)
            complexity = self._compute_complexity(cell_count)

            if qtype in ("area_single", "perimeter_single", "area_diff_two", "union_perimeter_two"):
                # draw K disjoint shapes on white background (one edge-connected component per color)
                k_shapes = rng.randint(2, 8)
                cell_colors, color_to_nodes = _paint_disjoint_shapes(rng, patch, g, tiling.name, k_shapes)
                if len(color_to_nodes) < (2 if qtype in ("area_diff_two", "union_perimeter_two") else 1):
                    continue

                if qtype == "area_single":
                    col = rng.choice(list(color_to_nodes.keys()))
                    area = len(color_to_nodes[col])
                    C = color_label(col)
                    question = rng.choice(PROMPTS_AREA).format(C=C)
                    answer = int(area)
                    meta_variant = {"measure": "area", "scope": "single_color", "color": col, "color_label": C}

                elif qtype == "perimeter_single":
                    col = rng.choice(list(color_to_nodes.keys()))
                    region = color_to_nodes[col]
                    perim = _perimeter_of_region(tiling.name, g, region)
                    C = color_label(col)
                    question = rng.choice(PROMPTS_PERIM).format(C=C)
                    answer = int(perim)
                    meta_variant = {"measure": "perimeter", "scope": "single_color", "color": col, "color_label": C}

                elif qtype == "area_diff_two":
                    c1, c2 = rng.sample(list(color_to_nodes.keys()), 2)
                    area1 = len(color_to_nodes[c1]); area2 = len(color_to_nodes[c2])
                    C1, C2 = color_label(c1), color_label(c2)
                    question = rng.choice(PROMPTS_AREA_DIFF_TWO).format(C1=C1, C2=C2)
                    answer = int(abs(area1 - area2))
                    meta_variant = {
                        "measure": "area_diff_abs", "scope": "two_colors",
                        "color_a": c1, "color_b": c2,
                        "labels": {"a": C1, "b": C2}
                    }

                else:  # union_perimeter_two — enforce edge-adjacent pair so union is continuous
                    cols = list(color_to_nodes.keys())
                    touching_pairs: List[Tuple[str, str]] = []
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            a, b = cols[i], cols[j]
                            if _sets_edge_touch(g, color_to_nodes[a], color_to_nodes[b]):
                                touching_pairs.append((a, b))
                    if not touching_pairs:
                        # resample this whole instance
                        continue
                    c1, c2 = rng.choice(touching_pairs)
                    union = set(color_to_nodes[c1]) | set(color_to_nodes[c2])
                    # sanity: union must be a single edge-connected component
                    if len(_components(g, union)) != 1:
                        continue
                    perim = _perimeter_of_region(tiling.name, g, union)
                    C1, C2 = color_label(c1), color_label(c2)
                    question = rng.choice(PROMPTS_UNION_PERIM_TWO).format(C1=C1, C2=C2)
                    answer = int(perim)
                    meta_variant = {
                        "measure": "perimeter_union", "scope": "two_colors_edge_connected",
                        "color_a": c1, "color_b": c2,
                        "labels": {"a": C1, "b": C2},
                        "union_connected": True
                    }

                composite = render_patch_crisp(
                    patch, cell_colors, size_px=tile_px,
                    background="white", outline_rgba=(0, 0, 0, 255), outline_px=1
                )
                meta = {
                    "pattern_kind": "tiles",
                    "pattern": self.name,
                    "grid": (1, 1),
                    "variant": meta_variant,
                    "question": question,
                    "answer": answer,
                    "tiling_kind": tiling.name,
                    "dims": (spec.width, spec.height),
                    "out_px": tile_px,
                    "composite_ready": True,
                    "board_cell_count": cell_count,
                    "complexity": complexity,
                    "complexity_score": complexity["score"],
                    "complexity_level": complexity["level"],
                    "complexity_version": complexity["version"],
                }
                return composite, [spec], meta

            else:  # holes_single
                cell_colors, region_color, hole_sets = _paint_region_with_holes(rng, patch, g, tiling.name)
                holes = len(hole_sets)
                if holes < 1:
                    continue

                # Verify hole count via topology
                region_nodes = {i for i, c in enumerate(cell_colors) if c == region_color}
                holes_check = _holes_inside_region(tiling.name, g, region_nodes)
                if holes_check != holes:
                    holes = holes_check

                C = color_label(region_color)
                question = rng.choice(PROMPTS_HOLES).format(C=C)
                answer = int(holes)
                meta_variant = {"measure": "holes", "scope": "single_color", "color": region_color, "color_label": C}

                composite = render_patch_crisp(
                    patch, cell_colors, size_px=tile_px,
                    background="white", outline_rgba=(0, 0, 0, 255), outline_px=1
                )
                meta = {
                    "pattern_kind": "tiles",
                    "pattern": self.name,
                    "grid": (1, 1),
                    "variant": meta_variant,
                    "question": question,
                    "answer": answer,
                    "tiling_kind": tiling.name,
                    "dims": (spec.width, spec.height),
                    "out_px": tile_px,
                    "composite_ready": True,
                    "board_cell_count": cell_count,
                    "complexity": complexity,
                    "complexity_score": complexity["score"],
                    "complexity_level": complexity["level"],
                    "complexity_version": complexity["version"],
                }
                return composite, [spec], meta

        raise RuntimeError(f"{self.name}: failed to sample a valid geometry instance after {self.max_retries} attempts.")
