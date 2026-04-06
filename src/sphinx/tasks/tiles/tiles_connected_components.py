# sphinx/tasks/tiles/tiles_connected_components.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from sphinx.base import Task
from .common import MIN_TILING_WH, _max_wh_for, _components, _out_px_for_dims
from sphinx.registry import register_task
from sphinx.utils.drawing import render_patch_crisp
from sphinx.config import MAX_BUILD_RETRIES, COLORS_NAMES
from ...utils.rng import _dirichlet, choice_weighted
from ...utils.colors import color_label

COLORS = list(COLORS_NAMES)

from sphinx.tilings import (
    TilingSpec, create_tiling, get_tiling_names, Colorer,
    build_dual_graph,
)

# ----------------------------- prompts -----------------------------
# 10 templates for edge-adjacency (polygons)
PROMPT_TEMPLATES_EDGE = [
    "Cells are adjacent if they share an edge. What is the {measure} {scope}?",
    "Using edge adjacency only, what is the {measure} {scope}?",
    "If tiles are connected by shared sides, what is the {measure} {scope}?",
    "Under edge connectivity, what is the {measure} {scope}?",
    "Considering edge-adjacent tiles as one region, what is the {measure} {scope}?",
    "Connectivity is defined by shared edges. What is the {measure} {scope}?",
    "With edge-only adjacency, what is the {measure} {scope}?",
    "Based on edge-connected tiles, what is the {measure} {scope}?",
    "When cells are joined by edges, what is the {measure} {scope}?",
    "Edge-connected cells form components. What is the {measure} {scope}?"
]


# 10 parallel templates for point-touch (circles)
PROMPT_TEMPLATES_TOUCH = [
    "Cells are adjacent if they touch at a point. What is the {measure} {scope}?",
    "Using point-touch adjacency, what is the {measure} {scope}?",
    "If cells are considered adjacent when they touch at a point, what is the {measure} {scope}?",
    "Under point connectivity, what is the {measure} {scope}?",
    "Considering tiles that meet at a point as connected, what is the {measure} {scope}?",
    "Connectivity is defined by point-touch. What is the {measure} {scope}?",
    "With point-touch adjacency, what is the {measure} {scope}?",
    "Based on tiles joined by point-touch, what is the {measure} {scope}?",
    "When cells connect at points, what is the {measure} {scope}?",
    "Point-connected cells form components. What is the {measure} {scope}?"
]

# Text fragments we plug into the templates
MEASURES = {
    "size_largest": "size (cells) of the largest connected component",
    "size_smallest": "size (cells) of the smallest connected component",
    "count_components": "number of connected components",
}

# sampling weights for tiling families
TILING_WEIGHTS = {
    "square": 1.0,
    "triangular": 0.25,
    "hexagonal": 1.0,
    "circles": 1.0,
    "rhombille": 0.25,
    "voronoi": 0.0,
    "orthogonal_split": 0.0,
}

# ----------------------------- helpers -----------------------------
def _sizes_from_comps(
    comps: List[Set[int]],
    weights: Optional[List[int]] = None,
) -> List[int]:
    if not comps:
        return []
    if weights is None:
        return [len(c) for c in comps]
    return [sum(weights[i] for i in c) for c in comps]

def _unique_extreme(values: List[int], kind: str) -> Optional[int]:
    """
    Enforce uniqueness for largest/smallest queries.
    Returns the extreme value if it occurs exactly once, else None.
    """
    if not values:
        return None
    val = max(values) if kind == "max" else min(values)
    if values.count(val) == 1:
        return val
    return None

# ----------------------------- task -----------------------------
@register_task
class TilesConnectedComponentTask(Task):
    """
    Show a single colored tiling (non-uniform colors). Ask an integer question about
    connected tiles under edge-adjacency, restricted to a specific color only.
    Answer is a single integer.

    Query types:
      - size_largest / size_smallest / count_components
    """
    name = "tiles_connected_component"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.min_colors = 2

    def _compute_complexity(self, component_count: int) -> Dict[str, Any]:
        """Normalize total component count across all colors."""
        min_comp = 1
        max_comp = 6
        span = max(1, max_comp - min_comp)
        normalized = (int(component_count) - min_comp) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "tiles-connected-components-total-v1",
            "range": {"min_components": min_comp, "max_components": max_comp},
            "component_count": int(component_count),
        }

    def _sample_tiling_and_colors(self, rng: random.Random):
        """Sample a tiling and assign non-uniform colors to its cells."""
        names = get_tiling_names()
        weights = [TILING_WEIGHTS.get(n, 1.0) for n in names]
        tname = choice_weighted(rng, names, weights)
        tiling = create_tiling(tname)

        hi = _max_wh_for(tiling.name)
        w = rng.randint(MIN_TILING_WH, hi)
        h = rng.randint(MIN_TILING_WH, hi)


        k_max = len(COLORS)
        k = rng.randint(self.min_colors, k_max)
        idxs = rng.sample(range(k_max), k)
        seed = rng.randint(0, 2 ** 31 - 1)

        alpha = rng.uniform(0.7, 1.5)
        p = _dirichlet(rng, len(idxs), alpha=alpha)

        spec = TilingSpec(
            tiling.name, seed,
            width=w, height=h,
            color_mode="non_uniform",
            non_uniform={"colors_idx": idxs, "p": p},
        )
        patch = tiling.generate(spec)
        Colorer().apply(tiling, patch, spec)

        cell_colors = [c.color for c in patch.cells]
        used_colors = sorted(set(cell_colors))

        # <<< key change: circles use point-touch adjacency >>>
        adjacency_mode = "touch" if tiling.name == "circles" else "edge"
        g = build_dual_graph(patch, connect_on_touch=(adjacency_mode == "touch"))

        return tiling, spec, patch, g, cell_colors, used_colors, adjacency_mode

    def _pick_query_and_answer(
            self,
            rng: random.Random,
            g: Dict[int, Set[int]],
            cell_colors: List[str],
            used_colors: List[str],
            weights: Optional[List[int]] = None,
            templates: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select a per-color query (measure) with a unique integer answer when
        uniqueness is relevant (largest/smallest). Counting queries are always unique.
        """
        templates = templates or PROMPT_TEMPLATES_EDGE

        # Precompute connected components and sizes per color
        per_color: Dict[str, Dict[str, Any]] = {}
        for col in used_colors:
            nodes = {i for i, c in enumerate(cell_colors) if c == col}
            cs = _components(g, nodes)
            per_color[col] = {
                "nodes": nodes,
                "comps": cs,
                "sizes": _sizes_from_comps(cs, weights),
            }

        # Candidate per-color queries to try (shuffle for diversity)
        candidates: List[Tuple[str, str]] = []  # (measure_key, color)
        rng_used = used_colors[:]
        rng.shuffle(rng_used)
        for col in rng_used[:3]:  # consider up to 3 colors per instance
            candidates.append(("size_largest", col))
            candidates.append(("size_smallest", col))
            candidates.append(("count_components", col))

        rng.shuffle(candidates)

        for measure_key, color in candidates:
            data = per_color.get(color, None)
            if not data or not data["sizes"]:
                continue

            sizes_c = data["sizes"]
            if measure_key == "count_components":
                ans = len(sizes_c)
            elif measure_key == "size_largest":
                ans = _unique_extreme(sizes_c, "max")
                if ans is None:
                    continue
            else:  # size_smallest
                ans = _unique_extreme(sizes_c, "min")
                if ans is None:
                    continue

            # Build the natural-language question
            template = rng.choice(templates)
            label = color_label(color)  # e.g., "red (#e6194b)"
            if "With color {scope}" in template:
                scope_text = label
            else:
                scope_text = f"within color {label}"

            q = template.format(
                measure=MEASURES[measure_key],
                scope=scope_text
            )

            return {
                "question": q,
                "measure": measure_key,
                "scope": "color",
                "color": color,
                "color_label": label,
                "answer_int": int(ans),
            }

        return None  # signal to resample

    # ----------------------------- public API -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        weights: Optional[List[int]] = None

        for _ in range(self.max_retries):
            tiling, spec, patch, g, cell_colors, used_colors, adjacency_mode = self._sample_tiling_and_colors(rng)
            weights = [1] * len(cell_colors)

            templates = PROMPT_TEMPLATES_TOUCH if adjacency_mode == "touch" else PROMPT_TEMPLATES_EDGE
            picked = self._pick_query_and_answer(rng, g, cell_colors, used_colors, weights, templates)
            if picked is None:
                continue

            tile_px = _out_px_for_dims(spec.width, spec.height)

            composite = render_patch_crisp(
                patch,
                cell_colors,
                size_px=tile_px,
                background="white",
                outline_rgba=(0, 0, 0, 255),
                outline_px=1,
            )

            total_components = sum(len(_components(g, {i for i, c in enumerate(cell_colors) if c == col}))
                                   for col in used_colors)
            complexity = self._compute_complexity(int(total_components))

            meta = {
                "pattern_kind": "tiles",
                "pattern": self.name,
                "grid": (1, 1),
                "variant": {
                    "measure": picked["measure"],
                    "scope": picked["scope"],
                    "color": picked["color"],
                },
                "question": picked["question"],
                "answer": picked["answer_int"],
                "colors_used": used_colors,
                "color_names": {c: COLORS_NAMES.get(c, "color") for c in used_colors},
                "tiling_kind": tiling.name,
                "adjacency_mode": adjacency_mode,  # NEW (provenance)
                "color_mode": "non_uniform",
                "dims": (spec.width, spec.height),
                "out_px": tile_px,
                "composite_ready": True,
                "total_components": int(total_components),
                "complexity": complexity,
                "complexity_score": complexity["score"],
                "complexity_level": complexity["level"],
                "complexity_version": complexity["version"],
            }
            return composite, [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a unique-answer sample after {self.max_retries} attempts.")
