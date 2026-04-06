# sphinx/tasks/tiles/tiles_recoloring.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from sphinx.base import Task
from .common import MIN_TILING_WH, _max_wh_for, _out_px_for_dims
from sphinx.registry import register_task
from sphinx.utils.drawing import render_patch_crisp
from sphinx.config import MAX_BUILD_RETRIES, COLORS_NAMES
from ...utils.rng import choice_weighted

COLORS = list(COLORS_NAMES)

from sphinx.tilings import (
    TilingSpec, create_tiling, get_tiling_names,
    build_dual_graph,
)

# -----------------------------------------------------------------------------
# Sampling knobs & bounds (as requested)
# -----------------------------------------------------------------------------
MIN_DIFF_CELLS = 1
MAX_DIFF_CELLS = 15

VARIANT_WEIGHTS = {
    "same_color": 1.0,     # add/remove only (uniform fill color)
    "color_change": 1.0,   # add/remove + recolor some overlapping cells
}

# sampling weights for tiling families (mirrors other Tiles tasks)
TILING_WEIGHTS = {
    "square": 1.0,
    "triangular": 0.25,
    "hexagonal": 0.75,
    "circles": 1.0,
    "rhombille": 0.25,
    "voronoi": 0.0,
    "orthogonal_split": 0.0,
}

SEP_PX = 40  # gap between left and right panels in the composite

# ----------------------------- prompts -----------------------------
PROMPTS_SAME = [
    "Two tiles are shown side by side (left and right). Count how many cells differ between them. Answer with an integer.",
    "Look at the left and right tiles. How many cells are different (filled on one side but not the other)? Respond with an integer.",
    "Compare the two tiles. How many cells do not match between left and right? Give an integer.",
    "How many cells are different between the left and right images? Answer as an integer.",
    "Count the number of cells that differ between the left and right tiles. Provide an integer.",
    "Between the two side-by-side tiles, how many cells are different? Reply with an integer.",
    "How many cells are not the same between the left tile and the right tile? Integer only.",
    "Compare left vs. right: how many cells differ? Give an integer.",
    "What is the count of differing cells between the two tiles (left/right)? Answer with an integer.",
    "How many cells are different across the two panels? Respond with an integer.",
]

PROMPTS_COLOR = [
    "Two tiles are shown (left/right). A cell counts as different if its color differs (including filled vs. blank). How many cells differ?",
    "Count cells whose colors do not match between left and right (filled vs. blank also counts). Provide an integer.",
    "Compare the two tiles. A cell is different if its left/right colors are not equal. How many such cells are there?",
    "How many cells differ in color between the left and right panels (including presence vs. absence)? Integer only.",
    "Count the cells that mismatch in color across the two tiles. Answer with an integer.",
    "Between the side-by-side tiles, how many cells have different colors (filled vs. blank included)?",
    "How many cells are different (color mismatch) between left and right? Respond with an integer.",
    "A difference is any cell with unequal colors (including blank vs. filled). How many differences are there?",
    "Consider color differences only (presence/absence included). How many cells differ between the tiles?",
    "How many cells do not match in color between the left and right images? Integer answer.",
]

# ----------------------------- small helpers -----------------------------
def _compose_left_right(
    left_img: "Image.Image",
    right_img: "Image.Image",
    sep_px: int = SEP_PX,
    bg: str = "white",
    panel_pad_px: int = 28,   # inner white pad between grid and border
    frame_px: int = 3,        # black border thickness
    white_thresh: int = 245,  # near-white background threshold
    bleed_px: int = 1,        # keep a tiny halo so grid lines aren't clipped
) -> "Image.Image":
    """
    Compose LEFT and RIGHT so **both the grid and the border** align.

    Steps for each panel image from render_patch_crisp:
      1) Remove the existing page frame by flood-filling non-white pixels that touch the image edges.
      2) Take the tight bbox of the remaining content (grid lines + fills).
      3) Rewrap with a uniform border/padding.
    Finally, paste the rewrapped panels side-by-side (top-aligned) with a fixed gap.
    """
    from PIL import Image, ImageDraw
    from collections import deque

    def _crop_to_grid(im: "Image.Image") -> "Image.Image":
        rgb = im.convert("RGB")
        w, h = rgb.size
        px = rgb.load()

        def is_nonwhite(r, g, b) -> bool:
            return not (r > white_thresh and g > white_thresh and b > white_thresh)

        # 1) Mark edge-connected non-white (this is the page frame / any artifacts touching edges)
        edge_conn = [[False] * w for _ in range(h)]
        q = deque()

        # seed with all edges
        for x in range(w):
            for y in (0, h - 1):
                r, g, b = px[x, y]
                if is_nonwhite(r, g, b) and not edge_conn[y][x]:
                    edge_conn[y][x] = True
                    q.append((x, y))
        for y in range(h):
            for x in (0, w - 1):
                r, g, b = px[x, y]
                if is_nonwhite(r, g, b) and not edge_conn[y][x]:
                    edge_conn[y][x] = True
                    q.append((x, y))

        while q:
            x, y = q.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h and not edge_conn[ny][nx]:
                    r, g, b = px[nx, ny]
                    if is_nonwhite(r, g, b):
                        edge_conn[ny][nx] = True
                        q.append((nx, ny))

        # 2) Tight bbox of non-white that is NOT edge-connected (i.e., the grid + fills)
        x0, y0, x1, y1 = w, h, -1, -1
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                if is_nonwhite(r, g, b) and not edge_conn[y][x]:
                    if x < x0: x0 = x
                    if y < y0: y0 = y
                    if x > x1: x1 = x
                    if y > y1: y1 = y

        # Fallback: if nothing detected (shouldn't happen), just return original
        if x1 < 0:
            return rgb

        # add a tiny bleed so outlines aren't clipped
        x0 = max(0, x0 - bleed_px)
        y0 = max(0, y0 - bleed_px)
        x1 = min(w - 1, x1 + bleed_px)
        y1 = min(h - 1, y1 + bleed_px)
        return rgb.crop((x0, y0, x1 + 1, y1 + 1))

    def _rewrap_with_uniform_border(cropped: "Image.Image") -> "Image.Image":
        W = cropped.width  + 2 * (panel_pad_px + frame_px)
        H = cropped.height + 2 * (panel_pad_px + frame_px)
        panel = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(panel)
        # crisp outer frame
        draw.rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0), width=frame_px)
        panel.paste(cropped, (panel_pad_px + frame_px, panel_pad_px + frame_px))
        return panel

    # Build clean, uniformly framed panels
    L = _rewrap_with_uniform_border(_crop_to_grid(left_img))
    R = _rewrap_with_uniform_border(_crop_to_grid(right_img))

    # Side-by-side composition, borders top-aligned
    W = L.width + sep_px + R.width
    H = max(L.height, R.height)
    canvas = Image.new("RGB", (W, H), bg)
    canvas.paste(L, (0, 0))
    canvas.paste(R, (L.width + sep_px, 0))
    return canvas

def _connected_bfs(g: Dict[int, Set[int]], start: int, target_size: int, rng: random.Random) -> Set[int]:
    """Grow a connected set to ~target_size via randomized BFS."""
    seen: Set[int] = {start}
    frontier = [start]
    while frontier and len(seen) < target_size:
        u = frontier.pop(rng.randrange(len(frontier)))
        nbrs = list(g[u]); rng.shuffle(nbrs)
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                frontier.append(v)
                if len(seen) >= target_size:
                    break
    return seen

def _frontier(g: Dict[int, Set[int]], region: Set[int]) -> List[int]:
    out: Set[int] = set()
    for u in region:
        for v in g[u]:
            if v not in region:
                out.add(v)
    return list(out)

def _is_connected(g: Dict[int, Set[int]], nodes: Set[int]) -> bool:
    if not nodes:
        return True
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in g[u]:
            if v in nodes and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)

def _grow_region(g: Dict[int, Set[int]], region: Set[int], k: int, rng: random.Random) -> int:
    """Add up to k cells from the frontier while keeping region connected."""
    added = 0
    for _ in range(k):
        fr = _frontier(g, region)
        if not fr:
            break
        v = rng.choice(fr)
        region.add(v)
        added += 1
    return added

def _shrink_region(g: Dict[int, Set[int]], region: Set[int], k: int, rng: random.Random) -> int:
    """Remove up to k boundary cells while preserving connectivity."""
    removed = 0
    for _ in range(k):
        # Prefer boundary cells
        candidates = [u for u in region if any(v not in region for v in g[u])]
        if not candidates:
            candidates = list(region)
        rng.shuffle(candidates)

        took = False
        for u in candidates:
            if len(region) <= 1:
                break
            region.remove(u)
            if _is_connected(g, region):
                removed += 1
                took = True
                break
            # undo if it disconnects
            region.add(u)
        if not took:
            break
    return removed

def _count_differences(a: List[str], b: List[str]) -> Tuple[int, List[int]]:
    ids = [i for i in range(len(a)) if a[i] != b[i]]
    return len(ids), ids

def _pick_three_distinct_colors(rng: random.Random) -> Tuple[str, str, str]:
    idxs = list(range(len(COLORS)))
    rng.shuffle(idxs)
    if len(idxs) >= 3:
        return COLORS[idxs[0]], COLORS[idxs[1]], COLORS[idxs[2]]
    # Fallbacks for tiny palettes
    if len(idxs) == 2:
        return "white", COLORS[idxs[0]], COLORS[idxs[1]]
    if len(idxs) == 1:
        return "white", COLORS[idxs[0]], COLORS[idxs[0]]
    return "white", "#1f77b4", "#d62728"

# -----------------------------------------------------------------------------
@register_task
class TilesRecoloringTask(Task):
    """
    Show two tiles (same geometry) side by side: LEFT and RIGHT.
    LEFT has a connected filled region in a single color. RIGHT is derived by:
      • Variant 'same_color': adding/removing a connected set of cells (same fill color).
      • Variant 'color_change': same as above, plus recoloring some overlapping cells to a second color.
    The question asks for the number of cells that differ between LEFT and RIGHT,
    where “different” is defined by unequal per-cell colors (so presence/absence counts as a color difference).
    """
    name = "tiles_recoloring"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, diff_cells: int) -> Dict[str, Any]:
        """Normalize the number of differing cells to EASY/HARD."""
        min_diff = int(MIN_DIFF_CELLS)
        max_diff = int(MAX_DIFF_CELLS)
        span = max(1, max_diff - min_diff)
        normalized = (int(diff_cells) - min_diff) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "tiles-recoloring-diff-v1",
            "range": {"min_diff": min_diff, "max_diff": max_diff},
            "diff_cells": int(diff_cells),
        }

    # ----------------------------- internal: sample patch & graph -----------------------------
    def _sample_tiling(self, rng: random.Random):
        names = get_tiling_names()
        weights = [TILING_WEIGHTS.get(n, 1.0) for n in names]
        tname = choice_weighted(rng, names, weights)
        tiling = create_tiling(tname)

        hi = _max_wh_for(tiling.name)
        w = rng.randint(MIN_TILING_WH, hi)
        h = rng.randint(MIN_TILING_WH, hi)

        seed = rng.randint(0, 2 ** 31 - 1)
        spec = TilingSpec(tiling.name, seed, width=w, height=h)
        patch = tiling.generate(spec)

        # circles use point-touch adjacency; polygons use shared-edge
        adjacency_mode = "touch" if tiling.name == "circles" else "edge"
        g = build_dual_graph(patch, connect_on_touch=(adjacency_mode == "touch"))

        return tiling, spec, patch, g, adjacency_mode

    # ----------------------------- public API -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        kinds = list(VARIANT_WEIGHTS.keys())
        w = [VARIANT_WEIGHTS[k] for k in kinds]
        variant = choice_weighted(rng, kinds, w)

        for _ in range(self.max_retries):
            tiling, spec, patch, g, adjacency_mode = self._sample_tiling(rng)
            n = len(patch.cells)
            if n < 10:
                continue

            # Choose palette: background, base fill (left/right), and alt (used only in color_change)
            bg_color, base_color, alt_color = _pick_three_distinct_colors(rng)

            # Build a connected LEFT region of moderate size
            target = max(8, min(n - 4, int(0.25 * n + rng.randint(-3, 3))))
            start = rng.randrange(n)
            left_region: Set[int] = _connected_bfs(g, start, target, rng)

            # Difference budget request
            want_min = max(1, int(MIN_DIFF_CELLS))
            want_max = max(want_min, int(MAX_DIFF_CELLS))
            diff_budget = rng.randint(want_min, want_max)

            # Plan split between shape edits and recolors (recolors only in color_change)
            if variant == "color_change":
                shape_budget = rng.randint(0, diff_budget)
                recolor_budget = diff_budget - shape_budget
            else:
                shape_budget = diff_budget
                recolor_budget = 0

            # Construct RIGHT region by grow/shrink from LEFT
            right_region = set(left_region)
            shape_done = 0

            if shape_budget > 0:
                # random choice: try shrink first or grow first
                try_shrink = rng.random() < 0.5 and len(right_region) > 1
                if try_shrink:
                    shape_done += _shrink_region(g, right_region, shape_budget - shape_done, rng)
                    if shape_done < shape_budget:
                        shape_done += _grow_region(g, right_region, shape_budget - shape_done, rng)
                else:
                    shape_done += _grow_region(g, right_region, shape_budget - shape_done, rng)
                    if shape_done < shape_budget and len(right_region) > 1:
                        shape_done += _shrink_region(g, right_region, shape_budget - shape_done, rng)

            # Pick recolor set within the overlap
            recolored: Set[int] = set()
            if recolor_budget > 0:
                overlap = list(left_region & right_region)
                rng.shuffle(overlap)
                k = min(recolor_budget, len(overlap))
                recolored = set(overlap[:k])

            # Paint LEFT/RIGHT
            left_colors: List[str] = [bg_color] * n
            for u in left_region:
                left_colors[u] = base_color

            right_colors: List[str] = [bg_color] * n
            for u in right_region:
                right_colors[u] = (alt_color if (variant == "color_change" and u in recolored) else base_color)

            ans, diff_ids = _count_differences(left_colors, right_colors)

            # If we undershot the minimum, try to top up (prefer recolor within overlap if available)
            guard = 0
            while ans < want_min and guard < 24:
                guard += 1
                if variant == "color_change":
                    # add more recolors inside current overlap if possible
                    overlap_more = list((left_region & right_region) - recolored)
                    if overlap_more:
                        u = rng.choice(overlap_more)
                        right_colors[u] = alt_color
                        recolored.add(u)
                    else:
                        # expand right region by one
                        _grow_region(g, right_region, 1, rng)
                        if u := next(iter(right_region - left_region), None):
                            right_colors[u] = base_color
                else:
                    # same_color: adjust shape (grow first, else shrink)
                    prev = len(right_region)
                    if _grow_region(g, right_region, 1, rng) == 0:
                        _shrink_region(g, right_region, 1, rng)
                    # repaint right colors after region change
                    right_colors = [bg_color] * n
                    for u in right_region:
                        right_colors[u] = base_color

                ans, diff_ids = _count_differences(left_colors, right_colors)

            # Sanity: we should now be within bounds (rarely, we might overshoot via shape expansions—re-roll)
            if not (want_min <= ans <= want_max):
                continue

            # Render each panel and compose side-by-side
            tile_px = _out_px_for_dims(spec.width, spec.height)

            left_img = render_patch_crisp(
                patch,
                left_colors,
                size_px=tile_px,
                background="white",
                outline_rgba=(0, 0, 0, 255),
                outline_px=1,
            )
            right_img = render_patch_crisp(
                patch,
                right_colors,
                size_px=tile_px,
                background="white",
                outline_rgba=(0, 0, 0, 255),
                outline_px=1,
            )

            composite = _compose_left_right(left_img, right_img, sep_px=SEP_PX, bg="white")

            # Pick prompt set based on variant (both are correct for either, but color ones are explicit)
            prompts = PROMPTS_COLOR if variant == "color_change" else PROMPTS_SAME
            question = rng.choice(prompts)
            complexity = self._compute_complexity(ans)

            meta = {
                "pattern_kind": "tiles",
                "pattern": self.name,
                "grid": (1, 2),
                "variant": {
                    "kind": variant,
                    "measure": "cellwise_color_difference",
                    "definition": "count cells whose left/right colors differ (presence/absence counts).",
                },
                "question": question,
                "answer": int(ans),
                "tiling_kind": tiling.name,
                "adjacency_mode": adjacency_mode,  # provenance for region growth
                "dims": (spec.width, spec.height),
                "out_px": tile_px,
                "composite_ready": True,
                "colors": {
                    "background": bg_color,
                    "base": base_color,
                    "alt": (alt_color if variant == "color_change" else None),
                },
                "indices": {
                    "diff_cell_ids": diff_ids,
                },
                "bounds": {
                    "min_diff": int(want_min),
                    "max_diff": int(want_max),
                },
                "sizes": {
                    "left_region": int(len(left_region)),
                    "right_region": int(len(right_region)),
                },
                "complexity": complexity,
                "complexity_score": complexity["score"],
                "complexity_level": complexity["level"],
                "complexity_version": complexity["version"],
            }
            return composite, [spec], meta

        raise RuntimeError(f"{self.name}: failed to sample a valid instance after {self.max_retries} attempts.")
