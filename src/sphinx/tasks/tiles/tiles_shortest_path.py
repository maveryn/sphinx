# sphinx/tasks/tiles/tiles_shortest_path.py
from __future__ import annotations
import random
import math
from typing import Any, Dict, Set, Tuple

from sphinx.base import Task
from .common import MIN_TILING_WH, _max_wh_for, _components, _out_px_for_dims
from sphinx.registry import register_task
from sphinx.utils.drawing import render_patch_crisp
from sphinx.config import MAX_BUILD_RETRIES, COLORS_NAMES
from ...utils.rng import choice_weighted
from ...utils.colors import color_label

COLORS = list(COLORS_NAMES)

from sphinx.tilings import (
    TilingSpec, create_tiling, get_tiling_names,
    build_dual_graph,  # edge-adjacency via shared boundary segments
)

# ----------------------------------------------------------------------
# Sampling knobs

# sampling weights for tiling families
TILING_WEIGHTS = {
    "square": 1.0,
    "triangular": 0.25,
    "hexagonal": 0.75,
    "circles": 1.0,
    "rhombille": 0.25,
    "voronoi": 0.0,
    "orthogonal_split": 0.0,
}


UNREACHABLE_PROB: float = 0.1  # 10% of the time, force unreachable
REGIME_WEIGHTS: Dict[str, float] = {
    "sparse": 0.30,
    "dense": 0.01,
    "balanced": 0.50,
    "patchy": 0.01,
}
# Obstacle mask constraints
MIN_OBSTACLES: int = 1
MIN_PASSABLE: int = 2

# Reachable distance shaping
REACHABLE_MIN_STEPS: int = 3            # require at least this many steps in reachable mode
PAIR_WEIGHTED_COMPONENTS: bool = True   # choose components ∝ number of unordered pairs

# Target length ~ Normal around the board "edge length" (≈ max(width, height))
TARGET_STEPS_SIGMA_FRAC: float = 0.35   # stdev as a fraction of scale_len
TARGET_STEPS_WINDOW_FRAC: float = 0.15  # acceptable deviation window (fraction of scale_len)
MAX_NEAR_TARGET_ATTEMPTS: int = 48      # tries to find a pair near the target distance

OBSTACLE_SHADE_CANDIDATES = ["#111111", "#1a1a1a", "#202020", "#262626", "#2c2c2c"]
PASSABLE_SHADE_CANDIDATES = ["#f9f9f9", "#f5f5f5", "#f0f0f0", "#ebebeb", "#e6e6e6"]


# ----------------------------- prompts -----------------------------
# Edge-adjacency (default for polygonal tilings)
PROMPT_TEMPLATES_EDGE = [
    "Cells are adjacent if they share an edge. Obstacles are dark gray; passable cells are light gray. Start is {start_label}, End is {end_label}. What is the minimum number of steps from Start to End? If no path exists, answer -1.",

    "Using edge adjacency, dark gray tiles are impassable and light gray tiles are passable. Start = {start_label}, End = {end_label}. What is the length of the shortest path in steps? If no path exists, answer -1.",

    "Cells that share a side are neighbors. Dark gray cells are obstacles; light gray cells can be traversed. From Start ({start_label}) to End ({end_label}), what is the minimum number of steps? If no path exists, answer -1.",

    "Movement is allowed only between cells sharing an edge. Dark gray tiles cannot be entered; light gray tiles are free. What is the shortest path length from Start ({start_label}) to End ({end_label})? If no path exists, answer -1.",

    "Under edge adjacency, dark gray cells block movement and only light gray cells can be used. Start = {start_label}, End = {end_label}. What is the fewest number of steps from Start to End? If no path exists, answer -1.",

    "Shared-edge connectivity applies: two cells are neighbors if they share a side. Dark gray cells are obstacles; light gray cells are passable. What is the shortest path length from Start ({start_label}) to End ({end_label})? If no path exists, answer -1.",

    "Neighbors are defined by shared edges. Dark gray cells are impassable; light gray cells are traversable. From Start = {start_label} to End = {end_label}, what is the minimum step count? If no path exists, answer -1.",

    "Using edge adjacency only, paths cannot pass through dark gray obstacles but may use light gray tiles. Start is {start_label}, End is {end_label}. What is the length of the shortest path? If no path exists, answer -1.",

    "Movement is restricted to side-sharing cells. Dark gray = blocked, light gray = free. From Start ({start_label}) to End ({end_label}), what is the minimum number of steps required? If no path exists, answer -1.",

    "With adjacency defined by shared edges, dark gray tiles are obstacles and light gray tiles are traversable. Start = {start_label}, End = {end_label}. What is the minimum path length in steps? If no path exists, answer -1."
]


# Point-touch (tangency) adjacency — used for circle packing tiling
PROMPT_TEMPLATES_TOUCH = [
    "Cells are adjacent if they touch at a point (tangency). Obstacles are dark gray; passable cells are light gray. Start is {start_label}, End is {end_label}. What is the minimum number of steps from Start to End? If no path exists, answer -1.",

    "Using point-touch adjacency, dark gray tiles are impassable and light gray tiles are passable. Start = {start_label}, End = {end_label}. What is the length of the shortest path in steps? If no path exists, answer -1.",

    "Cells that touch even at a single point are neighbors. Dark gray cells are obstacles; light gray cells are free. From Start ({start_label}) to End ({end_label}), what is the minimum number of steps? If no path exists, answer -1.",

    "Movement is allowed only between cells that meet at a point. Dark gray tiles cannot be entered; light gray tiles are traversable. What is the shortest path length from {start_label} (Start) to {end_label} (End)? If no path exists, answer -1.",

    "Under tangency adjacency (point-touch, not shared edges), dark gray cells block movement and only light gray cells can be used. Start = {start_label}, End = {end_label}. What is the fewest number of steps from Start to End? If no path exists, answer -1.",

    "Point-touch connectivity applies: two cells are neighbors if they meet at a corner. With dark gray obstacles and light gray passable tiles, what is the fewest steps from Start ({start_label}) to End ({end_label})? If no path exists, answer -1.",

    "Tiles that touch at a point are adjacent. Dark gray = blocked, light gray = allowed. What is the minimum number of steps from Start [{start_label}] to End [{end_label}]? If no path exists, answer -1.",

    "Using tangency adjacency only, dark gray tiles are impassable while light gray tiles are passable. Start is {start_label}; End is {end_label}. What is the shortest path length in steps? If no path exists, answer -1.",

    "With point-touch adjacency and avoiding dark gray obstacles, movement is through light gray tiles. From Start ({start_label}) to End ({end_label}), what is the minimal number of steps? If no path exists, answer -1.",

    "Movement is defined by point-touch adjacency. Dark gray tiles are obstacles; light gray tiles are traversable. Start = {start_label}, End = {end_label}. What is the shortest path length in steps? If no path exists, answer -1."
]



def _bfs_distance(g, src: int, dst: int, blocked: Set[int]) -> int:
    """Shortest number of edge-steps from src to dst avoiding 'blocked'. Returns -1 if unreachable."""
    if src in blocked or dst in blocked:
        return -1
    if src == dst:
        return 0
    from collections import deque
    q = deque([src])
    dist = {src: 0}
    while q:
        u = q.popleft()
        du = dist[u]
        for v in g[u]:
            if v in blocked or v in dist:
                continue
            dist[v] = du + 1
            if v == dst:
                return dist[v]
            q.append(v)
    return -1

# ---------- helpers for distance shaping ----------
def _sample_target_steps(rng: random.Random, scale_len: int, min_steps: int, max_steps: int) -> Tuple[int, float, float]:
    """Draw an integer target path length ~ Normal(mu=scale_len, sigma=fraction*scale_len), truncated."""
    mu = float(scale_len)
    sigma = max(1.0, float(scale_len) * TARGET_STEPS_SIGMA_FRAC)
    for _ in range(8):
        x = rng.normalvariate(mu, sigma)
        L = int(round(x))
        if min_steps <= L <= max_steps:
            return L, mu, sigma
    L = min(max_steps, max(min_steps, int(round(mu))))
    return L, mu, sigma

def _bfs_dists_within_comp(g, src: int, comp: Set[int], blocked: Set[int]) -> Dict[int, int]:
    from collections import deque
    q = deque([src])
    dist = {src: 0}
    seen = {src}
    while q:
        u = q.popleft()
        for v in g[u]:
            if v in blocked or v in seen or v not in comp:
                continue
            seen.add(v)
            dist[v] = dist[u] + 1
            q.append(v)
    return dist

def _far_pair_and_diameter(g, comp: Set[int], blocked: Set[int], rng: random.Random) -> Tuple[int, int, int]:
    """Double-sweep BFS to get a near-diameter pair within 'comp'."""
    s = rng.choice(tuple(comp))
    d0 = _bfs_dists_within_comp(g, s, comp, blocked)
    a = max(d0, key=d0.get)
    d1 = _bfs_dists_within_comp(g, a, comp, blocked)
    b = max(d1, key=d1.get)
    diam = d1[b]
    return a, b, diam

def _pick_pair_with_target_distance(
    rng: random.Random,
    g,
    comp: Set[int],
    blocked: Set[int],
    target_L: int,
    window_steps: int,
    attempts: int = MAX_NEAR_TARGET_ATTEMPTS,
) -> Tuple[int, int, int]:
    """
    Try to find (s,e) in 'comp' with dist ≈ target_L; expand tolerance if needed.
    Returns (s,e,dist). Falls back to the closest near-target pair found, else diameter-based.
    """
    nodes = tuple(comp)
    best_pair = None
    best_gap = None

    # Try multiple random anchors; progressively widen the window.
    for _ in range(attempts):
        s = rng.choice(nodes)
        dmap = _bfs_dists_within_comp(g, s, comp, blocked)
        if not dmap or len(dmap) <= 1:
            continue

        # Progressive widening around target
        base = max(window_steps, 1)
        for rad in range(base, base * 3 + 1):
            candidates = [v for v, d in dmap.items() if v != s and abs(d - target_L) <= rad]
            if candidates:
                e = rng.choice(candidates)
                d = dmap[e]
                gap = abs(d - target_L)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_pair = (s, e, d)
                # If we hit within the base window, accept immediately.
                if gap <= window_steps:
                    return s, e, d
                break

    # Use the best near-target candidate if we found one
    if best_pair is not None:
        return best_pair

    # Last resort: move from a diameter endpoint toward the target
    a, b, diam = _far_pair_and_diameter(g, comp, blocked, rng)
    dmap = _bfs_dists_within_comp(g, a, comp, blocked)
    if dmap:
        target = min((v for v in dmap if v != a), key=lambda v: abs(dmap[v] - target_L))
        return a, target, dmap[target]

    # Very degenerate fallback
    s, e = rng.sample(nodes, 2)
    d = _bfs_distance(g, s, e, blocked)
    return s, e, d

# ---------- connectivity helpers ----------

# ----------------------------- task -----------------------------
@register_task
class TilesShortestPathTask(Task):
    """
    Show a single tiling. Two special tiles are Start and End (distinct colors).
    Some tiles are obstacles (dark gray) and cannot be entered; others are passable (light gray).
    Ask for the minimum number of edge-steps from Start to End while avoiding obstacles.
    If no path exists, the correct answer is -1.
    """
    name = "tiles_shortest_path"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _compute_complexity(self, width: int, height: int, tiling_name: str) -> Dict[str, Any]:
        """Normalize board size (cell count) to EASY/HARD difficulty."""
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
            "version": "tiles-shortest-path-board-cells-v1",
            "range": {"min_cells": min_cells, "max_cells": max_cells},
            "cell_count": int(cell_count),
            "width": int(width),
            "height": int(height),
        }

    def _sample_tiling(self, rng: random.Random):
        """Sample a tiling patch and build its dual graph."""
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

        adjacency_mode = "touch" if tiling.name == "circles" else "edge"
        g = build_dual_graph(patch, connect_on_touch=(adjacency_mode == "touch"))

        return tiling, spec, patch, g, adjacency_mode

    def _sample_obstacle_field(self, rng: random.Random, n: int) -> Tuple[Set[int], float, str]:
        """
        Sample an obstacle set using Beta(alpha_obs, alpha_free) (Dirichlet-2).
        Enforce: at least MIN_OBSTACLES obstacles and at least MIN_PASSABLE passable cells.
        """
        regimes = {
            "sparse": (0.35, 2.7),   # low obstacle probability on average
            "dense":  (2.7, 0.35),   # high obstacle probability on average
            "balanced": (1.2, 1.2),  # around 50/50 but still variable
            "patchy": (0.6, 0.6),    # high variance, tends to near-0 or near-1
        }
        regime_names = list(regimes.keys())
        regime_w = [REGIME_WEIGHTS.get(k, 1.0) for k in regime_names]
        regime = choice_weighted(rng, regime_names, regime_w)

        a_obs, a_free = regimes[regime]
        min_blocked = max(MIN_OBSTACLES, 0)
        max_blocked = max(0, n - MIN_PASSABLE)  # leave room for Start/End

        # Try multiple times to satisfy counts under the chosen regime.
        for _ in range(64):
            p_obs = rng.betavariate(a_obs, a_free)
            blocked = {i for i in range(n) if rng.random() < p_obs}
            k = len(blocked)
            if min_blocked <= k <= max_blocked:
                return blocked, p_obs, regime

        # Fallback (rare): force exactly 'need' obstacles uniformly at random.
        need = min(max(min_blocked, 0), max_blocked)
        blocked = set(rng.sample(range(n), need)) if need > 0 else set()
        eff_p = (need / float(n)) if n else 0.0
        return blocked, eff_p, regime

    # ----------------------------- public API -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        want_unreachable = (rng.random() < UNREACHABLE_PROB)

        for _ in range(self.max_retries):
            tiling, spec, patch, g, adjacency_mode = self._sample_tiling(rng)
            n = len(patch.cells)

            # Try multiple masks / starts / ends against the chosen target
            for _inner in range(self.max_retries * 3):
                blocked, p_obs, regime = self._sample_obstacle_field(rng, n)

                # Need at least two passable cells for Start/End
                passable = [i for i in range(n) if i not in blocked]
                if len(passable) < 2:
                    continue

                comps = _components(g, set(passable))

                # Initialize reachable-shaping provenance (only set in reachable mode)
                target_L = mu_steps = sigma_steps = None  # type: ignore

                if want_unreachable:
                    # Require at least two distinct passable components
                    if len(comps) < 2:
                        continue
                    c1, c2 = rng.sample(comps, 2)
                    start = rng.choice(tuple(c1))
                    end = rng.choice(tuple(c2))
                else:
                    # Candidate components with at least two nodes
                    viable = [c for c in comps if len(c) >= 2]
                    if not viable:
                        continue

                    # Precompute near-diameter info per component
                    info = []
                    max_diam = 0
                    for c in viable:
                        a, b, diam = _far_pair_and_diameter(g, c, blocked, rng)
                        info.append((c, a, b, diam))
                        if diam > max_diam:
                            max_diam = diam

                    # Sample desired path length ~ Normal around the board's edge length
                    scale_len = max(spec.width, spec.height)
                    min_steps = max(1, REACHABLE_MIN_STEPS)
                    if max_diam < min_steps:
                        continue  # no component can host a long enough path
                    target_L, mu_steps, sigma_steps = _sample_target_steps(rng, scale_len, min_steps, max_diam)
                    win = max(1, int(round(TARGET_STEPS_WINDOW_FRAC * scale_len)))

                    # Choose a component whose diameter is close to the target, favoring larger comps
                    comp_items = []
                    comp_weights = []
                    for (c, _a, _b, diam) in info:
                        pair_weight = (len(c) * (len(c) - 1) // 2) if PAIR_WEIGHTED_COMPONENTS else 1.0
                        proximity = math.exp(-((diam - target_L) ** 2) / (2.0 * (sigma_steps ** 2))) + 1e-12
                        comp_items.append(c)
                        comp_weights.append(pair_weight * proximity)

                    if sum(comp_weights) == 0:
                        continue
                    comp = choice_weighted(rng, comp_items, comp_weights)

                    # Try to realize the target length inside this component (avoid boundary bias)
                    start, end, dist = _pick_pair_with_target_distance(rng, g, comp, blocked, target_L, win)

                    # Enforce minimum steps
                    if dist < min_steps:
                        continue

                # BFS for the step count; this will match the chosen mode by construction
                dist = _bfs_distance(g, start, end, blocked)
                if want_unreachable and dist != -1:
                    continue
                if (not want_unreachable) and dist == -1:
                    continue

                # Choose shades (not pure black/white) and marker colors
                obstacle_col = rng.choice(OBSTACLE_SHADE_CANDIDATES)
                passable_col = rng.choice(PASSABLE_SHADE_CANDIDATES)

                kmax = len(COLORS)
                if kmax >= 2:
                    idxs = rng.sample(range(kmax), 2)
                    start_col, end_col = COLORS[idxs[0]], COLORS[idxs[1]]
                else:
                    start_col, end_col = "#1f77b4", "#d62728"  # fallback

                # Paint cells
                cell_colors = [passable_col] * n
                for i in blocked:
                    cell_colors[i] = obstacle_col
                cell_colors[start] = start_col
                cell_colors[end] = end_col

                slabel = color_label(start_col)
                elabel = color_label(end_col)

                templates = PROMPT_TEMPLATES_TOUCH if adjacency_mode == "touch" else PROMPT_TEMPLATES_EDGE
                q_template = rng.choice(templates)
                question = q_template.format(start_label=slabel, end_label=elabel)

                tile_px = _out_px_for_dims(spec.width, spec.height)

                composite = render_patch_crisp(
                    patch,
                    cell_colors,
                    size_px=tile_px,
                    background="white",
                    outline_rgba=(0, 0, 0, 255),
                    outline_px=1,
                )

                complexity = self._compute_complexity(spec.width, spec.height, tiling.name)
                meta = {
                    "pattern_kind": "tiles",
                    "pattern": self.name,
                    "grid": (1, 1),
                    "variant": {"measure": "shortest_path_steps", "scope": "start_to_end"},
                    "question": question,
                    "answer": int(dist),
                    "tiling_kind": tiling.name,
                    "adjacency_mode": adjacency_mode,   # <<< provenance
                    "dims": (spec.width, spec.height),
                    "out_px": tile_px,
                    "composite_ready": True,
                    "colors": {
                        "obstacle": obstacle_col,
                        "passable": passable_col,
                        "start": start_col,
                        "end": end_col,
                    },
                    "mask_stats": {
                        "p_obstacle": p_obs,
                        "regime": regime,
                        "blocked_frac": len(blocked) / float(n),
                    },
                    "indices": {"start": start, "end": end},
                    "target_unreachable": bool(want_unreachable),
                    "complexity": complexity,
                    "complexity_score": complexity["score"],
                    "complexity_level": complexity["level"],
                    "complexity_version": complexity["version"],
                }

                # Only add reachable shaping provenance when applicable
                if (not want_unreachable) and (target_L is not None):
                    meta.update({
                        "target_steps": int(target_L),
                        "target_steps_mu": float(mu_steps),
                        "target_steps_sigma": float(sigma_steps),
                    })

                return composite, [spec], meta

        raise RuntimeError(f"{self.name}: failed to meet sampling target after {self.max_retries} attempts.")
