# sphinx/charts/common.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    # Optional: project palette (used for colored charts)
    from sphinx.config import COLORS  # type: ignore[attr-defined]
except Exception:
    # Fallback palette
    COLORS = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

CHART_MIN_K = 3
CHART_MAX_K = 10


def compute_chart_complexity(k: int) -> Dict[str, Any]:
    """Return normalized complexity metadata for charts parameterized by category count."""
    span = max(1, CHART_MAX_K - CHART_MIN_K)
    normalized = (k - CHART_MIN_K) / span
    normalized = max(0.0, min(1.0, normalized))

    if normalized < 0.75:
        level = "EASY"
    else:
        level = "HARD"

    return {
        "score": normalized,
        "level": level,
        "version": "charts-k-v1",
        "range": {"min_k": CHART_MIN_K, "max_k": CHART_MAX_K},
        "k": int(k),
    }


@dataclass
class ChartSpec:
    # generation
    seed: int
    chart_type: str                 # 'pie' | 'bar'
    labels: List[str]               # ['A','B',...]
    value_kind: str                 # 'count' | 'percentage'
    counts: List[int]               # counts or mirror of percentages
    percentages_int: List[int]      # integer percentages (sum to 100)
    colors: List[str]               # distinct colors
    color_mode: str                 # 'distinct'
    # rendering defaults
    width_px: int = 800
    height_px: int = 600
    render_mode: str = "color"

# ----------------------------- sampling ----------------------------------

def sample_category_labels(rng: random.Random, k: int) -> List[str]:
    import string
    letters = list(string.ascii_uppercase)
    rng.shuffle(letters)
    return letters[:k]

def _largest_remainder_integer_percentages(weights: List[float]) -> List[int]:
    tot = float(sum(weights)) or 1.0
    exact = [100.0 * (w / tot) for w in weights]
    floors = [int(x) for x in exact]
    rems = [x - f for x, f in zip(exact, floors)]
    need = 100 - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: (-rems[i], i))
    for i in range(need):
        floors[order[i % len(weights)]] += 1
    return floors

def sample_percentages_int(rng: random.Random, k: int, *, enforce_min1: bool = True) -> List[int]:
    w = [rng.random() + 1e-6 for _ in range(k)]
    p = _largest_remainder_integer_percentages(w)
    if enforce_min1:
        zeros = [i for i, v in enumerate(p) if v == 0]
        for z in zeros:
            j = max(range(k), key=lambda i: p[i])
            if p[j] <= 1:
                return sample_percentages_int(rng, k, enforce_min1=True)
            p[j] -= 1
            p[z] += 1
    return p

def sample_counts_and_percentages(
    rng: random.Random, k: int, *, total_min: int = 40, total_max: int = 200, enforce_min1: bool = True
) -> Tuple[List[int], List[int]]:
    total = rng.randint(max(total_min, k if enforce_min1 else 1), total_max)
    w = [rng.random() + 1e-6 for _ in range(k)]
    exact = [total * (x / sum(w)) for x in w]
    floors = [int(e) for e in exact]
    need = total - sum(floors)
    rems = [e - f for e, f in zip(exact, floors)]
    order = sorted(range(k), key=lambda i: (-rems[i], i))
    for i in range(need):
        floors[order[i % k]] += 1
    if enforce_min1:
        for i in range(k):
            if floors[i] < 1:
                j = max(range(k), key=lambda t: floors[t])
                floors[i] += 1
                floors[j] -= 1
    perc = _largest_remainder_integer_percentages(floors)
    return floors, perc

def choose_colors(rng: random.Random, k: int, mode: str = "distinct") -> Tuple[List[str], str]:
    """Always return k distinct colors; synthesize extras if needed."""
    n = len(COLORS)
    if n >= k:
        idxs = rng.sample(range(n), k)
        cols = [COLORS[i] for i in idxs]
    else:
        idxs = list(range(n)); rng.shuffle(idxs)
        cols = [COLORS[i] for i in idxs]
        # HSV sweep for extras (distinct-ish)
        import colorsys
        for i in range(k - n):
            h = (i + 0.31) / max(1, k - n)
            r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.90)
            cols.append("#%02x%02x%02x" % (int(255*r), int(255*g), int(255*b)))
    return cols, "distinct"
