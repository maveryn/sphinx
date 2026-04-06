# sphinx/utils/rng.py
import random
from typing import Sequence, Any, List, Mapping

def choice_weighted(rng: random.Random, items: Sequence[Any], weights: Sequence[float]):
    # Python <3.11 friendly helper
    total = float(sum(weights))
    r = rng.random() * total
    c = 0.0
    for it, w in zip(items, weights):
        c += w
        if r <= c:
            return it
    return items[-1]

def _dirichlet(rng: random.Random, n: int, alpha: float = 1.0) -> List[float]:
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(xs) or 1.0
    return [x / s for x in xs]


def weighted_order(
    rng: random.Random, items: Sequence[Any], weights: Mapping[Any, float]
) -> List[Any]:
    """Return ``items`` in a weighted random order.

    Uses the Efraimidis–Spirakis algorithm to generate a permutation without
    replacement where higher-weighted items are more likely to appear earlier in
    the result.
    """
    pairs = []
    for k in items:
        w = max(float(weights.get(k, 1.0)), 1e-9)
        u = max(rng.random(), 1e-12)
        pairs.append((u ** (1.0 / w), k))
    pairs.sort(reverse=True)
    return [k for _, k in pairs]
