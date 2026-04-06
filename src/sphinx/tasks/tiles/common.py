# sphinx/tasks/tiles/common.py

"""Shared helpers and constants for tile-based tasks."""

from typing import Dict, List, Optional, Set

from sphinx.config import OUT_CELL

MIN_TILING_WH = 3
MAX_TILING_WH_DEFAULT = 10

# These have ~2× faces, so cap a bit lower
MAX_TILING_WH = {
    "triangular": 8,
    "rhombille": 8,
}


def _max_wh_for(tiling_name: str) -> int:
    """Return the maximum width/height for the given tiling name."""
    return MAX_TILING_WH.get(tiling_name, MAX_TILING_WH_DEFAULT)


def _components(g: Dict[int, Set[int]], allowed: Optional[Set[int]] = None) -> List[Set[int]]:
    """Return connected tiles (as node id sets) of the subgraph induced by 'allowed'."""
    if allowed is None:
        allowed = set(g.keys())
    seen: Set[int] = set()
    comps: List[Set[int]] = []
    for u in allowed:
        if u in seen:
            continue
        stack = [u]
        comp = {u}
        seen.add(u)
        while stack:
            x = stack.pop()
            for v in g[x]:
                if v in allowed and v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def _out_px_for_dims(w: int, h: int) -> int:
    """Scale output image size with the larger board dimension."""
    m = max(int(w), int(h))
    if m <= 6:
        return int(OUT_CELL)
    scale = 1.0 + (m - 6) / 6.0  # in [1.0, 2.0]
    return int(round(OUT_CELL * min(2.0, scale)))


__all__ = [
    "MIN_TILING_WH",
    "MAX_TILING_WH_DEFAULT",
    "MAX_TILING_WH",
    "_max_wh_for",
    "_components",
    "_out_px_for_dims",
]

