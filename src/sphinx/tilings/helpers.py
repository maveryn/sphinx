# sphinx/tilings/helpers.py
from typing import List, Sequence, Iterable

from ..base import Vertex, Edge, Cell, TilingPatch


def _mk_patch(vertices_xy: Sequence[tuple],
              faces: Sequence[Sequence[int]],
              kinds: Sequence[str],
              coords: Sequence[tuple]) -> TilingPatch:
    """Build a :class:`TilingPatch` from raw vertex and face data."""
    vertices = [Vertex(i, xy) for i, xy in enumerate(vertices_xy)]
    cells = []
    for cid, (f, k, c) in enumerate(zip(faces, kinds, coords)):
        cells.append(Cell(cid, f, k, c))
    edges: List[Edge] = []  # dual graph is derived by shared boundaries
    return TilingPatch(
        vertices,
        edges,
        cells,
        {v.id: i for i, v in enumerate(vertices)},
        {},
        {c.id: i for i, c in enumerate(cells)},
    )


__all__ = ["_mk_patch"]

