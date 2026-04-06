# sphinx/tilings/graph.py
from __future__ import annotations
from typing import Dict, List, Tuple, Set
from ..base import TilingPatch

def _edge_key(a: Tuple[float,float], b: Tuple[float,float], q: float = 1e-6):
    ax, ay = round(a[0]/q)*q, round(a[1]/q)*q
    bx, by = round(b[0]/q)*q, round(b[1]/q)*q
    return tuple(sorted(((ax,ay), (bx,by))))

def _poly_edges(poly: List[Tuple[float,float]]):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i+1)%n]


def build_dual_graph(patch: TilingPatch, tol=1e-6, connect_on_touch: bool = False) -> Dict[int, Set[int]]:
    """Cells are nodes; edge when two cells share a boundary segment (up to quantization).
    If connect_on_touch=True, cells that meet at a single vertex are also considered adjacent.
    This is useful for circle packings (disks approximated by polygons) where adjacency occurs
    at tangency points rather than shared edges.
    """
    e2cells: Dict[Tuple, List[int]] = {}
    polys = patch.cell_polygons()
    for ci, poly in enumerate(polys):
        for a,b in _poly_edges(poly):
            k = _edge_key(a,b, tol)
            e2cells.setdefault(k, []).append(ci)

    g: Dict[int, Set[int]] = {i:set() for i in range(len(polys))}
    for cells in e2cells.values():
        if len(cells) == 2:
            a,b = cells
            g[a].add(b); g[b].add(a)

    if connect_on_touch:
        # Connect cells that share at least one vertex.
        v2cells: Dict[Tuple[float,float], Set[int]] = {}
        for ci, poly in enumerate(polys):
            for (x,y) in poly:
                key = _edge_key((x,y),(x,y), tol)[0]  # quantized point key
                v2cells.setdefault(key, set()).add(ci)
        for cells in v2cells.values():
            if len(cells) >= 2:
                clist = list(cells)
                for i in range(len(clist)):
                    for j in range(i+1, len(clist)):
                        a, b = clist[i], clist[j]
                        if a != b:
                            g[a].add(b); g[b].add(a)
    return g


def largest_color_component(g: Dict[int, Set[int]], cell_colors: List[str]) -> Tuple[int, Set[int]]:
    """Return (representative_node_id, node_ids) for the largest monochrome component."""
    seen: Set[int] = set()
    best = (-1, set())
    for u in g:
        if u in seen: continue
        col = cell_colors[u]
        stack = [u]; comp = set([u]); seen.add(u)
        while stack:
            x = stack.pop()
            for v in g[x]:
                if v not in seen and cell_colors[v] == col:
                    seen.add(v); comp.add(v); stack.append(v)
        if len(comp) > len(best[1]):
            best = (u, comp)
    return best

def reachable(g: Dict[int, Set[int]], src: int, dst: int, blocked: Set[int] | None = None) -> bool:
    blocked = blocked or set()
    if src in blocked or dst in blocked: return False
    q = [src]; seen = {src}
    while q:
        u = q.pop(0)
        if u == dst: return True
        for v in g[u]:
            if v not in seen and v not in blocked:
                seen.add(v); q.append(v)
    return False
