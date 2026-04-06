# sphinx/tilings/irregular.py
from __future__ import annotations
import math, random
from typing import Dict, List, Tuple

from ..schema import TilingSpec
from ..base import Tiling
from ..registry import register_tiling
from .helpers import _mk_patch

# ---------------------------------------------------------------------------
# Utility: unique vertex registry with coordinate quantization
# ---------------------------------------------------------------------------
class _VPool:
    def __init__(self, tol: float = 1e-9):
        self.tol = tol
        self.xy: List[Tuple[float,float]] = []
        self.map: Dict[Tuple[float,float], int] = {}

    def key(self, x: float, y: float) -> Tuple[float,float]:
        q = self.tol
        return (round(x/q)*q, round(y/q)*q)

    def add(self, x: float, y: float) -> int:
        k = self.key(x,y)
        i = self.map.get(k)
        if i is None:
            i = len(self.xy)
            self.xy.append((x,y))
            self.map[k] = i
        return i

# ---------------------------------------------------------------------------
# 1) Circle packing "tiling" (cells are disks approximated by regular m-gons)
#     - We place circle centers on a triangular lattice (densest equal-circle packing).
#     - We DO NOT fill gaps: faces are disjoint polygons approximating the circles.
#     - Connectivity (if needed) should treat cells as adjacent when their polygons
#       share a vertex (i.e., when two circles touch). See graph.build_dual_graph(..., connect_on_touch=True).
# ---------------------------------------------------------------------------
@register_tiling
class CirclePackingTiling(Tiling):
    name = "circles"
    supports_wythoffian = False

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        # Keep modest sizes by default; users can override width/height.
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(6,12), height=rng.randint(6,12),
                          uniform={"scheme": "same"})

    def generate(self, spec: TilingSpec) -> TilingPatch:
        s = self.clamp_spec(spec)
        # Geometric parameters
        r = 0.5                       # circle radius
        m = 24                        # polygon sides per circle (multiple of 6 to align with touch points)
        dtheta = 2.0*math.pi / m

        # Triangular lattice for circle centers: basis (2r, 0) and (r, sqrt(3) r)
        sqrt3 = math.sqrt(3.0)

        vpool = _VPool(tol=1e-9)
        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int,int]] = []

        for j in range(s.height):
            for i in range(s.width):
                cx = 2.0*r*i + (r if (j & 1) else 0.0)
                cy = sqrt3*r*j
                # Build an m-gon approximating the circle, with a vertex exactly on each
                # of the 6 touching directions (0, 60, 120, ... degrees).
                poly: List[int] = []
                for k in range(m):
                    ang = k*dtheta  # start at 0 so that 0,60,120,... are included
                    vx = cx + r*math.cos(ang)
                    vy = cy + r*math.sin(ang)
                    poly.append(vpool.add(vx, vy))
                faces.append(poly); kinds.append("circle"); coords.append((i,j))

        return _mk_patch(vpool.xy, faces, kinds, coords)

# ---------------------------------------------------------------------------
# 2) Rhombille tiling — 60°/120° rhombi. We implement it by subdividing
#    a flat-top hexagonal grid: each regular hexagon is partitioned into
#    three congruent rhombi meeting at the hex center.
# ---------------------------------------------------------------------------
@register_tiling
class RhombilleTiling(Tiling):
    name = "rhombille"
    supports_wythoffian = True  # admits 3-coloring by axial parity, like hex/tri

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(6,10), height=rng.randint(6,10),
                          uniform={"scheme": "wythoffian"})

    def generate(self, spec: TilingSpec) -> TilingPatch:
        s = self.clamp_spec(spec)

        R = 1.0  # same scale as hex tiling corner radius
        def hex_corners(cx, cy):
            ang0 = 0.0  # flat-top
            return [(cx + R*math.cos(ang0 + k*math.pi/3.0),
                     cy + R*math.sin(ang0 + k*math.pi/3.0)) for k in range(6)]

        vpool = _VPool(tol=1e-9)
        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int,int,int]] = []  # (q, r, sector)

        for r in range(s.height):
            for q in range(s.width):
                cx = (3.0/2.0) * R * q
                cy = (math.sqrt(3)/2.0) * (2*r + (q & 1)) * R
                corners = hex_corners(cx, cy)  # v0..v5
                cxy = (sum(x for x,_ in corners)/6.0, sum(y for _,y in corners)/6.0)
                vid = [vpool.add(x,y) for (x,y) in corners]
                cvid = vpool.add(*cxy)

                # Partition hex into three rhombi: [v0,v1,v2,c], [v2,v3,v4,c], [v4,v5,v0,c]
                triples = [(0,1,2),(2,3,4),(4,5,0)]
                for k,(a,b,c_) in enumerate(triples):
                    poly = [vid[a], vid[b], vid[c_], cvid]
                    faces.append(poly); kinds.append("rhombus"); coords.append((q,r,k))

        return _mk_patch(vpool.xy, faces, kinds, coords)

    # Simple 3-classing by axial (q - r) mod 3 like hexes
    def wythoffian_class_id(self, cell: Cell) -> int:
        q, r, _ = cell.coord
        return (q - r) % 3

# ---------------------------------------------------------------------------
# 3) Floret pentagonal tiling (6-fold pentille).
#    NOTE: This is a minimal, topology-correct constructive version that
#    aligns all edges to the three 60° directions and produces 5-sided faces
#    with one ~60° corner and four ~120° corners. It is implemented by
#    "pushing in" a short edge around each vertex of a triangular lattice.
#    The geometry is parameterized by two radii so that the limiting case
#    approaches the deltoidal trihexagonal tiling.
#
#    This is *not* a Wythoff-accurate dual of the snub trihexagonal tiling
#    (sr{3,6}) yet; it preserves the combinatorics (V3.3.3.3.6) and the
#    60°/120° edge directions. For most downstream tasks (graph structure,
#    coloration, connectivity) this suffices. If you need the exact
#    metric form, swap in a Wythoff-based generator later.
# ---------------------------------------------------------------------------

# @register_tiling
class FloretPentagonalTiling(Tiling):
    name = "floret_pentagonal"
    supports_wythoffian = True

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(6,10), height=rng.randint(6,10),
                          uniform={"scheme": "wythoffian"})

    def generate(self, spec: TilingSpec) -> TilingPatch:
        """
        TODO (implementation sketch):
          • Build the snub trihexagonal tiling sr{3,6} via a Wythoff construction in p6,
            or equivalently:
              - Start from a flat-top hex grid (centers at odd-q layout).
              - For each hex-edge, place a single "snub" point at a fixed fraction t of the
                way from one endpoint to the other, with a *global* chirality choice so
                that around each hex the snub points cycle consistently.
              - Polygons are formed by connecting each hex's 6 snub points (a chiral hexagon)
                and the 4 snub points around each original vertex (triangles).
          • Take the dual: place one vertex at the centroid of each face (hex or triangle)
            and connect centroids of faces that share an edge. The resulting faces are the
            desired 5-gons (V3.3.3.3.6), with four 120° and one 60° angles. See:
              - Wikipedia / Snub trihexagonal tiling (dual is the floret pentagonal tiling).
              - Polytope Wiki / Floret pentagonal tiling (edge/angle facts).
          • Choose scale t so that when the short edge length is 1, the long edges are 2,
            matching the canonical parameterization.

        For now we leave this unimplemented to avoid shipping a brittle approximation.
        """
        raise NotImplementedError("FloretPentagonalTiling.generate is planned (dual of sr{3,6}); see TODO in docstring.")


# ---------------------------------------------------------------------------
# 4) Voronoi tiling (jittered grid seeds; optional Lloyd relaxation)
#    - Visual: organic, convex polygons with varied side counts.
#    - Implementation: O(N^2) half-plane clipping (no external deps).
# ---------------------------------------------------------------------------
@register_tiling
class VoronoiTiling(Tiling):
    name = "voronoi"
    supports_wythoffian = False

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(6, 10), height=rng.randint(6, 10),
                          uniform={"scheme": "same",
                                   "jitter": 0.45,
                                   "relax_iters": 1})

    # -------------- small geometry helpers --------------
    @staticmethod
    def _clip_halfplane(poly: List[Tuple[float,float]], n: Tuple[float,float], c: float, eps=1e-12):
        """Clip convex/concave polygon by half-plane n·x <= c. Returns new list of points."""
        out: List[Tuple[float,float]] = []
        if not poly:
            return out
        nx, ny = n
        def sgn(pt):
            return nx*pt[0] + ny*pt[1] - c
        prev = poly[-1]
        sp = sgn(prev)
        for cur in poly:
            sc = sgn(cur)
            if sc <= eps:  # current is inside
                if sp > eps:
                    # edge crosses from outside to inside: add intersection
                    dx, dy = cur[0]-prev[0], cur[1]-prev[1]
                    den = nx*dx + ny*dy
                    if abs(den) > 1e-18:
                        t = (c - (nx*prev[0] + ny*prev[1])) / den
                        out.append((prev[0]+t*dx, prev[1]+t*dy))
                out.append(cur)
            elif sp <= eps:
                # leaving the half-plane: add intersection
                dx, dy = cur[0]-prev[0], cur[1]-prev[1]
                den = nx*dx + ny*dy
                if abs(den) > 1e-18:
                    t = (c - (nx*prev[0] + ny*prev[1])) / den
                    out.append((prev[0]+t*dx, prev[1]+t*dy))
            prev, sp = cur, sc
        return out

    @staticmethod
    def _poly_area_centroid(poly: List[Tuple[float,float]]):
        """Return (area, (cx, cy)) for a simple polygon; area signed."""
        if len(poly) < 3:
            return 0.0, (0.0, 0.0)
        A = 0.0
        Cx = 0.0
        Cy = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            cross = x1*y2 - x2*y1
            A += cross
            Cx += (x1 + x2) * cross
            Cy += (y1 + y2) * cross
        A *= 0.5
        if abs(A) < 1e-18:
            return 0.0, (poly[0][0], poly[0][1])
        return A, (Cx/(6.0*A), Cy/(6.0*A))

    def generate(self, spec: TilingSpec) -> TilingPatch:
        s = self.clamp_spec(spec)
        W, H = float(s.width), float(s.height)
        rng = random.Random(s.seed)

        jitter = float((getattr(s, "uniform", None) or {}).get("jitter", 0.45))
        jitter = max(0.0, min(0.49, jitter))  # keep seeds within their grid cells
        relax_iters = int((getattr(s, "uniform", None) or {}).get("relax_iters", 1))
        relax_iters = max(0, min(3, relax_iters))

        # Base seeds on a jittered width×height grid (one seed per cell)
        seeds = []
        for j in range(s.height):
            for i in range(s.width):
                jx = (rng.random()*2 - 1.0) * jitter
                jy = (rng.random()*2 - 1.0) * jitter
                x = i + 0.5 + jx
                y = j + 0.5 + jy
                seeds.append((x, y))

        # Lloyd relaxation (optional)
        def build_cells(cur_seeds):
            # Precompute constants for half-planes relative to each seed
            rect = [(0.0,0.0), (W,0.0), (W,H), (0.0,H)]
            cells: List[List[Tuple[float,float]]] = []
            for a_idx, a in enumerate(cur_seeds):
                poly = rect[:]  # start with bounding box
                ax, ay = a
                aa = ax*ax + ay*ay
                for b_idx, b in enumerate(cur_seeds):
                    if a_idx == b_idx:
                        continue
                    bx, by = b
                    n = (bx - ax, by - ay)                 # outward normal from a to b
                    c = (bx*bx + by*by - aa) * 0.5         # bisector
                    poly = self._clip_halfplane(poly, n, c)
                    if not poly:
                        break
                cells.append(poly)
            return cells

        cells = build_cells(seeds)
        for _ in range(relax_iters):
            # compute centroids and rebuild
            centroids = []
            for poly in cells:
                _, (cx, cy) = self._poly_area_centroid(poly)
                # clamp to domain to avoid drift
                centroids.append((min(max(cx, 0.0), W), min(max(cy, 0.0), H)))
            seeds = centroids
            cells = build_cells(seeds)

        # Build TilingPatch with vertex pooling
        vpool = _VPool(tol=1e-9)
        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int,int]] = []

        # Ensure CCW orientation (area positive)
        def is_ccw(poly):
            A = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i+1) % len(poly)]
                A += x1*y2 - x2*y1
            return A > 0

        idx = 0
        for poly in cells:
            if len(poly) < 3:
                continue
            if not is_ccw(poly):
                poly = list(reversed(poly))
            face = [vpool.add(x, y) for (x, y) in poly]
            faces.append(face)
            kinds.append("voronoi")
            coords.append((idx // s.width, idx % s.width))
            idx += 1

        return _mk_patch(vpool.xy, faces, kinds, coords)


# ---------------------------------------------------------------------------
# 5) Orthogonal split tiling (Mondrian-style rectangles)
#    - Visual: axis-aligned rectangles of varied sizes (very different from rhombi & circles).
# ---------------------------------------------------------------------------
@register_tiling
class OrthogonalSplitTiling(Tiling):
    name = "orthogonal_split"
    supports_wythoffian = False

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        # Aim for ~width*height rectangles
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(5, 9), height=rng.randint(5, 9),
                          uniform={"scheme": "same",
                                   "min_dim": 0.6})

    @staticmethod
    def _split_rect(rng: random.Random, rect, min_dim):
        x0, y0, x1, y1 = rect
        w = x1 - x0
        h = y1 - y0
        if w < 2*min_dim and h < 2*min_dim:
            return None  # too small to split safely
        # prefer splitting along the longer dimension
        if (w > h and w >= 2*min_dim) or (h < 2*min_dim):
            # vertical split at x = x0 + t*w
            lo = x0 + min_dim
            hi = x1 - min_dim
            if hi <= lo:
                return None
            t = rng.uniform(lo, hi)
            return ((x0, y0, t, y1), (t, y0, x1, y1))
        else:
            # horizontal split at y = y0 + t*h
            lo = y0 + min_dim
            hi = y1 - min_dim
            if hi <= lo:
                return None
            t = rng.uniform(lo, hi)
            return ((x0, y0, x1, t), (x0, t, x1, y1))

    def generate(self, spec: TilingSpec) -> TilingPatch:
        s = self.clamp_spec(spec)
        rng = random.Random(s.seed)
        W, H = float(s.width), float(s.height)
        target = max(4, s.width * s.height)  # roughly this many rectangles
        min_dim = float((getattr(s, "uniform", None) or {}).get("min_dim", 0.6))

        rects = [(0.0, 0.0, W, H)]
        # grow by splitting the currently largest rectangle
        guard = 0
        while len(rects) < target and guard < 10000:
            guard += 1
            # choose largest by area
            areas = [(i, (r[2]-r[0])*(r[3]-r[1])) for i, r in enumerate(rects)]
            areas.sort(key=lambda t: t[1], reverse=True)
            idx = areas[0][0]
            r = rects[idx]
            sp = self._split_rect(rng, r, min_dim)
            if sp is None:
                # cannot split; try next largest
                if len(areas) > 1:
                    idx = areas[1][0]
                    r = rects[idx]
                    sp = self._split_rect(rng, r, min_dim)
            if sp is None:
                break
            rects[idx] = sp[0]
            rects.append(sp[1])

        # Build patch
        vpool = _VPool(tol=1e-9)
        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int,int]] = []

        for i, (x0, y0, x1, y1) in enumerate(rects):
            poly = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
            face = [vpool.add(x, y) for (x, y) in poly]
            faces.append(face)
            kinds.append("rect")
            coords.append((i, 0))

        return _mk_patch(vpool.xy, faces, kinds, coords)
