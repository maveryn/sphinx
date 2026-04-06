# sphinx/tilings/regular.py
from __future__ import annotations
import math, random
from typing import Dict, List, Tuple


from ..schema import TilingSpec
from ..base import Tiling
from ..registry import register_tiling
from .helpers import _mk_patch

# -------------------------------
# Square tiling
# -------------------------------
@register_tiling
class SquareTiling(Tiling):
    name = "square"
    supports_wythoffian = True

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(6,12), height=rng.randint(6,12),
                          uniform={"scheme": "wythoffian"})

    def generate(self, spec: TilingSpec) -> TilingPatch:
        s = self.clamp_spec(spec)
        verts: List[Tuple[float,float]] = []
        vid: Dict[Tuple[int,int], int] = {}
        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int,int]] = []

        for j in range(s.height+1):
            for i in range(s.width+1):
                vid[(i,j)] = len(verts); verts.append((i, j))
        for j in range(s.height):
            for i in range(s.width):
                a = vid[(i, j)]
                b = vid[(i+1, j)]
                c = vid[(i+1, j+1)]
                d = vid[(i, j+1)]
                faces.append([a,b,c,d]); kinds.append("square"); coords.append((i,j))
        return _mk_patch(verts, faces, kinds, coords)

    # 4-class parity on the checkerboard grid — good proxy for Wythoff orbits
    def wythoffian_class_id(self, cell: Cell) -> int:
        i,j = cell.coord
        return ((i & 1) << 1) | (j & 1)

# -------------------------------
# Triangular tiling (equilateral)
# -------------------------------
@register_tiling
class TriangularTiling(Tiling):
    name = "triangular"
    supports_wythoffian = True

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(8,14), height=rng.randint(8,14),
                          uniform={"scheme": "wythoffian"})

    def generate(self, spec: TilingSpec) -> TilingPatch:
        """
        Axis-aligned, point-up/point-down equilateral triangles.
        Rows are horizontal, every other row is shifted by L/2 horizontally.
        The patch silhouette is rectangular (with sawtooth edges), not a rhombus.
        """
        s = self.clamp_spec(spec)

        L = 1.0                       # side length
        H = math.sqrt(3.0) / 2.0 * L  # row-to-row vertical step

        verts: List[Tuple[float, float]] = []
        vid: Dict[Tuple[int, int], int] = {}

        def V(col: int, row: int) -> int:
            """
            Vertex at integer 'col' on given 'row', with odd rows offset by L/2.
            col is allowed to go up to width+1 on odd rows to close the strip.
            """
            key = (col, row)
            if key not in vid:
                x = L * (col + 0.5 * (row & 1))
                y = H * row
                vid[key] = len(verts)
                verts.append((x, y))
            return vid[key]

        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int, int]] = []

        # Build 'height' strips; each strip between row r and r+1 contributes 2 * width triangles
        for r in range(s.height):
            for i in range(s.width):
                # vertices on row r
                a = V(i,     r)
                b = V(i + 1, r)
                # vertices on row r+1 (shifted by half a cell when r is odd)
                c = V(i + (r & 1),     r + 1)
                d = V(i + 1 + (r & 1), r + 1)

                # Up triangle (base on row r, apex on row r+1)
                faces.append([a, b, c]); kinds.append("triangle")
                coords.append((2 * i + (r & 1), 2 * r))

                # Down triangle (base on row r+1, apex on row r)
                faces.append([b, d, c]); kinds.append("triangle")
                coords.append((2 * i + 1 + (r & 1), 2 * r))

        return _mk_patch(verts, faces, kinds, coords)

    # 3-coloring by (i + 2j) mod 3 (kept the same contract)
    def wythoffian_class_id(self, cell: Cell) -> int:
        i, j = cell.coord
        return (i + 2 * j) % 3

# -------------------------------
# Hexagonal tiling (flat-top, odd-q)
# -------------------------------
@register_tiling
class HexagonalTiling(Tiling):
    name = "hexagonal"
    supports_wythoffian = True

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        return TilingSpec(self.name, rng.randint(0, 2**31-1),
                          width=rng.randint(6,10), height=rng.randint(6,10),
                          uniform={"scheme": "wythoffian"})

    def generate(self, spec: TilingSpec) -> TilingPatch:
        s = self.clamp_spec(spec)

        # Use the same R for both corners and center spacing.
        R = 1.0  # corner radius (distance center→vertex); absolute scale doesn't matter

        def hex_corners(cx, cy):
            ang0 = 0.0  # <<< flat-top orientation
            return [(cx + R*math.cos(ang0 + k*math.pi/3.0),
                     cy + R*math.sin(ang0 + k*math.pi/3.0)) for k in range(6)]

        verts: List[Tuple[float,float]] = []
        faces: List[List[int]] = []
        kinds: List[str] = []
        coords: List[Tuple[int,int]] = []

        # odd-q vertical offset layout for flat-top hexes
        for r in range(s.height):
            for q in range(s.width):
                cx = (3.0/2.0) * R * q
                cy = (math.sqrt(3)/2.0) * (2*r + (q & 1)) * R
                corners = hex_corners(cx, cy)
                base = len(verts)
                verts.extend(corners)
                faces.append([base + k for k in range(6)])
                kinds.append("hex")
                coords.append((q, r))

        return _mk_patch(verts, faces, kinds, coords)

    # 3-coloring by (q - r) mod 3 (axial coords) — unchanged
    def wythoffian_class_id(self, cell: Cell) -> int:
        q, r = cell.coord
        return (q - r) % 3
