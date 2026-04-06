# sphinx/motifs/polyhex.py
import math
import random
from typing import List, Tuple
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

HexCell = Tuple[int, int]  # axial coordinates (q, r), pointy-top orientation


def _hex_neighbors(q: int, r: int) -> List[HexCell]:
    """
    Pointy-top axial neighbors for a regular hex grid.
      Using the standard axial basis:
        x = a * sqrt(3) * (q + r/2)
        y = a * 1.5 * r
    """
    return [
        (q + 1, r),     # +q
        (q - 1, r),     # -q
        (q, r + 1),     # +r
        (q, r - 1),     # -r
        (q + 1, r - 1), # +q -r
        (q - 1, r + 1), # -q +r
    ]


def _make_growth_path(seed: int, max_len: int) -> List[HexCell]:
    """
    Deterministic connected growth path of hex-cells (edge-adjacent).
    Returns a list of (q, r) axial cells; prefix of length k is a k-hex polyhex.
    """
    rng = random.Random(seed)
    start: HexCell = (0, 0)
    cells = {start}
    order: List[HexCell] = [start]

    while len(order) < max_len:
        front = set()
        for (q, r) in cells:
            for nb in _hex_neighbors(q, r):
                if nb not in cells:
                    front.add(nb)

        if not front:
            # extremely unlikely; but restart from a deterministic variant if it happens
            rng.seed(seed + len(order) * 9973)
            q, r = order[-1]
            front = {_hex_neighbors(q, r)[0]}

        nb = rng.choice(list(front))
        cells.add(nb)
        order.append(nb)

    # Make coordinates non-negative for nicer centering downstream
    minq = min(q for q, _ in order)
    minr = min(r for _, r in order)
    order = [(q - minq, r - minr) for (q, r) in order]
    return order


@register_motif
class PolyhexMotif(Motif):
    """
    Single polyhex (piece built from edge-adjacent regular hexagons).

    Attributes:
      - count: number of unit hexes (>=1)
      - thickness: outline thickness (integer, for crisp supersampled edges)
      - extra.scale: overall size multiplier
      - extra.rotation: rotation angle snapped to 60° increments (0..300)
      - extra.path: reusable axial growth path (list[(q,r)])
      - extra.gap_frac: centroid-scaling factor to create a small seam between hexes
    """
    name = "polyhex"
    attr_ranges = {"count": (3, 15), "thickness": (2, 4)}  # tweak to taste

    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(*self.attr_ranges["count"])
        thickness = rng.randint(*self.attr_ranges["thickness"])

        # Path long enough for any reasonable progression
        max_len = max(self.attr_ranges["count"][1], 18)
        path = _make_growth_path(seed, max_len)

        extra = {
            "scale": rng.uniform(0.95, 1.10),
            "rotation": rng.choice([0, 60, 120, 180, 240, 300]),
            "path": path,
            "gap_frac": rng.uniform(0.06, 0.12),
        }
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=count,
            thickness=thickness,
            size=1.0,
            extra=extra,
        )

    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        path = ex.get("path")
        if not path:
            path = _make_growth_path(spec.seed, max(self.attr_ranges["count"][1], 18))
            ex["path"] = path
        max_count = len(path)

        c = int(spec.count) if hasattr(spec, "count") else self.attr_ranges["count"][0]
        c = max(1, min(max_count, c))

        thick = max(1, int(getattr(spec, "thickness", self.attr_ranges["thickness"][0])))

        rot = int(float(ex.get("rotation", 0.0))) % 360
        ex["rotation"] = (rot // 60) * 60  # snap to multiples of 60°

        ex["scale"] = float(ex.get("scale", 1.0))
        ex["gap_frac"] = float(ex.get("gap_frac", 0.1))

        return spec.clone(count=c, thickness=thick, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)
        path: List[HexCell] = s.extra["path"]
        k = int(s.count)
        cells = path[:k]

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        color = COLORS[s.color_idx]

        # Layout
        scale = float(s.extra.get("scale", 1.0))
        rot = int(s.extra.get("rotation", 0)) % 360
        # Use a slightly larger margin because 60° rotations can grow the bbox
        margin = int(SS_CELL * 0.14)
        usable_w = SS_CELL - 2 * margin
        usable_h = SS_CELL - 2 * margin

        # Bounding in terms of the hex side length 'a'.
        # Pointy-top hex geometry:
        #   center.x = a * sqrt(3) * (q + r/2)
        #   center.y = a * 1.5 * r
        #   horizontal half-extent (apothem) = a * sqrt(3) / 2
        #   vertical half-extent (radius)    = a
        u_vals = [q + 0.5 * r for (q, r) in cells]
        min_u = min(u_vals)
        max_u = max(u_vals)
        min_r = min(r for _, r in cells)
        max_r = max(r for _, r in cells)
        u_range = max_u - min_u
        r_range = max_r - min_r

        # Total width/height of the union (axis-aligned) as functions of 'a'
        # width_total(a)  = sqrt(3)*a*(u_range + 1)
        # height_total(a) = a*(1.5*r_range + 2)
        eps = 1e-6
        a_from_w = usable_w / max(math.sqrt(3) * (u_range + 1.0), eps)
        a_from_h = usable_h / max(1.5 * r_range + 2.0, eps)
        a = int(max(2 * SUPERSAMPLE, min(a_from_w, a_from_h) * scale))

        # Ensure it fits after rounding
        def piece_size(side_len: int):
            w = math.sqrt(3) * side_len * (u_range + 1.0)
            h = side_len * (1.5 * r_range + 2.0)
            # be conservative with ceil so we don't clip
            return int(math.ceil(w)), int(math.ceil(h))

        pw, ph = piece_size(a)
        while (pw > usable_w or ph > usable_h) and a > 2 * SUPERSAMPLE:
            a -= 1
            pw, ph = piece_size(a)

        # Offsets: left/top boundary locations at this 'a'
        left_extent = math.sqrt(3) * a * (min_u - 0.5)  # cx_min - apothem
        top_extent = a * (1.5 * min_r - 1.0)            # cy_min - radius

        ox = margin + (usable_w - pw) // 2 - int(round(left_extent))
        oy = margin + (usable_h - ph) // 2 - int(round(top_extent))

        # Visual separation between hexes via centroid scaling
        gap_frac = float(s.extra.get("gap_frac", 0.1))
        inset_scale = max(0.6, 1.0 - gap_frac)  # keep reasonable solidity

        outline_w = max(1, int(s.thickness) * SUPERSAMPLE)

        def hex_vertices(cx: int, cy: int, side: float):
            # Pointy-top hex, vertex order clockwise starting at top
            ap = side                            # radius (center → point)
            at = (math.sqrt(3) / 2.0) * side     # apothem
            pts = [
                (cx,               cy - ap),  # top
                (cx + at,          cy - 0.5 * ap),
                (cx + at,          cy + 0.5 * ap),
                (cx,               cy + ap),
                (cx - at,          cy + 0.5 * ap),
                (cx - at,          cy - 0.5 * ap),
            ]
            # centroid scaling for the seam
            out = []
            for (x, y) in pts:
                sx = cx + (x - cx) * inset_scale
                sy = cy + (y - cy) * inset_scale
                out.append((int(round(sx)), int(round(sy))))
            return out

        # Draw each hex
        for (q, r) in cells:
            cx = ox + int(round(math.sqrt(3) * a * (q + 0.5 * r)))
            cy = oy + int(round(1.5 * a * r))
            verts = hex_vertices(cx, cy, float(a))
            # filled face
            draw.polygon(verts, fill=color)
            # crisp outline (polygon-outline width isn't consistent across Pillow versions)
            for i in range(6):
                x0, y0 = verts[i]
                x1, y1 = verts[(i + 1) % 6]
                draw.line((x0, y0, x1, y1), fill="black", width=outline_w)

        if rot:
            img = img.rotate(rot, resample=Image.NEAREST, expand=False)

        return _down_on_background(img)
