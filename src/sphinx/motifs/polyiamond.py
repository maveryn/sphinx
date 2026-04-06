# sphinx/motifs/polyiamond.py
import math
import random
from typing import List, Tuple
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

TriCell = Tuple[int, int, int]  # (gx, gy, orient) with orient: 0=up, 1=down


def _tri_neighbors(x: int, y: int, o: int) -> List[TriCell]:
    """
    Edge-adjacent neighbors on an equilateral triangle grid (edge-to-edge).

    Coordinate system (half-step lattice):
      - Horizontal unit = a/2
      - Vertical unit   = (sqrt(3)/2 * a) / 2
      - For UP (o=0): base is at y+2, for DOWN (o=1): base is at y-2.
      - Edge-sharing neighbors (derived so shared edges coincide exactly):
        UP(x,y):
          across base → DOWN(x,   y+4)
          left edge   → DOWN(x-1, y+2)
          right edge  → DOWN(x+1, y+2)
        DOWN(x,y):
          across base → UP(x,   y-4)
          left edge   → UP(x-1, y-2)
          right edge  → UP(x+1, y-2)
    """
    if o == 0:  # up
        return [(x, y + 4, 1), (x - 1, y + 2, 1), (x + 1, y + 2, 1)]
    else:       # down
        return [(x, y - 4, 0), (x - 1, y - 2, 0), (x + 1, y - 2, 0)]


def _make_growth_path(seed: int, max_len: int) -> List[TriCell]:
    """
    Deterministic connected growth path of triangles (edge-adjacent).
    Returns a list of (gx, gy, orient); the prefix of length k is a k-iamond.
    """
    rng = random.Random(seed)
    start: TriCell = (0, 0, 0)  # begin with an up-pointing triangle
    cells = {start}
    order: List[TriCell] = [start]

    while len(order) < max_len:
        front = set()
        for (x, y, o) in cells:
            for nb in _tri_neighbors(x, y, o):
                if nb not in cells:
                    front.add(nb)

        if not front:
            # Very unlikely; fall back to a deterministic neighbor from the last cell
            rng.seed(seed + len(order) * 9973)
            lx, ly, lo = order[-1]
            nb = _tri_neighbors(lx, ly, lo)[0]
        else:
            nb = rng.choice(list(front))

        cells.add(nb)
        order.append(nb)

    # Normalize to top-left for nicer centering
    minx = min(x for x, _, _ in order)
    miny = min(y for _, y, _ in order)
    order = [(x - minx, y - miny, o) for (x, y, o) in order]
    return order


@register_motif
class PolyiamondMotif(Motif):
    """
    Single polyiamond (piece built from edge-adjacent equilateral triangles).

    Attributes:
      - count: number of unit triangles (>=1)
      - thickness: outline thickness (integer, for crisp supersampled edges)
      - extra.scale: overall size multiplier (float)
      - extra.rotation: rotation angle snapped to 60° increments (0..300)
      - extra.path: reusable growth path (list[(gx,gy,orient)])
    """
    name = "polyiamond"
    attr_ranges = {"count": (3, 15), "thickness": (2, 4)}

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

        return spec.clone(count=c, thickness=thick, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)
        path: List[TriCell] = s.extra["path"]
        k = int(s.count)
        cells = path[:k]

        # Bounds in half-steps (units of a/2 horizontally, h/2 vertically)
        # For any triangle (gx,gy,o), x spans [gx-1, gx+1]; y spans:
        #   up:   [gy,   gy+2]
        #   down: [gy-2, gy  ]
        min_xu = min(gx - 1 for gx, _, _ in cells)
        max_xu = max(gx + 1 for gx, _, _ in cells)
        min_yu = min((gy if o == 0 else gy - 2) for _, gy, o in cells)
        max_yu = max((gy + 2 if o == 0 else gy) for _, gy, o in cells)

        width_units = max(1, max_xu - min_xu)     # in (a/2) steps
        height_units = max(1, max_yu - min_yu)    # in (h/2) steps

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        color = COLORS[s.color_idx]

        # Layout
        scale = float(s.extra.get("scale", 1.0))
        rot = int(s.extra.get("rotation", 0)) % 360
        margin = int(SS_CELL * 0.14)  # 60° rotations can enlarge bbox
        usable_w = SS_CELL - 2 * margin
        usable_h = SS_CELL - 2 * margin

        # Triangle metrics
        eps = 1e-6
        a_from_w = (2.0 * usable_w) / max(width_units, eps)
        a_from_h = (4.0 * usable_h) / (math.sqrt(3) * max(height_units, eps))
        a = int(max(2 * SUPERSAMPLE, min(a_from_w, a_from_h) * scale))

        def tri_height(side: int) -> int:
            return int(round(side * math.sqrt(3) / 2.0))

        h = tri_height(a)
        piece_w = (a // 2) * width_units + ((a % 2) and (width_units // 2))  # integer-safe
        piece_h = (h // 2) * height_units + ((h % 2) and (height_units // 2))

        while (piece_w > usable_w or piece_h > usable_h) and a > 2 * SUPERSAMPLE:
            a -= 1
            h = tri_height(a)
            piece_w = (a // 2) * width_units + ((a % 2) and (width_units // 2))
            piece_h = (h // 2) * height_units + ((h % 2) and (height_units // 2))

        # Convert grid half-steps to pixel coordinates; center within the cell
        ox = margin + (usable_w - piece_w) // 2 - int(min_xu * (a / 2.0))
        oy = margin + (usable_h - piece_h) // 2 - int(min_yu * (h / 2.0))

        outline_w = max(1, int(s.thickness) * SUPERSAMPLE)

        # Draw triangles exactly edge-to-edge (no insetting)
        for (gx, gy, o) in cells:
            if o == 0:  # up
                apex = (ox + int(round(gx * (a / 2.0))), oy + int(round(gy * (h / 2.0))))
                br = (ox + int(round((gx + 1) * (a / 2.0))), oy + int(round((gy + 2) * (h / 2.0))))
                bl = (ox + int(round((gx - 1) * (a / 2.0))), oy + int(round((gy + 2) * (h / 2.0))))
                verts = [apex, br, bl]
            else:       # down
                apex = (ox + int(round(gx * (a / 2.0))), oy + int(round(gy * (h / 2.0))))
                bl = (ox + int(round((gx - 1) * (a / 2.0))), oy + int(round((gy - 2) * (h / 2.0))))
                br = (ox + int(round((gx + 1) * (a / 2.0))), oy + int(round((gy - 2) * (h / 2.0))))
                verts = [apex, bl, br]

            # filled face
            draw.polygon(verts, fill=color)
            # crisp outline (explicit lines keep width consistent across Pillow versions)
            draw.line([verts[0], verts[1]], fill="black", width=outline_w)
            draw.line([verts[1], verts[2]], fill="black", width=outline_w)
            draw.line([verts[2], verts[0]], fill="black", width=outline_w)

        if rot:
            img = img.rotate(rot, resample=Image.NEAREST, expand=False)

        return _down_on_background(img)
