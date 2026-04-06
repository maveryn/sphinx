# sphinx/motifs/polygon.py
import math, random
from PIL import Image, ImageDraw
from ..base import Motif
from ..schema import MotifSpec
from ..utils.geom import polygon_area
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _non_self_intersecting_polygon, _down_on_background

@register_motif
class PolygonMotif(Motif):
    """
    Irregular vs regular polygon controlled by extra['mode']:

      • mode="asym": random non–self-intersecting polygon with N sides
      • mode="sym" : regular N-gon (equal edges), optionally rotated by extra['rotation'] degrees.

    Fields used:
      - size (float): overall scale multiplier for bounding radius.
      - thickness (int): outline width (scaled by SUPERSAMPLE).
      - extra:
          * sides (int): number of sides
          * mode  (str): "sym" or "asym"
          * rotation (float, deg) — used in "sym" mode

    """
    name = "polygon"
    attr_ranges = {
        "size": (1.0, 1.2),
        "thickness": (3, 6),
        "sides": (3, 6),
        "count": (1, 1),
    }

    # --- sampling ---
    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        sides = rng.randint(*self.attr_ranges["sides"])
        mode = rng.choice(["asym", "sym"])
        extra = {
            "sides": sides,
            "mode": mode,
            "rotation": rng.uniform(0, 360.0),  # primarily used in "sym" mode
        }
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,
            size=rng.uniform(*self.attr_ranges["size"]),
            angle=0.0,
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        # Preserve sampled extras instead of discarding them.
        ex = dict(spec.extra or {})

        # sides
        smin, smax = self.attr_ranges["sides"]
        sides = int(ex.get("sides", getattr(spec, "sides", 5)))
        sides = max(int(smin), min(int(smax), sides))

        # mode (keep the randomly sampled 'sym'/'asym' if present)
        mode = ex.get("mode", "asym")
        if mode not in ("asym", "sym"):
            mode = "asym"

        # rotation (normalize to [0, 360))
        rot = float(ex.get("rotation", 0.0)) % 360.0

        # size & thickness
        zmin, zmax = self.attr_ranges["size"]
        size = float(getattr(spec, "size", 1.0))
        size = max(float(zmin), min(float(zmax), size))

        tmin, tmax = self.attr_ranges["thickness"]
        thickness = int(getattr(spec, "thickness", int(tmin)))
        thickness = max(int(tmin), min(int(tmax), thickness))

        ex.update({"sides": sides, "mode": mode, "rotation": rot})
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        rng = random.Random(s.seed)

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # layout / bounds
        pad = int(20 * SUPERSAMPLE)
        cx, cy = SS_CELL / 2.0, SS_CELL / 2.0
        max_r_base = (min(SS_CELL - 2 * pad, SS_CELL - 2 * pad) / 2.0) * 0.95 * float(s.size)
        max_r_base = max(max_r_base, 0.15 * SS_CELL)

        sides = int(s.extra["sides"])
        mode = s.extra["mode"]
        color = COLORS[s.color_idx]
        outline_w = max(SUPERSAMPLE, int(s.thickness) * SUPERSAMPLE)

        target_area = float(0.1) * (SS_CELL * SS_CELL)

        if mode == "sym":
            # Regular N-gon. Choose circumradius to meet area target (or cap by bounds).
            # Area of regular polygon with circumradius R: A = 0.5 * n * R^2 * sin(2π/n)
            k = sides * math.sin(2.0 * math.pi / sides)
            if k <= 1e-9:
                k = 1e-9
            R_needed = math.sqrt(2.0 * target_area / k)
            R = min(max_r_base, R_needed * 1.02)  # slight slack
            rot = float(s.extra.get("rotation", 0.0))
            verts = [
                (
                    cx + R * math.cos(math.radians(rot) + 2.0 * math.pi * i / sides),
                    cy + R * math.sin(math.radians(rot) + 2.0 * math.pi * i / sides),
                )
                for i in range(sides)
            ]
        else:
            # mode == "asym": sample non-self-intersecting polygon; enforce area floor.
            bounds = (cx - max_r_base, cy - max_r_base, cx + max_r_base, cy + max_r_base)
            verts = None
            for _ in range(64):
                cand = _non_self_intersecting_polygon(rng, sides, bounds)
                if polygon_area(cand) >= target_area:
                    verts = cand
                    break
            if verts is None:
                # Fallback: regular N-gon sized by target area (same as 'sym', but randomized phase)
                k = sides * math.sin(2.0 * math.pi / sides) or 1e-6
                R_needed = math.sqrt(2.0 * target_area / k)
                R = min(max_r_base, R_needed * 1.02)
                phase = rng.random() * 2.0 * math.pi
                verts = [
                    (
                        cx + R * math.cos(phase + 2.0 * math.pi * i / sides),
                        cy + R * math.sin(phase + 2.0 * math.pi * i / sides),
                    )
                    for i in range(sides)
                ]

        draw.polygon(verts, fill=color, outline="black", width=outline_w)
        return _down_on_background(img)
