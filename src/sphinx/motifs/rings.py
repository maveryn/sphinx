# sphinx/motifs/rings.py
import math, random
from PIL import Image, ImageDraw
from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

@register_motif
class RingsMotif(Motif):
    """
    Concentric rings vs. notched annulus, controlled by extra['mode']:

      • mode="sym"  : k concentric ring *outlines* (colored strokes).
      • mode="asym" : single filled annulus with a wedge-shaped notch at angle_deg.

    Extras (spec.extra):
      - mode       (str)  : "sym" or "asym"
      - angle_deg  (float): notch center angle in degrees (used in "asym", ignored in "sym")
      - inner_frac (float): (asym) inner radius as fraction of half-canvas
      - outer_frac (float): (asym) outer radius as fraction of half-canvas
      - notch_deg  (float): (asym) notch angular width in degrees
    """
    name = "rings"
    attr_ranges = {"count": (1, 8), "size": (0.9, 1.2), "thickness": (2, 4)}

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        mode = rng.choice(["sym", "asym"])
        extra = {"mode": mode}

        if mode == "asym":
            inner = rng.uniform(0.30, 0.45)
            outer = rng.uniform(0.55, 0.75)
            if inner >= outer:
                inner, outer = 0.35, 0.65
            extra.update({
                "inner_frac": inner,
                "outer_frac": outer,
                "notch_deg":  rng.uniform(28, 65),
                "angle_deg":  rng.uniform(0, 360),
            })
            count = 1   # single annulus in asym mode
        else:
            extra["angle_deg"] = rng.uniform(0, 360)  # exposed for weighting, not used in render
            count = rng.randint(*self.attr_ranges["count"])

        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=count,
            size=rng.uniform(*self.attr_ranges["size"]),
            angle=0.0,
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        mode = ex.get("mode", "sym")
        if mode not in ("sym", "asym"):
            mode = "sym"

        # clamp basics
        zlo, zhi = self.attr_ranges["size"]
        size = max(zlo, min(zhi, float(getattr(spec, "size", 1.0))))

        tlo, thi = self.attr_ranges["thickness"]
        thickness = max(int(tlo), min(int(thi), int(getattr(spec, "thickness", int(tlo)))))

        clo, chi = self.attr_ranges["count"]
        if mode == "sym":
            count = max(int(clo), min(int(chi), int(getattr(spec, "count", 3))))
        else:
            count = 1  # asym is a single annulus

        # orientation always normalized (used only in asym)
        ang = float(ex.get("angle_deg", 0.0)) % 360.0
        ex["angle_deg"] = ang
        ex["mode"] = mode

        if mode == "asym":
            inner = float(ex.get("inner_frac", 0.38))
            outer = float(ex.get("outer_frac", 0.65))
            notch = float(ex.get("notch_deg", 40.0))

            inner = max(0.15, min(0.80, inner))
            outer = max(0.20, min(0.95, outer))
            # ensure a visible ring band
            min_band = 0.06
            if outer - inner < min_band:
                grow = (min_band - (outer - inner)) * 0.5
                outer = min(0.95, outer + grow)
                inner = max(0.15, inner - grow)
            notch = max(12.0, min(110.0, notch))

            ex.update({"inner_frac": inner, "outer_frac": outer, "notch_deg": notch})
        else:
            # strip asym-only fields if present
            ex.pop("inner_frac", None)
            ex.pop("outer_frac", None)
            ex.pop("notch_deg", None)

        return spec.clone(count=count, size=size, thickness=thickness, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)

        cx, cy = SS_CELL / 2.0, SS_CELL / 2.0
        color = COLORS[s.color_idx]
        mode = s.extra["mode"]

        if mode == "sym":
            # k concentric ring outlines
            k = max(1, int(s.count))
            outer = SS_CELL * 0.40 * float(s.size)
            ring_w = max(2 * SUPERSAMPLE, int(outer / (2 * k)))

            # stroke width in pixels-at-SS
            width_px = max(2 * SUPERSAMPLE, int(max(1, round(float(s.thickness)))) * SUPERSAMPLE)

            for i in range(k):
                r = outer - i * (2 * ring_w)
                if r <= ring_w:
                    break
                bbox = (cx - r, cy - r, cx + r, cy + r)
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                d.ellipse((x0, y0, x1, y1), outline=color, width=width_px)

        else:
            # single notched annulus
            scl = float(getattr(s, "size", 1.0))
            half = SS_CELL / 2.0
            R0 = int(half * float(s.extra["inner_frac"]) * 0.9 * scl)
            R1 = int(half * float(s.extra["outer_frac"]) * 0.9 * scl)

            # pixel guardrails
            R1 = min(R1, int(half) - 2 * SUPERSAMPLE)
            R0 = max(2 * SUPERSAMPLE, min(R0, R1 - 3 * SUPERSAMPLE))

            ow = max(1, int(s.thickness) * SUPERSAMPLE)

            # Outer disk (filled), then punch inner hole and wedge by drawing transparent regions
            d.ellipse([int(cx - R1), int(cy - R1), int(cx + R1), int(cy + R1)],
                      fill=color, outline="black", width=ow)
            # inner hole
            d.ellipse([int(cx - R0), int(cy - R0), int(cx + R0), int(cy + R0)],
                      fill=(255, 255, 255, 0), outline=None)

            # notch (wedge)
            A = math.radians(float(s.extra["angle_deg"]))
            W = math.radians(float(s.extra.get("notch_deg", 40.0)))
            a0, a1 = A - W / 2.0, A + W / 2.0
            p0 = (int(cx + R1 * math.cos(a0)), int(cy + R1 * math.sin(a0)))
            p1 = (int(cx + R1 * math.cos(a1)), int(cy + R1 * math.sin(a1)))
            d.polygon([(int(cx), int(cy)), p0, p1], fill=(255, 255, 255, 0))

        return _down_on_background(img)
