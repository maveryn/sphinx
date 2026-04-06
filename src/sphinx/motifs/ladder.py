# sphinx/motifs/ladder.py
# sphinx/motifs/ladder.py
import random
from PIL import Image, ImageDraw
from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

@register_motif
class LadderMotif(Motif):
    """
    Two vertical rails with evenly spaced horizontal rungs.

    Conventions:
      - `spec.count` controls the number of rungs (≥3 by default).
      - Rotation is limited to {0, 90} for clear orientation differences, not counted towards DoF

    Extras (spec.extra):
      - scale         (float): global size multiplier affecting vertical margins.
      - rotation      (int):   0 or 90 degrees.
      - rail_gap_frac (float): rail separation as a fraction of full width (≈ 0.32–0.40).
    """
    name = "ladder"
    attr_ranges = {"count": (3, 9), "thickness": (2, 5)}

    # --- sampling ---
    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(*self.attr_ranges["count"])
        thickness = rng.randint(*self.attr_ranges["thickness"])
        extra = {
            "scale": rng.uniform(0.9, 1.1),
            "rotation": rng.choice([0, 90]),
            "rail_gap_frac": rng.uniform(0.32, 0.40),
        }
        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=count, thickness=thickness, size=1.0, extra=extra
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        ex.pop("mode", None)  # ignore generic symmetry-mode flags if present

        cmin, cmax = self.attr_ranges["count"]
        count = max(int(cmin), min(int(cmax), int(getattr(spec, "count", 6))))
        thick = max(1, int(getattr(spec, "thickness", 3)))

        scale = float(ex.get("scale", 1.0))
        scale = max(0.75, min(1.30, scale))

        rot = int(float(ex.get("rotation", 0))) % 360
        rot = 90 if 45 <= rot < 135 or 225 <= rot < 315 else 0  # snap to {0,90}

        gap = float(ex.get("rail_gap_frac", 0.36))
        gap = max(0.25, min(0.45, gap))  # keep rails inside the canvas comfortably

        ex.update({"scale": scale, "rotation": rot, "rail_gap_frac": gap})
        return spec.clone(count=count, thickness=thick, size=1.0, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        color = COLORS[s.color_idx]

        scale = float(s.extra["scale"])
        rot = int(s.extra["rotation"])
        thick_px = max(1, int(s.thickness) * SUPERSAMPLE)

        # Vertical margins for rungs; horizontal rails are positioned by gap fraction
        M = int(SS_CELL * 0.16 * scale)    # top/bottom margin
        rail_gap_frac = float(s.extra["rail_gap_frac"])

        left = int(SS_CELL * (0.5 - rail_gap_frac / 2.0))
        right = SS_CELL - left

        # Rails (black)
        d.line((left, M, left, SS_CELL - M), fill="black", width=max(1, thick_px))
        d.line((right, M, right, SS_CELL - M), fill="black", width=max(1, thick_px))

        # Rungs: evenly spaced between margins
        usable_h = SS_CELL - 2 * M
        if s.count > 1:
            step = usable_h / (int(s.count) - 1)
        else:
            step = 0

        for i in range(int(s.count)):
            y = int(M + i * step)
            d.line((left, y, right, y), fill=color, width=thick_px)

        if rot:
            img = img.rotate(rot, resample=Image.NEAREST, expand=False)

        return _down_on_background(img)
