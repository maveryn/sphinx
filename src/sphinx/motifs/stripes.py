# sphinx/motifs/stripes.py
from PIL import Image, ImageDraw
from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

@register_motif
class StripesMotif(Motif):
    """
    Parallel stripes (single mode).

    Top-level fields:
      - count (int): number of colored stripes across the tile (approximate)
      - angle (deg): orientation; 0 = vertical, 90 = horizontal (45 supported too)
      - size  (float): scales stripe width
      - thickness (int): minimum pixel width floor for stripe visibility
    """
    name = "stripes"
    attr_ranges = {
        "count": (2, 6),
        "size": (0.9, 1.2),
        "angle": (0, 90),
        "thickness": (3, 6),
    }

    # --- sampling ---
    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=rng.randint(*self.attr_ranges["count"]),
            size=rng.uniform(*self.attr_ranges["size"]),
            angle=rng.choice([0, 45, 90]),  # keep discrete orientations for clean visuals
            thickness=rng.randint(*self.attr_ranges["thickness"]),
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        # count
        cmin, cmax = self.attr_ranges["count"]
        count = int(getattr(spec, "count", 3))
        count = max(int(cmin), min(int(cmax), count))

        # size
        smin, smax = self.attr_ranges["size"]
        size = float(getattr(spec, "size", 1.0))
        size = max(float(smin), min(float(smax), size))

        # thickness (visibility floor in pixels@1x, scaled by SUPERSAMPLE later)
        tmin, tmax = self.attr_ranges["thickness"]
        thickness = int(getattr(spec, "thickness", int(tmin)))
        thickness = max(int(tmin), min(int(tmax), thickness))

        # angle normalized to [0, 90]
        amin, amax = self.attr_ranges["angle"]
        angle = float(getattr(spec, "angle", 0.0))
        # map any angle to [0,180) then clamp to [0,90] (stripe symmetry)
        angle = angle % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        angle = max(float(amin), min(float(amax), angle))

        return spec.clone(count=count, size=size, angle=angle, thickness=thickness)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        k = max(1, int(s.count))
        color = COLORS[s.color_idx]

        # Base stripe width from target count, scaled by 'size'
        base_w = int((SS_CELL / (2.0 * k)) * float(s.size))

        # Visibility floor from 'thickness'
        floor_w = max(2 * SUPERSAMPLE, int(round(float(s.thickness))) * SUPERSAMPLE)

        w = max(base_w, floor_w)

        # Center the alternating pattern (stripe width w, gap w) in the tile
        period = 2 * w
        start_x = -int((SS_CELL % period) / 2)

        x = start_x
        while x < SS_CELL:
            # draw colored stripe, let PIL clip if x<0 or x+w>SS_CELL
            draw.rectangle([x, 0, x + w, SS_CELL], fill=color, outline=None)
            x += period

        if abs(float(s.angle)) > 1e-3:
            img = img.rotate(float(s.angle), resample=Image.BICUBIC, expand=False)

        return _down_on_background(img)
