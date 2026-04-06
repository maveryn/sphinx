# sphinx/motifs/dot.py
import random
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background


@register_motif
class DotMotif(Motif):
    """
    Single dot centered in the tile.

    Extras: none
    """
    name = "dot"
    attr_ranges = {"count": (1, 1), "size": (0.9, 1.2), "thickness": (3, 6)}

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,  # always one dot
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
        )

    # --- normalization ---
    def clamp_spec(self, spec: MotifSpec):
        ex = dict(spec.extra or {})
        ex.pop("mode", None)  # ignore generic flags if present

        size = float(getattr(spec, "size", 1.0))
        size = max(self.attr_ranges["size"][0], min(self.attr_ranges["size"][1], size))
        thick = max(1, int(getattr(spec, "thickness", 3)))

        return spec.clone(count=1, size=size, thickness=thick, extra=ex)

    # --- rendering ---
    def render(self, spec: MotifSpec):
        s = self.clamp_spec(spec)

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        color = COLORS[s.color_idx]

        # Centered dot; keep a small margin so outlines don't clip
        margin = int(SS_CELL * 0.12)
        usable = SS_CELL - 2 * margin

        # Dot radius ~35% of usable span, scaled by size
        r = max(3 * SUPERSAMPLE, int(usable * 0.35 * float(s.size)))

        cx = cy = SS_CELL // 2
        ow = max(1, int(s.thickness) * SUPERSAMPLE)

        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="black", width=ow)

        return _down_on_background(img)
