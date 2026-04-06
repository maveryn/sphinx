# sphinx/motifs/keyhole.py
import math, random
from PIL import Image, ImageDraw, ImageChops, ImageFilter

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


@register_motif
class KeyholeMotif(Motif):
    """
    Circular head with a rectangular notch cutout (single mode).

    Extras:
      - head_frac : head radius / half-cell
      - notch_w   : notch width relative to diameter (2*R)
      - notch_len : notch length in units of R
      - rotation  : degrees
      - aa        : extra supersample factor (optional; default = max(2, SUPERSAMPLE))
    """
    name = "keyhole"
    attr_ranges = {"size": (0.95, 1.10), "thickness": (2, 4), "count": (1, 1)}

    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        extra = {
            "head_frac":  rng.uniform(0.40, 0.48),
            "notch_w":    rng.uniform(0.18, 0.28),
            "notch_len":  rng.uniform(0.70, 1.10),
            "rotation":   rng.uniform(0, 360),
            # let clamp_spec choose aa
        }
        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=1, size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        s_lo, s_hi = self.attr_ranges["size"]
        t_lo, t_hi = self.attr_ranges["thickness"]

        size = max(s_lo, min(s_hi, float(getattr(spec, "size", 1.0))))
        thickness = max(t_lo, min(t_hi, int(getattr(spec, "thickness", t_lo))))

        head_frac = max(0.30, min(0.60, float(ex.get("head_frac", 0.44))))
        notch_w   = max(0.08, min(0.40, float(ex.get("notch_w", 0.22))))
        notch_len = max(0.40, min(1.40, float(ex.get("notch_len", 0.90))))
        rotation  = float(ex.get("rotation", 0.0)) % 360.0

        aa = int(ex.get("aa", max(2, SUPERSAMPLE)))
        aa = max(1, min(4, aa))

        ex.update({
            "head_frac": head_frac,
            "notch_w": notch_w,
            "notch_len": notch_len,
            "rotation": rotation,
            "aa": aa,
        })
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)
        fill_rgba = _to_rgba(COLORS[s.color_idx])
        AA = int(s.extra["aa"])

        W = H = SS_CELL * AA
        Cx = Cy = W // 2

        half = (SS_CELL / 2.0) * float(s.size)
        R = max(6 * AA, int(half * float(s.extra["head_frac"]) * AA))

        # Masks (grayscale)
        head = Image.new("L", (W, H), 0)
        ImageDraw.Draw(head).ellipse((Cx - R, Cy - R, Cx + R, Cy + R), fill=255)

        notch = Image.new("L", (W, H), 0)
        notch_w_px   = max(2 * AA, int(2 * R * float(s.extra["notch_w"])))
        notch_len_px = max(3 * AA, int(R * float(s.extra["notch_len"])))
        x0 = Cx
        y0 = Cy - notch_w_px // 2
        x1 = Cx + notch_len_px
        y1 = Cy + notch_w_px // 2
        ImageDraw.Draw(notch).rectangle((x0, y0, x1, y1), fill=255)
        notch = notch.rotate(float(s.extra["rotation"]), resample=Image.BICUBIC, center=(Cx, Cy), expand=False)

        mask = ImageChops.subtract(head, notch)

        # Fill with alpha = mask
        fill = Image.new("RGBA", (W, H), fill_rgba)
        fill.putalpha(mask)

        # Uniform outline: edge detection + dilation to requested width
        ow = max(AA, int(s.thickness) * AA)
        edge = mask.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 0 else 0)
        edge = edge.filter(ImageFilter.MaxFilter(max(3, (ow | 1))))
        outline = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        outline.putalpha(edge)

        shape = Image.alpha_composite(fill, outline)

        # Downsample
        img = shape.resize((SS_CELL, SS_CELL), Image.LANCZOS)

        return _down_on_background(img)
