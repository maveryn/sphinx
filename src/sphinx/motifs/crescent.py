# sphinx/motifs/crescent.py
import math, random
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFilter

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


@register_motif
class CrescentMotif(Motif):
    """
    Crescent = primary disk MINUS a shifted disk (single mode).

    Render pipeline (anti-aliased):
      1) Build masks A (base disk) and B (offset disk) at supersampled resolution (SS_CELL).
      2) cres_mask = A − B  (ImageChops.subtract clamps at 0).
      3) Colorize by applying cres_mask as alpha to a solid RGBA layer.
      4) Outline = (FIND_EDGES of cres_mask) → dilate to requested outline width.
      5) Composite fill + outline, then `_down_on_white` handles high-quality downsample.
    """
    name = "crescent"
    attr_ranges = {"size": (1.0, 1.75), "thickness": (2, 4), "count": (1, 1)}  # thickness = outline width

    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        extra = {
            "radius_frac": rng.uniform(0.34, 0.45),   # circle radius / half-cell
            "offset_frac": rng.uniform(0.35, 0.65),   # center offset in radii
            "rotation":    rng.uniform(0, 360),       # cut direction (deg)
        }
        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=1,
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        smin, smax = self.attr_ranges["size"]
        tmin, tmax = self.attr_ranges["thickness"]

        size = max(smin, min(smax, float(getattr(spec, "size", 1.0))))
        thickness = max(tmin, min(tmax, int(getattr(spec, "thickness", tmin))))

        rfrac = max(0.20, min(0.70, float(ex.get("radius_frac", 0.40))))
        off_frac = max(0.10, min(0.95, float(ex.get("offset_frac", 0.50))))
        rot = float(ex.get("rotation", 0.0)) % 360.0

        ex.update({"radius_frac": rfrac, "offset_frac": off_frac, "rotation": rot})
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)
        fill_rgba = _to_rgba(COLORS[s.color_idx])
        ow = max(SUPERSAMPLE, int(s.thickness) * SUPERSAMPLE)  # outline width in SS pixels

        W = H = SS_CELL
        Cx = Cy = SS_CELL // 2

        # Radius in supersampled pixels
        half = (SS_CELL // 2) * float(s.size)
        R = max(6 * SUPERSAMPLE, int(half * float(s.extra["radius_frac"])))

        # Disk centers
        off = float(s.extra["offset_frac"]) * R
        ang = math.radians(float(s.extra["rotation"]))
        Ax, Ay = Cx, Cy
        Bx = int(round(Cx + off * math.cos(ang)))
        By = int(round(Cy + off * math.sin(ang)))

        # Masks
        A = Image.new("L", (W, H), 0)
        B = Image.new("L", (W, H), 0)
        ImageDraw.Draw(A).ellipse((Ax - R, Ay - R, Ax + R, Ay + R), fill=255)
        ImageDraw.Draw(B).ellipse((Bx - R, By - R, Bx + R, By + R), fill=255)

        cres_mask = ImageChops.subtract(A, B)

        # Fill layer with mask as alpha
        cres = Image.new("RGBA", (W, H), fill_rgba)
        cres.putalpha(cres_mask)

        # Outline from edges (dilate to 'ow')
        edge = cres_mask.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 0 else 0)
        k = max(3, (ow | 1))  # odd kernel size
        edge = edge.filter(ImageFilter.MaxFilter(k))
        outline = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        outline.putalpha(edge)

        img = Image.alpha_composite(cres, outline)
        return _down_on_background(img)
