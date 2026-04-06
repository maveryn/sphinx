# sphinx/motifs/gear.py
import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageChops, ImageFilter

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


@register_motif
class GearMotif(Motif):
    """
    Cog/gear with (optionally) one missing tooth.

    Render pipeline (anti-aliased):
      1) Build a filled **gear mask** from the outer contour polygon.
      2) Subtract the inner circular hole from that mask.
      3) Colorize by using the mask as alpha on a solid RGBA layer.
      4) Create a crisp **outline** as a morphological ring: (dilate − erode) of the mask.
      5) Composite on white at high-res, then downsample with LANCZOS.

    Extras (spec.extra):
      - teeth (int): number of teeth (7..13 in sampling).
      - inner_frac (float): inner hole radius / half-cell.
      - base_frac  (float): valley radius / half-cell.
      - tip_frac   (float): tip radius / half-cell.
      - tip_width  (float): fraction of tooth angle occupied by the flat tip (0..1).
      - rotation   (deg): phase of the first tooth.
      - missing_idx (int): index of a missing tooth (rendered as valley height).

      Optional:
      - aa (int): extra supersampling factor (1..4). If absent, defaults to max(2, SUPERSAMPLE).
    """
    name = "gear"
    attr_ranges = {"size": (0.95, 1.05), "thickness": (2, 4), "count": (1, 1)}

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        teeth = rng.randint(7, 13)
        inner = rng.uniform(0.42, 0.58)
        base  = rng.uniform(0.62, 0.75)
        tip   = rng.uniform(0.78, 0.90)
        # ensure ordering: inner < base < tip
        if inner >= base: inner = max(0.35, base - 0.08)
        if base  >= tip:  base  = max(0.60, tip  - 0.08)

        extra = {
            "teeth": teeth,
            "inner_frac": inner,            # inner hole
            "base_frac":  base,             # valley radius
            "tip_frac":   tip,              # tooth tip radius
            "tip_width":  rng.uniform(0.34, 0.55),  # portion of tooth angle used by tip
            "rotation":   rng.uniform(0, 360),
            "missing_idx": rng.randrange(teeth),    # one missing tooth (valley-height)
            # 'aa' omitted; set in clamp_spec
        }
        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=1, size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        size = float(getattr(spec, "size", 1.0))
        thickness = max(self.attr_ranges["thickness"][0],
                        min(self.attr_ranges["thickness"][1], int(getattr(spec, "thickness", 2))))

        # geometry clamps
        k = int(ex.get("teeth", 9));  k = max(5, min(30, k))
        inner = float(ex.get("inner_frac", 0.50))
        base  = float(ex.get("base_frac",  0.68))
        tip   = float(ex.get("tip_frac",   0.84))
        # keep strict ordering with minimum gaps
        if inner >= base: inner = max(0.20, base - 0.06)
        if base  >= tip:  base  = max(0.40, tip  - 0.06)
        tipw = float(ex.get("tip_width", 0.45)); tipw = max(0.15, min(0.80, tipw))

        rot  = float(ex.get("rotation", 0.0)) % 360.0
        miss = int(ex.get("missing_idx", 0)) % max(1, k)

        # robust AA factor
        aa_raw = ex.get("aa", None)
        try:
            aa = int(aa_raw)
        except Exception:
            aa = max(2, SUPERSAMPLE)
        aa = max(1, min(4, aa))

        ex.update({
            "teeth": k, "inner_frac": inner, "base_frac": base, "tip_frac": tip,
            "tip_width": tipw, "rotation": rot, "missing_idx": miss, "aa": aa
        })
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)

        # High-res canvas for anti-aliasing
        AA = int(s.extra["aa"])
        W = H = SS_CELL * AA
        cx = cy = W // 2

        color = _to_rgba(COLORS[s.color_idx])
        ow = max(AA, int(s.thickness) * AA)  # outline width in AA pixels (used as morphology kernel)

        # Radii in AA pixels
        half = (SS_CELL / 2.0) * float(s.size) * AA
        R_inner = int(half * float(s.extra["inner_frac"]))
        R_base  = int(half * float(s.extra["base_frac"]))
        R_tip   = int(half * float(s.extra["tip_frac"]))

        k      = int(s.extra["teeth"])
        tipw   = float(s.extra["tip_width"])      # fraction of tooth angle taken by tip flat
        start  = math.radians(float(s.extra["rotation"]))
        miss   = int(s.extra["missing_idx"])

        # ---- 1) Build outer contour polygon (valley→tipL→tipR→valley per tooth) ----
        tooth_ang = 2.0 * math.pi / k
        verts: List[Tuple[int, int]] = []
        for i in range(k):
            a_center = start + i * tooth_ang
            left_a   = a_center - 0.5 * tooth_ang
            right_a  = a_center + 0.5 * tooth_ang
            tip_l_a  = a_center - 0.5 * tooth_ang * tipw
            tip_r_a  = a_center + 0.5 * tooth_ang * tipw

            # Missing tooth rendered at valley height
            tipR = tipL = R_base if i == miss else R_tip

            # valley-left
            verts.append((int(cx + R_base * math.cos(left_a)),  int(cy + R_base * math.sin(left_a))))
            # tip-left
            verts.append((int(cx + tipL   * math.cos(tip_l_a)), int(cy + tipL   * math.sin(tip_l_a))))
            # tip-right
            verts.append((int(cx + tipR   * math.cos(tip_r_a)), int(cy + tipR   * math.sin(tip_r_a))))
            # valley-right
            verts.append((int(cx + R_base * math.cos(right_a)), int(cy + R_base * math.sin(right_a))))

        # ---- 2) Masks: outer polygon filled, subtract inner circular hole ----
        outer_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(outer_mask).polygon(verts, fill=255)

        inner_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(inner_mask).ellipse((cx - R_inner, cy - R_inner, cx + R_inner, cy + R_inner), fill=255)

        gear_mask = ImageChops.subtract(outer_mask, inner_mask)

        # ---- 3) Colorize (mask → alpha) ----
        fill_layer = Image.new("RGBA", (W, H), color)
        fill_layer.putalpha(gear_mask)

        # ---- 4) Uniform outline via morphological ring ----
        # Use an odd kernel roughly equal to 'ow' to get a stroke-like ring
        ksz = max(3, (ow | 1))
        dil = gear_mask.filter(ImageFilter.MaxFilter(ksz))
        ero = gear_mask.filter(ImageFilter.MinFilter(ksz))
        ring = ImageChops.subtract(dil, ero)

        outline = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        outline.putalpha(ring)

        # Composite fill + outline at high-res
        img_big = Image.alpha_composite(fill_layer, outline)

        # Downsample AA canvas back to SS_CELL, then project helper downsamples SS→1x
        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)
        return _down_on_background(img)
