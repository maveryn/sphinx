# sphinx/motifs/glyph.py
import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background


@register_motif
class GlyphMotif(Motif):
    """
    Block-style letters & symbols rendered from stroke segments (no fonts).

    Supported glyphs:
      Letters:  A E F H I K L M N T V W X Y Z
      Symbols:  plus times equals minus hash asterisk caret chevron_l chevron_r
    """
    name = "glyph"
    LETTERS = ["A","E","F","H","I","K","L","M","N","T","V","W","X","Y","Z"]
    SYMBOLS = ["plus","times","equals","minus","hash","asterisk","caret","chevron_l","chevron_r"]
    ALL = LETTERS + SYMBOLS

    # Glyphs that should NOT be used with asym mode (rotationally/symmetrically invariant).
    # Keep lowercase for consistency when checking with glyph.lower().
    ASYM_BLOCKLIST = {"plus", "times", "asterisk", "hash", "equals", "minus", "h", "i", "x"}

    attr_ranges = {
        "size": (0.92, 1.12),
        "thickness": (2, 5),
        "count": (1, 1),
    }

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)

        # choose mode first
        mode = rng.choice(["sym", "asym"])

        # choose glyph from allowed pool depending on mode
        blocked = self.ASYM_BLOCKLIST  # already lowercase
        if mode == "asym":
            pool = [g for g in self.ALL if g.lower() not in blocked]
            # safety fallback (shouldn't happen)
            if not pool:
                pool = self.ALL
        else:
            pool = self.ALL
        glyph = rng.choice(pool)

        # signed tilt only if we actually use asym
        if mode == "asym":
            tilt_mag = rng.uniform(30.0, 60.0)
            tilt_deg = rng.choice([-1.0, 1.0]) * tilt_mag
        else:
            tilt_deg = 0.0

        extra = {
            "glyph": glyph,
            "glyph_id": self.ALL.index(glyph),
            "rotation": rng.choice([0, 90, 180, 270]),
            "stroke_frac": rng.uniform(0.10, 0.18),
            "mode": mode,
            "tilt_deg": tilt_deg,  # 0 if sym
            # let clamp choose 'aa'
        }
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        smin, smax = self.attr_ranges["size"]
        tmin, tmax = self.attr_ranges["thickness"]

        size = max(smin, min(smax, float(getattr(spec, "size", 1.0))))
        thickness = max(tmin, min(tmax, int(getattr(spec, "thickness", tmin))))

        # mode first
        mode = str(ex.get("mode", "sym"))
        if mode not in ("sym", "asym"):
            mode = "sym"

        # glyph + id (with possible re-sample for asym)
        glyph = ex.get("glyph", "H")
        if glyph not in self.ALL:
            glyph = "A"

        blocked = self.ASYM_BLOCKLIST
        if mode == "asym" and glyph.lower() in blocked:
            # Deterministic re-sample using seed so clamp is stable
            rng = random.Random(int(getattr(spec, "seed", 0)) ^ 0xA5A5_5A5A)
            pool = [g for g in self.ALL if g.lower() not in blocked] or self.ALL
            glyph = pool[rng.randrange(len(pool))]

        gid = int(ex.get("glyph_id", self.ALL.index(glyph)))
        # ensure glyph_id matches chosen glyph (especially after re-sample)
        if not (0 <= gid < len(self.ALL)) or self.ALL[gid] != glyph:
            gid = self.ALL.index(glyph)

        rotation = int(float(ex.get("rotation", 0))) % 360
        rotation = (rotation // 90) * 90  # snap for crispness

        stroke_frac = float(ex.get("stroke_frac", 0.14))
        stroke_frac = max(0.07, min(0.24, stroke_frac))

        # tilt handling
        tilt_raw = float(ex.get("tilt_deg", 0.0))
        if mode == "asym":
            if abs(tilt_raw) < 1e-6:
                tilt_raw = 45.0
            sign = -1.0 if tilt_raw < 0 else 1.0
            mag = max(30.0, min(60.0, abs(tilt_raw)))
            tilt_deg = sign * mag
        else:
            tilt_deg = 0.0

        aa = int(ex.get("aa", max(2, SUPERSAMPLE)))
        if mode == "asym":
            aa = max(3, aa)
        aa = max(1, min(4, aa))

        ex.update({
            "glyph": glyph,
            "glyph_id": gid,
            "rotation": rotation,
            "stroke_frac": stroke_frac,
            "mode": mode,
            "tilt_deg": float(tilt_deg),
            "aa": aa,
        })
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        glyph = s.extra["glyph"]
        mode = s.extra.get("mode", "sym")
        tilt_deg = float(s.extra.get("tilt_deg", 0.0)) if mode == "asym" else 0.0

        AA = int(s.extra["aa"])
        W = H = SS_CELL * AA
        cx = cy = W // 2

        # glyph box (leave margin); shrink slightly if tilted to avoid clipping
        tilt_mag = abs(tilt_deg)
        base_scale = 0.78 - 0.10 * min(1.0, tilt_mag / 60.0)
        box = int(min(W, H) * base_scale * float(s.size))
        stroke_w = max(AA, int(s.extra["stroke_frac"] * box))
        col = COLORS[s.color_idx]
        ow = max(AA, int(s.thickness) * AA)  # outline width for polygon edges

        img_big = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        d = ImageDraw.Draw(img_big)

        # Build segments in normalized coords [-0.5, 0.5]
        segs = self._strokes_for_glyph(glyph)

        # Draw each segment as a rotated rectangle (thick stroke)
        cx_i = cx; cy_i = cy
        for (x0, y0, x1, y1) in segs:
            p0x = cx_i + int(x0 * box)
            p0y = cy_i + int(y0 * box)
            p1x = cx_i + int(x1 * box)
            p1y = cy_i + int(y1 * box)
            poly = _rect_around_segment(p0x, p0y, p1x, p1y, stroke_w)
            # fill
            d.polygon(poly, fill=col)
            # crisp edge
            d.polygon(poly, outline="black", width=ow)

        # 1) snap rotation (multiples of 90°) for crispness
        rot = int(s.extra["rotation"])
        if rot:
            img_big = img_big.rotate(rot, resample=Image.BICUBIC, center=(cx, cy), expand=False)

        # 2) asym tilt (non-orthogonal), applied AFTER the 90° snap
        if mode == "asym" and abs(tilt_deg) >= 1.0:
            img_big = img_big.rotate(tilt_deg, resample=Image.BICUBIC, center=(cx, cy), expand=False)

        # Downsample
        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)
        return _down_on_background(img)

    # --- glyph definitions ---
    def _strokes_for_glyph(self, g: str) -> List[Tuple[float,float,float,float]]:
        L, R, T, B = -0.45, 0.45, -0.45, 0.45
        Mx, My = 0.0, 0.0
        qx = 0.22
        qy = 0.22

        g = g.lower()
        segs: List[Tuple[float,float,float,float]] = []

        # Letters
        if g == "h":
            segs += [(L,T,L,B), (R,T,R,B), (L,My,R,My)]
        elif g == "l":
            segs += [(L,T,L,B), (L,B,R,B)]
        elif g == "e":
            segs += [(L,T,L,B), (L,T,R,T), (L,My,R*0.75,My), (L,B,R,B)]
        elif g == "f":
            segs += [(L,T,L,B), (L,T,R,T), (L,My,R*0.75,My)]
        elif g == "t":
            segs += [(L,T,R,T), (Mx,T,Mx,B)]
        elif g == "i":
            segs += [(Mx,T,Mx,B)]
        elif g == "m":
            segs += [(L,T,L,B), (R,T,R,B), (L,B,Mx,T), (Mx,T,R,B)]
        elif g == "n":
            segs += [(L,T,L,B), (R,T,R,B), (L,B,R,T)]
        elif g == "v":
            segs += [(-0.35,T, Mx,B), (0.35,T, Mx,B)]
        elif g == "w":
            segs += [(-0.40,T, -0.15,B), (-0.15,B, 0.0,T), (0.0,T, 0.15,B), (0.15,B, 0.40,T)]
        elif g == "x":
            segs += [(L,T,R,B), (R,T,L,B)]
        elif g == "y":
            segs += [(-0.30,T, Mx,My), (0.30,T, Mx,My), (Mx,My, Mx,B)]
        elif g == "z":
            segs += [(L,T,R,T), (R,T,L,B), (L,B,R,B)]
        elif g == "k":
            segs += [(L,T,L,B), (L,My,R,T), (L,My,R,B)]
        elif g == "a":
            segs += [(-0.33,B, Mx,T), (Mx,T, 0.33,B), (-0.20,My, 0.20,My)]
        else:  # fallback to H
            segs += [(L,T,L,B), (R,T,R,B), (L,My,R,My)]

        # Symbols
        if g == "plus":
            segs = [(L,My,R,My), (Mx,T,Mx,B)]
        elif g == "times":
            segs = [(L,T,R,B), (R,T,L,B)]
        elif g == "equals":
            segs = [(L, My - qy, R, My - qy), (L, My + qy, R, My + qy)]
        elif g == "minus":
            segs = [(L,My,R,My)]
        elif g == "hash":
            segs = [(-qx,T,-qx,B), (qx,T,qx,B), (L,-qy,R,-qy), (L,qy,R,qy)]
        elif g == "asterisk":
            arms = []
            for k in range(6):
                a = math.radians(60 * k)
                arms.append((0.0, 0.0, 0.40*math.cos(a), 0.40*math.sin(a)))
            segs = arms
        elif g == "caret":
            segs = [(-0.32, My+0.12, Mx, T), (Mx, T, 0.32, My+0.12)]
        elif g == "chevron_l":
            segs = [(-0.18, My, 0.20, T), (-0.18, My, 0.20, B)]
        elif g == "chevron_r":
            segs = [(0.18, My, -0.20, T), (0.18, My, -0.20, B)]

        return segs


def _rect_around_segment(x0: int, y0: int, x1: int, y1: int, w: int) -> List[Tuple[int, int]]:
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        hw = max(1, w // 2)
        return [(x0 - hw, y0 - hw), (x0 + hw, y0 - hw), (x0 + hw, y0 + hw), (x0 - hw, y0 + hw)]
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    hw = w / 2.0
    ax = x0 + nx * hw
    ay = y0 + ny * hw
    bx = x0 - nx * hw
    by = y0 - ny * hw
    cx = x1 - nx * hw
    cy = y1 - ny * hw
    dxp = x1 + nx * hw
    dyp = y1 + ny * hw
    return [(int(round(ax)), int(round(ay))),
            (int(round(bx)), int(round(by))),
            (int(round(cx)), int(round(cy))),
            (int(round(dxp)), int(round(dyp)))]
