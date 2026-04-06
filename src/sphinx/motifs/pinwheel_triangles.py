# sphinx/motifs/pinwheel_triangles.py
import math, random
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background


@register_motif
class PinwheelTrianglesMotif(Motif):
    """
    Pinwheel made of isosceles triangular blades.

    Modes (spec.extra["mode"]):
      - "sym"  : all blades identical
      - "asym" : one blade is scaled (short or tall)

    Asymmetry fields:
      - short_idx   (int): which blade is altered
      - asym_type   (str): "short" or "tall"
      - short_scale (float): multiplicative factor for that blade's tip radius
    """
    name = "pinwheel_triangles"
    attr_ranges = {
        "size": (0.85, 0.95),
        "thickness": (2, 4),
        "count": (1, 1),
        "blades": (3, 8),            # number of blades
        "aperture_deg": (18, 48),    # angular width of each blade (deg)
        "inner_frac": (0.06, 0.22),  # apex radius / half-cell
        "outer_frac": (0.72, 0.94),  # tip radius / half-cell
    }

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        blades = rng.randint(*self.attr_ranges["blades"])
        mode = rng.choice(["sym", "asym"])

        extra = {
            "mode": mode,
            "blades": blades,
            "aperture_deg": rng.uniform(*self.attr_ranges["aperture_deg"]),
            "inner_frac": rng.uniform(*self.attr_ranges["inner_frac"]),
            "outer_frac": rng.uniform(*self.attr_ranges["outer_frac"]),
            "rotation": rng.uniform(0, 360),
        }
        if mode == "asym":
            asym_type = rng.choice(["short", "tall"])  # 50/50
            short_idx = rng.randrange(blades)
            short_scale = (
                rng.uniform(0.50, 0.75) if asym_type == "short"
                else rng.uniform(1.25, 1.5)
            )
            extra.update({
                "short_idx": short_idx,
                "short_scale": short_scale,
                "asym_type": asym_type,
            })

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

        # basics
        sz_lo, sz_hi = self.attr_ranges["size"]
        th_lo, th_hi = self.attr_ranges["thickness"]
        size = max(sz_lo, min(sz_hi, float(getattr(spec, "size", 1.0))))
        thickness = max(th_lo, min(th_hi, int(getattr(spec, "thickness", th_lo))))

        # mode
        mode = ex.get("mode", "sym")
        if mode not in ("sym", "asym"):
            mode = "sym"

        # geometry
        b_lo, b_hi = self.attr_ranges["blades"]
        blades = max(b_lo, min(b_hi, int(ex.get("blades", 4))))

        ap_lo, ap_hi = self.attr_ranges["aperture_deg"]
        aperture_deg = max(ap_lo, min(ap_hi, float(ex.get("aperture_deg", 30.0))))

        in_lo, in_hi = self.attr_ranges["inner_frac"]
        out_lo, out_hi = self.attr_ranges["outer_frac"]
        inner_frac = max(in_lo, min(in_hi, float(ex.get("inner_frac", 0.12))))
        outer_frac = max(out_lo, min(out_hi, float(ex.get("outer_frac", 0.86))))
        if outer_frac <= inner_frac + 0.10:
            outer_frac = min(out_hi, inner_frac + 0.10)

        rotation = float(ex.get("rotation", 0.0)) % 360.0

        # asym-only fields
        if mode == "asym":
            short_idx = int(ex.get("short_idx", 0)) % max(1, blades)
            asym_type = ex.get("asym_type", "short")
            if asym_type not in ("short", "tall"):
                asym_type = "short"
            # Default + clamp short_scale depending on type
            default_scale = 0.7 if asym_type == "short" else 1.2
            short_scale = float(ex.get("short_scale", default_scale))
            if asym_type == "short":
                short_scale = max(0.40, min(0.95, short_scale))
            else:
                short_scale = max(1.05, min(1.5, short_scale))
        else:
            short_idx = 0
            asym_type = "short"
            short_scale = 1.0

        ex.update({
            "mode": mode,
            "blades": blades,
            "aperture_deg": aperture_deg,
            "inner_frac": inner_frac,
            "outer_frac": outer_frac,
            "rotation": rotation,
            "short_idx": short_idx,
            "short_scale": short_scale,
            "asym_type": asym_type,
        })
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    # --- rendering (anti-aliased via supersampled canvas + LANCZOS) ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        ex = s.extra
        mode = ex["mode"]
        k = int(ex["blades"])

        # modest extra AA to smooth edges
        AA = max(2, SUPERSAMPLE)
        W = H = SS_CELL * AA
        img_big = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        d = ImageDraw.Draw(img_big)

        cx = cy = W // 2
        half_base = (SS_CELL // 2) * float(s.size)

        r0 = int(half_base * float(ex["inner_frac"]) * AA)
        r1_base = int(half_base * float(ex["outer_frac"]) * AA)
        r1_cap = int(half_base * AA * 0.98)  # keep tips within tile

        start = math.radians(float(ex["rotation"]))
        ap = math.radians(float(ex["aperture_deg"]))
        short_idx = int(ex.get("short_idx", 0))
        short_scale = float(ex.get("short_scale", 1.0))

        col = COLORS[s.color_idx]
        ow = max(2 * AA, int(s.thickness) * AA)  # outline width at AA scale

        for i in range(k):
            scale = short_scale if (mode == "asym" and i == short_idx) else 1.0
            r1 = int(r1_base * scale)
            if r1 > r1_cap:
                r1 = r1_cap

            ang = start + i * (2 * math.pi / k)
            aL = ang - ap / 2.0
            aR = ang + ap / 2.0
            pts = [
                (cx + int(r0 * math.cos(ang)), cy + int(r0 * math.sin(ang))),  # apex
                (cx + int(r1 * math.cos(aL)),  cy + int(r1 * math.sin(aL))),   # base left
                (cx + int(r1 * math.cos(aR)),  cy + int(r1 * math.sin(aR))),   # base right
            ]
            d.polygon(pts, fill=col, outline="black", width=ow)

        # downsample to SS_CELL with high-quality resampling, then finalize
        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)
        return _down_on_background(img)
