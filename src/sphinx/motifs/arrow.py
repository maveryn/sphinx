# sphinx/motifs/arrow.py
import math, random
from PIL import Image, ImageDraw, ImageFilter, ImageChops

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


def _sample_split(rng, neg_range, pos_range):
    a, b = neg_range if rng.random() < 0.5 else pos_range
    return rng.uniform(a, b)


@register_motif
class ArrowMotif(Motif):
    """
    k arrows incident at center, equally spaced in angle. Each arrow = rectangular shaft + triangular head.

    Modes:
      - "sym"  : all arrows identical (equal length & spacing)
      - "asym" : exactly one arrow is distorted (by length, angle, or both)
                 (when count == 1, asym == sym)

    """
    name = "arrow"
    attr_ranges = {"size": (0.95, 1.05), "thickness": (2, 4), "count": (1, 8)}

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(*self.attr_ranges["count"])
        mode = rng.choice(["sym", "asym"])
        asym_mode = rng.choice(["length", "angle", "both"]) if mode == "asym" else "length"

        extra = {
            "mode": mode,
            "rotation": rng.uniform(0, 360),
            "shaft_frac": rng.uniform(0.40, 0.55),
            "head_frac":  rng.uniform(0.22, 0.32),
            # asym (used only if mode='asym' and count>1)
            "asym_idx": rng.randrange(count) if count > 1 else 0,
            "asym_mode": asym_mode,  # {"length","angle","both"}
            # split-range sampling (away from identity):
            "asym_scale": _sample_split(rng, (0.60, 0.8), (1.2, 1.4)),
            "asym_angle": _sample_split(rng, (-30.0, -15.0), (15.0, 30.0)),
            # DO NOT set 'aa' here—clamp_spec will provide a numeric default
        }
        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=count,
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        ex.pop("mode_hint", None)

        # count / basics
        cmin, cmax = self.attr_ranges["count"]
        count = max(cmin, min(cmax, int(getattr(spec, "count", 1))))
        size  = max(self.attr_ranges["size"][0],
                    min(self.attr_ranges["size"][1], float(getattr(spec, "size", 1.0))))
        thick = max(self.attr_ranges["thickness"][0],
                    min(self.attr_ranges["thickness"][1], int(getattr(spec, "thickness", 2))))

        # mode
        mode = ex.get("mode", "sym")
        if mode not in ("sym", "asym") or count <= 1:
            mode = "sym"

        rot = float(ex.get("rotation", 0.0)) % 360.0
        shaft_frac = max(0.20, min(0.70, float(ex.get("shaft_frac", 0.48))))
        head_frac  = max(0.12, min(0.45, float(ex.get("head_frac", 0.28))))

        # robust AA: handle missing/None/NaN/non-numeric
        aa_raw = ex.get("aa", None)
        try:
            aa = int(aa_raw)
        except Exception:
            aa = max(3, SUPERSAMPLE)
        aa = max(1, min(4, aa))

        # asym controls
        asym_idx  = int(ex.get("asym_idx", 0)) % max(1, count)
        asym_mode = ex.get("asym_mode", "length")
        if asym_mode not in ("length", "angle", "both"):
            asym_mode = "length"

        raw_scale = float(ex.get("asym_scale", 1.2))
        asym_scale = max(1.15, min(1.35, raw_scale)) if raw_scale >= 1.0 else max(0.60, min(0.85, raw_scale))

        raw_angle = float(ex.get("asym_angle", 20.0))
        asym_angle = (max(15.0, min(30.0, raw_angle))
                      if raw_angle >= 0 else
                      -max(15.0, min(30.0, abs(raw_angle))))

        ex.update({
            "mode": mode, "rotation": rot,
            "shaft_frac": shaft_frac, "head_frac": head_frac, "aa": aa,
            "asym_idx": asym_idx, "asym_mode": asym_mode,
            "asym_scale": asym_scale, "asym_angle": asym_angle,
        })
        return spec.clone(count=count, size=size, thickness=thick, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        AA = int(s.extra["aa"])
        W = H = SS_CELL * AA
        cx = cy = W // 2

        fill_rgba = _to_rgba(COLORS[s.color_idx])

        # High-res accumulators
        fill_acc = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        outline_acc = Image.new("RGBA", (W, H), (255, 255, 255, 0))

        # Geometry base (center→tip)
        half = W // 2
        pad  = int(0.16 * W)
        L    = max(6 * AA, int((half - pad) * float(s.size)))
        head_len  = max(3 * AA, int(L * float(s.extra["head_frac"])))
        shaft_len = max(6 * AA, L - head_len)

        # Thickness
        shaft_h = max(2 * AA, int(L * float(s.extra["shaft_frac"]) * 0.20))
        y_half  = max(AA, shaft_h // 2)

        # Outline kernel (odd ≥3)
        ow  = max(AA, int(s.thickness) * AA)
        k_outline = max(3, (ow | 1))

        def build_poly(L_local, head_len_local, shaft_len_local):
            head_half = max(y_half, int(0.45 * head_len_local))
            return [
                (0, -y_half),
                (shaft_len_local, -y_half),
                (shaft_len_local, -head_half),
                (L_local, 0),
                (shaft_len_local, head_half),
                (shaft_len_local, y_half),
                (0, y_half),
            ]

        def rotate_points(poly, angle_deg):
            a = math.radians(angle_deg)
            ca, sa = math.cos(a), math.sin(a)
            return [(cx + int(round(x * ca - y * sa)),
                     cy + int(round(x * sa + y * ca))) for (x, y) in poly]

        def draw_arrow(angle_deg, L_local, head_len_local, shaft_len_local):
            poly = build_poly(L_local, head_len_local, shaft_len_local)
            pts = rotate_points(poly, angle_deg)

            # mask
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).polygon(pts, fill=255)

            # fill
            fill_layer = Image.new("RGBA", (W, H), fill_rgba)
            fill_layer.putalpha(mask)
            fill_acc.alpha_composite(fill_layer)

            # uniform outline: (dilate - erode)
            dil = mask.filter(ImageFilter.MaxFilter(k_outline))
            ero = mask.filter(ImageFilter.MinFilter(k_outline))
            ring = ImageChops.subtract(dil, ero)
            out_layer = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            out_layer.putalpha(ring)
            outline_acc.alpha_composite(out_layer)

        # Draw arrows
        k = int(s.count)
        base_rot = float(s.extra["rotation"])
        step = 360.0 / k if k > 0 else 0.0

        asym_idx   = int(s.extra["asym_idx"])
        asym_mode  = s.extra["asym_mode"]
        asym_scale = float(s.extra["asym_scale"])
        asym_angle = float(s.extra["asym_angle"])
        mode       = s.extra["mode"]

        for i in range(k):
            angle_i = base_rot + i * step
            L_i, head_i, shaft_i = L, head_len, shaft_len

            if mode == "asym" and k > 1 and i == asym_idx:
                if asym_mode in ("angle", "both"):
                    angle_i += asym_angle
                if asym_mode in ("length", "both"):
                    L_i = max(6 * AA, int(L * asym_scale))
                    head_i = max(3 * AA, int(L_i * float(s.extra["head_frac"])))
                    shaft_i = max(6 * AA, L_i - head_i)

            draw_arrow(angle_i, L_i, head_i, shaft_i)

        # Center hub
        if k > 1:
            hub_r = max(AA, min(y_half, ow))
            hub_mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(hub_mask).ellipse((cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r), fill=255)
            hub_fill = Image.new("RGBA", (W, H), fill_rgba)
            hub_fill.putalpha(hub_mask)
            fill_acc.alpha_composite(hub_fill)

            hub_d = max(3, (AA | 1))
            hub_ring = hub_mask.filter(ImageFilter.MaxFilter(hub_d))
            hub_ring = ImageChops.subtract(hub_ring, hub_mask.filter(ImageFilter.MinFilter(hub_d)))
            hub_out = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            hub_out.putalpha(hub_ring)
            outline_acc.alpha_composite(hub_out)

        # Compose at high-res, then downsample with LANCZOS (no halos)
        composed = Image.alpha_composite(fill_acc, outline_acc)
        img = composed.resize((SS_CELL, SS_CELL), Image.BICUBIC)
        return _down_on_background(img)

@register_motif
class SingleArrowMotif(ArrowMotif):
    """
    Single arrow motif (exactly one arrow). Reuses ArrowMotif rendering.
    - count is fixed to 1
    - mode is forced to "sym" (no asym distortion)
    - all other knobs (size, thickness, rotation, stroke/head fractions, color, aa) behave the same
    """
    name = "single_arrow"
    # keep same size/thickness ranges; fix count to 1
    attr_ranges = {"size": (0.95, 1.05), "thickness": (2, 4), "count": (1, 1)}

    # sample like ArrowMotif, then force count=1 and sym-mode
    def sample_spec(self, rng: random.Random):
        spec = super().sample_spec(rng)
        ex = dict(spec.extra or {})
        ex["mode"] = "sym"
        ex["asym_idx"] = 0
        # keep rotation/shaft_frac/head_frac as sampled
        return spec.clone(count=1, extra=ex)

    # clamp to ensure count=1 & sym, regardless of what the caller passes
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        ex.update({"mode": "sym", "asym_idx": 0})
        # force count=1 pre-clamp, then run parent clamp
        base = super().clamp_spec(spec.clone(count=1, extra=ex))
        ex2 = dict(base.extra or {})
        ex2.update({"mode": "sym", "asym_idx": 0})
        return base.clone(count=1, extra=ex2)
