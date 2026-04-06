# sphinx/motifs/star_polygon.py
import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageChops, ImageFilter

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


@register_motif
class StarPolygonMotif(Motif):
    """
    N-point star (single star) with optional asymmetry on exactly one tip.

    Modes (spec.extra["mode"]):
      - "sym"  : canonical star polygon (alternating outer radius R and inner radius r).
      - "asym" : pick one tip (index 0..n_points-1) and distort it by:
                   • size  (outer radius scaled), or
                   • angle (tip rotated by ±Δ), or
                   • both  (apply both).
                 When distortion is small, symmetry-breaking remains clear.

    Rendering pipeline (anti-aliased):
      1) Build a high-res **star mask** (AA factor).
      2) Punch a tiny **notch** circle from the rim (if enabled).
      3) Colorize (mask → alpha) and synthesize a **uniform outline** via morphology (dilate − erode).
      4) Composite on white at high-res and **downsample with LANCZOS** for crisp edges.

    """
    name = "star_polygon"
    attr_ranges = {"size": (0.95, 1.05), "thickness": (2, 4), "count": (1, 1)}

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)

        # Core star params
        n_points   = rng.randint(5, 9)
        inner_frac = rng.uniform(0.38, 0.55)
        rotation   = rng.uniform(0, 360)
        notch_angle = rng.uniform(0, 360)
        notch_frac  = rng.uniform(0.04, 0.07)

        # Mode + asym controls (only used in "asym")
        mode = rng.choice(["sym", "asym"])
        asym_idx  = rng.randrange(n_points) if mode == "asym" else 0
        asym_mode = rng.choice(["size", "angle", "both"]) if mode == "asym" else "size"

        # Split-range sampling: avoid near-identity distortions
        def sample_split(neg, pos):
            a, b = neg if rng.random() < 0.5 else pos
            return rng.uniform(a, b)

        asym_scale = sample_split((0.5, 0.75), (1.25, 1.5))   # for size
        asym_angle = sample_split((-30.0, -15.0), (15.0, 30.0)) # degrees for tip rotation

        extra = {
            "mode": mode,
            "n_points": n_points,
            "inner_frac": inner_frac,
            "rotation": rotation,
            "notch_angle": notch_angle,
            "notch_frac": notch_frac,
            # asym bits (ignored in "sym")
            "asym_idx": asym_idx,
            "asym_mode": asym_mode,
            "asym_scale": asym_scale,
            "asym_angle": asym_angle,
            # 'aa' omitted → set in clamp_spec
        }

        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=1,
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})

        size  = max(self.attr_ranges["size"][0], min(self.attr_ranges["size"][1], float(getattr(spec, "size", 1.0))))
        thick = max(self.attr_ranges["thickness"][0], min(self.attr_ranges["thickness"][1], int(getattr(spec, "thickness", 2))))

        n_points = max(4, min(16, int(ex.get("n_points", 7))))
        inner_frac = max(0.20, min(0.80, float(ex.get("inner_frac", 0.45))))
        rotation   = float(ex.get("rotation", 0.0)) % 360.0

        notch_angle = float(ex.get("notch_angle", 0.0)) % 360.0
        notch_frac  = max(0.0, min(0.20, float(ex.get("notch_frac", 0.05))))

        mode = ex.get("mode", "sym")
        if mode not in ("sym", "asym"):
            mode = "sym"

        # AA factor (robust)
        aa_raw = ex.get("aa", None)
        try:
            aa = int(aa_raw)
        except Exception:
            aa = max(2, SUPERSAMPLE)
        aa = max(1, min(4, aa))

        # Asym controls
        asym_idx  = int(ex.get("asym_idx", 0)) % max(1, n_points)
        asym_mode = ex.get("asym_mode", "size")
        if asym_mode not in ("size", "angle", "both"):
            asym_mode = "size"

        raw_scale = float(ex.get("asym_scale", 1.20))
        # keep away from identity using split clamps
        asym_scale = max(1.15, min(1.35, raw_scale)) if raw_scale >= 1.0 else max(0.65, min(0.85, raw_scale))

        raw_angle = float(ex.get("asym_angle", 20.0))
        asym_angle = (max(15.0, min(30.0, raw_angle))
                      if raw_angle >= 0 else
                      -max(15.0, min(30.0, abs(raw_angle))))

        ex.update({
            "mode": mode,
            "n_points": n_points,
            "inner_frac": inner_frac,
            "rotation": rotation,
            "notch_angle": notch_angle,
            "notch_frac": notch_frac,
            "aa": aa,
            "asym_idx": asym_idx,
            "asym_mode": asym_mode,
            "asym_scale": asym_scale,
            "asym_angle": asym_angle,
        })
        return spec.clone(count=1, size=size, thickness=thick, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)

        # AA canvas
        AA = int(s.extra["aa"])
        W = H = SS_CELL * AA
        cx = cy = W // 2

        color = _to_rgba(COLORS[s.color_idx])
        ow = max(AA, int(s.thickness) * AA)  # outline width in AA px

        # Layout: margin and radii in AA pixels
        margin = int(SS_CELL * 0.12 * AA)
        max_radius = int((SS_CELL * AA - 2 * margin) * 0.5 * float(s.size))
        R = max(6 * AA, max_radius)
        r = max(2 * AA, int(R * float(s.extra["inner_frac"])))

        n = int(s.extra["n_points"])
        start = math.radians(float(s.extra["rotation"]))
        step = math.pi / n

        mode = s.extra["mode"]
        asym_idx = int(s.extra["asym_idx"])
        asym_mode = s.extra["asym_mode"]
        asym_scale = float(s.extra["asym_scale"])
        asym_delta = math.radians(float(s.extra["asym_angle"]))

        # Build star vertices (2n); modify the chosen tip in asym mode
        pts: List[Tuple[int, int]] = []
        for i in range(2 * n):
            is_tip = (i % 2 == 0)
            tip_id = i // 2
            a = start + i * step
            rad = R if is_tip else r

            if mode == "asym" and is_tip and (tip_id == asym_idx):
                if asym_mode in ("size", "both"):
                    rad = int(max(2 * AA, R * asym_scale))
                if asym_mode in ("angle", "both"):
                    a += asym_delta

            x = cx + int(round(rad * math.cos(a)))
            y = cy + int(round(rad * math.sin(a)))
            pts.append((x, y))

        # ---- Mask: filled star polygon ----
        star_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(star_mask).polygon(pts, fill=255)

        # ---- Notch: punch a small circle from the rim (optional) ----
        notch_r = int(max(0, float(s.extra["notch_frac"])) * R)
        if notch_r > 0:
            a_notch = math.radians(float(s.extra["notch_angle"]))
            nx = cx + int(round(R * math.cos(a_notch)))
            ny = cy + int(round(R * math.sin(a_notch)))
            notch = Image.new("L", (W, H), 0)
            ImageDraw.Draw(notch).ellipse((nx - notch_r, ny - notch_r, nx + notch_r, ny + notch_r), fill=255)
            star_mask = ImageChops.subtract(star_mask, notch)

        # ---- Fill & outline (morphological ring) ----
        fill_layer = Image.new("RGBA", (W, H), color)
        fill_layer.putalpha(star_mask)

        ksz = max(3, (ow | 1))  # odd ≥3
        dil = star_mask.filter(ImageFilter.MaxFilter(ksz))
        ero = star_mask.filter(ImageFilter.MinFilter(ksz))
        ring = ImageChops.subtract(dil, ero)

        outline = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        outline.putalpha(ring)

        # Composite at high-res, then LANCZOS downsample
        img_big = Image.alpha_composite(fill_layer, outline)
        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)

        return _down_on_background(img)
