# sphinx/motifs/segment.py
import math
import random
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _rot2d
@register_motif
class SegmentMotif(Motif):
    """
    A simple straight line segment between two points.
    Renders as a single solid-colored line (no border/halo).

    Drive it with either:
      1) Explicit endpoints (tile-normalized): extra.p0, extra.p1 in [0,1]^2
      2) Parametric: extra.center, extra.angle_deg, extra.length_frac
    """
    name = "segment"
    attr_ranges = {"thickness": (3, 6)}

    # -------------------------- sampling & clamping --------------------------

    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        thickness = rng.randint(*self.attr_ranges["thickness"])

        cx = rng.uniform(0.25, 0.75)
        cy = rng.uniform(0.25, 0.75)
        angle = rng.uniform(0, 360)
        length_frac = rng.uniform(0.35, 0.8)

        extra = {
            "center": (cx, cy),
            "angle_deg": angle,
            "length_frac": length_frac,
            "rotation": int(rng.uniform(0, 360)) % 360,  # kept for compatibility
            "scale": 1.0,
        }

        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,
            thickness=thickness,
            size=1.0,
            extra=extra,
        )

    def clamp_spec(self, spec: MotifSpec):
        ex = dict(spec.extra or {})

        # thickness >= 1 (pixel-space thickness is applied after AA scaling)
        thick = max(1, int(getattr(spec, "thickness", self.attr_ranges["thickness"][0])))

        # canonicalize fields
        rot = int(float(ex.get("rotation", 0.0))) % 360
        ex["rotation"] = rot
        ex["scale"] = float(ex.get("scale", 1.0))

        def _clamp01(v): return max(0.0, min(1.0, float(v)))

        if "p0" in ex and "p1" in ex:
            p0 = tuple(ex["p0"]); p1 = tuple(ex["p1"])
            p0 = (_clamp01(p0[0]), _clamp01(p0[1]))
            p1 = (_clamp01(p1[0]), _clamp01(p1[1]))
            # avoid degenerate zero-length (bump a hair)
            if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 1e-4:
                p1 = (min(1.0, p1[0] + 1e-2), p1[1])
            ex["p0"], ex["p1"] = p0, p1
        else:
            cx, cy = ex.get("center", (0.5, 0.5))
            cx, cy = _clamp01(cx), _clamp01(cy)
            angle = float(ex.get("angle_deg", 0.0)) % 180.0  # 180°-periodic
            length_frac = float(ex.get("length_frac", 0.5))
            length_frac = max(0.01, min(1.5, length_frac))
            ex["center"] = (cx, cy)
            ex["angle_deg"] = angle
            ex["length_frac"] = length_frac

        return spec.clone(thickness=thick, extra=ex)

    # ------------------------------ rendering --------------------------------

    @staticmethod
    def _adaptive_aa_factor(angle_deg: float) -> int:
        """
        Choose a local AA multiplier based on angle's distance from axis-aligned.
        0° or 90° => 1x (already crisp); ~45° => 3x.
        """
        # distance to nearest axis (0 or 90)
        a = abs(((angle_deg + 45) % 90) - 45)  # 0 at 45°, 45 at axis
        if a < 15:
            return 3  # very diagonal -> more AA
        elif a < 30:
            return 2
        return 1

    def _draw_segment(self, draw: ImageDraw.ImageDraw,
                      x0: float, y0: float, x1: float, y1: float,
                      color, stroke_px: float):
        """Draw a thick segment with round caps at high-res canvas."""
        w = max(1, int(round(stroke_px)))
        draw.line((x0, y0, x1, y1), fill=color, width=w)
        # round caps for prettier diagonals
        r = w / 2.0
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=color)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=color)

    def render(self, spec: MotifSpec):
        s = self.clamp_spec(spec)
        color = COLORS[s.color_idx]

        # base geometry box (float space)
        base_margin = SS_CELL * 0.12  # keep as float to preserve subpixel accuracy

        # Determine final endpoints/angle in *normalized tile space* [0,1]^2,
        # applying rotation/scale analytically to avoid raster resampling.
        ex = s.extra or {}
        rot = int(ex.get("rotation", 0)) % 360
        scale = float(ex.get("scale", 1.0))
        scale = max(0.2, min(2.0, scale))  # sane bounds

        if "p0" in ex and "p1" in ex:
            cx, cy = 0.5, 0.5
            (u0x, u0y) = ex["p0"]; (u1x, u1y) = ex["p1"]

            # apply scale & rotation about tile center analytically
            v0x, v0y = (u0x - cx) * scale, (u0y - cy) * scale
            v1x, v1y = (u1x - cx) * scale, (u1y - cy) * scale
            if rot:
                v0x, v0y = _rot2d(v0x, v0y, rot)
                v1x, v1y = _rot2d(v1x, v1y, rot)
            u0x, u0y = cx + v0x, cy + v0y
            u1x, u1y = cx + v1x, cy + v1y

            # determine drawing angle for AA heuristic
            ang = math.degrees(math.atan2(u1y - u0y, u1x - u0x)) % 180.0
            aa = self._adaptive_aa_factor(ang)

        else:
            (cx, cy) = ex["center"]
            ang = (float(ex.get("angle_deg", 0.0)) + rot) % 180.0  # fold-in rotation
            aa = self._adaptive_aa_factor(ang)

            # compute half-length in normalized space (scaled)
            length_frac = float(ex.get("length_frac", 0.5)) * scale
            length_frac = max(0.01, min(1.5, length_frac))

            # we'll convert to pixels later; just keep center+dir for now
            u0x = u0y = u1x = u1y = None  # delay until we know pixel box

        # Allocate adaptive AA canvas
        AA = aa
        S = int(SS_CELL * AA)
        img = Image.new("RGBA", (S, S), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Pixel-space drawing region (float, padded to avoid clipping)
        stroke_px = max(1.0, float(s.thickness) * SUPERSAMPLE * AA)
        pad = max(1.0, stroke_px * 0.5)
        margin = base_margin * AA
        left = margin + pad
        top = margin + pad
        right = S - margin - pad
        bottom = S - margin - pad

        if "p0" in ex and "p1" in ex:
            # map normalized endpoints into pixel box
            x0 = left + u0x * (right - left)
            y0 = top + u0y * (bottom - top)
            x1 = left + u1x * (right - left)
            y1 = top + u1y * (bottom - top)

        else:
            # center in pixels
            cxp = left + cx * (right - left)
            cyp = top + cy * (bottom - top)

            # direction unit vector from angle
            a = math.radians(ang)
            ux, uy = math.cos(a), math.sin(a)

            # requested half-length in pixels
            L_req = length_frac * min((right - left), (bottom - top))
            Lh_req = 0.5 * max(0.0, L_req)

            # bound to box so caps don't clip
            INF = 1e9

            def bound_along(pos, d, lo, hi):
                if abs(d) < 1e-9:
                    return INF
                return (hi - pos) / d if d > 0 else (lo - pos) / d

            Lh_max = max(0.0, min(
                bound_along(cxp, ux, left, right),
                bound_along(cxp, -ux, left, right),
                bound_along(cyp, uy, top, bottom),
                bound_along(cyp, -uy, top, bottom),
            ))
            Lh = min(Lh_req, Lh_max)

            x0, y0 = cxp - Lh * ux, cyp - Lh * uy
            x1, y1 = cxp + Lh * ux, cyp + Lh * uy

        # Draw the segment at high-res with round caps
        self._draw_segment(draw, x0, y0, x1, y1, color, stroke_px)

        # If we used adaptive AA > 1, downsample back to SS_CELL with LANCZOS
        if AA != 1:
            img = img.resize((SS_CELL, SS_CELL), resample=Image.LANCZOS)

        # Compose onto configured background and downscale to OUT_CELL
        return _down_on_background(img)
