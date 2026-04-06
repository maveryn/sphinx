# sphinx/motifs/arc.py
import math
import random
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from ..utils.geom import clamp, unit, perp
from .helpers import _down_on_background, _rot2d
def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


@register_motif
class ArcMotif(Motif):
    """
    A curved/arc segment between two points.
    Renders as a single solid-colored stroke (no border/halo).

    Drive it with either:
      1) Explicit endpoints (tile-normalized): extra.p0, extra.p1 in [0,1]^2
      2) Parametric straight baseline (like SegmentMotif):
           extra.center, extra.angle_deg (0..360), extra.length_frac

    Curve is chosen by extra.arc_type ∈ {
        "circle", "quadratic", "cubic", "sine", "cubic_bump", "s_shape",
        "semiellipse", "superellipse", "gaussian", "catenary", "sigmoid"
    }.

    Rendering improvements:
      - No raster rotation; rotation is folded into geometry.
      - Adaptive oversampling (1x/2x/3x) based on diagonalness/curvature.
      - Densified polylines + round caps at endpoints and interior joins.
    """
    name = "arc"
    attr_ranges = {"thickness": (2, 5)}

    # ------------------------ Spec sampling ------------------------

    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        thickness = rng.randint(*self.attr_ranges["thickness"])

        cx = rng.uniform(0.25, 0.75)
        cy = rng.uniform(0.25, 0.75)
        angle = rng.uniform(0, 360)
        length_frac = rng.uniform(0.35, 0.9)

        arc_types = [
            "circle", "quadratic", "cubic", "sine", "cubic_bump", "s_shape",
            "semiellipse", "superellipse", "gaussian", "catenary", "sigmoid",
        ]
        arc_type = rng.choice(arc_types)

        extra = {
            "center": (cx, cy),
            "angle_deg": angle,
            "length_frac": length_frac,
            "rotation": int(rng.uniform(0, 360)) % 360,
            "scale": 1.0,  # affects baseline length in param mode
            "arc_type": arc_type,
            "curv_frac": rng.uniform(0.12, 0.55),
            "cycles": rng.uniform(0.6, 1.8),
            "phase": rng.uniform(0.0, 2.0 * math.pi),
            "super_m": rng.uniform(1.2, 4.0),
            "sigma": rng.uniform(0.12, 0.35),
            "kappa": rng.uniform(0.6, 2.0),
            "beta": rng.uniform(2.0, 6.0),
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

    # ------------------------ Spec clamping ------------------------

    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        thick = max(1, int(getattr(spec, "thickness", self.attr_ranges["thickness"][0])))

        rot = int(float(ex.get("rotation", 0.0))) % 360
        ex["rotation"] = rot
        ex["scale"] = float(ex.get("scale", 1.0))

        if "p0" in ex and "p1" in ex:
            p0 = tuple(ex["p0"]); p1 = tuple(ex["p1"])
            p0 = (_clamp01(float(p0[0])), _clamp01(float(p0[1])))
            p1 = (_clamp01(float(p1[0])), _clamp01(float(p1[1])))
            if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 1e-4:
                p1 = (min(1.0, p1[0] + 1e-2), p1[1])
            ex["p0"], ex["p1"] = p0, p1
        else:
            cx, cy = ex.get("center", (0.5, 0.5))
            cx = clamp(float(cx), 0.0, 1.0)
            cy = clamp(float(cy), 0.0, 1.0)
            angle = float(ex.get("angle_deg", 0.0)) % 360.0
            length_frac = clamp(float(ex.get("length_frac", 0.5)), 0.01, 1.5)
            ex["center"] = (cx, cy); ex["angle_deg"] = angle; ex["length_frac"] = length_frac

        valid_types = {
            "circle", "quadratic", "cubic", "sine", "cubic_bump", "s_shape",
            "semiellipse", "superellipse", "gaussian", "catenary", "sigmoid",
        }
        arc_type = ex.get("arc_type", "circle")
        ex["arc_type"] = arc_type if arc_type in valid_types else "circle"

        ex["curv_frac"] = clamp(float(ex.get("curv_frac", 0.3)), 0.0, 1.0)
        ex["cycles"] = clamp(float(ex.get("cycles", 1.0)), 0.25, 3.0)
        ex["phase"] = float(ex.get("phase", 0.0))
        ex["super_m"] = clamp(float(ex.get("super_m", 2.0)), 1.05, 6.0)
        ex["sigma"] = clamp(float(ex.get("sigma", 0.2)), 0.05, 0.5)
        ex["kappa"] = clamp(float(ex.get("kappa", 1.2)), 0.2, 3.0)
        ex["beta"] = clamp(float(ex.get("beta", 3.0)), 0.5, 10.0)

        return spec.clone(thickness=thick, extra=ex)

    # ------------------------ AA & geometry helpers ------------------------

    @staticmethod
    def _aa_factor_for_arc(angle_deg: float, arc_type: str, thickness: int, cycles: float) -> int:
        """
        Choose AA multiplier based on diagonalness of the baseline + curve complexity.
        """
        # distance to axis (0 at 45°, 45 near axis)
        a = abs(((angle_deg + 45) % 90) - 45)
        base = 3 if a < 15 else 2 if a < 30 else 1
        if arc_type in {"sine", "s_shape", "cubic", "superellipse"}:
            base = max(base, 2)
        if cycles > 1.2:
            base = 3
        if thickness <= 2:
            base = max(base, 2)
        return base

    @staticmethod
    def _densify(pts: List[Tuple[float, float]], max_seg_len: float) -> List[Tuple[float, float]]:
        """Ensure no segment exceeds max_seg_len by linear interpolation."""
        if len(pts) < 2:
            return pts
        out = [pts[0]]
        for i in range(1, len(pts)):
            x0, y0 = out[-1]
            x1, y1 = pts[i]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            n = max(1, int(math.ceil(L / max_seg_len)))
            for k in range(1, n + 1):
                t = k / n
                out.append((x0 + dx * t, y0 + dy * t))
        return out

    def _polyline(self, draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], color, width: int):
        """
        Draw a connected polyline with round caps at endpoints and soft joins:
        stamp small disks at interior vertices to hide miter artifacts.
        """
        if len(pts) < 2:
            return
        draw.line(pts, fill=color, width=max(1, int(round(width))))
        r = max(0.5, width / 2.0)
        # endpoints
        x0, y0 = pts[0]; x1, y1 = pts[-1]
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=color)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=color)
        # interior joins
        for (x, y) in pts[1:-1]:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    # ------------------------ Endpoint computation ------------------------

    def _compute_endpoints_px(self, s, bounds, AA: int) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Return pixel endpoints (x0,y0),(x1,y1) and baseline angle (deg) used for AA."""
        (left, top, right, bottom) = bounds
        ex = s.extra or {}
        rot = int(ex.get("rotation", 0)) % 360
        scale = float(ex.get("scale", 1.0))
        scale = max(0.2, min(2.0, scale))

        if "p0" in ex and "p1" in ex:
            cx, cy = 0.5, 0.5
            u0x, u0y = ex["p0"]; u1x, u1y = ex["p1"]
            # apply scale+rotation about tile center in normalized space
            v0x, v0y = (u0x - cx) * scale, (u0y - cy) * scale
            v1x, v1y = (u1x - cx) * scale, (u1y - cy) * scale
            if rot:
                v0x, v0y = _rot2d(v0x, v0y, rot)
                v1x, v1y = _rot2d(v1x, v1y, rot)
            u0x, u0y = _clamp01(cx + v0x), _clamp01(cy + v0y)
            u1x, u1y = _clamp01(cx + v1x), _clamp01(cy + v1y)

            x0 = left + u0x * (right - left); y0 = top + u0y * (bottom - top)
            x1 = left + u1x * (right - left); y1 = top + u1y * (bottom - top)
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
            return (x0, y0), (x1, y1), ang

        # Parametric baseline: fold rotation into angle and scale into length
        cxu, cyu = ex["center"]
        cx = left + cxu * (right - left)
        cy = top + cyu * (bottom - top)
        ang = (float(ex.get("angle_deg", 0.0)) + rot) % 360.0
        a = math.radians(ang)
        ux, uy = math.cos(a), math.sin(a)

        L_req = float(ex.get("length_frac", 0.5)) * scale * min((right - left), (bottom - top))
        Lh_req = 0.5 * max(0.0, L_req)

        INF = 1e9
        def bound_along(pos, d, lo, hi):
            if abs(d) < 1e-9: return INF
            return (hi - pos) / d if d > 0 else (lo - pos) / d

        Lh_max = max(0.0, min(
            bound_along(cx,  ux, left, right),
            bound_along(cx, -ux, left, right),
            bound_along(cy,  uy, top, bottom),
            bound_along(cy, -uy, top, bottom),
        ))
        Lh = min(Lh_req, Lh_max)
        return (cx - Lh * ux, cy - Lh * uy), (cx + Lh * ux, cy + Lh * uy), (ang % 180.0)

    # ------------------------ Curve constructors (unchanged math) ------------------------

    def _curve_from_offset_fn(self, p0, p1, bounds, curv_frac, g_fn, step_scale=1.8) -> List[Tuple[float, float]]:
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-3:
            return [p0, p1]

        tx, ty = unit(dx, dy)
        nx, ny = unit(*perp(tx, ty))
        amp_req = max(1.0, curv_frac * L)

        def b(pos, d, lo, hi):
            if abs(d) < 1e-9: return float("inf")
            return (hi - pos) / d if d > 0 else (lo - pos) / d

        (left, top, right, bottom) = bounds
        samples = 25
        min_allow = float("inf")
        for i in range(samples):
            t = i / (samples - 1)
            bx = x0 + t * dx
            by = y0 + t * dy
            gv = abs(g_fn(t))
            if gv < 1e-6:
                continue
            allow = min(b(bx,  nx, left, right), b(bx, -nx, left, right),
                        b(by,  ny, top, bottom), b(by, -ny, top, bottom)) / gv
            min_allow = min(min_allow, allow)

        amp = amp_req if min_allow == float("inf") else min(amp_req, max(0.0, min_allow) * 0.9)
        steps = max(24, int(L / (step_scale * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            bx = x0 + t * dx
            by = y0 + t * dy
            off = amp * g_fn(t)
            pts.append((bx + off * nx, by + off * ny))
        return pts

    def _curve_circle(self, p0, p1, bounds, curv_frac) -> List[Tuple[float, float]]:
        (left, top, right, bottom) = bounds
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-3: return [p0, p1]
        (tx, ty) = unit(dx, dy)
        (nx, ny) = unit(*perp(tx, ty))
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5

        def bound_along(pos, d, lo, hi):
            if abs(d) < 1e-9: return float("inf")
            return (hi - pos) / d if d > 0 else (lo - pos) / d

        s_bound = max(0.0, min(
            bound_along(mx,  nx, left, right),
            bound_along(mx, -nx, left, right),
            bound_along(my,  ny, top, bottom),
            bound_along(my, -ny, top, bottom),
        ))
        sign = 1.0 if random.random() < 0.5 else -1.0
        s = sign * min(s_bound * 0.9, max(1.0, curv_frac * L))

        r = (L * L) / (8.0 * abs(s)) + abs(s) / 2.0
        d_center = (r - abs(s))
        cx = mx + (nx * sign) * d_center
        cy = my + (ny * sign) * d_center

        a0 = math.atan2(y0 - cy, x0 - cx)
        a1 = math.atan2(y1 - cy, x1 - cx)
        if sign > 0:
            if a1 < a0: a1 += 2 * math.pi
        else:
            if a1 > a0: a1 -= 2 * math.pi

        steps = max(16, int(L / (2.0 * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            ang = a0 + t * (a1 - a0)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        return pts

    def _curve_quadratic(self, p0, p1, bounds, curv_frac) -> List[Tuple[float, float]]:
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-3: return [p0, p1]
        tx, ty = unit(dx, dy); nx, ny = unit(*perp(tx, ty))
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5

        (left, top, right, bottom) = bounds
        def b(pos, d, lo, hi):
            if abs(d) < 1e-9: return float("inf")
            return (hi - pos) / d if d > 0 else (lo - pos) / d

        amp_max = min(
            min(b(x0,  nx, left, right), b(x0, -nx, left, right), b(y0,  ny, top, bottom), b(y0, -ny, top, bottom)),
            min(b(mx,  nx, left, right), b(mx, -nx, left, right), b(my,  ny, top, bottom), b(my, -ny, top, bottom)),
            min(b(x1,  nx, left, right), b(x1, -nx, left, right), b(y1,  ny, top, bottom), b(y1, -ny, top, bottom)),
        ) * 0.9
        amp = max(1.0, min(amp_max, curv_frac * L)) * (1.0 if random.random() < 0.5 else -1.0)

        cx, cy = mx + amp * nx, my + amp * ny
        steps = max(16, int(L / (2.0 * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps; omt = 1.0 - t
            px = omt * omt * x0 + 2.0 * omt * t * cx + t * t * x1
            py = omt * omt * y0 + 2.0 * omt * t * cy + t * t * y1
            pts.append((px, py))
        return pts

    def _curve_cubic(self, p0, p1, bounds, curv_frac) -> List[Tuple[float, float]]:
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy)
        if L < 1e-3: return [p0, p1]
        tx, ty = unit(dx, dy); nx, ny = unit(*perp(tx, ty))

        tlen1 = (0.2 + 0.25 * random.random()) * L
        tlen2 = (0.2 + 0.25 * random.random()) * L
        amp1 = (0.3 + 0.5 * random.random()) * curv_frac * L * (1.0 if random.random() < 0.5 else -1.0)
        amp2 = (0.3 + 0.5 * random.random()) * curv_frac * L * (1.0 if random.random() < 0.5 else -1.0)

        c1 = (x0 + tlen1 * tx + amp1 * nx, y0 + tlen1 * ty + amp1 * ny)
        c2 = (x1 - tlen2 * tx + amp2 * nx, y1 - tlen2 * ty + amp2 * ny)

        steps = max(20, int(L / (1.8 * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps; omt = 1.0 - t
            px = (omt**3) * x0 + 3 * (omt**2) * t * c1[0] + 3 * omt * (t**2) * c2[0] + (t**3) * x1
            py = (omt**3) * y0 + 3 * (omt**2) * t * c1[1] + 3 * omt * (t**2) * c2[1] + (t**3) * y1
            pts.append((px, py))
        return pts

    def _curve_sine(self, p0, p1, bounds, curv_frac, cycles, phase) -> List[Tuple[float, float]]:
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy)
        if L < 1e-3: return [p0, p1]
        tx, ty = unit(dx, dy); nx, ny = unit(*perp(tx, ty))

        (left, top, right, bottom) = bounds
        def b(pos, d, lo, hi):
            if abs(d) < 1e-9: return float("inf")
            return (hi - pos) / d if d > 0 else (lo - pos) / d

        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        amp_max = min(
            min(b(x0,  nx, left, right), b(x0, -nx, left, right), b(y0,  ny, top, bottom), b(y0, -ny, top, bottom)),
            min(b(mx,  nx, left, right), b(mx, -nx, left, right), b(my,  ny, top, bottom), b(my, -ny, top, bottom)),
            min(b(x1,  nx, left, right), b(x1, -nx, left, right), b(y1,  ny, top, bottom), b(y1, -ny, top, bottom)),
        ) * 0.9
        amp = max(1.0, min(amp_max, curv_frac * L))

        steps = max(32, int(L / (1.6 * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            bx = x0 + t * dx; by = y0 + t * dy
            off = amp * math.sin(2.0 * math.pi * cycles * t + phase)
            pts.append((bx + off * nx, by + off * ny))
        return pts

    def _curve_poly_bump(self, p0, p1, curv_frac) -> List[Tuple[float, float]]:
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy)
        if L < 1e-3: return [p0, p1]
        tx, ty = unit(dx, dy); nx, ny = unit(*perp(tx, ty))
        amp = curv_frac * L * (1.0 if random.random() < 0.5 else -1.0)
        steps = max(24, int(L / (1.8 * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            bump = 4.0 * t * (1.0 - t)
            bx = x0 + t * dx; by = y0 + t * dy
            pts.append((bx + amp * bump * nx, by + amp * bump * ny))
        return pts

    def _curve_poly_s(self, p0, p1, curv_frac) -> List[Tuple[float, float]]:
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy)
        if L < 1e-3: return [p0, p1]
        tx, ty = unit(dx, dy); nx, ny = unit(*perp(tx, ty))
        amp = curv_frac * L
        steps = max(24, int(L / (1.8 * SUPERSAMPLE)))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            sshape = (2.0 * t - 1.0) * t * (1.0 - t) * 6.0 / 1.5
            bx = x0 + t * dx; by = y0 + t * dy
            pts.append((bx + amp * sshape * nx, by + amp * sshape * ny))
        return pts

    def _curve_semiellipse(self, p0, p1, bounds, curv_frac) -> List[Tuple[float, float]]:
        sign = 1.0 if random.random() < 0.5 else -1.0
        def g(t):
            u = 2.0 * t - 1.0
            return sign * math.sqrt(max(0.0, 1.0 - u * u))
        return self._curve_from_offset_fn(p0, p1, bounds, curv_frac, g, step_scale=1.6)

    def _curve_superellipse(self, p0, p1, bounds, curv_frac, m) -> List[Tuple[float, float]]:
        m = max(1.05, float(m))
        sign = 1.0 if random.random() < 0.5 else -1.0
        def g(t):
            u = abs(2.0 * t - 1.0)
            return sign * (max(0.0, 1.0 - (u ** m))) ** (1.0 / m)
        return self._curve_from_offset_fn(p0, p1, bounds, curv_frac, g, step_scale=1.6)

    def _curve_gaussian(self, p0, p1, bounds, curv_frac, sigma) -> List[Tuple[float, float]]:
        sigma = max(1e-3, float(sigma))
        base = math.exp(-0.25 / (2.0 * sigma * sigma))
        denom = max(1e-6, 1.0 - base)
        def g(t):
            val = math.exp(-((t - 0.5) ** 2) / (2.0 * sigma * sigma))
            return (val - base) / denom
        return self._curve_from_offset_fn(p0, p1, bounds, curv_frac, g, step_scale=1.6)

    def _curve_catenary(self, p0, p1, bounds, curv_frac, kappa) -> List[Tuple[float, float]]:
        k = max(1e-3, float(kappa))
        cosh_k = math.cosh(k)
        denom = max(1e-6, 1.0 - cosh_k)
        def g(t):
            u = 2.0 * t - 1.0
            return (math.cosh(k * u) - cosh_k) / denom
        return self._curve_from_offset_fn(p0, p1, bounds, curv_frac, g, step_scale=1.6)

    def _curve_sigmoid(self, p0, p1, bounds, curv_frac, beta) -> List[Tuple[float, float]]:
        b = float(beta)
        def raw(t): return t * (1.0 - t) * math.tanh(b * (t - 0.5))
        grid = 129
        gmax = 0.0
        for i in range(grid):
            t = i / (grid - 1)
            gmax = max(gmax, abs(raw(t)))
        gmax = max(gmax, 1e-6)
        def g(t): return raw(t) / gmax
        return self._curve_from_offset_fn(p0, p1, bounds, curv_frac, g, step_scale=1.8)

    # ------------------------ Rendering ------------------------

    def render(self, spec):
        s = self.clamp_spec(spec)
        color = COLORS[s.color_idx]

        # Float layout (avoid early rounding)
        base_margin = SS_CELL * 0.12

        # Decide AA factor from baseline angle & complexity (use param fields)
        ex = s.extra or {}
        baseline_angle = float(ex.get("angle_deg", 0.0))
        rot = int(ex.get("rotation", 0)) % 360
        arc_type = ex.get("arc_type", "circle")
        cycles = float(ex.get("cycles", 1.0))
        # Heuristic AA choice
        AA = self._aa_factor_for_arc((baseline_angle + rot) % 180.0, arc_type, s.thickness, cycles)

        S = int(SS_CELL * AA)
        img = Image.new("RGBA", (S, S), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        stroke_px = max(1.0, float(s.thickness) * SUPERSAMPLE * AA)
        pad = max(1.0, stroke_px * 0.5)
        margin = base_margin * AA
        left = margin + pad
        top = margin + pad
        right = S - margin - pad
        bottom = S - margin - pad
        bounds = (left, top, right, bottom)

        # Endpoints + actual baseline angle (post rotation/scale)
        p0, p1, ang = self._compute_endpoints_px(s, bounds, AA)

        # Build curve points
        curv_frac = float(ex.get("curv_frac", 0.3))
        phase = float(ex.get("phase", 0.0))

        if arc_type == "circle":
            pts = self._curve_circle(p0, p1, bounds, curv_frac)
        elif arc_type == "quadratic":
            pts = self._curve_quadratic(p0, p1, bounds, curv_frac)
        elif arc_type == "cubic":
            pts = self._curve_cubic(p0, p1, bounds, curv_frac)
        elif arc_type == "sine":
            pts = self._curve_sine(p0, p1, bounds, curv_frac, cycles, phase)
        elif arc_type == "cubic_bump":
            pts = self._curve_poly_bump(p0, p1, curv_frac)
        elif arc_type == "s_shape":
            pts = self._curve_poly_s(p0, p1, curv_frac)
        elif arc_type == "semiellipse":
            pts = self._curve_semiellipse(p0, p1, bounds, curv_frac)
        elif arc_type == "superellipse":
            pts = self._curve_superellipse(p0, p1, bounds, curv_frac, ex.get("super_m", 2.0))
        elif arc_type == "gaussian":
            pts = self._curve_gaussian(p0, p1, bounds, curv_frac, ex.get("sigma", 0.2))
        elif arc_type == "catenary":
            pts = self._curve_catenary(p0, p1, bounds, curv_frac, ex.get("kappa", 1.2))
        elif arc_type == "sigmoid":
            pts = self._curve_sigmoid(p0, p1, bounds, curv_frac, ex.get("beta", 3.0))
        else:
            pts = [p0, p1]

        # Densify based on stroke thickness so joins don't stair-step
        max_seg_len = max(1.0, 0.6 * stroke_px)
        pts = self._densify(pts, max_seg_len)

        # Draw stroke with round endpoints and softened joins
        self._polyline(draw, pts, color=color, width=stroke_px)

        # Downsample if we oversampled
        if AA != 1:
            img = img.resize((SS_CELL, SS_CELL), resample=Image.LANCZOS)

        return _down_on_background(img)
