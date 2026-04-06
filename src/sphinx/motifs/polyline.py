# sphinx/motifs/polyline.py
import math
import random
from typing import List, Tuple
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from ..utils.geom import unit, perp
from .helpers import _down_on_background, _rot2d
from ..utils.rng import choice_weighted

Pt = Tuple[float, float]  # normalized [0,1]^2


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _dist(a: Pt, b: Pt) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
def _apply_rot_scale_norm(pts: List[Pt], rot_deg: float, scale: float) -> List[Pt]:
    """Apply rotation and scale about tile center (0.5,0.5) in normalized space."""
    cx, cy = 0.5, 0.5
    scale = max(0.2, min(2.0, float(scale)))
    out: List[Pt] = []
    if abs(rot_deg) % 360 == 0 and abs(scale - 1.0) < 1e-6:
        return [( _clamp01(x), _clamp01(y) ) for (x, y) in pts]
    for (x, y) in pts:
        vx, vy = (x - cx) * scale, (y - cy) * scale
        if rot_deg:
            vx, vy = _rot2d(vx, vy, rot_deg)
        out.append((_clamp01(cx + vx), _clamp01(cy + vy)))
    return out


def _random_poly_points(rng: random.Random, n: int) -> List[Pt]:
    """Bounded random walk in [0,1]^2 with min step; clamps to avoid degeneracy."""
    n = max(2, int(n))
    pts: List[Pt] = []
    x, y = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    pts.append((x, y))
    for _ in range(1, n):
        tries = 0
        while True:
            tries += 1
            ang = rng.uniform(0, 2 * math.pi)
            step = rng.uniform(0.12, 0.28)
            nx = _clamp01(x + step * math.cos(ang))
            ny = _clamp01(y + step * math.sin(ang))
            if _dist((x, y), (nx, ny)) >= 0.05 or tries > 10:
                x, y = nx, ny
                pts.append((x, y))
                break
    return pts


@register_motif
class PolylineMotif(Motif):
    """
    Polyline motif with straight and/or quadratic-curved edges.

    Rendering quality upgrades:
      - Rotation/scale applied analytically in normalized space (no bitmap rotate).
      - Angle/curvature-aware adaptive AA (1x/2x/3x) + LANCZOS downsample.
      - Densify segments relative to stroke width.
      - Round endpoints and soft interior joins (small disks at vertices).
    """
    name = "polyline"
    attr_ranges = {"thickness": (2, 5), "points": (3, 7)}

    # ------------------------ spec sampling ------------------------
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        thickness = rng.randint(*self.attr_ranges["thickness"])
        n = rng.randint(*self.attr_ranges["points"])

        mode_weights = [1.0, 1.0, 1.0]
        poly_mode = choice_weighted(rng, ["straight", "curved", "mixed"], mode_weights)

        extra = {
            "points": _random_poly_points(rng, n),
            "num_points": n,
            "poly_mode": poly_mode,
            "mode_weights": mode_weights,
            "switch_keep_prob": rng.uniform(0.4, 0.6),
            "block_len_range": (2, 4),
            "curv_frac": rng.uniform(0.18, 0.4),
            "curv_jitter": 0.25,
            "rotation": int(rng.uniform(0, 360)) % 360,
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

    # ------------------------ clamping ------------------------
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})

        thick = max(1, int(getattr(spec, "thickness", self.attr_ranges["thickness"][0])))
        rot = int(float(ex.get("rotation", 0.0))) % 360
        ex["rotation"] = rot
        ex["scale"] = float(ex.get("scale", 1.0))

        pts = ex.get("points")
        n_req = int(ex.get("num_points", self.attr_ranges["points"][0]))
        if not isinstance(pts, list) or len(pts) < 2:
            pts = _random_poly_points(random.Random(spec.seed), max(2, n_req))
        else:
            pts = [(_clamp01(float(x)), _clamp01(float(y))) for (x, y) in pts]
            if len(pts) < 2:
                pts = _random_poly_points(random.Random(spec.seed), max(2, n_req))
        ex["points"] = pts
        ex["num_points"] = max(2, int(len(pts)))

        weights = ex.get("mode_weights", [1.0, 1.0, 1.0])
        if not isinstance(weights, (list, tuple)) or len(weights) != 3 or sum(float(w) for w in weights) <= 0:
            weights = [1.0, 1.0, 1.0]
        weights = [max(0.0, float(w)) for w in weights]
        ex["mode_weights"] = weights

        mode = ex.get("poly_mode")
        if mode not in ("straight", "curved", "mixed"):
            rng_local = random.Random((spec.seed << 1) ^ thick)
            mode = choice_weighted(rng_local, ["straight", "curved", "mixed"], weights)
        ex["poly_mode"] = mode

        block_rng = ex.get("block_len_range", (2, 4))
        try:
            bmin, bmax = int(block_rng[0]), int(block_rng[1])
        except Exception:
            bmin, bmax = 2, 4
        if bmin > bmax:
            bmin, bmax = bmax, bmin
        bmin = max(1, bmin)
        bmax = max(bmin, bmax)
        ex["block_len_range"] = (bmin, bmax)
        ex["switch_keep_prob"] = min(1.0, max(0.0, float(ex.get("switch_keep_prob", 0.5))))

        ex["curv_frac"] = max(0.0, min(1.0, float(ex.get("curv_frac", 0.25))))
        ex["curv_jitter"] = max(0.0, min(0.95, float(ex.get("curv_jitter", 0.25))))

        return spec.clone(thickness=thick, extra=ex)

    # ------------------------ AA heuristics ------------------------
    @staticmethod
    def _aa_factor_for_poly(angles: List[float], has_curves: bool, thickness: int) -> int:
        """
        Choose AA multiplier based on diagonalness across segments and curve presence.
        'angles' are degrees of each segment (atan2) modulo 180.
        """
        if not angles:
            return 1
        # distance to nearest axis (0 or 90) for each segment: 0=most diagonal
        axis_dist = [abs(((a + 45) % 90) - 45) for a in angles]
        worst = min(axis_dist)  # most diagonal segment
        base = 3 if worst < 15 else 2 if worst < 30 else 1
        if has_curves:
            base = max(base, 2)
        if thickness <= 2:
            base = max(base, 2)
        return base

    # ------------------------ drawing helpers ------------------------
    @staticmethod
    def _map_to_pixels(pt: Pt, left: float, top: float, right: float, bottom: float) -> Tuple[float, float]:
        x = left + pt[0] * (right - left)
        y = top + pt[1] * (bottom - top)
        return (x, y)

    @staticmethod
    def _max_amp_for_sign(mx: float, my: float, nx: float, ny: float,
                          left: float, top: float, right: float, bottom: float, sign: float) -> float:
        """Max |amp| along ±normal keeping (mx,my)+sign*amp*(nx,ny) inside bounds."""
        def bound(pos, d, lo, hi):
            if abs(d) < 1e-9:
                return float("inf")
            return (hi - pos) / d if d > 0 else (lo - pos) / d
        allow_plus = min(bound(mx, nx, left, right), bound(my, ny, top, bottom))
        allow_minus = min(bound(mx, -nx, left, right), bound(my, -ny, top, bottom))
        allow = allow_plus if sign >= 0 else allow_minus
        return max(0.0, allow * 0.9)

    @staticmethod
    def _quad_curve_points(p0: Tuple[float, float], p1: Tuple[float, float], amp: float, steps: int) -> List[Tuple[float, float]]:
        """Quadratic Bezier from p0 to p1 with control at midpoint + amp*normal."""
        x0, y0 = p0; x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        tx, ty = unit(dx, dy)
        nx, ny = perp(tx, ty)
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        cx, cy = mx + amp * nx, my + amp * ny

        pts: List[Tuple[float, float]] = []
        steps = max(6, steps)
        for i in range(steps + 1):
            t = i / steps
            omt = 1.0 - t
            x = omt * omt * x0 + 2.0 * omt * t * cx + t * t * x1
            y = omt * omt * y0 + 2.0 * omt * t * cy + t * t * y1
            pts.append((x, y))
        return pts

    @staticmethod
    def _densify_segment(p0: Tuple[float, float], p1: Tuple[float, float], max_len: float) -> List[Tuple[float, float]]:
        """Linear interpolation so no subsegment exceeds max_len."""
        x0, y0 = p0; x1, y1 = p1
        L = math.hypot(x1 - x0, y1 - y0)
        if L <= max_len:
            return [p0, p1]
        n = max(1, int(math.ceil(L / max_len)))
        out = []
        for k in range(n + 1):
            t = k / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        return out

    def _draw_with_round_joins(self, draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], color, width: float, join_pts: List[Tuple[float, float]]):
        """Draw polyline and stamp disks at endpoints and interior join points."""
        if len(pts) < 2:
            return
        w = max(1, int(round(width)))
        draw.line(pts, fill=color, width=w)
        r = max(0.5, w / 2.0)
        # endpoints
        x0, y0 = pts[0]; x1, y1 = pts[-1]
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=color)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=color)
        # interior joins
        for (x, y) in join_pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    # ------------------------ rendering ------------------------
    def render(self, spec):
        s = self.clamp_spec(spec)
        color = COLORS[s.color_idx]
        ex = s.extra or {}

        # Local deterministic RNG for render-time choices
        rng = random.Random((s.seed << 2) ^ 0x9E3779B1)

        # Apply rotation/scale analytically in normalized space
        pts_n: List[Pt] = ex["points"]
        pts_n = _apply_rot_scale_norm(pts_n, rot_deg=int(ex.get("rotation", 0)) % 360, scale=float(ex.get("scale", 1.0)))

        # AA factor: derive from segment angles and whether we have curves
        mode = ex["poly_mode"]
        has_curves = (mode != "straight")
        # compute angles from normalized points
        angles = []
        for i in range(len(pts_n) - 1):
            x0, y0 = pts_n[i]; x1, y1 = pts_n[i + 1]
            if x1 == x0 and y1 == y0:
                continue
            angles.append((math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0))
        AA = self._aa_factor_for_poly(angles, has_curves, s.thickness)

        # High-res canvas
        S = int(SS_CELL * AA)
        img = Image.new("RGBA", (S, S), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Float layout & stroke
        base_margin = SS_CELL * 0.12
        stroke_px = max(1.0, float(s.thickness) * SUPERSAMPLE * AA)
        pad = max(1.0, stroke_px * 0.5)
        margin = base_margin * AA
        left = margin + pad
        top = margin + pad
        right = S - margin - pad
        bottom = S - margin - pad

        # Map to pixels after transform
        pts_px = [self._map_to_pixels(p, left, top, right, bottom) for p in pts_n]
        if len(pts_px) < 2:
            return _down_on_background(img)  # nothing to draw

        # Edge styles
        edges = len(pts_px) - 1
        if mode in ("straight", "curved"):
            styles = ["S" if mode == "straight" else "C"] * edges
        else:
            bmin, bmax = ex["block_len_range"]
            keep_p = float(ex.get("switch_keep_prob", 0.5))
            cur = rng.choice(["S", "C"])
            styles = []
            i = 0
            while i < edges:
                run_len = rng.randint(bmin, bmax)
                for _ in range(run_len):
                    if i >= edges: break
                    styles.append(cur); i += 1
                if rng.random() > keep_p:
                    cur = "C" if cur == "S" else "S"

        # Build a single master polyline by concatenating per-edge point lists,
        # densified relative to stroke width. Also record original vertices for join disks.
        max_seg_len = max(1.0, 0.6 * stroke_px)
        curv_frac = float(ex.get("curv_frac", 0.25))
        jitter = float(ex.get("curv_jitter", 0.25))

        master_pts: List[Tuple[float, float]] = []
        join_pts: List[Tuple[float, float]] = []  # pixel coords of original vertices (interior only)

        for i in range(edges):
            p0 = pts_px[i]
            p1 = pts_px[i + 1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-3:
                continue

            if styles[i] == "S":
                seg_pts = self._densify_segment(p0, p1, max_seg_len)
            else:
                tx, ty = unit(dx, dy)
                nx, ny = perp(tx, ty)
                mx, my = (p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5
                sign = 1.0 if rng.random() < 0.5 else -1.0
                amp_req = seg_len * curv_frac * (1.0 + jitter * (rng.random() * 2.0 - 1.0))
                amp_allow = self._max_amp_for_sign(mx, my, nx, ny, left, top, right, bottom, sign)
                amp = sign * min(amp_req, amp_allow)
                steps = max(12, int(seg_len / (1.6 * SUPERSAMPLE)))
                seg_pts = self._quad_curve_points(p0, p1, amp, steps)
                # densify again if needed
                dense: List[Tuple[float, float]] = [seg_pts[0]]
                for j in range(1, len(seg_pts)):
                    chunk = self._densify_segment(dense[-1], seg_pts[j], max_seg_len)
                    dense.pop()  # remove duplicate start
                    dense.extend(chunk)
                seg_pts = dense

            if not master_pts:
                master_pts.extend(seg_pts)
            else:
                # avoid duplicating the connecting point
                if master_pts[-1] == seg_pts[0]:
                    master_pts.extend(seg_pts[1:])
                else:
                    master_pts.extend(seg_pts)

            # record interior join (original vertex at p1), except final endpoint
            if i + 1 < len(pts_px) - 1:
                join_pts.append(pts_px[i + 1])

        # Draw once, with round endpoints and softened joins
        self._draw_with_round_joins(draw, master_pts, color=color, width=stroke_px, join_pts=join_pts)

        # Downsample if oversampled
        if AA != 1:
            img = img.resize((SS_CELL, SS_CELL), resample=Image.LANCZOS)

        return _down_on_background(img)
