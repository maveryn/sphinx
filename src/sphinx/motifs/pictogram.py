# sphinx/motifs/pictogram.py
# sphinx/motifs/pictogram.py
import math, random
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

Point = Tuple[int, int]


@register_motif
class PictogramMotif(Motif):
    """
    Pictograms: boat, rocket, kite, house, person (very simple stick/icon style).

    Families (spec.extra["family"]):
      - "boat"
      - "rocket"
      - "kite"     (diamond is outline-only; never filled)
      - "house"    (body/roof are outline-only; never filled; windows in 2 columns)
      - "person"   (stick figure with circular head; rounded joints)

    Common extras:
      - rotation (deg) : rotation about tile center
      - aa (int)       : extra supersample factor (1–4) on top of SUPERSAMPLE
      - filled (bool)  : only applies to boat/rocket subparts; ignored by kite/house/person
    """
    name = "pictogram"
    FAMS = ("boat", "rocket", "kite", "house", "person")

    attr_ranges = {
        "size": (0.92, 1.15),
        "thickness": (3, 7),
        "count": (1, 1),
    }

    # ---------- sampling ----------
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        fam = rng.choice(self.FAMS)
        extra: Dict = {
            "family": fam,
            "rotation": rng.uniform(0, 360),
            # favor higher AA for crisper downsampling
            "aa": rng.choice([max(4, SUPERSAMPLE), max(3, SUPERSAMPLE)]),
            "filled": rng.random() < 0.2,  # ignored for kite/house/person
        }

        # Family-specific knobs (kept minimal to preserve simplicity)
        if fam == "boat":
            extra.update({"two_sails": rng.random() < 0.35})
        elif fam == "rocket":
            extra.update({"flame": rng.random() < 0.75})
        elif fam == "kite":
            extra.update({"bows": rng.randint(2, 5)})
        elif fam == "person":
            extra.update({"arms": rng.choice(["down", "out", "up"])})
        elif fam == "house":
            w = rng.choice([1, 2])                           # 50–50
            extra.update({"windows": w})
            if w == 1:
                extra.update({"window_side": rng.choice(["left", "right"])})

        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # ---------- normalization ----------
    def clamp_spec(self, spec):
        smin, smax = self.attr_ranges["size"]
        tmin, tmax = self.attr_ranges["thickness"]

        size = float(getattr(spec, "size", 1.0))
        size = max(smin, min(smax, size))

        thickness = int(getattr(spec, "thickness", 4))
        thickness = max(tmin, min(tmax, thickness))

        ex = dict(spec.extra or {})
        fam = ex.get("family", "boat")
        if fam not in self.FAMS:
            fam = "boat"

        rotation = float(ex.get("rotation", 0.0)) % 360.0
        aa = int(ex.get("aa", max(3, SUPERSAMPLE)))
        aa = max(1, min(4, aa))
        filled = bool(ex.get("filled", False))

        # Per-family clamps
        if fam == "boat":
            ex["two_sails"] = bool(ex.get("two_sails", False))
        elif fam == "rocket":
            ex["flame"] = bool(ex.get("flame", True))
        elif fam == "kite":
            bows = int(ex.get("bows", 3))
            ex["bows"] = max(0, min(6, bows))
        elif fam == "person":
            arms = ex.get("arms", "down")
            if arms not in ("down", "out", "up"):
                arms = "down"
            ex["arms"] = arms
        elif fam == "house":
            windows = 1 if int(ex.get("windows", 2)) <= 1 else 2
            if windows == 1:
                side = ex.get("window_side", "right")
                if side not in ("left", "right"):
                    side = "right"
                ex.update({"windows": 1, "window_side": side})
            else:
                ex.update({"windows": 2})
                ex.pop("window_side", None)

        ex.update({"family": fam, "rotation": rotation, "aa": aa, "filled": filled})
        return spec.clone(count=1, size=size, thickness=thickness, extra=ex)

    # ---------- rendering ----------
    def render(self, spec):
        s = self.clamp_spec(spec)
        col = COLORS[s.color_idx]
        AA = int(s.extra["aa"])
        W = H = SS_CELL * AA
        cx = cy = W // 2
        ow = max(AA, int(s.thickness) * AA)      # silhouette stroke width @ AA scale
        tw = max(AA, (ow * 6) // 10)             # detail stroke width

        # Base canvas at AA scale
        img_big = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        d = ImageDraw.Draw(img_big)

        # Square working bbox centered, scaled by 'size'
        side = int(0.86 * min(W, H) * float(s.size))
        x0 = (W - side) // 2
        y0 = (H - side) // 2
        bbox = (x0, y0, side, side)

        fam = s.extra["family"]
        if fam == "boat":
            self._draw_boat(d, bbox, col, ow, tw, s.extra)
        elif fam == "rocket":
            self._draw_rocket(d, bbox, col, ow, tw, s.extra)
        elif fam == "kite":
            self._draw_kite(d, bbox, col, ow, tw, s.extra)   # outline-only diamond
        elif fam == "house":
            self._draw_house(d, bbox, col, ow, tw, s.extra)  # outline-only body/roof
        else:
            self._draw_person(d, bbox, col, ow, tw, s.extra)

        rot = float(s.extra["rotation"])
        if abs(rot) > 1e-6:
            img_big = img_big.rotate(rot, resample=Image.BICUBIC, center=(cx, cy), expand=False)

        # Downsample once
        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)
        return _down_on_background(img)

    # ---------- helpers ----------
    @staticmethod
    def _U(bbox):
        x, y, w, h = bbox
        def f(ux: float, uy: float) -> Point:
            return (int(round(x + ux * w)), int(round(y + uy * h)))
        return f

    @staticmethod
    def _ellipse(d: ImageDraw.ImageDraw, cx, cy, rx, ry, outline, width, fill=None):
        x0, y0 = int(round(cx - rx)), int(round(cy - ry))
        x1, y1 = int(round(cx + rx)), int(round(cy + ry))
        d.ellipse((x0, y0, x1, y1), outline=outline, width=width, fill=fill)

    @staticmethod
    def _circle(d, cx, cy, r, outline, width, fill=None):
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        d.ellipse((x0, y0, x1, y1), outline=outline, width=width, fill=fill)

    @staticmethod
    def _capped_line(d, p0: Point, p1: Point, col, width):
        # Line with round end caps (drawn as small filled discs)
        d.line([p0, p1], fill=col, width=width)
        r = max(1, width // 2)
        for (x, y) in (p0, p1):
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)

    @staticmethod
    def _poly(d, pts: List[Point], outline, width, fill=None, close=True, rounded=True):
        # Fill first (pre-AA), then outline using capped polyline with rounded vertices.
        if fill is not None:
            d.polygon(pts, fill=fill)
        if close:
            edges = list(zip(pts, pts[1:] + [pts[0]]))
        else:
            edges = list(zip(pts, pts[1:]))

        # draw segments
        for a, b in edges:
            d.line([a, b], fill=outline, width=width)

        if rounded:
            r = max(1, width // 2)
            verts = pts if close else (pts if len(pts) <= 2 else pts[1:-1])
            for (x, y) in verts:
                d.ellipse((x - r, y - r, x + r, y + r), fill=outline)

    @staticmethod
    def _line(d, p0: Point, p1: Point, col, width):
        d.line([p0, p1], fill=col, width=width)

    # ---------- families ----------
    def _draw_boat(self, d, bbox, col, ow, tw, ex):
        u = self._U(bbox)
        filled = bool(ex.get("filled", False))
        two_sails = bool(ex.get("two_sails", False))

        # Hull (trapezoid)
        base_y = 0.84
        hull = [u(0.18, base_y), u(0.82, base_y), u(0.72, base_y - 0.14), u(0.28, base_y - 0.14)]
        self._poly(d, hull, outline=col, width=ow, fill=(col if filled else None))

        # Mast
        self._capped_line(d, u(0.50, base_y - 0.14), u(0.50, 0.18), col, ow)

        # Sail(s), outline-only (keeps airy look)
        sail_right = [u(0.50, 0.20), u(0.50, base_y - 0.14), u(0.78, base_y - 0.14)]
        self._poly(d, sail_right, outline=col, width=tw, fill=None)
        if two_sails:
            sail_left = [u(0.50, 0.26), u(0.50, base_y - 0.14), u(0.30, base_y - 0.20)]
            self._poly(d, sail_left, outline=col, width=tw, fill=None)

        # Flag (small filled triangle)
        flag = [u(0.50, 0.18), u(0.58, 0.21), u(0.50, 0.24)]
        self._poly(d, flag, outline=col, width=tw, fill=(col if filled else None))

    def _draw_rocket(self, d, bbox, col, ow, tw, ex):
        u = self._U(bbox)
        filled = bool(ex.get("filled", False))
        flame = bool(ex.get("flame", True))

        # Body (rect) + nose + fins — using rounded vertices
        bx0, by0 = u(0.34, 0.22)
        bx1, by1 = u(0.66, 0.84)
        d.rectangle((bx0, by0, bx1, by1), outline=col, width=ow, fill=(col if filled else None))
        # Nose
        nose = [u(0.34, 0.32), u(0.50, 0.04), u(0.66, 0.32)]
        self._poly(d, nose, outline=col, width=ow, fill=(col if filled else None))
        # Fins
        left_fin = [u(0.34, 0.79), u(0.18, 0.86), u(0.34, 0.86)]
        right_fin = [u(0.66, 0.79), u(0.82, 0.86), u(0.66, 0.86)]
        self._poly(d, left_fin, outline=col, width=ow, fill=(col if filled else None))
        self._poly(d, right_fin, outline=col, width=ow, fill=(col if filled else None))

        # Window (outline-only) with rounded stroke
        self._circle(d, *u(0.50, 0.40), max(2, ow), outline=col, width=tw, fill=None)

        # Flame
        if flame:
            flame_pts = [u(0.50, 0.96), u(0.42, 0.86), u(0.58, 0.86)]
            self._poly(d, flame_pts, outline=col, width=tw, fill=(col if filled else None))

    def _draw_kite(self, d, bbox, col, ow, tw, ex):
        u = self._U(bbox)
        bows = int(ex.get("bows", 3))

        # Diamond (STRICTLY outline-only) with rounded corners
        top, bottom, left, right = u(0.50, 0.16), u(0.50, 0.78), u(0.28, 0.48), u(0.72, 0.48)
        diamond = [top, right, bottom, left]
        self._poly(d, diamond, outline=col, width=ow, fill=None)

        # Spars
        self._capped_line(d, left, right, col, tw)
        self._capped_line(d, top, bottom, col, tw)

        # String path (slight curve via two segments)
        anchor = bottom
        s1, s2, s3 = u(0.68, 0.90), u(0.56, 0.98), u(0.66, 1.10)  # may leave tile; clipped
        self._capped_line(d, anchor, s1, col, tw)
        self._capped_line(d, s1, s2, col, tw)
        self._capped_line(d, s2, s3, col, tw)

        # Bows along string (placed by arclength)
        if bows > 0:
            segs = [(s1, s2), (s2, s3)]
            lens = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in segs]
            total = sum(lens)
            if total > 1e-3:
                step = total / (bows + 1)
                acc = 0.0
                cur = 0
                a, b = segs[cur]
                L = lens[cur]
                for k in range(bows):
                    target = (k + 1) * step
                    while acc + L < target and cur < len(segs) - 1:
                        acc += L
                        cur += 1
                        a, b = segs[cur]
                        L = lens[cur]
                    t = max(0.0, min(1.0, (target - acc) / max(L, 1e-6)))
                    px = int(round(a[0] + (b[0] - a[0]) * t))
                    py = int(round(a[1] + (b[1] - a[1]) * t))
                    bw = int(0.045 * bbox[2])
                    bh = int(0.020 * bbox[3])
                    self._poly(d, [(px, py), (px - bw, py - bh), (px - bw, py + bh)], outline=col, width=tw, fill=None)
                    self._poly(d, [(px, py), (px + bw, py - bh), (px + bw, py + bh)], outline=col, width=tw, fill=None)

    def _draw_house(self, d, bbox, col, ow, tw, ex):
        """
        House is outline-only for body & roof.
        Windows always use a 2-column layout (two fixed x-positions). If windows == 1,
        one column is left empty; side chosen by 'window_side' (left|right).
        """
        u = self._U(bbox)
        windows = int(ex.get("windows", 2))
        side = ex.get("window_side", "right")

        # Body (STRICT outline-only)
        bx0, by0 = u(0.15, 0.40)
        bx1, by1 = u(0.85, 0.90)
        d.rectangle((bx0, by0, bx1, by1), outline=col, width=ow, fill=None)

        # Roof (outline-only) — endpoints sit exactly on body top to avoid hairline gaps
        apex = u(0.50, 0.20)
        left, right = (bx0, by0), (bx1, by0)
        self._poly(d, [left, apex, right], outline=col, width=ow, fill=None)

        # Door (outline-only)
        dx0, dy0 = u(0.30, 0.60)
        dx1, dy1 = u(0.48, 0.90)
        d.rectangle((dx0, dy0, dx1, dy1), outline=col, width=tw, fill=None)
        # knob
        self._circle(d, dx1 - max(3, tw // 2), (dy0 + dy1) // 2, max(2, tw // 2),
                     outline=col, width=1, fill=col)

        # Window grid: always two column centers on the right half
        house_w = bx1 - bx0
        c_left  = int(round(bx0 + 0.60 * house_w))
        c_right = int(round(bx0 + 0.80 * house_w))

        win_w = max(int(0.14 * bbox[2]), 2 * tw)
        win_h = win_w
        wy0 = int(round((by0 + by1) * 0.5 - win_h * 0.5))
        wy1 = wy0 + win_h

        def draw_window(cx):
            wx0 = cx - win_w // 2
            wx1 = cx + win_w // 2
            # frame
            d.rectangle((wx0, wy0, wx1, wy1), outline=col, width=tw, fill=None)
            # mullions: keep 2 columns (one vertical), plus one horizontal → 2x2 panes
            self._capped_line(d, ( (wx0 + wx1)//2, wy0 ), ( (wx0 + wx1)//2, wy1 ), col, max(1, tw))
            self._capped_line(d, ( wx0, (wy0 + wy1)//2 ), ( wx1, (wy0 + wy1)//2 ), col, max(1, tw))

        if windows == 2:
            for cx in (c_left, c_right):
                draw_window(cx)
        else:
            draw_window(c_left if side == "left" else c_right)
            # leave the other column empty to preserve 2-column layout

    def _draw_person(self, d, bbox, col, ow, tw, ex):
        """
        Very simple person: head circle + torso + arms + legs.
        All limbs use capped lines (rounded joints) for much cleaner rendering.
        """
        u = self._U(bbox)
        arms_mode = ex.get("arms", "down")

        # Head (outline-only to keep icon feel)
        hx, hy = u(0.50, 0.28)
        hr = int(0.10 * min(bbox[2], bbox[3]))
        self._circle(d, hx, hy, hr, outline=col, width=tw, fill=None)

        # Torso (neck->hip)
        neck = u(0.50, 0.38)
        hip  = u(0.50, 0.68)
        self._capped_line(d, neck, hip, col, ow)

        # Shoulder & arm anchors
        shoulder_y = 0.46
        Ls = u(0.44, shoulder_y)
        Rs = u(0.56, shoulder_y)

        if arms_mode == "out":
            # straight line across shoulders
            self._capped_line(d, Ls, Rs, col, ow)
        elif arms_mode == "up":
            # V-up to neck
            self._capped_line(d, Ls, neck, col, ow)
            self._capped_line(d, Rs, neck, col, ow)
        else:  # "down"
            # relaxed: slight diagonal down
            self._capped_line(d, Ls, u(0.46, 0.60), col, ow)
            self._capped_line(d, Rs, u(0.54, 0.60), col, ow)

        # Hips/legs
        Lh = hip
        self._capped_line(d, hip, u(0.40, 0.92), col, ow)
        self._capped_line(d, hip, u(0.60, 0.92), col, ow)
