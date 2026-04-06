# sphinx/tasks/geometric/geometric_position.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES

# Reuse background from geometry module
from sphinx.geometry import graph_paper_rgb

# Optional RGBA paste helper (like geometric_sort)
try:
    from sphinx.utils.drawing import paste_rgba  # type: ignore
except Exception:
    paste_rgba = None

# ------------------------------------------------------------
# Tunables
# ------------------------------------------------------------

COLOR_POOL = [
    ("blue",   "#1f77b4"),
    ("red",    "#d62728"),
    ("green",  "#2ca02c"),
    ("purple", "#9467bd"),
    ("orange", "#ff7f0e"),
]

# Number of big shapes PMF (k, probability)
BIG_SHAPE_COUNT_PMF: List[Tuple[int, float]] = [
    (1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25)
]
BIG_SHAPE_COUNT_VALUES = [int(k) for k, _ in BIG_SHAPE_COUNT_PMF]
BIG_SHAPE_COUNT_MIN = min(BIG_SHAPE_COUNT_VALUES)
BIG_SHAPE_COUNT_MAX = max(BIG_SHAPE_COUNT_VALUES)

DEFAULT_CANVAS = (768, 512)  # WxH

SMALL_SHAPES_MIN_DEFAULT = 10
SMALL_SHAPES_MAX_DEFAULT = 40

ANSWER_MIN_DEFAULT = 1
ANSWER_MAX_DEFAULT = 20

SHAPE_KINDS_DEFAULT = ("circle", "triangle", "square", "pentagon", "hexagon")
RELATIONS_DEFAULT   = ("inside", "outside", "above", "below", "left", "right")

# Draw style
BIG_OUTLINE_RGB          = (0, 0, 0)
SMALL_SHAPE_FILL_RGB     = (0, 0, 0)
SMALL_SHAPE_OUTLINE_RGB  = (0, 0, 0)

# Rendering quality (ported style from geometric_sort)
AA_SCALE           = 2                # supersampling factor for anti-aliased edges
BIG_OUTLINE_W      = 5                # px outline width for big colored shapes
SMALL_OUTLINE_W    = 2                # px outline width for small black shapes
DOWNSAMPLE_FILTER  = getattr(Image, "LANCZOS", Image.BICUBIC)

# --- visual clearance so shapes never look like they touch (accounts for outlines & AA) ---
AA_FUDGE = 1.0  # extra px to keep a visible gap after downsampling
SEP_REGION = 10
SEP_SHAPE = 10
SEP_REGION_VISUAL = SEP_REGION + (BIG_OUTLINE_W * 0.5) + (SMALL_OUTLINE_W * 0.5) + AA_FUDGE
SEP_SHAPE_VISUAL  = SEP_SHAPE  + (SMALL_OUTLINE_W) + AA_FUDGE


# placement & packing
OUTER_PAD         = 18           # keep away from canvas border
GRID_STEP         = 36           # candidate lattice spacing for small shapes
R_MIN, R_MAX      = 10, 22       # small-shape "radius" in pixels
SEP_SHAPE         = 2            # extra clearance between small shapes (pixels)
SEP_REGION        = 2            # clearance to big shape boundary for small shapes (pixels)
SEP_BIG           = 8            # minimum gap between big shapes' bounding boxes (pixels)
PACK_TRIES_PER_SHAPE = 5000       # attempts per small shape before giving up
GLOBAL_RETRIES        = 20       # times we rebuild the whole scene before giving up
PLACE_BIG_TRIES       = 500      # attempts per big shape before restart

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)

def _dist_point_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    projx, projy = x1 + t * vx, y1 + t * vy
    return math.hypot(px - projx, py - projy)

def _triangle_vertices_equilateral(cx: float, cy: float, side: float, rot_deg: float) -> List[Tuple[float, float]]:
    # For an equilateral triangle, distance from centroid to vertex = side / sqrt(3)
    Rv = side / math.sqrt(3.0)
    rot = math.radians(rot_deg)
    return [(cx + Rv * math.cos(rot + 2 * math.pi * i / 3.0),
             cy + Rv * math.sin(rot + 2 * math.pi * i / 3.0)) for i in range(3)]

def _point_in_triangle(px: float, py: float, verts: List[Tuple[float, float]]) -> bool:
    (x1, y1), (x2, y2), (x3, y3) = verts
    # Barycentric technique (signs)
    def _sign(xa, ya, xb, yb, xc, yc):
        return (xa - xc) * (yb - yc) - (xb - xc) * (ya - yc)
    b1 = _sign(px, py, x1, y1, x2, y2) < 0.0
    b2 = _sign(px, py, x2, y2, x3, y3) < 0.0
    b3 = _sign(px, py, x3, y3, x1, y1) < 0.0
    return (b1 == b2) and (b2 == b3)

def _bbox_from_points(pts: Iterable[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = int(math.floor(min(xs))), int(math.ceil(max(xs)))
    y0, y1 = int(math.floor(min(ys))), int(math.ceil(max(ys)))
    return (x0, y0, x1, y1)

def _bbox_inside_canvas(b: Tuple[int, int, int, int], W: int, H: int, pad: int) -> bool:
    x0, y0, x1, y1 = b
    return (x0 >= pad) and (y0 >= pad) and (x1 <= W - pad) and (y1 <= H - pad)

def _aabb_separated(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], sep: int) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax1 + sep <= bx0) or (bx1 + sep <= ax0) or (ay1 + sep <= by0) or (by1 + sep <= ay0)

# ------------------------------------------------------------
# Big shape classes
# ------------------------------------------------------------

class BigShape:
    kind: str
    name: str       # color name
    rgb: Tuple[int, int, int]

    def bbox(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def draw(self, dr: ImageDraw.ImageDraw):
        """(legacy path) direct draw — not used after AA patches introduced."""
        raise NotImplementedError

    # --- geometry for strict relations ---
    def leftmost(self) -> float:  raise NotImplementedError
    def rightmost(self) -> float: raise NotImplementedError
    def topmost(self) -> float:   raise NotImplementedError
    def bottommost(self) -> float:raise NotImplementedError

    def contains_pt(self, px: float, py: float) -> bool:
        raise NotImplementedError

    def inside_margin(self, px: float, py: float) -> float:
        """If inside region, return minimum distance to boundary; else return < 0."""
        raise NotImplementedError

    def dist_to_region(self, px: float, py: float) -> float:
        """If outside region, return Euclidean distance to region; else return 0."""
        raise NotImplementedError

    def relation_of_shape(self, s: "SmallShape", sep_region: float = 0.0) -> Dict[str, bool]:
        """Strict radius-aware relations for a circular footprint."""
        cx, cy, r = s.cx, s.cy, s.r
        inside  = self.contains_pt(cx, cy) and (self.inside_margin(cx, cy) > r + sep_region)
        outside = (not self.contains_pt(cx, cy)) and (self.dist_to_region(cx, cy) >= r + sep_region)

        left   = (cx + r) < (self.leftmost()  - sep_region)
        right  = (cx - r) > (self.rightmost() + sep_region)
        above  = (cy + r) < (self.topmost()   - sep_region)
        below  = (cy - r) > (self.bottommost()+ sep_region)

        return {
            "inside": inside, "outside": outside,
            "left": left, "right": right, "above": above, "below": below
        }

@dataclass
class BigRect(BigShape):
    name: str
    rgb: Tuple[int, int, int]
    x: int
    y: int
    w: int
    h: int
    kind: str = "rectangle"

    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def draw(self, dr: ImageDraw.ImageDraw):
        dr.rectangle(self.bbox(), fill=self.rgb, outline=BIG_OUTLINE_RGB, width=BIG_OUTLINE_W)

    def leftmost(self) -> float:   return float(self.x)
    def rightmost(self) -> float:  return float(self.x + self.w)
    def topmost(self) -> float:    return float(self.y)
    def bottommost(self) -> float: return float(self.y + self.h)

    def contains_pt(self, px: float, py: float) -> bool:
        return (self.x < px < self.x + self.w) and (self.y < py < self.y + self.h)

    def inside_margin(self, px: float, py: float) -> float:
        if not self.contains_pt(px, py):
            return -1.0
        return min(px - self.x, (self.x + self.w) - px, py - self.y, (self.y + self.h) - py)

    def dist_to_region(self, px: float, py: float) -> float:
        x0, y0, x1, y1 = self.bbox()
        dx = 0.0 if x0 <= px <= x1 else (x0 - px if px < x0 else px - x1)
        dy = 0.0 if y0 <= py <= y1 else (y0 - py if py < y0 else py - y1)
        return math.hypot(dx, dy)

@dataclass
class BigCircle(BigShape):
    name: str
    rgb: Tuple[int, int, int]
    cx: float
    cy: float
    R: float
    kind: str = "circle"

    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(self.cx - self.R), int(self.cy - self.R),
                int(self.cx + self.R), int(self.cy + self.R))

    def draw(self, dr: ImageDraw.ImageDraw):
        dr.ellipse(self.bbox(), fill=self.rgb, outline=BIG_OUTLINE_RGB, width=BIG_OUTLINE_W)

    def leftmost(self) -> float:   return float(self.cx - self.R)
    def rightmost(self) -> float:  return float(self.cx + self.R)
    def topmost(self) -> float:    return float(self.cy - self.R)
    def bottommost(self) -> float: return float(self.cy + self.R)

    def contains_pt(self, px: float, py: float) -> bool:
        return _euclid(px, py, self.cx, self.cy) < self.R

    def inside_margin(self, px: float, py: float) -> float:
        d = _euclid(px, py, self.cx, self.cy)
        if d >= self.R:
            return -1.0
        return self.R - d

    def dist_to_region(self, px: float, py: float) -> float:
        d = _euclid(px, py, self.cx, self.cy)
        return max(0.0, d - self.R)

@dataclass
class BigTriangle(BigShape):
    name: str
    rgb: Tuple[int, int, int]
    cx: float
    cy: float
    side: float
    rot_deg: float
    kind: str = "triangle"

    @property
    def verts(self) -> List[Tuple[float, float]]:
        return _triangle_vertices_equilateral(self.cx, self.cy, self.side, self.rot_deg)

    def bbox(self) -> Tuple[int, int, int, int]:
        return _bbox_from_points(self.verts)

    def draw(self, dr: ImageDraw.ImageDraw):
        # Kept for completeness; main path now uses AA patch paste.
        poly = self.verts
        dr.polygon(poly, fill=self.rgb, outline=BIG_OUTLINE_RGB)
        # Outline width handled better by AA patches; no extra line here.

    def leftmost(self) -> float:
        return float(min(x for x, _ in self.verts))

    def rightmost(self) -> float:
        return float(max(x for x, _ in self.verts))

    def topmost(self) -> float:
        return float(min(y for _, y in self.verts))

    def bottommost(self) -> float:
        return float(max(y for _, y in self.verts))

    def contains_pt(self, px: float, py: float) -> bool:
        return _point_in_triangle(px, py, self.verts)

    def inside_margin(self, px: float, py: float) -> float:
        if not self.contains_pt(px, py):
            return -1.0
        v = self.verts
        dists = [_dist_point_to_segment(px, py, v[i][0], v[i][1], v[(i + 1) % 3][0], v[(i + 1) % 3][1]) for i in range(3)]
        return float(min(dists))

    def dist_to_region(self, px: float, py: float) -> float:
        if self.contains_pt(px, py):
            return 0.0
        v = self.verts
        dists = [_dist_point_to_segment(px, py, v[i][0], v[i][1], v[(i + 1) % 3][0], v[(i + 1) % 3][1]) for i in range(3)]
        return float(min(dists))

# ------------------------------------------------------------
# Small shapes (black)
# ------------------------------------------------------------

@dataclass
class SmallShape:
    kind: str                  # circle|triangle|square|pentagon|hexagon
    cx: float
    cy: float
    r: float                   # conservative bounding-circle radius (circumradius for polygons)
    rot_deg: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

def _regular_ngon(cx: float, cy: float, r: float, n: int, rot_deg: float) -> List[Tuple[float, float]]:
    rot = math.radians(rot_deg)
    return [(cx + r * math.cos(rot + 2*math.pi*i/n),
             cy + r * math.sin(rot + 2*math.pi*i/n)) for i in range(n)]

# ------------------------- AA rendering helpers -------------------------

def _render_rect_patch(x0: int, y0: int, x1: int, y1: int,
                       fill_rgb: Tuple[int,int,int], outline_rgb: Tuple[int,int,int], outline_w: int
                       ) -> Tuple[Image.Image, Tuple[int,int]]:
    pad = (outline_w // 2) + 1
    w = (x1 - x0) + 2 * pad
    h = (y1 - y0) + 2 * pad
    W, H = max(1, w * AA_SCALE), max(1, h * AA_SCALE)
    patch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(patch)
    rx0 = pad * AA_SCALE
    ry0 = pad * AA_SCALE
    rx1 = rx0 + (x1 - x0) * AA_SCALE
    ry1 = ry0 + (y1 - y0) * AA_SCALE
    dr.rectangle([rx0, ry0, rx1, ry1],
                 fill=(*fill_rgb, 255),
                 outline=(*outline_rgb, 255),
                 width=max(1, outline_w * AA_SCALE))
    patch = patch.resize((w, h), resample=DOWNSAMPLE_FILTER)
    return patch, (x0 - pad, y0 - pad)

def _render_ellipse_patch(cx: float, cy: float, R: float,
                          fill_rgb: Tuple[int,int,int], outline_rgb: Tuple[int,int,int], outline_w: int
                          ) -> Tuple[Image.Image, Tuple[int,int]]:
    x0, y0 = int(cx - R), int(cy - R)
    x1, y1 = int(cx + R), int(cy + R)
    pad = (outline_w // 2) + 1
    w = (x1 - x0) + 2 * pad
    h = (y1 - y0) + 2 * pad
    W, H = max(1, w * AA_SCALE), max(1, h * AA_SCALE)
    patch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(patch)
    ex0 = pad * AA_SCALE
    ey0 = pad * AA_SCALE
    ex1 = ex0 + (x1 - x0) * AA_SCALE
    ey1 = ey0 + (y1 - y0) * AA_SCALE
    dr.ellipse([ex0, ey0, ex1, ey1],
               fill=(*fill_rgb, 255),
               outline=(*outline_rgb, 255),
               width=max(1, outline_w * AA_SCALE))
    patch = patch.resize((w, h), resample=DOWNSAMPLE_FILTER)
    return patch, (x0 - pad, y0 - pad)

def _render_polygon_patch(verts: List[Tuple[float, float]],
                          fill_rgb: Tuple[int,int,int], outline_rgb: Tuple[int,int,int], outline_w: int
                          ) -> Tuple[Image.Image, Tuple[int,int]]:
    x0, y0, x1, y1 = _bbox_from_points(verts)
    pad = (outline_w // 2) + 1
    w = (x1 - x0) + 2 * pad
    h = (y1 - y0) + 2 * pad
    W, H = max(1, w * AA_SCALE), max(1, h * AA_SCALE)
    patch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(patch)
    pts = [((x - x0 + pad) * AA_SCALE, (y - y0 + pad) * AA_SCALE) for (x, y) in verts]
    dr.polygon(pts,
               fill=(*fill_rgb, 255),
               outline=(*outline_rgb, 255))
    # draw a proper thick outline by stroking edges
    if outline_w > 0:
        W_OUT = max(1, outline_w * AA_SCALE)
        for i in range(len(pts)):
            xA, yA = pts[i]
            xB, yB = pts[(i + 1) % len(pts)]
            dr.line([xA, yA, xB, yB], fill=(*outline_rgb, 255), width=W_OUT, joint="curve")
    patch = patch.resize((w, h), resample=DOWNSAMPLE_FILTER)
    return patch, (x0 - pad, y0 - pad)

def _draw_small_shape_rgba(bg: Image.Image, s: "SmallShape"):
    fill = SMALL_SHAPE_FILL_RGB
    outline = SMALL_SHAPE_OUTLINE_RGB
    w_out = SMALL_OUTLINE_W

    if s.kind == "circle":
        patch, (x, y) = _render_ellipse_patch(s.cx, s.cy, s.r, fill, outline, w_out)
    else:
        nmap = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}
        n = nmap.get(s.kind, 4)
        verts = _regular_ngon(s.cx, s.cy, s.r, n, s.rot_deg)
        patch, (x, y) = _render_polygon_patch(verts, fill, outline, w_out)

    if paste_rgba is not None:
        paste_rgba(bg, patch, (int(x), int(y)))
    else:
        bg.paste(patch, (int(x), int(y)), patch)

# ------------------------------------------------------------
# Big-shape sampling & placement
# ------------------------------------------------------------

def _sample_from_pmf(rng: random.Random, items: List[Tuple[int, float]]) -> int:
    total = sum(max(0.0, p) for _, p in items)
    if total <= 0:
        # fallback uniform over keys
        keys = [k for k, _ in items]
        return rng.choice(keys)
    x = rng.uniform(0.0, total)
    acc = 0.0
    for k, p in items:
        acc += max(0.0, p)
        if x <= acc:
            return k
    return items[-1][0]

def _aabb_of_bigshape(s: BigShape) -> Tuple[int, int, int, int]:
    return s.bbox()

def _non_overlapping_with_all(bbox: Tuple[int, int, int, int], existing: List[BigShape], sep: int) -> bool:
    return all(_aabb_separated(bbox, _aabb_of_bigshape(e), sep) for e in existing)

def _choose_big_kind(rng: random.Random) -> str:
    return rng.choice(["rectangle", "circle", "triangle"])

def _size_ranges_for_k(k: int, W: int, H: int) -> Dict[str, Tuple[float, float]]:
    """
    Return size ranges tuned by count k to keep placement feasible.
    - Rectangles: fractions of W/H.
    - Circle: radius as fraction of min(W,H).
    - Triangle: side length as fraction of min(W,H).
    """
    mw = float(W); mh = float(H); m = float(min(W, H))
    if k <= 1:
        return {
            "rect_w": (0.50 * mw, 0.66 * mw), "rect_h": (0.46 * mh, 0.60 * mh),
            "circle_R": (0.22 * m, 0.30 * m), "tri_side": (0.50 * m, 0.70 * m)
        }
    if k == 2:
        return {
            "rect_w": (0.42 * mw, 0.52 * mw), "rect_h": (0.44 * mh, 0.56 * mh),
            "circle_R": (0.20 * m, 0.26 * m), "tri_side": (0.45 * m, 0.62 * m)
        }
    if k == 3:
        return {
            "rect_w": (0.36 * mw, 0.48 * mw), "rect_h": (0.40 * mh, 0.50 * mh),
            "circle_R": (0.16 * m, 0.23 * m), "tri_side": (0.40 * m, 0.56 * m)
        }
    # k >= 4
    return {
        "rect_w": (0.30 * mw, 0.42 * mw), "rect_h": (0.34 * mh, 0.46 * mh),
        "circle_R": (0.14 * m, 0.20 * m), "tri_side": (0.34 * m, 0.48 * m)
    }

def _spawn_big_shapes(rng: random.Random, W: int, H: int, k: int) -> List[BigShape]:
    """
    Place k big shapes (rectangle/circle/triangle) with unique colors, no overlapping AABBs.
    """
    assert k >= 1
    names_hex = rng.sample(COLOR_POOL, k)
    placed: List[BigShape] = []
    sr = _size_ranges_for_k(k, W, H)

    for i in range(k):
        nm, hx = names_hex[i]
        rgb = _hex_to_rgb(hx)
        kind = _choose_big_kind(rng)

        for _ in range(PLACE_BIG_TRIES):
            if kind == "rectangle":
                w = int(rng.uniform(*sr["rect_w"]))
                h = int(rng.uniform(*sr["rect_h"]))
                x = rng.randint(OUTER_PAD, max(OUTER_PAD, W - w - OUTER_PAD))
                y = rng.randint(OUTER_PAD, max(OUTER_PAD, H - h - OUTER_PAD))
                candidate = BigRect(nm, rgb, x, y, w, h)
            elif kind == "circle":
                R = float(rng.uniform(*sr["circle_R"]))
                cx = rng.uniform(OUTER_PAD + R, W - OUTER_PAD - R)
                cy = rng.uniform(OUTER_PAD + R, H - OUTER_PAD - R)
                candidate = BigCircle(nm, rgb, cx, cy, R)
            else:  # triangle
                side = float(rng.uniform(*sr["tri_side"]))
                rot = rng.uniform(0.0, 360.0)
                cx = rng.uniform(OUTER_PAD + 0.5 * side, W - OUTER_PAD - 0.5 * side)
                cy = rng.uniform(OUTER_PAD + 0.5 * side, H - OUTER_PAD - 0.5 * side)
                candidate = BigTriangle(nm, rgb, cx, cy, side, rot)

            bb = candidate.bbox()
            if not _bbox_inside_canvas(bb, W, H, OUTER_PAD):
                continue
            if not _non_overlapping_with_all(bb, placed, SEP_BIG):
                continue
            placed.append(candidate)
            break
        else:
            # restart all if one can't be placed
            return _spawn_big_shapes(rng, W, H, k)

    return placed

# ---------- small-shape placement (no overlaps, no boundary crossings) ----------

def _fits_wrt_bigshapes(cx: float, cy: float, r: float, regions: List[BigShape], clearance: float = SEP_REGION_VISUAL) -> bool:
    """
    A circle of radius r centered at (cx,cy) either lies completely inside a region
    with clearance 'clearance' or completely outside all regions with the same clearance.
    No small shape may come within 'clearance' of ANY big-shape boundary.
    """
    for rg in regions:
        inside_m = rg.inside_margin(cx, cy)
        if inside_m >= 0:
            if inside_m <= r + clearance:
                return False  # would touch/cross boundary from the inside
        else:
            if rg.dist_to_region(cx, cy) < r + clearance:
                return False  # would touch/cross boundary from the outside
    return True


def _fits_wrt_shapes(cx: float, cy: float, r: float, placed: List[SmallShape], sep: float = SEP_SHAPE_VISUAL) -> bool:
    for s in placed:
        if _euclid(cx, cy, s.cx, s.cy) < (r + s.r + sep):
            return False
    return True

def _fits_canvas(cx: float, cy: float, r: float, W: int, H: int) -> bool:
    return (OUTER_PAD + r <= cx <= W - OUTER_PAD - r) and (OUTER_PAD + r <= cy <= H - OUTER_PAD - r)

def _place_shapes_no_overlap(
    rng: random.Random,
    W: int, H: int,
    regions: List[BigShape],
    n_shapes: int,
    shapes_allowed: Tuple[str, ...],
) -> Optional[List[SmallShape]]:  # unchanged logic; rendering is separate
    """Rejection-sample centers on a jittered grid, enforcing:
       - no shape–shape overlap,
       - no crossing of any big shape boundary,
       - all shapes remain inside canvas with OUTER_PAD.
    Returns a list of placed shapes or None if packing failed.
    """
    xs = list(range(OUTER_PAD, W - OUTER_PAD, GRID_STEP))
    ys = list(range(OUTER_PAD, H - OUTER_PAD, GRID_STEP))
    cands = [(x + rng.uniform(-0.35*GRID_STEP, 0.35*GRID_STEP),
              y + rng.uniform(-0.35*GRID_STEP, 0.35*GRID_STEP))
             for x in xs for y in ys]
    rng.shuffle(cands)

    shapes: List[SmallShape] = []

    for _ in range(n_shapes):
        r0 = rng.uniform(R_MIN, R_MAX)
        kind = rng.choice(shapes_allowed)
        rot = rng.uniform(0, 360)

        placed = False
        tries = 0
        while tries < PACK_TRIES_PER_SHAPE:
            tries += 1
            if cands:
                cx, cy = cands.pop()
            else:
                cx = rng.uniform(OUTER_PAD + r0, W - OUTER_PAD - r0)
                cy = rng.uniform(OUTER_PAD + r0, H - OUTER_PAD - r0)

            r = r0
            if tries > PACK_TRIES_PER_SHAPE // 2:  # tiny shrink late in search
                r = max(R_MIN * 0.9, 0.92 * r0)

            if not _fits_canvas(cx, cy, r, W, H):          continue
            if not _fits_wrt_bigshapes(cx, cy, r, regions): continue
            if not _fits_wrt_shapes(cx, cy, r, shapes):    continue

            shapes.append(SmallShape(kind, cx, cy, r, rot))
            placed = True
            break

        if not placed:
            return None

    return shapes

# ----------------------- PROMPTS (strict, clarified) -----------------------

PROMPTS = {
    "inside": [
        "How many {shape}s are strictly inside the {color} {ref}?",
        "What is the number of {shape}s strictly inside the {color} {ref}?",
        "How many {shape}s lie strictly inside the {color} {ref}?",
        "What is the count of {shape}s strictly inside the {color} {ref}?",
        "How many {shape}s can be found strictly inside the {color} {ref}?",
        "What total of {shape}s are strictly contained in the {color} {ref}?",
        "How many {shape}s appear strictly inside the {color} {ref}?",
        "What is the total number of {shape}s strictly inside the {color} {ref}?",
        "How many {shape}s are strictly enclosed by the {color} {ref}?",
        "What is the exact count of {shape}s strictly inside the {color} {ref}?",
    ],
    "outside": [
        "How many {shape}s are strictly outside the {color} {ref} (no touching)?",
        "What is the number of {shape}s strictly outside the {color} {ref}?",
        "How many {shape}s lie strictly outside the {color} {ref}?",
        "What is the count of {shape}s strictly outside the {color} {ref}?",
        "How many {shape}s can be found strictly outside the {color} {ref}?",
        "What total of {shape}s are strictly beyond the {color} {ref}?",
        "How many {shape}s appear strictly outside the {color} {ref}?",
        "What is the total number of {shape}s strictly outside the {color} {ref}?",
        "How many {shape}s are strictly located outside the {color} {ref}?",
        "What is the exact count of {shape}s strictly outside the {color} {ref}?",
    ],
    "left": [
        "How many {shape}s are strictly to the left of the {color} {ref} (i.e., left of its leftmost point)?",
        "What is the number of {shape}s strictly to the left of the {color} {ref} (left of its leftmost point)?",
        "How many {shape}s lie strictly to the left of the {color} {ref} (left of its leftmost point)?",
        "What is the count of {shape}s strictly to the left of the {color} {ref} (left of its leftmost point)?",
        "How many {shape}s can be found strictly to the left of the {color} {ref} (left of its leftmost point)?",
        "What total of {shape}s are strictly positioned to the left of the {color} {ref} (left of its leftmost point)?",
        "How many {shape}s appear strictly to the left of the {color} {ref} (left of its leftmost point)?",
        "What is the total number of {shape}s strictly to the left of the {color} {ref} (left of its leftmost point)?",
        "How many {shape}s are strictly located to the left of the {color} {ref} (left of its leftmost point)?",
        "What is the exact count of {shape}s strictly to the left of the {color} {ref} (left of its leftmost point)?",
    ],
    "right": [
        "How many {shape}s are strictly to the right of the {color} {ref} (i.e., right of its rightmost point)?",
        "What is the number of {shape}s strictly to the right of the {color} {ref} (right of its rightmost point)?",
        "How many {shape}s lie strictly to the right of the {color} {ref} (right of its rightmost point)?",
        "What is the count of {shape}s strictly to the right of the {color} {ref} (right of its rightmost point)?",
        "How many {shape}s can be found strictly to the right of the {color} {ref} (right of its rightmost point)?",
        "What total of {shape}s are strictly positioned to the right of the {color} {ref} (right of its rightmost point)?",
        "How many {shape}s appear strictly to the right of the {color} {ref} (right of its rightmost point)?",
        "What is the total number of {shape}s strictly to the right of the {color} {ref} (right of its rightmost point)?",
        "How many {shape}s are strictly located to the right of the {color} {ref} (right of its rightmost point)?",
        "What is the exact count of {shape}s strictly to the right of the {color} {ref} (right of its rightmost point)?",
    ],
    "above": [
        "How many {shape}s are strictly above the {color} {ref} (i.e., above its topmost point)?",
        "What is the number of {shape}s strictly above the {color} {ref} (above its topmost point)?",
        "How many {shape}s lie strictly above the {color} {ref} (above its topmost point)?",
        "What is the count of {shape}s strictly above the {color} {ref} (above its topmost point)?",
        "How many {shape}s can be found strictly above the {color} {ref} (above its topmost point)?",
        "What total of {shape}s are strictly positioned above the {color} {ref} (above its topmost point)?",
        "How many {shape}s appear strictly above the {color} {ref} (above its topmost point)?",
        "What is the total number of {shape}s strictly above the {color} {ref} (above its topmost point)?",
        "How many {shape}s are strictly located above the {color} {ref} (above its topmost point)?",
        "What is the exact count of {shape}s strictly above the {color} {ref} (above its topmost point)?",
    ],
    "below": [
        "How many {shape}s are strictly below the {color} {ref} (i.e., below its bottommost point)?",
        "What is the number of {shape}s strictly below the {color} {ref} (below its bottommost point)?",
        "How many {shape}s lie strictly below the {color} {ref} (below its bottommost point)?",
        "What is the count of {shape}s strictly below the {color} {ref} (below its bottommost point)?",
        "How many {shape}s can be found strictly below the {color} {ref} (below its bottommost point)?",
        "What total of {shape}s are strictly positioned below the {color} {ref} (below its bottommost point)?",
        "How many {shape}s appear strictly below the {color} {ref} (below its bottommost point)?",
        "What is the total number of {shape}s strictly below the {color} {ref} (below its bottommost point)?",
        "How many {shape}s are strictly located below the {color} {ref} (below its bottommost point)?",
        "What is the exact count of {shape}s strictly below the {color} {ref} (below its bottommost point)?",
    ],
}

# ----------------------- Spec & Task -----------------------

@dataclass
class GeoPosSpec:
    seed: int
    canvas: Tuple[int, int]
    rects: List[Tuple[str, Tuple[int, int, int, int]]]   # bbox snapshots (for all big shapes)
    n_small: int
    shapes_allowed: Tuple[str, ...]
    relations_allowed: Tuple[str, ...]
    answer_bounds: Tuple[int, int]

@register_task
class GeometricPositionTask(Task):
    """
    Spatial-position counting over colored big shapes (rectangles, circles, triangles).

    Guarantees:
      • Big shapes do not overlap (AABBs separated by ≥ SEP_BIG).
      • No two small shapes overlap (pairwise circle separation ≥ r_i + r_j + SEP_SHAPE).
      • No small shape touches/crosses any big shape boundary (clearance ≥ SEP_REGION);
        every small shape is wholly inside some big shape or wholly outside all big shapes.

    Rendering:
      • Big and small shapes are drawn on supersampled RGBA patches (AA_SCALE),
        then downsampled (LANCZOS) and alpha-pasted for crisp edges, mirroring the
        approach used in geometric_sort.

    Relations are **strict** and radius-aware for counting.
    """
    name = "geometric_position"

    def __init__(
        self,
        canvas: Tuple[int, int] = DEFAULT_CANVAS,
        big_rect_count_min: int = 1,      # retained for compatibility; acts as a clamp
        big_rect_count_max: int = 4,      # default now 4 so PMF can sample up to 4
        small_shapes_min: int = SMALL_SHAPES_MIN_DEFAULT,
        small_shapes_max: int = SMALL_SHAPES_MAX_DEFAULT,
        answer_min: int = ANSWER_MIN_DEFAULT,
        answer_max: int = ANSWER_MAX_DEFAULT,
        shapes_allowed: Tuple[str, ...] = SHAPE_KINDS_DEFAULT,
        relations_allowed: Tuple[str, ...] = RELATIONS_DEFAULT,
    ):
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.W, self.H = int(canvas[0]), int(canvas[1])

        self.big_rect_count_min = int(max(1, big_rect_count_min))
        self.big_rect_count_max = int(max(self.big_rect_count_min, big_rect_count_max))

        self.small_shapes_min = int(max(1, small_shapes_min))
        self.small_shapes_max = int(max(self.small_shapes_min, small_shapes_max))

        self.answer_min = int(max(1, answer_min))
        self.answer_max = int(max(self.answer_min, answer_max))

        self.shapes_allowed = tuple(shapes_allowed)
        self.relations_allowed = tuple(relations_allowed)

    def _compute_complexity(self, big_shape_count: int, answer: Optional[int] = None) -> Dict[str, Any]:
        """Normalize big-shape counts to [0,1] and map to categorical levels."""
        span = max(1, BIG_SHAPE_COUNT_MAX - BIG_SHAPE_COUNT_MIN)
        normalized = (int(big_shape_count) - BIG_SHAPE_COUNT_MIN) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        result: Dict[str, Any] = {
            "score": normalized,
            "level": level,
            "version": "geometric-position-bigshape-v1",
            "range": {"min_big_shapes": BIG_SHAPE_COUNT_MIN, "max_big_shapes": BIG_SHAPE_COUNT_MAX},
            "big_shape_count": int(big_shape_count),
        }
        if answer is not None:
            result["answer"] = int(answer)
        return result

    # --------------- public API ----------------

    def generate_instance(self, motif_impls: Dict, rng: random.Random):
        for _ in range(self.max_retries):
            seed = rng.randint(0, 2**31 - 1)
            lrng = random.Random(seed)

            # Canvas & background
            bg = graph_paper_rgb(self.W, self.H).convert("RGBA")

            # Number of big shapes from PMF (clamped by min/max)
            k_sample = _sample_from_pmf(lrng, BIG_SHAPE_COUNT_PMF)
            K = max(self.big_rect_count_min, min(self.big_rect_count_max, int(k_sample)))

            # Place big shapes (rect/circle/triangle), all non-overlapping (AABB)
            bigs: List[BigShape] = _spawn_big_shapes(lrng, self.W, self.H, K)

            # --- Render big shapes with AA patches (like geometric_sort composition) ---
            for b in bigs:
                if isinstance(b, BigRect):
                    x0, y0, x1, y1 = b.bbox()
                    patch, (px, py) = _render_rect_patch(x0, y0, x1, y1, b.rgb, BIG_OUTLINE_RGB, BIG_OUTLINE_W)
                elif isinstance(b, BigCircle):
                    patch, (px, py) = _render_ellipse_patch(b.cx, b.cy, b.R, b.rgb, BIG_OUTLINE_RGB, BIG_OUTLINE_W)
                else:  # triangle
                    patch, (px, py) = _render_polygon_patch(b.verts, b.rgb, BIG_OUTLINE_RGB, BIG_OUTLINE_W)

                if paste_rgba is not None:
                    paste_rgba(bg, patch, (int(px), int(py)))
                else:
                    bg.paste(patch, (int(px), int(py)), patch)

            # Place small shapes with *no overlaps* and *no boundary crossings*
            n_small = lrng.randint(self.small_shapes_min, self.small_shapes_max)

            shapes = None
            for _pack_try in range(GLOBAL_RETRIES):
                shapes = _place_shapes_no_overlap(
                    lrng, self.W, self.H, bigs, n_small, self.shapes_allowed
                )
                if shapes is not None:
                    break
            if shapes is None:
                # Rebuild the whole scene (new big shapes/seed)
                continue

            # --- Render the small shapes on AA patches (black fill/outline) ---
            for s in shapes:
                _draw_small_shape_rgba(bg, s)

            # Choose which big shape, relation, and small-shape kind to ask about
            target_big = lrng.choice(bigs)
            present_kinds = {s.kind for s in shapes}
            allowed_kinds = [k for k in self.shapes_allowed if k in present_kinds] or list(self.shapes_allowed)
            shape_kind = lrng.choice(allowed_kinds)

            # Try relations until we hit answer bounds (strict, radius-aware)
            relation = None
            answer = None
            for _try_rel in range(28):
                rel = lrng.choice(self.relations_allowed)
                cnt = 0
                for s in shapes:
                    if s.kind != shape_kind:
                        continue
                    rels = target_big.relation_of_shape(s, SEP_REGION_VISUAL)
                    if rels.get(rel, False):
                        cnt += 1
                if self.answer_min <= cnt <= self.answer_max:
                    relation, answer = rel, cnt
                    break

            if relation is None:
                # Try again (full rebuild)
                continue

            color_name = target_big.name
            ref_kind   = target_big.kind  # "rectangle" | "circle" | "triangle"
            prompt = lrng.choice(PROMPTS[relation]).format(shape=shape_kind, color=color_name, ref=ref_kind)

            spec = GeoPosSpec(
                seed=seed,
                canvas=(self.W, self.H),
                rects=[(b.name, b.bbox()) for b in bigs],  # bbox snapshots for all bigs
                n_small=n_small,
                shapes_allowed=self.shapes_allowed,
                relations_allowed=self.relations_allowed,
                answer_bounds=(self.answer_min, self.answer_max),
            )

            complexity = self._compute_complexity(len(bigs), int(answer))

            meta = {
                "composite_ready": True,
                "pattern_kind": "geometry",
                "pattern": self.name,
                "variant": {
                    "n_big_shapes": len(bigs),
                    "big_shape_kinds": [b.kind for b in bigs],
                    "shapes_minmax": [self.small_shapes_min, self.small_shapes_max],
                    "answer_minmax": [self.answer_min, self.answer_max],
                    "relation": relation,
                    "small_shape_kind": shape_kind,
                    "target_color": color_name,
                    "target_kind": ref_kind,
                    "no_overlap": True,
                    "strict_classification": True,
                    "sep_small_small": SEP_SHAPE,
                    "sep_small_big": SEP_REGION,
                    "sep_big_big": SEP_BIG,
                    "aa_scale": AA_SCALE,
                    "big_outline_w": BIG_OUTLINE_W,
                    "small_outline_w": SMALL_OUTLINE_W,
                },
                "question": prompt,
                "answer": int(answer),
                "dims": (self.W, self.H),
                "big_shapes": [
                    (
                        {"kind": "rectangle", "color": b.name, "bbox": b.bbox()}
                        if isinstance(b, BigRect) else
                        {"kind": "circle", "color": b.name, "cx": float(b.cx), "cy": float(b.cy), "R": float(b.R)}
                        if isinstance(b, BigCircle) else
                        {"kind": "triangle", "color": b.name, "verts": [(float(x), float(y)) for (x, y) in b.verts]}
                    ) for b in bigs
                ],
                "rectangles": [  # backward-compat convenience (rectangles only)
                    {"color": b.name, "bbox": b.bbox()}
                    for b in bigs if isinstance(b, BigRect)
                ],
                "shapes": [
                    {"kind": s.kind, "center": [float(s.cx), float(s.cy)], "r": float(s.r), "rot_deg": float(s.rot_deg)}
                    for s in shapes
                ],
                "sep_small_small": SEP_SHAPE_VISUAL,
                "sep_small_big": SEP_REGION_VISUAL,
            }

            meta["complexity"] = complexity
            meta["complexity_score"] = complexity["score"]
            meta["complexity_level"] = complexity["level"]
            meta["complexity_version"] = complexity["version"]

            return bg.convert("RGB"), [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
