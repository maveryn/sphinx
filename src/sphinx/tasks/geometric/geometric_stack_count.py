# sphinx/tasks/geometric/geometric_stack_count.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES
from sphinx.geometry import graph_paper_rgb  # background

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

# Number of stacked big sheets PMF (k, probability).
# We'll still sample from this PMF, then clamp/fallback into [STACK_COUNT_MIN, STACK_COUNT_MAX].
STACK_COUNT_PMF: List[Tuple[int, float]] = [
    (1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25), (5, 0.25)
]

# Bounds for how many large sheets per sample
STACK_COUNT_MIN = 2
STACK_COUNT_MAX = 4

# ---- Overlap constraints for the stacked sheets ----
# Pairwise overlap ratio = intersection_area / min(area_i, area_j)
# Enforce that for ANY OVERLAPPING PAIR:  MIN <= ratio <= MAX.
# Non-overlapping pairs are allowed.
MIN_PAIR_OVERLAP_RATIO = 0.05   # minimum allowed ratio (overlapping pairs only)
MAX_PAIR_OVERLAP_RATIO = 0.75   # reduce heavy overlaps



# Require EVERY sheet to have at least one neighbor pair in the band above.
REQUIRE_NEIGHBOR_FOR_EACH_SHEET = True

DEFAULT_CANVAS = (768, 512)  # WxH

ANSWER_MIN_DEFAULT = 1
ANSWER_MAX_DEFAULT = 20

# Small objects
SMALL_SHAPES_MIN_DEFAULT = 10
SMALL_SHAPES_MAX_DEFAULT = 40
R_MIN, R_MAX      = 8, 15       # small-shape "radius" in pixels
GRID_STEP         = 20          # candidate lattice spacing
OUTER_PAD         = 5           # padding to canvas edges
SEP_SHAPE         = 1           # clearance between small shapes

# Strict relation margin (small object vs. target sheet boundary)
SEP_REGION        = 10

# VISUAL strictness for "strictly inside" and for placement (accounts for outlines & AA)
SMALL_SHAPE_OUTLINE_PX = 5
SHEET_OUTLINE_WIDTH     = 3
AA_FUDGE                = 1.0  # extra px for antialiasing/downsampling cushion

# Margin between any small shape and any sheet boundary (used in placement AND counting)
SEP_REGION_VISUAL = SEP_REGION + 0.5 * SHEET_OUTLINE_WIDTH + 0.5 * SMALL_SHAPE_OUTLINE_PX + AA_FUDGE

# Small–small visible clearance (prevents “kissing” outlines after AA)
SEP_SHAPE_VISUAL  = SEP_SHAPE  +      SMALL_SHAPE_OUTLINE_PX                              + AA_FUDGE

# Counting must match what we enforce during placement
COUNT_STRICT_MARGIN_PX = SEP_REGION_VISUAL


# Allow tangency when enforcing "no intersection with sheets"
TANGENCY_TOL = 0.5

# High-quality rendering for small shapes (supersampling)
SMALL_SHAPE_SS          = 3      # 3x supersampling
SMALL_SHAPE_OUTLINE_PX  = 2      # base-space visual outline thickness
RENDER_R_SCALE          = 0.96   # draw shapes slightly inside the bounding circle

# Tries
PACK_TRIES_PER_SHAPE = 500
GLOBAL_RETRIES        = 10

# Draw style
BIG_OUTLINE_RGB         = (0, 0, 0)
SMALL_SHAPE_FILL_RGB    = (0, 0, 0)
SMALL_SHAPE_OUTLINE_RGB = (0, 0, 0)

# Big sheet kinds allowed for stacking in a sample
STACK_KINDS_DEFAULT = ("rectangle", "circle", "triangle")  # equilateral for triangle

# Relations for this task (questions use only "inside")
RELATIONS_DEFAULT = ("inside",)

# ------------------------------------------------------------
# Utilities
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

def _sample_from_pmf(rng: random.Random, items: List[Tuple[int, float]]) -> int:
    total = sum(max(0.0, p) for _, p in items)
    if total <= 0:
        keys = [k for k, _ in items]
        return rng.choice(keys)
    x = rng.uniform(0.0, total)
    acc = 0.0
    for k, p in items:
        acc += max(0.0, p)
        if x <= acc:
            return k
    return items[-1][0]

# ------------------------------------------------------------
# Big sheet classes (with strict relation geometry)
# ------------------------------------------------------------

class BigSheet:
    kind: str
    name: str       # color name
    rgb: Tuple[int, int, int]

    def bbox(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def draw(self, dr: ImageDraw.ImageDraw):
        raise NotImplementedError

    # extreme points
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
        """Strict, visibility-agnostic relations for a circular footprint."""
        cx, cy, r = s.cx, s.cy, s.r
        inside  = self.contains_pt(cx, cy) and (self.inside_margin(cx, cy) > r + sep_region)
        outside = (not self.contains_pt(cx, cy)) and (self.dist_to_region(cx, cy) >= r + sep_region)
        return {"inside": inside, "outside": outside}

@dataclass
class SheetRect(BigSheet):
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
        dr.rectangle(self.bbox(), fill=self.rgb, outline=BIG_OUTLINE_RGB, width=SHEET_OUTLINE_WIDTH)

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
class SheetCircle(BigSheet):
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
        dr.ellipse(self.bbox(), fill=self.rgb, outline=BIG_OUTLINE_RGB, width=SHEET_OUTLINE_WIDTH)

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
class SheetTriangle(BigSheet):
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
        poly = self.verts
        dr.polygon(poly, fill=self.rgb, outline=BIG_OUTLINE_RGB)
        dr.line(poly + [poly[0]], fill=BIG_OUTLINE_RGB, width=SHEET_OUTLINE_WIDTH)

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
# Small objects (black) — high-quality rendering
# ------------------------------------------------------------

@dataclass
class SmallShape:
    kind: str                  # circle|triangle|square
    cx: float
    cy: float
    r: float                   # bounding-circle radius for placement/geometry
    rot_deg: float

def _regular_ngon(cx: float, cy: float, r: float, n: int, rot_deg: float) -> List[Tuple[float, float]]:
    rot = math.radians(rot_deg)
    return [(cx + r * math.cos(rot + 2*math.pi*i/n),
             cy + r * math.sin(rot + 2*math.pi*i/n)) for i in range(n)]

def _oriented_rect_points(cx: float, cy: float, half_w: float, half_h: float, rot_deg: float) -> List[Tuple[float, float]]:
    """Axis-aligned rectangle (±half_w, ±half_h) rotated about center by rot_deg."""
    ca, sa = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    pts_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    return [(cx + x*ca - y*sa, cy + x*sa + y*ca) for (x, y) in pts_local]

def _draw_small_shapes_supersampled(base_img: Image.Image, shapes: List[SmallShape]) -> Image.Image:
    """
    Draw small shapes on an RGBA supersampled overlay, then downsample and alpha-composite.
    This dramatically improves edge quality of triangles/squares.
    """
    W, H = base_img.size
    SS = int(max(2, SMALL_SHAPE_SS))
    hi = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(hi)

    # Scaled outline width (aim for ~2 px in base space)
    ow = max(1, int(round(SMALL_SHAPE_OUTLINE_PX * SS)))

    for s in shapes:
        # Slightly shrink the visual shape so "strictly inside" also looks clearly inside.
        Rv = float(s.r) * float(RENDER_R_SCALE)

        if s.kind == "circle":
            box = [
                int(round((s.cx - Rv) * SS)), int(round((s.cy - Rv) * SS)),
                int(round((s.cx + Rv) * SS)), int(round((s.cy + Rv) * SS))
            ]
            dr.ellipse(box, fill=SMALL_SHAPE_FILL_RGB + (255,), outline=SMALL_SHAPE_OUTLINE_RGB + (255,), width=ow)

        elif s.kind == "triangle":
            pts = _regular_ngon(s.cx, s.cy, Rv, 3, s.rot_deg)
            pts_s = [(int(round(x * SS)), int(round(y * SS))) for (x, y) in pts]
            dr.polygon(pts_s, fill=SMALL_SHAPE_FILL_RGB + (255,), outline=None)
            dr.line(pts_s + [pts_s[0]], fill=SMALL_SHAPE_OUTLINE_RGB + (255,), width=ow, joint="curve")

        else:  # square as true oriented rectangle with equal sides
            side = Rv * math.sqrt(2.0)
            half = side * 0.5
            pts = _oriented_rect_points(s.cx, s.cy, half, half, s.rot_deg)
            pts_s = [(int(round(x * SS)), int(round(y * SS))) for (x, y) in pts]
            dr.polygon(pts_s, fill=SMALL_SHAPE_FILL_RGB + (255,), outline=None)
            dr.line(pts_s + [pts_s[0]], fill=SMALL_SHAPE_OUTLINE_RGB + (255,), width=ow, joint="curve")

    # Downsample and composite
    overlay = hi.resize((W, H), resample=Image.LANCZOS)
    if base_img.mode != "RGBA":
        base_rgba = base_img.convert("RGBA")
    else:
        base_rgba = base_img
    out = Image.alpha_composite(base_rgba, overlay)
    return out.convert("RGB")

# ------------------------------------------------------------
# Geometry: areas and intersections for overlap constraints (same kind only)
# ------------------------------------------------------------

def _rect_area(sr: SheetRect) -> float:
    return float(sr.w) * float(sr.h)

def _circle_area(sc: SheetCircle) -> float:
    return math.pi * float(sc.R) * float(sc.R)

def _tri_area(st: SheetTriangle) -> float:
    # Equilateral triangle area = (sqrt(3)/4) * side^2
    return (math.sqrt(3.0) / 4.0) * (float(st.side) ** 2)

def _rect_rect_intersection_area(a: SheetRect, b: SheetRect) -> float:
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    dx = max(0, min(ax1, bx1) - max(ax0, bx0))
    dy = max(0, min(ay1, by1) - max(ay0, by0))
    return float(dx * dy)

def _circle_circle_intersection_area(a: SheetCircle, b: SheetCircle) -> float:
    r0 = float(a.R); r1 = float(b.R)
    d = _euclid(a.cx, a.cy, b.cx, b.cy)
    if d >= r0 + r1:
        return 0.0
    if d <= abs(r0 - r1):
        # one inside the other
        return math.pi * (min(r0, r1) ** 2)
    r0_2, r1_2, d2 = r0 * r0, r1 * r1, d * d
    alpha = math.acos(max(-1.0, min(1.0, (d2 + r0_2 - r1_2) / (2.0 * d * r0))))
    beta  = math.acos(max(-1.0, min(1.0, (d2 + r1_2 - r0_2) / (2.0 * d * r1))))
    area = r0_2 * alpha + r1_2 * beta - 0.5 * math.sqrt(
        max(0.0, (-d + r0 + r1) * (d + r0 - r1) * (d - r0 + r1) * (d + r0 + r1))
    )
    return float(area)

def _signed_area(poly: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s

def _ensure_ccw(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return poly if _signed_area(poly) >= 0 else list(reversed(poly))

def _line_intersection(p1, p2, p3, p4) -> Tuple[float, float]:
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return ((x2 + x3) * 0.5, (y2 + y3) * 0.5)
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (px, py)

def _suth_hodg_clip(subject: List[Tuple[float, float]], clip: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    def inside(p, a, b) -> bool:
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0

    out = subject[:]
    clip_ccw = _ensure_ccw(clip)
    for i in range(len(clip_ccw)):
        input_list = out
        out = []
        A = clip_ccw[i]
        B = clip_ccw[(i + 1) % len(clip_ccw)]
        if not input_list:
            break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    out.append(_line_intersection(S, E, A, B))
                out.append(E)
            elif inside(S, A, B):
                out.append(_line_intersection(S, E, A, B))
            S = E
    return out

def _tri_tri_intersection_area(a: SheetTriangle, b: SheetTriangle) -> float:
    poly_a = _ensure_ccw(a.verts)
    poly_b = _ensure_ccw(b.verts)
    inter_poly = _suth_hodg_clip(poly_a, poly_b)
    if len(inter_poly) < 3:
        return 0.0
    return abs(_signed_area(inter_poly))

def _sheet_area(s: BigSheet) -> float:
    if isinstance(s, SheetRect):    return _rect_area(s)
    if isinstance(s, SheetCircle):  return _circle_area(s)
    if isinstance(s, SheetTriangle):return _tri_area(s)
    return 0.0

def _sheet_intersection_area(a: BigSheet, b: BigSheet) -> float:
    if isinstance(a, SheetRect) and isinstance(b, SheetRect):
        return _rect_rect_intersection_area(a, b)
    if isinstance(a, SheetCircle) and isinstance(b, SheetCircle):
        return _circle_circle_intersection_area(a, b)
    if isinstance(a, SheetTriangle) and isinstance(b, SheetTriangle):
        return _tri_tri_intersection_area(a, b)
    return 0.0

def _pair_overlap_ratio(a: BigSheet, b: BigSheet) -> float:
    inter = _sheet_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    denom = max(1e-9, min(_sheet_area(a), _sheet_area(b)))
    return float(inter / denom)

def _overlap_constraints_ok(sheets: List[BigSheet]) -> bool:
    """
    Valid iff:
      • For every overlapping pair (ratio > 0), MIN_PAIR_OVERLAP_RATIO ≤ ratio ≤ MAX_PAIR_OVERLAP_RATIO.
      • Every sheet has at least one neighbor with ratio in [MIN, MAX] (if REQUIRE_NEIGHBOR_FOR_EACH_SHEET).
      • Non-overlapping pairs are allowed.
    """
    n = len(sheets)
    if n < 2:
        return False
    neighbor_ok = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            r = _pair_overlap_ratio(sheets[i], sheets[j])
            if r > 0.0:
                if r < MIN_PAIR_OVERLAP_RATIO or r > MAX_PAIR_OVERLAP_RATIO:
                    return False
                neighbor_ok[i] = True
                neighbor_ok[j] = True
    return all(neighbor_ok) if REQUIRE_NEIGHBOR_FOR_EACH_SHEET else any(neighbor_ok)

# ------------------------------------------------------------
# Stack sampling & placement  (EQUAL AREA for all big sheets)
# ------------------------------------------------------------

def _size_ranges_for_k(kind: str, k: int, W: int, H: int) -> Dict[str, Tuple[float, float]]:
    """
    Larger ranges to cover more area; tuned by k.
    """
    mw = float(W); mh = float(H); m = float(min(W, H))
    if kind == "rectangle":
        if k <= 2: return {"w": (0.58*mw, 0.74*mw), "h": (0.54*mh, 0.68*mh)}
        if k == 3: return {"w": (0.50*mw, 0.64*mw), "h": (0.48*mh, 0.62*mh)}
        if k == 4: return {"w": (0.46*mw, 0.58*mw), "h": (0.46*mh, 0.58*mh)}
        return          {"w": (0.42*mw, 0.54*mw), "h": (0.44*mh, 0.56*mh)}  # k>=5
    if kind == "circle":
        if k <= 2: return {"R": (0.25*m, 0.35*m)}
        if k == 3: return {"R": (0.22*m, 0.31*m)}
        if k == 4: return {"R": (0.20*m, 0.29*m)}
        return          {"R": (0.18*m, 0.27*m)}
    # triangle (equilateral)
    if k <= 2: return {"side": (0.54*m, 0.76*m)}
    if k == 3: return {"side": (0.50*m, 0.70*m)}
    if k == 4: return {"side": (0.46*m, 0.66*m)}
    return          {"side": (0.44*m, 0.62*m)}

def _spawn_stacked_sheets_once(
    rng: random.Random,
    W: int, H: int,
    k: int,
    kind: str,
) -> List[BigSheet]:
    """
    Equal-area placement with neighbor-overlap requirement:
      • Sample ONE common size parameter (area) for the whole stack.
          - rectangle:   same (w,h) for every sheet  -> same area w*h
          - circle:      same radius R               -> same area πR²
          - triangle:    same side length            -> same area (√3/4) s²
      • First sheet near center; each subsequent sheet must:
          - overlap at least one placed sheet with ratio in [MIN, MAX], and
          - for every overlapping pair it forms, ratio also in [MIN, MAX].
      • Non-overlap with other previously placed sheets is allowed.
    """
    assert k >= 1 and kind in ("rectangle", "circle", "triangle")
    names_hex = rng.sample(COLOR_POOL, min(k, len(COLOR_POOL)))
    sizes = _size_ranges_for_k(kind, k, W, H)
    sheets: List[BigSheet] = []

    # Common anchor near the canvas center; offset range tuned per kind
    cx0 = rng.uniform(0.40*W, 0.60*W)
    cy0 = rng.uniform(0.40*H, 0.60*H)
    if kind == "rectangle":
        offW, offH = 0.08*W, 0.08*H
    elif kind == "circle":
        offW, offH = 0.08*W, 0.08*H
    else:  # triangle
        offW, offH = 0.07*W, 0.07*H

    # --------- choose common size (equal area) ---------
    if kind == "rectangle":
        w_common = int(rng.uniform(*sizes["w"]))
        h_common = int(rng.uniform(*sizes["h"]))
    elif kind == "circle":
        R_common = float(rng.uniform(*sizes["R"]))
    else:
        side_common = float(rng.uniform(*sizes["side"]))

    for i in range(k):
        nm, hx = names_hex[i % len(names_hex)]
        rgb = _hex_to_rgb(hx)

        success = False
        for _ in range(360):
            # Sample pose with the shared size
            if kind == "rectangle":
                w, h = w_common, h_common
                ox = rng.uniform(-offW, offW); oy = rng.uniform(-offH, offH)
                x = int(cx0 + ox - w/2); y = int(cy0 + oy - h/2)
                cand = SheetRect(nm, rgb, x, y, w, h)
            elif kind == "circle":
                R = R_common
                ox = rng.uniform(-offW, offW); oy = rng.uniform(-offH, offH)
                cx = cx0 + ox; cy = cy0 + oy
                cand = SheetCircle(nm, rgb, cx, cy, R)
            else:
                side = side_common
                rot = rng.uniform(0.0, 360.0)
                ox = rng.uniform(-offW, offW); oy = rng.uniform(-offH, offH)
                cx = cx0 + ox; cy = cy0 + oy
                cand = SheetTriangle(nm, rgb, cx, cy, side, rot)

            if not _bbox_inside_canvas(cand.bbox(), W, H, OUTER_PAD):
                continue

            if i == 0:
                sheets.append(cand)
                success = True
                break

            # Subsequent: must have at least one neighbor in-band among previous.
            has_band_neighbor = False
            bad_overlap = False
            for prev in sheets:
                r = _pair_overlap_ratio(cand, prev)
                if r > 0.0:
                    if r < MIN_PAIR_OVERLAP_RATIO or r > MAX_PAIR_OVERLAP_RATIO:
                        bad_overlap = True
                        break
                    has_band_neighbor = True
            if bad_overlap or not has_band_neighbor:
                continue

            sheets.append(cand)
            success = True
            break

        if not success:
            return []  # fail this attempt

    return sheets

def _spawn_stacked_sheets(
    rng: random.Random,
    W: int, H: int,
    k: int,
    kind: str,
) -> List[BigSheet]:
    """
    Try multiple times to build a stack that satisfies the overlap band and
    the per-sheet neighbor rule (with equal area across sheets).
    """
    for _attempt in range(160):
        sheets = _spawn_stacked_sheets_once(rng, W, H, k, kind)
        if not sheets or len(sheets) != k:
            continue
        if _overlap_constraints_ok(sheets):
            return sheets
    return []

# ------------------------------------------------------------
# Small-object placement (no overlap among small shapes AND no boundary crossing with sheets)
# ------------------------------------------------------------

def _fits_canvas(cx: float, cy: float, r: float, W: int, H: int) -> bool:
    return (OUTER_PAD + r <= cx <= W - OUTER_PAD - r) and (OUTER_PAD + r <= cy <= H - OUTER_PAD - r)

def _fits_wrt_small_shapes(cx: float, cy: float, r: float, placed: List[SmallShape]) -> bool:
    for s in placed:
        if _euclid(cx, cy, s.cx, s.cy) < (r + s.r + SEP_SHAPE_VISUAL):
            return False
    return True


def _fits_wrt_sheets_no_crossing(cx: float, cy: float, r: float, sheets: List[BigSheet]) -> bool:
    """
    Enforce a *visual margin* from EVERY sheet boundary. A small circle of radius r centered at (cx,cy)
    must be either wholly inside a sheet or wholly outside it, and in both cases it must remain at least
    SEP_REGION_VISUAL away from that sheet's boundary (so no edge-touch illusions).
    """
    for sh in sheets:
        inside_m = sh.inside_margin(cx, cy)
        if inside_m >= 0.0:
            # inside: need distance to boundary strictly greater than r + visual margin
            if inside_m <= r + SEP_REGION_VISUAL:
                return False
        else:
            # outside: need distance to region strictly greater than r + visual margin
            if sh.dist_to_region(cx, cy) < r + SEP_REGION_VISUAL:
                return False
    return True


def _place_small_shapes_anywhere(
    rng: random.Random,
    W: int, H: int,
    n_shapes: int,
    shapes_allowed: Tuple[str, ...],
    sheets: List[BigSheet],
) -> Optional[List[SmallShape]]:
    """Place small shapes anywhere on the canvas (drawn on top of sheets),
    with:
      • no small–small overlap,
      • no crossing of any sheet boundary (touching allowed),
      • canvas padding respected.
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
            if tries > PACK_TRIES_PER_SHAPE // 2:
                r = max(R_MIN * 0.9, 0.92 * r0)

            if not _fits_canvas(cx, cy, r, W, H):              continue
            if not _fits_wrt_small_shapes(cx, cy, r, shapes):  continue
            if not _fits_wrt_sheets_no_crossing(cx, cy, r, sheets): continue

            shapes.append(SmallShape(kind, cx, cy, r, rot))
            placed = True
            break

        if not placed:
            return None

    return shapes

# ----------------------- PROMPTS (strict; inside only) -----------------------
# {ref} ∈ {"rectangle","circle","triangle"}.
PROMPTS = {
    "inside": [
        "Colored sheets of equal area are stacked first (they may overlap and hide parts). Small black {shape}s are placed on top afterward. How many small black {shape}s lie strictly inside the {color} {ref}?",
        "First, several equal-area colored sheets are stacked, including the {color} {ref}. Then the small black {shape}s are drawn on top. How many small black {shape}s are strictly inside the {color} {ref}?",
        "The equal-area sheets are laid down first and may occlude each other; the black {shape}s are added afterwards on top. How many small black {shape}s are strictly inside the boundary of the {color} {ref}?",
        "Stack the same-area colored sheets first; after that, place the small black {shape}s on top. Using the true boundary of the {color} {ref} (ignore any occlusion), how many small black {shape}s are strictly inside?",
        "Several same-area colored sheets are stacked (overlaps allowed). Next, the small black {shape}s are placed on top. How many small black {shape}s fall strictly inside the {color} {ref}? Do not count border-touching.",
        "Equal-area colored sheets are placed first; then the black {shape}s are overlaid on top. Even if the {color} {ref} is partly hidden, how many small black {shape}s are strictly inside its boundary?",
        "A stack of equal-area colored sheets is created first; afterwards, the small black {shape}s are drawn on top. How many small black {shape}s lie strictly within the {color} {ref}? (Edge contact is not counted.)",
        "The {color} {ref} is one of several equal-area colored sheets stacked first. The small black {shape}s come later and sit on top. How many of those small black {shape}s are strictly inside the {color} {ref}?",
        "Equal-area colored sheets are stacked first, and the small black {shape}s are added on top. How many small black {shape}s are strictly inside the {color} {ref}?",
        "Colored sheets—all equal in area—are stacked first; small black {shape}s are added on top afterward. Count the small black {shape}s that are strictly inside the {color} {ref}; ignore visibility and don’t count border-touching.",
    ],
}



# ----------------------- Spec & Task -----------------------

@dataclass
class GeoStackSpec:
    seed: int
    canvas: Tuple[int, int]
    stack_kind: str
    n_big: int
    n_small: int
    shapes_allowed: Tuple[str, ...]
    relations_allowed: Tuple[str, ...]
    answer_bounds: Tuple[int, int]

@register_task
class GeometricStackCountTask(Task):
    """
    Count small objects strictly inside the contour of a chosen sheet
    when several same-kind sheets (rectangles/circles/equilateral triangles)
    are stacked. Small objects are drawn on top of sheets.

    Overlap policy (pairwise, based on intersection_area / min(area_i, area_j)):
      • For any overlapping pair, enforce MIN_PAIR_OVERLAP_RATIO ≤ ratio ≤ MAX_PAIR_OVERLAP_RATIO.
      • Non-overlapping pairs are allowed.
      • Every sheet must overlap with at least one other sheet in that ratio band.

    Small objects are limited to: circle, triangle, square.

    Additional guarantees for small objects:
      • They never cross the boundary of ANY sheet (they are either fully inside or fully outside each sheet).
      • Tangency to a sheet boundary is allowed, but such objects will NOT be counted as “strictly inside”.
      • The question is NEVER asked about the topmost sheet; we always target a non-top sheet.

    NEW: All large sheets in a scene have the SAME AREA (reported in the prompt).
    """
    name = "geometric_stack_count"

    def __init__(
        self,
        canvas: Tuple[int, int] = DEFAULT_CANVAS,
        stack_kinds_allowed: Tuple[str, ...] = STACK_KINDS_DEFAULT,
        small_shapes_min: int = SMALL_SHAPES_MIN_DEFAULT,
        small_shapes_max: int = SMALL_SHAPES_MAX_DEFAULT,
        answer_min: int = ANSWER_MIN_DEFAULT,
        answer_max: int = ANSWER_MAX_DEFAULT,
        relations_allowed: Tuple[str, ...] = RELATIONS_DEFAULT,  # ("inside",)
    ):
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.W, self.H = int(canvas[0]), int(canvas[1])

        self.stack_kinds_allowed = tuple(k for k in stack_kinds_allowed if k in ("rectangle","circle","triangle")) or STACK_KINDS_DEFAULT

        self.small_shapes_min = int(max(1, small_shapes_min))
        self.small_shapes_max = int(max(self.small_shapes_min, small_shapes_max))

        self.answer_min = int(max(1, answer_min))
        self.answer_max = int(max(self.answer_min, answer_max))

        # Only "inside" used for questions
        self.relations_allowed = ("inside",)

        # Small objects options - restricted to 3 kinds
        self.shapes_allowed: Tuple[str, ...] = ("circle", "triangle", "square")

    def _compute_complexity(self, stack_count: int, answer: Optional[int] = None) -> Dict[str, Any]:
        """Normalize stack counts to [0,1] and assign EASY/HARD labels."""
        span = max(1, STACK_COUNT_MAX - STACK_COUNT_MIN)
        normalized = (int(stack_count) - STACK_COUNT_MIN) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "geometric-stack-count-stack-v1",
            "range": {"min_stack": STACK_COUNT_MIN, "max_stack": STACK_COUNT_MAX},
            "stack_count": int(stack_count),
            **({"answer": int(answer)} if answer is not None else {}),
        }

    # --------------- public API ----------------

    def generate_instance(self, motif_impls: Dict, rng: random.Random):
        for _ in range(self.max_retries):
            seed = rng.randint(0, 2**31 - 1)
            lrng = random.Random(seed)

            # Canvas & background
            bg = graph_paper_rgb(self.W, self.H).convert("RGB")
            dr = ImageDraw.Draw(bg)

            # Choose stack kind
            stack_kind = lrng.choice(self.stack_kinds_allowed)

            # Sample K from PMF, then clamp/fallback to ensure within bounds
            k_sample = _sample_from_pmf(lrng, STACK_COUNT_PMF)
            if not (STACK_COUNT_MIN <= k_sample <= STACK_COUNT_MAX):
                K = lrng.randint(STACK_COUNT_MIN, STACK_COUNT_MAX)
            else:
                K = int(k_sample)

            # Build a valid equal-area stack (neighbor-aware placement)
            sheets: List[BigSheet] = _spawn_stacked_sheets(lrng, self.W, self.H, K, stack_kind)
            if not sheets or len(sheets) != K:
                continue

            # Draw stacked sheets first (bottom->top)
            for sh in sheets:
                sh.draw(dr)

            # Small objects (placed anywhere; drawn on top)
            n_small = lrng.randint(self.small_shapes_min, self.small_shapes_max)
            shapes = None
            for _pack_try in range(GLOBAL_RETRIES):
                shapes = _place_small_shapes_anywhere(
                    lrng, self.W, self.H, n_small, self.shapes_allowed, sheets
                )
                if shapes is not None:
                    break
            if shapes is None:
                continue

            # Render the small shapes on a high-quality overlay
            bg = _draw_small_shapes_supersampled(bg, shapes)

            # --------- Choose target (non-top sheet ONLY) and a small-shape kind ----------
            if len(sheets) >= 2:
                candidate_indices = list(range(0, len(sheets) - 1))  # exclude topmost (last)
            else:
                candidate_indices = [0]

            lrng.shuffle(candidate_indices)

            options: List[Tuple[int, str, int]] = []
            for idx in candidate_indices:
                sheet = sheets[idx]
                counts_by_kind: Dict[str, int] = {}
                present_kinds = {s.kind for s in shapes}
                for knd in present_kinds:
                    cnt = 0
                    for obj in shapes:
                        if obj.kind != knd:
                            continue
                        if sheet.relation_of_shape(obj, COUNT_STRICT_MARGIN_PX).get("inside", False):
                            cnt += 1
                    counts_by_kind[knd] = cnt

                for knd, c in counts_by_kind.items():
                    if self.answer_min <= c <= self.answer_max:
                        options.append((idx, knd, c))

            if not options:
                continue

            non_trivial = [opt for opt in options if opt[2] >= max(2, self.answer_min)]
            chosen_idx, shape_kind, answer = lrng.choice(non_trivial or options)

            target_sheet = sheets[chosen_idx]
            relation = "inside"

            color_name = target_sheet.name
            ref_kind   = target_sheet.kind
            big_area_int = int(round(_sheet_area(sheets[0])))

            prompt = lrng.choice(PROMPTS[relation]).format(
                shape=shape_kind, color=color_name, ref=ref_kind
            )

            spec = GeoStackSpec(
                seed=seed,
                canvas=(self.W, self.H),
                stack_kind=stack_kind,
                n_big=len(sheets),
                n_small=n_small,
                shapes_allowed=self.shapes_allowed,
                relations_allowed=self.relations_allowed,
                answer_bounds=(self.answer_min, self.answer_max),
            )

            complexity = self._compute_complexity(len(sheets), int(answer))

            meta = {
                "composite_ready": True,
                "pattern_kind": "geometry",
                "pattern": self.name,
                "variant": {
                    "stack_kind": stack_kind,
                    "n_sheets": len(sheets),
                    "stack_count_bounds": [STACK_COUNT_MIN, STACK_COUNT_MAX],
                    "overlap_constraints": {
                        "pair_overlap_ratio_range": [MIN_PAIR_OVERLAP_RATIO, MAX_PAIR_OVERLAP_RATIO],
                        "applies_to": "overlapping pairs only",
                        "require_neighbor_for_each_sheet": REQUIRE_NEIGHBOR_FOR_EACH_SHEET,
                        "ratio_denominator": "min(area_i, area_j)",
                    },
                    "small_shapes_minmax": [self.small_shapes_min, self.small_shapes_max],
                    "answer_minmax": [self.answer_min, self.answer_max],
                    "relation": relation,
                    "small_shape_kind": shape_kind,
                    "target_color": color_name,
                    "target_kind": ref_kind,
                    "strict": True,
                    "sep_small_small": SEP_SHAPE_VISUAL,
                    "sep_small_target": SEP_REGION_VISUAL,
                    "count_strict_margin_px": COUNT_STRICT_MARGIN_PX,
                    "objects_drawn_on_top": True,
                    "question_about_non_top": True,
                    "target_sheet_index": int(chosen_idx),
                    "target_is_top": (chosen_idx == len(sheets) - 1),
                    "no_cross_sheet_boundary_for_small_shapes": True,
                    "small_shape_supersample": SMALL_SHAPE_SS,
                    "small_shape_outline_px": SMALL_SHAPE_OUTLINE_PX,
                    "render_r_scale": RENDER_R_SCALE,
                    "equal_big_area_px2": float(_sheet_area(sheets[0])),
                },
                "question": prompt,
                "answer": int(answer),
                "dims": (self.W, self.H),
                "stack_sheets": [
                    (
                        {"kind": "rectangle", "color": sh.name, "bbox": sh.bbox()}
                        if isinstance(sh, SheetRect) else
                        {"kind": "circle", "color": sh.name, "cx": float(sh.cx), "cy": float(sh.cy), "R": float(sh.R)}
                        if isinstance(sh, SheetCircle) else
                        {"kind": "triangle", "color": sh.name, "verts": [(float(x), float(y)) for (x, y) in sh.verts]}
                    ) for sh in sheets
                ],
                "shapes": [
                    {"kind": s.kind, "center": [float(s.cx), float(s.cy)], "r": float(s.r), "rot_deg": float(s.rot_deg)}
                    for s in shapes
                ],
            }

            meta["complexity"] = complexity
            meta["complexity_score"] = complexity["score"]
            meta["complexity_level"] = complexity["level"]
            meta["complexity_version"] = complexity["version"]

            return bg, [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
