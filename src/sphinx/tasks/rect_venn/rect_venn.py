# sphinx/tasks/rect_venn/rect_venn.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from bisect import bisect_right

from PIL import Image, ImageDraw, ImageFont

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES

# ----------------------------- config -----------------------------
W_DEFAULT, H_DEFAULT = 1024, 768
NSETS_MIN_DEFAULT = 2
NSETS_MAX_DEFAULT = 4

# choose either rectangles or ellipses with equal probability
SHAPE_FAMILY_WEIGHTS = {"rect": 1.0, "ellipse": 0.0}

# Which kinds of questions to ask (weights sum to 1.0)
QUESTION_MODE_WEIGHTS = {
    "butnot": 0.1,       # intersection of {inc} minus {exc}
    "only": 0.4,         # exactly the included sets, no others (mask == include)
    "intersection": 0.4, # plain intersection of {inc}, no exclusions
    "union": 0.1,        # in any of the {inc}
}


# rectangles: avoid micro-cells by enforcing minimum edge separation (px)
GRID_MIN_SPAN_PX = 30

EDGE_W = 6
PAD = 12
NUM_MIN, NUM_MAX = 1, 9

# font
FONT_BASE_FRAC = 0.085
FONT_MIN_PX    = 18
PX_STEP        = 2

# grid for ellipses (regular partition, independent of edges)
ELL_TARGET_CELL = 44   # ≈ target cell size in px (both axes)

# Five easily distinguishable edge colors on white (no yellow)
COLOR_POOL = [
    ("blue",   "#1f77b4"),
    ("red",    "#d62728"),
    ("green",  "#2ca02c"),
    ("purple", "#9467bd"),
    ("orange", "#ff7f0e"),
]

# ----------------------------- prompt templates (10 each) -----------------------------
# Rectangles
PROMPTS_BUTNOT_RECT = [
    "What is the sum of the numbers inside the {inc} rectangle(s) but not in the {exc} rectangle(s)?",
    "Calculate the total of values that lie within the {inc} rectangle(s), excluding those also in the {exc} rectangle(s).",
    "Find the sum of all numbers contained in the {inc} rectangle(s) while leaving out any in the {exc} rectangle(s).",
    "Add together the numbers that appear inside the {inc} rectangle(s) but not inside the {exc} rectangle(s).",
    "What is the combined sum of numbers located in the {inc} rectangle(s), excluding the {exc} rectangle(s)?",
    "Compute the sum of values present in the {inc} rectangle(s) but absent from the {exc} rectangle(s).",
    "What is the total of the numbers in the {inc} rectangle(s) only, ignoring those that also fall in the {exc} rectangle(s)?",
    "Determine the sum of numbers that lie in the {inc} rectangle(s) but not in the {exc} rectangle(s).",
    "Add up all values found inside the {inc} rectangle(s), excluding any values from the {exc} rectangle(s).",
    "What is the sum of numbers that are within the {inc} rectangle(s) but outside the {exc} rectangle(s)?",
]


PROMPTS_ONLY_RECT = [
    "What is the sum of numbers that lie only inside the {inc} rectangle(s), and not in any other rectangle(s)?",
    "Compute the total of numbers found exclusively within the {inc} rectangle(s).",
    "Add together the values that occur solely inside the {inc} rectangle(s), with no overlap in other rectangle(s).",
    "Find the sum of entries that are unique to the {inc} rectangle(s), excluding all other regions.",
    "What is the total of numbers that appear only in the {inc} rectangle(s) and nowhere else?",
    "Calculate the sum of values located strictly inside the {inc} rectangle(s), not in any other rectangle(s).",
    "What is the combined sum of numbers that belong exclusively to the {inc} rectangle(s)?",
    "Add up all numbers that are present only in the {inc} rectangle(s), outside every other rectangle.",
    "Determine the sum of values that exist solely within the {inc} rectangle(s), not in any overlapping area.",
    "What is the sum of numbers contained only in the {inc} rectangle(s), with no part shared with others?",
]

# --- Rectangles: plain intersection (no exclusions) ---
PROMPTS_INTERSECT_RECT = [
    "What is the sum of numbers in the intersection of the {inc} rectangle(s)?",
    "Compute the total of values that lie in the shared region of the {inc} rectangle(s).",
    "Find the sum of all numbers contained in all of the {inc} rectangle(s).",
    "Add together the numbers located where the {inc} rectangle(s) overlap.",
    "What is the combined sum of numbers common to the {inc} rectangle(s)?",
    "Determine the total of numbers that are inside the shared region of the {inc} rectangle(s).",
    "Compute the sum of values in the region common to the {inc} rectangle(s).",
    "Add up the numbers that fall in the overlap of the {inc} rectangle(s).",
    "What is the total of the numbers present simultaneously in the {inc} rectangle(s)?",
    "Find the combined sum of numbers that lie in the intersection of the {inc} rectangle(s).",
]

# --- Rectangles: union (any of the included) ---
PROMPTS_UNION_RECT = [
    "What is the sum of the numbers inside any of the {inc} rectangle(s)?",
    "Compute the total of values that lie in the {inc} rectangle(s).",
    "Find the sum of all numbers contained in the {inc} rectangle(s), counting each number once.",
    "Add together the numbers located inside the {inc} rectangle(s).",
    "What is the combined sum of numbers in any of the {inc} rectangle(s)?",
    "Determine the overall sum of numbers covered by the {inc} rectangle(s).",
    "Compute the sum of values found within the {inc} rectangle(s).",
    "Add up all numbers that lie inside the {inc} rectangle(s).",
    "What is the total sum of numbers appearing anywhere in the {inc} rectangle(s)?",
    "Find the combined sum of numbers anywhere inside the {inc} rectangle(s).",
]


# --- Ellipses: plain intersection (parity with rectangles) ---
PROMPTS_INTERSECT_ELL = [
    "What is the sum of numbers in the intersection of the {inc} ellipse(s)?",
    "Compute the total of values that lie in the shared region of the {inc} ellipse(s).",
    "Find the sum of all numbers contained in every one of the {inc} ellipse(s).",
    "Add together the numbers located where the {inc} ellipse(s) overlap.",
    "What is the combined sum of numbers common to the {inc} ellipse(s)?",
    "Determine the total of numbers that are inside all {inc} ellipse(s) at once.",
    "Compute the sum of values in the region common to the {inc} ellipse(s).",
    "Add up the numbers that fall in the overlap of the {inc} ellipse(s).",
    "What is the total of the numbers present simultaneously in the {inc} ellipse(s)?",
    "Find the combined sum of numbers that lie in the intersection of the {inc} ellipse(s).",
]

# --- Ellipses: union (any of the included) ---
PROMPTS_UNION_ELL = [
    "What is the sum of the numbers inside any of the {inc} ellipse(s)?",
    "Compute the total of values that lie in at least one of the {inc} ellipse(s).",
    "Find the sum of all numbers contained in the {inc} ellipse(s), counting each number once.",
    "Add together the numbers located inside the {inc} ellipse(s).",
    "What is the combined sum of numbers in any of the {inc} ellipse(s)?",
    "Determine the total of numbers that appear in one or more of the {inc} ellipse(s).",
    "Compute the sum of values found within the {inc} ellipse(s).",
    "Add up all numbers that lie in the union of the {inc} ellipse(s).",
    "What is the total of numbers located in at least one of the {inc} ellipse(s)?",
    "Find the combined sum of numbers anywhere inside the {inc} ellipse(s).",
]


# Ellipses
PROMPTS_BUTNOT_ELL = [
    "What is the sum of the numbers in the {inc} ellipse(s) but not in the {exc} ellipse(s)?",
    "Compute the total of values inside the {inc} ellipse(s) and outside the {exc} ellipse(s).",
    "Add the numbers that lie in the {inc} ellipse(s) while excluding the {exc} ellipse(s).",
    "Find the sum of entries contained by the {inc} ellipse(s) yet not contained by the {exc} ellipse(s).",
    "What do you get when you sum numbers in the {inc} ellipse(s) but omit those in the {exc} ellipse(s)?",
    "Sum all values located within the {inc} ellipse(s), excluding the {exc} ellipse(s).",
    "Consider only the {inc} ellipse(s) and ignore the {exc} ellipse(s). What is the sum?",
    "Add up the numbers that fall inside the {inc} ellipse(s) but not the {exc} ellipse(s).",
    "Total the values in the {inc} ellipse(s) while removing any that are also in the {exc} ellipse(s).",
    "What is the total of the numbers in the {inc} ellipse(s) but outside the {exc} ellipse(s)?",
]
PROMPTS_ONLY_ELL = [
    "What is the sum of the numbers only in the {inc} ellipse(s)?",
    "Add the values that appear exclusively in the {inc} ellipse(s).",
    "Compute the total of numbers that lie only inside the {inc} ellipse(s).",
    "Sum the entries that belong to the {inc} ellipse(s) and to no others.",
    "What do you get if you add the numbers that are solely in the {inc} ellipse(s)?",
    "Add up all numbers that are found only in the {inc} ellipse(s).",
    "Consider the {inc} ellipse(s) alone. What is the sum of their numbers?",
    "Find the total for numbers that occur exclusively within the {inc} ellipse(s).",
    "Total the values that are unique to the {inc} ellipse(s).",
    "What is the sum of values that reside only inside the {inc} ellipse(s)?",
]

# ----------------------------- helpers -----------------------------
def _hex_to_rgba(h: str, a: int = 255) -> Tuple[int,int,int,int]:
    h = h.lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), a)

def _safe_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
    ]
    for p in cands:
        try: return ImageFont.truetype(p, px)
        except Exception: pass
    return ImageFont.load_default()

def _draw_text_centered(d: ImageDraw.ImageDraw, xy: Tuple[int,int], txt: str,
                        font: ImageFont.ImageFont):
    bbox = d.textbbox((0,0), txt, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = int(round(xy[0]-w/2)); y = int(round(xy[1]-h/2))
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx or dy: d.text((x+dx,y+dy), txt, font=font, fill=(255,255,255))
    d.text((x,y), txt, font=font, fill=(0,0,0))
    return (x,y,x+w,y+h)

def _join_names(nms: List[str]) -> str:
    return nms[0] if len(nms)==1 else ", ".join(nms[:-1]) + " and " + nms[-1]

# ----------------------------- grid partition: rectangles -----------------------------
def _grid_partition_rects(rects: List[Tuple[int,int,int,int]], W: int, H: int):
    def _coalesce_sorted(vals: List[int], min_span: int) -> List[int]:
        vals = sorted(vals)
        if not vals: return vals
        out = [vals[0]]
        for v in vals[1:]:
            if v - out[-1] >= min_span:
                out.append(v)
        return out

    xs_raw = sorted({x for (x0,y0,x1,y1) in rects for x in (x0, x1)})
    ys_raw = sorted({y for (x0,y0,x1,y1) in rects for y in (y0, y1)})
    xs = _coalesce_sorted(xs_raw, GRID_MIN_SPAN_PX)
    ys = _coalesce_sorted(ys_raw, GRID_MIN_SPAN_PX)
    if len(xs) < 2 or len(ys) < 2:
        return xs, ys, []

    xmid = [(xs[i] + xs[i+1]) // 2 for i in range(len(xs) - 1)]
    ymid = [(ys[j] + ys[j+1]) // 2 for j in range(len(ys) - 1)]

    mask: List[List[int]] = [[0]*(len(xs)-1) for _ in range(len(ys)-1)]
    for j, yc in enumerate(ymid):
        for i, xc in enumerate(xmid):
            m = 0
            for k, (x0,y0,x1,y1) in enumerate(rects):
                if x0 < xc < x1 and y0 < yc < y1:
                    m |= (1 << k)
            mask[j][i] = m
    return xs, ys, mask

# ----------------------------- grid partition: ellipses (regular grid) ----------------
def _inside_ellipse(x: float, y: float, cx: float, cy: float, a: float, b: float) -> bool:
    # shrink a,b slightly so midpoints near the outline don't count as inside
    aa = max(1.0, a - EDGE_W*0.6); bb = max(1.0, b - EDGE_W*0.6)
    return ((x - cx) / aa) ** 2 + ((y - cy) / bb) ** 2 < 1.0

def _grid_partition_ellipses(ells: List[Tuple[float,float,float,float]], W: int, H: int):
    # regular grid over the whole canvas; cell ~ ELL_TARGET_CELL px
    nx = max(10, min(int(round(W / float(ELL_TARGET_CELL))), 36))
    ny = max(8,  min(int(round(H / float(ELL_TARGET_CELL))), 28))
    xs = [int(round(i * W / nx)) for i in range(nx+1)]
    ys = [int(round(j * H / ny)) for j in range(ny+1)]

    xmid = [(xs[i] + xs[i+1]) * 0.5 for i in range(len(xs) - 1)]
    ymid = [(ys[j] + ys[j+1]) * 0.5 for j in range(len(ys) - 1)]

    mask: List[List[int]] = [[0]*(len(xs)-1) for _ in range(len(ys)-1)]
    for j, yc in enumerate(ymid):
        for i, xc in enumerate(xmid):
            m = 0
            for k, (cx,cy,a,b) in enumerate(ells):
                if _inside_ellipse(xc, yc, cx, cy, a, b):
                    m |= (1 << k)
            mask[j][i] = m
    return xs, ys, mask

# ----------------------------- components (shared) -----------------------------
def _components_by_mask(mask: List[List[int]]) -> Dict[int, List[List[Tuple[int,int]]]]:
    if not mask: return {}
    H = len(mask); W = len(mask[0])
    seen = [[False]*W for _ in range(H)]
    out: Dict[int, List[List[Tuple[int,int]]]] = {}

    for j in range(H):
        for i in range(W):
            m = mask[j][i]
            if m == 0 or seen[j][i]: continue
            comp: List[Tuple[int,int]] = []
            stack = [(i,j)]; seen[j][i] = True
            while stack:
                x,y = stack.pop()
                comp.append((x,y))
                for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if 0 <= nx < W and 0 <= ny < H and not seen[ny][nx] and mask[ny][nx] == m:
                        seen[ny][nx] = True
                        stack.append((nx,ny))
            out.setdefault(m, []).append(comp)
    return out

# ----------------------------- task spec -----------------------------
@dataclass
class RectVennSpec:
    seed: int
    canvas: Tuple[int,int]
    n_sets: int
    set_names: List[str]
    family: str                    # 'rect' or 'ellipse'
    rects: List[Dict[str,Any]]
    ellipses: List[Dict[str,Any]]
    region_sums: Dict[str,int]     # bitstring -> sum
    question: str
    answer: int

# ----------------------------- task -----------------------------
@register_task
class RectVennTask(Task):
    """
    Colored-set inclusion/exclusion with either axis-aligned **rectangles** (edge-partition grid)
    or axis-aligned **ellipses** (regular coarse grid). One number is placed in *every* non-empty
    atomic region (bitmask), with a forced fallback for skinny components.
    """
    name = "rect_venn"

    def __init__(self, W: int = W_DEFAULT, H: int = H_DEFAULT,
                 nsets_min: int = NSETS_MIN_DEFAULT, nsets_max: int = NSETS_MAX_DEFAULT):
        self.W, self.H = int(W), int(H)
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.nmin = int(nsets_min); self.nmax = int(nsets_max)

    def _compute_complexity(self, n_sets: int) -> Dict[str, Any]:
        """Normalize number of sets to [0,1] and derive a categorical level."""
        min_sets = int(getattr(self, "nmin", NSETS_MIN_DEFAULT))
        max_sets = int(getattr(self, "nmax", NSETS_MAX_DEFAULT))
        span = max(1, max_sets - min_sets)
        normalized = (int(n_sets) - min_sets) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "rect-venn-nsets-v1",
            "range": {"min_sets": min_sets, "max_sets": max_sets},
            "n_sets": int(n_sets),
        }

    # -------------------- sample rectangles (connected union) --------------------
    def _sample_rects_connected(self, rng: random.Random, n: int) -> Optional[List[Tuple[int,int,int,int]]]:
        W,H = self.W, self.H
        rects: List[Tuple[int,int,int,int]] = []
        xs_seen: List[int] = []; ys_seen: List[int] = []

        def _rand_rect():
            w = int(rng.uniform(0.30, 0.50) * W)
            h = int(rng.uniform(0.22, 0.40) * H)
            x0 = rng.randint(PAD, max(PAD, W - PAD - w))
            y0 = rng.randint(PAD, max(PAD, H - PAD - h))
            return (x0, y0, x0 + w, y0 + h)

        def _area_inter(a, b) -> int:
            x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
            x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
            return max(0, x1-x0) * max(0, y1-y0)

        def _edges_far_enough(x0, x1, y0, y1) -> bool:
            return all(abs(x0 - x) >= GRID_MIN_SPAN_PX and abs(x1 - x) >= GRID_MIN_SPAN_PX for x in xs_seen) and \
                   all(abs(y0 - y) >= GRID_MIN_SPAN_PX and abs(y1 - y) >= GRID_MIN_SPAN_PX for y in ys_seen)

        for i in range(n):
            ok = False
            for _ in range(240):
                r = _rand_rect()
                x0,y0,x1,y1 = r
                if i == 0:
                    ok = True
                else:
                    if not any(_area_inter(r, s) > 0 for s in rects):
                        continue
                    if not _edges_far_enough(x0,x1,y0,y1):
                        continue
                    ok = True
                if ok:
                    rects.append(r); xs_seen.extend([x0,x1]); ys_seen.extend([y0,y1]); break
            if not ok:
                return None
        return rects

    # -------------------- sample ellipses (connected union) --------------------
    def _sample_ellipses_connected(self, rng: random.Random, n: int) -> Optional[List[Tuple[float,float,float,float]]]:
        W,H = self.W, self.H
        ells: List[Tuple[float,float,float,float]] = []

        for i in range(n):
            placed = False
            for _ in range(240):
                a = rng.uniform(0.18*W, 0.30*W)  # semi-axis x
                b = rng.uniform(0.18*H, 0.30*H)  # semi-axis y
                if i == 0:
                    cx = rng.uniform(PAD + a, W - PAD - a)
                    cy = rng.uniform(PAD + b, H - PAD - b)
                    ells.append((cx,cy,a,b)); placed = True; break
                else:
                    # pick a base ellipse and place near it to ensure overlap
                    bx, by, ba, bb = rng.choice(ells)
                    cx = bx + rng.uniform(-0.6*ba, 0.6*ba)
                    cy = by + rng.uniform(-0.6*bb, 0.6*bb)
                    cx = min(max(PAD + a, cx), W - PAD - a)
                    cy = min(max(PAD + b, cy), H - PAD - b)

                    # overlap check (Minkowski-sum style; conservative)
                    ov = False
                    for (px,py,pa,pb) in ells:
                        dx = (cx - px) / (a + pa)
                        dy = (cy - py) / (b + pb)
                        if dx*dx + dy*dy < 0.95:  # generous overlap test
                            ov = True; break
                    if ov:
                        ells.append((cx,cy,a,b)); placed = True; break
            if not placed:
                return None
        return ells

    # -------------------- draw --------------------
    def _compose_rects(self, rects, names_hex):
        img = Image.new("RGBA", (self.W, self.H), (255,255,255,255))
        d = ImageDraw.Draw(img, "RGBA")
        for (xyxy, (nm,hx)) in zip(rects, names_hex):
            d.rectangle(xyxy, outline=_hex_to_rgba(hx,255), width=EDGE_W)
        return img

    def _compose_ells(self, ells, names_hex):
        img = Image.new("RGBA", (self.W, self.H), (255,255,255,255))
        d = ImageDraw.Draw(img, "RGBA")
        for ((cx,cy,a,b), (nm,hx)) in zip(ells, names_hex):
            d.ellipse((cx-a,cy-b,cx+a,cy+b), outline=_hex_to_rgba(hx,255), width=EDGE_W)
        return img

    # -------------------- questions --------------------
    def _sum_by_mode(self, sums_by_mask: Dict[int,int], include: int, exclude: int, mode: str) -> int:
        """
        mode:
        - 'butnot'      : in all include sets AND in none of exclude sets
        - 'only'        : in exactly the include sets (no others)  -> mask == include
        - 'intersection': in all include sets (ignores others)
        - 'union'       : in any of the include sets
        """
        if mode == "butnot":
            return sum(s for m, s in sums_by_mask.items() if (m & include) == include and (m & exclude) == 0)
        if mode == "only":
            return sum(s for m, s in sums_by_mask.items() if m == include)
        if mode == "intersection":
            return sum(s for m, s in sums_by_mask.items() if (m & include) == include)
        if mode == "union":
            return sum(s for m, s in sums_by_mask.items() if (m & include) != 0)
        # default safety
        return 0


    def _prompt(self, rng: random.Random, family: str, mode: str, inc: List[str], exc: List[str]) -> str:
        inc_s = _join_names(inc)
        if family == "rect":
            if mode == "butnot":
                return rng.choice(PROMPTS_BUTNOT_RECT).format(inc=inc_s, exc=_join_names(exc))
            if mode == "only":
                return rng.choice(PROMPTS_ONLY_RECT).format(inc=inc_s)
            if mode == "intersection":
                return rng.choice(PROMPTS_INTERSECT_RECT).format(inc=inc_s)
            if mode == "union":
                return rng.choice(PROMPTS_UNION_RECT).format(inc=inc_s)
        else:  # ellipse
            if mode == "butnot":
                return rng.choice(PROMPTS_BUTNOT_ELL).format(inc=inc_s, exc=_join_names(exc))
            if mode == "only":
                return rng.choice(PROMPTS_ONLY_ELL).format(inc=inc_s)
            if mode == "intersection":
                return rng.choice(PROMPTS_INTERSECT_ELL).format(inc=inc_s)
            if mode == "union":
                return rng.choice(PROMPTS_UNION_ELL).format(inc=inc_s)
        # fallback
        return rng.choice(PROMPTS_UNION_RECT if family == "rect" else PROMPTS_UNION_ELL).format(inc=inc_s)


    def _make_question(self, rng: random.Random, set_names: List[str],
                    sums_by_mask: Dict[int,int], family: str) -> Tuple[str,int,Dict[str,Any]]:
        n = len(set_names)
        idx = list(range(n))

        # small helper to draw a mode by weight
        total_w = sum(QUESTION_MODE_WEIGHTS.values())
        def pick_mode() -> str:
            r = rng.random() * total_w
            s = 0.0
            for m, w in QUESTION_MODE_WEIGHTS.items():
                s += w
                if r <= s:
                    return m
            return "union"

        for _ in range(80):
            mode = pick_mode()

            # choose how many to include
            if mode == "intersection":
                k_in = rng.randint(2 if n >= 2 else 1, min(n, 3))  # prefer ≥2 for intersection
            else:
                k_in = rng.randint(1, min(n, 3))

            inc = set(rng.sample(idx, k_in))
            rest = [i for i in idx if i not in inc]

            # exclusions depend on mode
            if mode == "butnot":
                if not rest:
                    continue
                exc = set(rest) if rng.random() < 0.6 else set(rng.sample(rest, rng.randint(1, min(2, len(rest)))))
            elif mode == "only":
                exc = set(rest)  # exclude all others by definition
            else:
                exc = set()      # intersection/union have no explicit exclusions

            include_mask = sum(1 << i for i in inc)
            exclude_mask = sum(1 << j for j in exc) if exc else 0

            ans = self._sum_by_mode(sums_by_mask, include_mask, exclude_mask, mode)
            if ans > 0:
                inc_names = [set_names[i] for i in inc]
                exc_names = [set_names[j] for j in exc]
                q = self._prompt(rng, family, mode, inc_names, exc_names)
                qmeta = {
                    "mode": mode,
                    "include_mask": include_mask,
                    "exclude_mask": exclude_mask,
                    "include_names": inc_names,
                    "exclude_names": exc_names,
                }
                return q, int(ans), qmeta

        # fallback: simple union of a single set that must be > 0
        for i in idx:
            include_mask = (1 << i); exclude_mask = 0
            ans = self._sum_by_mode(sums_by_mask, include_mask, exclude_mask, "union")
            if ans > 0:
                inc_names = [set_names[i]]
                q = self._prompt(rng, family, "union", inc_names, [])
                qmeta = {"mode": "union", "include_mask": include_mask, "exclude_mask": 0,
                        "include_names": inc_names, "exclude_names": []}
                return q, int(ans), qmeta

        # if everything failed (extremely unlikely), ask "only first"
        include_mask = 1; exclude_mask = sum(1 << j for j in range(1, n))
        inc_names = [set_names[0]]; exc_names = set_names[1:]
        q = self._prompt(rng, family, "butnot", inc_names, list(exc_names))
        ans = self._sum_by_mode(sums_by_mask, include_mask, exclude_mask, "butnot")
        qmeta = {"mode": "butnot", "include_mask": include_mask, "exclude_mask": exclude_mask,
                "include_names": inc_names, "exclude_names": list(exc_names)}
        return q, int(ans), qmeta


    # -------------------- shared number placement from a grid --------------------
    def _place_from_grid(
        self,
        rng: random.Random,
        img: Image.Image,
        xs: List[int], ys: List[int], mask: List[List[int]]
    ) -> Tuple[List[Dict[str,Any]], Dict[int,int]]:
        d = ImageDraw.Draw(img, "RGBA")
        base_px = max(FONT_MIN_PX, int(round(FONT_BASE_FRAC * min(self.W, self.H))))

        comps_by_mask = _components_by_mask(mask)
        placed: List[Dict[str,Any]] = []
        sums_by_mask: Dict[int,int] = {}
        occupied: List[Tuple[int,int,int,int]] = []

        def _fits(b: Tuple[int,int,int,int]) -> bool:
            x0,y0,x1,y1 = b
            if x0 < 2 or y0 < 2 or x1 > self.W-2 or y1 > self.H-2: return False
            for (a0,b0,a1,b1) in occupied:
                if not (x1 < a0 or a1 < x0 or y1 < b0 or b1 < y0): return False
            return True

        # Try to place one number in EACH component
        for m, comps in comps_by_mask.items():
            for cid, comp in enumerate(comps):
                # cells sorted by area
                comp_sorted = sorted(
                    comp,
                    key=lambda ij: (xs[ij[0]+1] - xs[ij[0]]) * (ys[ij[1]+1] - ys[ij[1]]),
                    reverse=True
                )
                comp_set = set(comp)
                placed_here = False
                for (i,j) in comp_sorted:
                    w = xs[i+1]-xs[i]; h = ys[j+1]-ys[j]
                    if w <= 8 or h <= 8: continue
                    margin = EDGE_W*2 + 6
                    allow_w = max(1, w - 2*margin)
                    allow_h = max(1, h - 2*margin)
                    if allow_w <= 8 or allow_h <= 8: continue
                    cx = (xs[i]+xs[i+1])//2; cy = (ys[j]+ys[j+1])//2

                    px = base_px
                    while px >= FONT_MIN_PX and not placed_here:
                        font = _safe_font(px)
                        val = rng.randint(NUM_MIN, NUM_MAX); txt = str(val)
                        tb = d.textbbox((0,0), txt, font=font)
                        tw, th = tb[2]-tb[0], tb[3]-tb[1]
                        if tw <= allow_w and th <= allow_h:
                            box = (cx - tw//2, cy - th//2, cx + (tw - tw//2), cy + (th - th//2))
                            # ensure corners of box map back to the same component
                            ok = True
                            for (pxp,pyp) in [(box[0],box[1]), (box[2]-1,box[1]), (box[0],box[3]-1), (box[2]-1,box[3]-1)]:
                                ii = bisect_right(xs, pxp) - 1
                                jj = bisect_right(ys, pyp) - 1
                                if not (0 <= ii < len(xs)-1 and 0 <= jj < len(ys)-1) or (ii,jj) not in comp_set:
                                    ok = False; break
                            if ok and _fits(box):
                                _draw_text_centered(d, (cx, cy), txt, font=font)
                                occupied.append(box)
                                placed.append({"value": int(val), "x": int(cx), "y": int(cy),
                                               "mask": int(m), "component": int(cid)})
                                sums_by_mask[m] = sums_by_mask.get(m, 0) + int(val)
                                placed_here = True
                                break
                        px -= PX_STEP

                # (soft) fallback per component: try bounding box center with small font
                if not placed_here:
                    ii = [i for (i,_) in comp]; jj = [j for (_,j) in comp]
                    xL, xR = xs[min(ii)], xs[max(ii)+1]; yT, yB = ys[min(jj)], ys[max(jj)+1]
                    cx = (xL + xR) // 2; cy = (yT + yB) // 2
                    px = max(FONT_MIN_PX, base_px // 2)
                    while px >= FONT_MIN_PX and not placed_here:
                        font = _safe_font(px)
                        val = rng.randint(NUM_MIN, NUM_MAX); txt = str(val)
                        tb = d.textbbox((0,0), txt, font=font)
                        tw, th = tb[2]-tb[0], tb[3]-tb[1]
                        box = (cx - tw//2, cy - th//2, cx + (tw - tw//2), cy + (th - th//2))
                        # require at least the center to be in the component and avoid collisions
                        ii = bisect_right(xs, cx) - 1; jj = bisect_right(ys, cy) - 1
                        if (ii,jj) in comp_set and _fits(box):
                            _draw_text_centered(d, (cx, cy), txt, font=font)
                            occupied.append(box)
                            placed.append({"value": int(val), "x": int(cx), "y": int(cy),
                                           "mask": int(m), "component": int(cid)})
                            sums_by_mask[m] = sums_by_mask.get(m, 0) + int(val)
                            placed_here = True
                            break
                        px -= PX_STEP

        # HARD guarantee: at least one number per *mask value*
        # If a mask has components but none got a number (very skinny), place into the largest cell.
        for m, comps in comps_by_mask.items():
            if not comps: continue
            if all(p["mask"] != m for p in placed):
                # find largest single cell with this mask anywhere
                best = None; best_area = -1
                for j in range(len(mask)):
                    for i in range(len(mask[0])):
                        if mask[j][i] != m: continue
                        area = (xs[i+1]-xs[i]) * (ys[j+1]-ys[j])
                        if area > best_area:
                            best_area = area; best = (i,j)
                if best is not None:
                    i,j = best
                    cx = (xs[i]+xs[i+1])//2; cy = (ys[j]+ys[j+1])//2
                    font = _safe_font(FONT_MIN_PX)
                    val = rng.randint(NUM_MIN, NUM_MAX); txt = str(val)
                    _draw_text_centered(d, (cx,cy), txt, font=font)
                    tb = d.textbbox((0,0), txt, font=font)
                    box = (cx - (tb[2]-tb[0])//2, cy - (tb[3]-tb[1])//2,
                           cx + (tb[2]-tb[0] - (tb[2]-tb[0])//2), cy + (tb[3]-tb[1] - (tb[3]-tb[1])//2))
                    occupied.append(box)
                    placed.append({"value": int(val), "x": int(cx), "y": int(cy),
                                   "mask": int(m), "component": 0})
                    sums_by_mask[m] = sums_by_mask.get(m, 0) + int(val)

        return placed, sums_by_mask

    # -------------------- public API --------------------
    def generate_instance(self, motif_impls: Dict[str,Any], rng: random.Random):
        for _ in range(self.max_retries):
            seed = rng.randrange(2**31 - 1)
            lrng = random.Random(seed)

            # family choice
            fams, weights = zip(*SHAPE_FAMILY_WEIGHTS.items())
            t = lrng.random() * sum(weights)
            s = 0.0; family = fams[-1]
            for f,w in SHAPE_FAMILY_WEIGHTS.items():
                s += w
                if t <= s:
                    family = f; break

            n = lrng.randint(self.nmin, self.nmax)
            pool = COLOR_POOL[:]; lrng.shuffle(pool)
            names_hex = pool[:n]
            set_names = [nm for (nm, _) in names_hex]

            if family == "rect":
                rects = self._sample_rects_connected(lrng, n)
                if not rects: continue
                rgba = self._compose_rects(rects, names_hex)
                xs, ys, mask = _grid_partition_rects(rects, self.W, self.H)
                placed, sums_by_mask = self._place_from_grid(lrng, rgba, xs, ys, mask)

                if not placed: continue
                q, ans, qmeta = self._make_question(lrng, set_names, sums_by_mask, "rect")
                region_sums = {format(m, f"0{n}b"): int(s) for m,s in sums_by_mask.items()}

                img_rgb = rgba.convert("RGB")
                spec = RectVennSpec(
                    seed=seed, canvas=(img_rgb.width, img_rgb.height),
                    n_sets=n, set_names=list(set_names), family="rect",
                    rects=[{"xyxy": tuple(map(int, r)), "color": nm, "hex": hx}
                           for r,(nm,hx) in zip(rects, names_hex)],
                    ellipses=[],
                    region_sums=region_sums, question=q, answer=int(ans),
                )
                complexity = self._compute_complexity(n)
                meta = {
                    "pattern_kind": "set_reasoning",
                    "pattern": self.name,
                    "variant": {"style": "colored_sets", "n_sets": n, "family": "rect"},
                    "dims": (img_rgb.width, img_rgb.height),
                    "colors": [{"name": nm, "hex": hx} for (nm,hx) in names_hex],
                    "rectangles": [{"xyxy": tuple(map(int, r))} for r in rects],
                    "numbers": placed,
                    "region_sums": region_sums,
                    "question": q, "answer": str(int(ans)),
                    "query_meta": qmeta,
                    "value_range": [NUM_MIN, NUM_MAX],
                    "composite_ready": True,
                }
                meta["complexity"] = complexity
                meta["complexity_score"] = complexity["score"]
                meta["complexity_level"] = complexity["level"]
                meta["complexity_version"] = complexity["version"]
                return img_rgb, [spec], meta

            else:  # ellipse family
                ells = self._sample_ellipses_connected(lrng, n)
                if not ells: continue
                rgba = self._compose_ells(ells, names_hex)
                xs, ys, mask = _grid_partition_ellipses(ells, self.W, self.H)
                placed, sums_by_mask = self._place_from_grid(lrng, rgba, xs, ys, mask)

                if not placed: continue
                q, ans, qmeta = self._make_question(lrng, set_names, sums_by_mask, "ellipse")
                region_sums = {format(m, f"0{n}b"): int(s) for m,s in sums_by_mask.items()}

                img_rgb = rgba.convert("RGB")
                spec = RectVennSpec(
                    seed=seed, canvas=(img_rgb.width, img_rgb.height),
                    n_sets=n, set_names=list(set_names), family="ellipse",
                    rects=[], ellipses=[{"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b),
                                         "color": nm, "hex": hx}
                                        for (cx,cy,a,b),(nm,hx) in zip(ells, names_hex)],
                    region_sums=region_sums, question=q, answer=int(ans),
                )
                complexity = self._compute_complexity(n)
                meta = {
                    "pattern_kind": "set_reasoning",
                    "pattern": self.name,
                    "variant": {"style": "colored_sets", "n_sets": n, "family": "ellipse"},
                    "dims": (img_rgb.width, img_rgb.height),
                    "colors": [{"name": nm, "hex": hx} for (nm,hx) in names_hex],
                    "ellipses": [{"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b)} for (cx,cy,a,b) in ells],
                    "numbers": placed,
                    "region_sums": region_sums,
                    "question": q, "answer": str(int(ans)),
                    "query_meta": qmeta,
                    "value_range": [NUM_MIN, NUM_MAX],
                    "composite_ready": True,
                }
                meta["complexity"] = complexity
                meta["complexity_score"] = complexity["score"]
                meta["complexity_level"] = complexity["level"]
                meta["complexity_version"] = complexity["version"]
                return img_rgb, [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
