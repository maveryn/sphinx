# sphinx/tasks/shape_count.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, Set, Callable

from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES

# Optional charts palette (soft, distinct colors)
try:
    from sphinx.charts import choose_colors   # returns (hex_colors, _)
    _HAVE_CHARTS = True
except Exception:
    _HAVE_CHARTS = False

# Weighted choice (tiles-style)
try:
    from sphinx.utils.rng import choice_weighted
except Exception:
    # Minimal fallback if utils.rng is not available
    def choice_weighted(rng: random.Random, items: List[Any], weights: List[float]) -> Any:
        total = sum(max(0.0, float(w)) for w in weights)
        if total <= 0:
            return rng.choice(items)
        x = rng.random() * total
        acc = 0.0
        for it, w in zip(items, weights):
            acc += max(0.0, float(w))
            if x <= acc:
                return it
        return items[-1]

# =============================  TOP‑LEVEL TUNABLES  =============================
# 1) Which *shape* to ask about (weights mirror tiles tasks style).
ASK_SHAPE_WEIGHTS: Dict[str, float] = {
    "rectangles": 1.0,
    "squares": 4.0,
    "parallelograms": 1.0,
    "triangles": 1.0,
}

# 2) Within each asked shape, which *sampler family* to use.
#    Keys below are stable semantic names; you can tweak weights freely.
#    Irregular samplers (line‑drop, polyomino/skew, inscribed‑with‑drops) are upweighted by default.
SHAPE_SAMPLER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "rectangles": {
        "grid_irregular": 5.0,   # line‑drop rectangular grids
        "grid_regular": 1.0,     # plain rectangular grids
        "polyomino_axis": 5.0,   # protruding (axis‑aligned) polyomino
    },
    "squares": {
        "grid_irregular": 2.0,   # line‑drop square grids
        "grid_regular": 1.0,     # plain square grids
        "staircase": 0,        # staircase squares
        "inscribed": 2,       # inscribed square◄►circle with visibility drops
        "polyomino_axis": 25,   # protruding (axis‑aligned) polyomino
    },
    "parallelograms": {
        "skew_irregular": 5,   # line‑drop skew grids
        "skew_regular": 1.0,     # plain skew grids
        "polyomino_skew": 5,   # protruding (skew) polyomino
    },
    "triangles": {
        "tri_lattice": 1.0,      # triangular lattice (only option)
    },
}

# 3) Global multiplier applied to *irregular* sampler weights (extra bias to pick them more often).
IRREGULARITY_WEIGHT_MULT: float = 1.35

# 4) “Protrusion” controls for polyomino builders (more attachments => more irregular outlines).
#    Range of extra attached rectangles + a bias that nudges towards the higher end.
POLY_EXTRAS_RANGE: Tuple[int, int] = (1, 3)
POLY_PROTRUSION_BIAS: float = 0.55   # ~55% chance to bump selected extras by +1 when feasible

# ===============================================================================
# ----------------------------- global config -----------------------------
COUNT_MIN_DEFAULT = 5
COUNT_MAX_DEFAULT = 20

LINE_W = 4
PAD_FRAC = 0.12
FILL_ALPHA = 70          # soft tint

# Minimum readable inner size for the inscribed alternation stack:
MIN_INNER_PX   = 24       # absolute floor in pixels
MIN_INNER_FRAC = 0.06     # 6% of the shorter canvas side

# Randomization for irregularity already present inside samplers
INSCRIBE_DROP_PROB_RANGE   = (0.15, 0.45)
GRID_LINE_DROP_PROB_RANGE  = (0.20, 0.50)

# ----------------------------- prompts (10 per shape; all end with '?') -----------------------------
PROMPTS_BY_SHAPE: Dict[str, List[str]] = {
    "rectangles": [
        "How many rectangles are there in the figure?",
        "Count all rectangles in the diagram—what is the total?",
        "What is the number of rectangles present in the figure?",
        "How many distinct rectangles can you find in the image?",
        "What is the total count of rectangles in the drawing?",
        "How many rectangles appear in the picture?",
        "What is the total number of rectangles shown?",
        "How many rectangles are contained in the figure?",
        "Count the rectangles in the diagram—how many are there?",
        "What is the rectangle count in the figure?",
    ],
    "squares": [
        "How many squares are there in the figure?",
        "Count all squares in the diagram—what is the total?",
        "What is the number of squares present in the figure?",
        "How many distinct squares can you find in the image?",
        "What is the total count of squares in the drawing?",
        "How many squares appear in the picture?",
        "What is the total number of squares shown?",
        "How many squares are contained in the figure?",
        "Count the squares in the diagram—how many are there?",
        "What is the square count in the figure?",
    ],
    "parallelograms": [
        "How many parallelograms are there in the figure?",
        "Count all parallelograms in the diagram—what is the total?",
        "What is the number of parallelograms present in the figure?",
        "How many distinct parallelograms can you find in the image?",
        "What is the total count of parallelograms in the drawing?",
        "How many parallelograms appear in the picture?",
        "What is the total number of parallelograms shown?",
        "How many parallelograms are contained in the figure?",
        "Count the parallelograms in the diagram—how many are there?",
        "What is the parallelogram count in the figure?",
    ],
    "triangles": [
        "How many triangles are there in the figure?",
        "Count all triangles in the diagram—what is the total?",
        "What is the number of triangles present in the figure?",
        "How many distinct triangles can you find in the image?",
        "What is the total count of triangles in the drawing?",
        "How many triangles appear in the picture?",
        "What is the total number of triangles shown?",
        "How many triangles are contained in the figure?",
        "Count the triangles in the diagram—how many are there?",
        "What is the triangle count in the figure?",
    ],
}

# ----------------------------- helpers: math -----------------------------
def comb2(n: int) -> int:
    return (n * (n - 1)) // 2

def count_rectangles_in_grid(rows: int, cols: int) -> int:
    return comb2(rows + 1) * comb2(cols + 1)

def count_squares_in_grid(rows: int, cols: int) -> int:
    m = min(rows, cols)
    return sum((rows - k + 1) * (cols - k + 1) for k in range(1, m + 1))

def count_parallelograms_in_skew(nu_lines: int, nv_lines: int) -> int:
    return comb2(nu_lines) * comb2(nv_lines)

def count_squares_in_staircase(n: int) -> int:
    return (n * (n + 1) * (n + 2)) // 6

def count_triangles_in_triangular_lattice(n: int) -> int:
    # Classic equilateral triangular grid inside a big triangle (order n)
    # counts *all* triangles of all sizes and both orientations.
    return (n * (n + 2) * (2 * n + 1)) // 8

# ---------- counts on an arbitrary occupancy mask (rows×cols, bool) ----------
def _count_rectangles_occ(occ: List[List[bool]]) -> int:
    """Count all axis-aligned rectangles fully covered by True cells."""
    R = len(occ); C = len(occ[0]) if R else 0
    total = 0
    # O(R^2 * C) using row-pair runs
    for top in range(R):
        good = [True] * C
        for bot in range(top, R):
            for c in range(C):
                good[c] = good[c] and occ[bot][c]
            run = 0
            for c in range(C):
                if good[c]:
                    run += 1
                    total += run
                else:
                    run = 0
    return total

def _count_squares_occ(occ: List[List[bool]]) -> int:
    """Count all k×k squares fully covered by True cells."""
    R = len(occ); C = len(occ[0]) if R else 0
    total = 0
    K = min(R, C)
    for k in range(1, K + 1):
        for r in range(R - k + 1):
            for c in range(C - k + 1):
                ok = True
                for rr in range(r, r + k):
                    if not ok: break
                    for cc in range(c, c + k):
                        if not occ[rr][cc]:
                            ok = False; break
                if ok:
                    total += 1
    return total

# ----------------------------- helpers: colors & canvas -----------------------------
def _hex_to_rgba(h: str, alpha: int = FILL_ALPHA) -> Tuple[int, int, int, int]:
    h = h.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (r, g, b, alpha)

def _pick_soft_rgba(rng: random.Random, n: int, alpha: int = FILL_ALPHA) -> List[Tuple[int,int,int,int]]:
    if _HAVE_CHARTS:
        hex_cols, _ = choose_colors(rng, n)
        return [_hex_to_rgba(h, alpha) for h in hex_cols]
    fallback = ["#bcd3f2", "#f7c9a9", "#c9e4de", "#ffe29a", "#e6c2ff", "#bfe1b0", "#f25f84"]
    return [_hex_to_rgba(rng.choice(fallback), alpha) for _ in range(n)]

def _safe_pad(w: int, h: int) -> int:
    return int(round(PAD_FRAC * min(w, h)))

def _white_rgba(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), (255, 255, 255, 255))

def _transparent(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))

def _compose_on_white(w: int, h: int, layers: List[Image.Image]) -> Image.Image:
    bg = _white_rgba(w, h)
    for lay in layers:
        bg.paste(lay, (0, 0), lay)
    return bg.convert("RGB")

# ----------------------------- helpers: drawing -----------------------------
def _poly(d: ImageDraw.ImageDraw, pts: List[Tuple[float, float]],
          outline_w: int = LINE_W, fill_rgba: Optional[Tuple[int,int,int,int]] = None):
    if fill_rgba is not None:
        d.polygon(pts, fill=fill_rgba, outline=(0,0,0,255))
    d.line(pts + [pts[0]], fill=(0,0,0,255), width=outline_w, joint="curve")

def _line(d: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], w: int = LINE_W):
    d.line(pts, fill=(0, 0, 0, 255), width=w, joint="curve")

# ----------------------------- base renderers (RGBA) -----------------------------
def render_axis_grid_square_cells(
    w: int, h: int, rows: int, cols: int,
    transparent: bool = False,
    fill_rgba: Optional[Tuple[int,int,int,int]] = None,
    keep_row_lines: Optional[List[bool]] = None,
    keep_col_lines: Optional[List[bool]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    img = _transparent(w, h) if transparent else _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)

    side = min(w, h) - 2 * pad
    unit = side / max(rows, cols)
    grid_w = cols * unit
    grid_h = rows * unit
    x0 = (w - grid_w) / 2
    y0 = (h - grid_h) / 2
    x1 = x0 + grid_w
    y1 = y0 + grid_h

    _poly(d, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], outline_w=LINE_W, fill_rgba=fill_rgba)

    if keep_col_lines is None:
        keep_col_lines = [True] * (cols + 1)
    if keep_row_lines is None:
        keep_row_lines = [True] * (rows + 1)
    keep_col_lines[0] = True; keep_col_lines[-1] = True
    keep_row_lines[0] = True; keep_row_lines[-1] = True

    kept_c = [i for i, k in enumerate(keep_col_lines) if k]
    kept_r = [i for i, k in enumerate(keep_row_lines) if k]

    for c in range(1, cols):
        if keep_col_lines[c]:
            x = x0 + c * unit
            _line(d, [(x, y0), (x, y1)])
    for r in range(1, rows):
        if keep_row_lines[r]:
            y = y0 + r * unit
            _line(d, [(x0, y), (x1, y)])

    meta: Dict[str, Any] = {"kept_rows": kept_r, "kept_cols": kept_c}
    rects = comb2(len(kept_c)) * comb2(len(kept_r))
    meta["rectangles_irregular"] = rects

    if rows == cols:
        # squares from matching row/col gaps
        def _pairdiff(idxs: List[int]) -> Dict[int,int]:
            out: Dict[int,int] = {}
            m = len(idxs)
            for a in range(m):
                for b in range(a+1, m):
                    dkey = idxs[b] - idxs[a]
                    out[dkey] = out.get(dkey, 0) + 1
            return out
        dc = _pairdiff(kept_c); dr = _pairdiff(kept_r)
        sq = 0
        for dkey, vc in dc.items():
            if dkey in dr:
                sq += vc * dr[dkey]
        meta["squares_irregular"] = sq
    else:
        meta["squares_irregular"] = None
    return img, meta

def render_parallelogram_grid(
    w: int, h: int, nu: int, nv: int, shear: float = 0.25,
    transparent: bool = False,
    fill_rgba: Optional[Tuple[int,int,int,int]] = None,
    keep_u_lines: Optional[List[bool]] = None,
    keep_v_lines: Optional[List[bool]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    img = _transparent(w, h) if transparent else _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)

    U = (0.65 * (w - 2 * pad), 0.0)
    V = (shear * (w - 2 * pad), 0.65 * (h - 2 * pad))
    cx, cy = w / 2.0, h / 2.0
    A = (cx - 0.5 * (U[0] + V[0]), cy - 0.5 * (U[1] + V[1]))
    B = (A[0] + U[0], A[1] + U[1])
    D = (A[0] + V[0], A[1] + V[1])
    C = (B[0] + V[0], B[1] + V[1])

    _poly(d, [A, B, C, D], outline_w=LINE_W, fill_rgba=fill_rgba)

    if keep_u_lines is None:
        keep_u_lines = [True] * nu
    if keep_v_lines is None:
        keep_v_lines = [True] * nv

    kept_u = [i for i, k in enumerate(keep_u_lines) if k]
    kept_v = [j for j, k in enumerate(keep_v_lines) if k]

    for i in range(nu):
        if keep_u_lines[i]:
            t = i / (nu - 1) if nu > 1 else 0.0
            p = (A[0] + t * V[0], A[1] + t * V[1])
            q = (B[0] + t * V[0], B[1] + t * V[1])
            if 0 < i < nu-1:
                _line(d, [p, q])

    for j in range(nv):
        if keep_v_lines[j]:
            s = j / (nv - 1) if nv > 1 else 0.0
            p = (A[0] + s * U[0], A[1] + s * U[1])
            q = (D[0] + s * U[0], D[1] + s * U[1])
            if 0 < j < nv-1:
                _line(d, [p, q])

    meta = {"kept_u": kept_u, "kept_v": kept_v}
    meta["parallelograms_irregular"] = comb2(len(kept_u)) * comb2(len(kept_v))
    return img, meta

# ----------------------------- occupancy renderers -----------------------------
def _render_occupancy(
    w: int, h: int, occ: List[List[bool]],
    fill_rgba: Optional[Tuple[int,int,int,int]] = None
) -> Image.Image:
    """Axis-aligned unit squares occupancy."""
    img = _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)
    R = len(occ); C = len(occ[0]) if R else 0
    if R == 0 or C == 0:
        return img

    side_w = w - 2 * pad
    side_h = h - 2 * pad
    unit = min(side_w / C, side_h / R)
    W = C * unit
    Hh = R * unit
    x0 = (w - W) / 2
    y0 = (h - Hh) / 2

    for r in range(R):
        for c in range(C):
            if not occ[r][c]:
                continue
            xL = x0 + c * unit
            yT = y0 + r * unit
            xR = xL + unit
            yB = yT + unit
            _poly(d, [(xL, yT), (xR, yT), (xR, yB), (xL, yB)],
                  outline_w=LINE_W, fill_rgba=fill_rgba)
    return img

def _render_skew_occupancy(
    w: int, h: int, occ: List[List[bool]], shear: float = 0.25,
    fill_rgba: Optional[Tuple[int,int,int,int]] = None
) -> Image.Image:
    """Parallelogram-aligned occupancy drawn as skewed cells."""
    img = _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)

    R = len(occ); C = len(occ[0]) if R else 0
    if R == 0 or C == 0:
        return img

    U = (0.70 * (w - 2 * pad), 0.0)
    V = (shear * (w - 2 * pad), 0.70 * (h - 2 * pad))
    cx, cy = w / 2.0, h / 2.0
    A = (cx - 0.5 * (U[0] + V[0]), cy - 0.5 * (U[1] + V[1]))
    u_step = (U[0] / C, U[1] / C)
    v_step = (V[0] / R, V[1] / R)

    for r in range(R):
        for c in range(C):
            if not occ[r][c]:
                continue
            p00 = (A[0] + c * u_step[0] + r * v_step[0], A[1] + c * u_step[1] + r * v_step[1])
            p10 = (p00[0] + u_step[0], p00[1] + u_step[1])
            p11 = (p10[0] + v_step[0], p10[1] + v_step[1])
            p01 = (p00[0] + v_step[0], p00[1] + v_step[1])
            _poly(d, [p00, p10, p11, p01], outline_w=LINE_W, fill_rgba=fill_rgba)
    return img

# ----------------------------- shapes we already had -----------------------------
def render_staircase_squares(w: int, h: int, n: int,
                             transparent: bool = False,
                             fill_rgba: Optional[Tuple[int,int,int,int]] = None) -> Image.Image:
    img = _transparent(w, h) if transparent else _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)

    side = min(w, h) - 2 * pad
    unit = side / (n + 1)
    x0 = (w - n * unit) / 2
    y0 = h - pad

    if fill_rgba is not None:
        _poly(d, [(x0, y0 - n * unit), (x0 + n * unit, y0), (x0, y0)], outline_w=LINE_W, fill_rgba=fill_rgba)

    for r in range(1, n + 1):
        y_top = y0 - r * unit
        y_bot = y0 - (r - 1) * unit
        for c in range(r):
            x_left = x0 + c * unit
            x_right = x_left + unit
            _poly(d, [(x_left, y_top), (x_right, y_top), (x_right, y_bot), (x_left, y_bot)], outline_w=LINE_W)

    _line(d, [(x0, y0 - n * unit), (x0 + n * unit, y0)])
    return img

def render_triangular_lattice(w: int, h: int, n: int,
                              transparent: bool = False,
                              fill_rgba: Optional[Tuple[int,int,int,int]] = None) -> Image.Image:
    """Proper triangular lattice (fixes the earlier single-filled-triangle bug)."""
    img = _transparent(w, h) if transparent else _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)

    side = min(w, h) - 2 * pad
    A = (w / 2 - side / 2, h / 2 + side * math.sqrt(3) / 6)
    B = (w / 2 + side / 2, h / 2 + side * math.sqrt(3) / 6)
    C = (w / 2,            h / 2 - side * math.sqrt(3) / 3)

    def lerp(P, Q, t): return (P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1]))

    _poly(d, [A, B, C], outline_w=LINE_W, fill_rgba=fill_rgba)

    for i in range(1, n):
        t = i / n
        _line(d, [lerp(C, A, t), lerp(C, B, t)])
        _line(d, [lerp(A, B, t), lerp(C, B, t)])
        _line(d, [lerp(A, B, 1 - t), lerp(C, A, t)])
    return img

# ----------------------------- inscribed alternation (ask only "squares") -----------------------------
def render_inscribed_alt_square_circle(
    w: int, h: int, k: int, start: str = "square",
    min_inner_px: int = MIN_INNER_PX,
    transparent: bool = False,
    keep_mask: Optional[List[bool]] = None,
) -> Tuple[Image.Image, int, int, int]:
    img = _transparent(w, h) if transparent else _white_rgba(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)
    cx, cy = w / 2.0, h / 2.0
    safe = min(w, h) / 2.0 - pad

    m_inner = max(int(min_inner_px), 6 * LINE_W, int(round(MIN_INNER_FRAC * min(w, h))))

    if start == "square":
        s_tmp = 2.0 * safe; r_tmp = s_tmp / 2.0
    else:
        r_tmp = safe; s_tmp = math.sqrt(2.0) * r_tmp

    cur_tmp = start; k_eff = 0
    for _ in range(k):
        ok = (s_tmp >= m_inner) if cur_tmp == "square" else (2.0 * r_tmp >= m_inner)
        if not ok: break
        k_eff += 1
        if cur_tmp == "square":
            r_tmp = s_tmp / 2.0; cur_tmp = "circle"
        else:
            s_tmp = math.sqrt(2.0) * r_tmp; cur_tmp = "square"
        s_tmp *= 0.999; r_tmp *= 0.999

    if keep_mask is None:
        keep_mask = [True] * k_eff
    else:
        keep_mask = (keep_mask + [True] * k_eff)[:k_eff]

    if start == "square":
        s = 2.0 * safe; r = s / 2.0; cur = "square"
    else:
        r = safe; s = math.sqrt(2.0) * r; cur = "circle"
    n_sq = 0; n_ci = 0

    for i in range(k_eff):
        draw_this = keep_mask[i]
        if cur == "square":
            if draw_this:
                x0, y0 = cx - s / 2.0, cy - s / 2.0
                x1, y1 = cx + s / 2.0, cy + s / 2.0
                _poly(ImageDraw.Draw(img, "RGBA"), [(x0,y0),(x1,y0),(x1,y1),(x0,y1)], outline_w=LINE_W)
                n_sq += 1
            r = s / 2.0; cur = "circle"
        else:
            if draw_this:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0, 255), width=LINE_W)
                n_ci += 1
            s = math.sqrt(2.0) * r; cur = "square"
        s *= 0.999; r *= 0.999

    return img, n_sq, n_ci, k_eff

# ----------------------------- decorative containers (transparent layers) -----------------------------
def container_triangle(w: int, h: int, fill_rgba: Tuple[int,int,int,int]) -> Image.Image:
    img = _transparent(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)
    side = min(w, h) - 2 * pad
    A = (w / 2 - side / 2, h / 2 + side * math.sqrt(3) / 6)
    B = (w / 2 + side / 2, h / 2 + side * math.sqrt(3) / 6)
    C = (w / 2,            h / 2 - side * math.sqrt(3) / 3)
    _poly(d, [A, B, C], outline_w=max(5, LINE_W), fill_rgba=fill_rgba)
    return img

def container_circle(w: int, h: int, fill_rgba: Tuple[int,int,int,int]) -> Image.Image:
    img = _transparent(w, h)
    d = ImageDraw.Draw(img, "RGBA")
    pad = _safe_pad(w, h)
    r = min(w, h) / 2 - pad
    cx, cy = w/2, h/2
    d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=(0,0,0,255), width=max(5, LINE_W), fill=fill_rgba)
    return img

# ----------------------------- polyomino helpers -----------------------------
def _build_polyomino_cells(
    rng: random.Random, base_r: int, base_c: int, n_extras: int
) -> Set[Tuple[int,int]]:
    """Start with base_r×base_c block at [0..r-1]×[0..c-1], then attach n_extras rectangles along sides."""
    cells: Set[Tuple[int,int]] = set((r, c) for r in range(base_r) for c in range(base_c))

    for _ in range(n_extras):
        side = rng.choice(["left", "right", "top", "bottom"])
        if side in ("left", "right"):
            h = rng.randint(1, base_r)
            y0 = rng.randint(0, base_r - h)
            w = rng.randint(1, min(2, base_c))  # modest
            if side == "left":
                x0 = -w
                for rr in range(y0, y0 + h):
                    for cc in range(x0, 0):
                        cells.add((rr, cc))
            else:
                x0 = base_c
                for rr in range(y0, y0 + h):
                    for cc in range(x0, x0 + w):
                        cells.add((rr, cc))
        else:
            w = rng.randint(1, base_c)
            x0 = rng.randint(0, base_c - w)
            h = rng.randint(1, min(2, base_r))
            if side == "top":
                y0 = -h
                for rr in range(y0, 0):
                    for cc in range(x0, x0 + w):
                        cells.add((rr, cc))
            else:  # bottom
                y0 = base_r
                for rr in range(y0, y0 + h):
                    for cc in range(x0, x0 + w):
                        cells.add((rr, cc))
    return cells

def _cells_to_occ(cells: Set[Tuple[int,int]]) -> List[List[bool]]:
    minr = min(r for r, _ in cells); maxr = max(r for r, _ in cells)
    minc = min(c for _, c in cells); maxc = max(c for _, c in cells)
    R = maxr - minr + 1
    C = maxc - minc + 1
    occ = [[False] * C for _ in range(R)]
    for (r, c) in cells:
        occ[r - minr][c - minc] = True
    return occ

# ----------------------------- spec + task -----------------------------
@dataclass
class ShapeCountSpec:
    seed: int
    canvas: Tuple[int, int]
    family: str            # grid | skew | staircase | tri_lattice | inscribed | polyomino | poly_skew
    ask_shape: str         # rectangles | squares | parallelograms | triangles
    params: Dict[str, Any]
    answer: int

@register_task
class ShapeCountTask(Task):
    """
    “How many X are in the figure?” with verifiable counts.

    New knobs:
      • ASK_SHAPE_WEIGHTS — weight which shape to ask.
      • SHAPE_SAMPLER_WEIGHTS — weight samplers within each shape (irregulars favored by default).
      • IRREGULARITY_WEIGHT_MULT — extra global upweight for irregular samplers.
      • POLY_EXTRAS_RANGE / POLY_PROTRUSION_BIAS — bias polyominoes toward more attachments.

    Mirrors the weighted-selection style used in tiles tasks. Irregular samplers include line‑drop grids,
    polyomino (axis & skew), and inscribed with dropped layers. Regular samplers include perfect grids,
    staircase, and triangular lattice. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    """
    name = "shape_count"

    def __init__(self, W: int = 640, H: int = 480,
                 count_min: int = COUNT_MIN_DEFAULT, count_max: int = COUNT_MAX_DEFAULT,
                 ask_shape_weights: Optional[Dict[str, float]] = None,
                 sampler_weights: Optional[Dict[str, Dict[str, float]]] = None,
                 irregularity_weight_mult: Optional[float] = None,
                 poly_extras_range: Optional[Tuple[int,int]] = None,
                 poly_protrusion_bias: Optional[float] = None):
        self.W, self.H = int(W), int(H)
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.cmin = int(count_min)
        self.cmax = int(count_max)
        if self.cmin < 1 or self.cmax < self.cmin:
            raise ValueError("Invalid (count_min, count_max) for ShapeCountTask")

        # Copy defaults and apply optional overrides
        self.ask_weights: Dict[str, float] = dict(ASK_SHAPE_WEIGHTS)
        if ask_shape_weights:
            for k, v in ask_shape_weights.items():
                if k in self.ask_weights:
                    self.ask_weights[k] = float(v)

        self.sampler_weights: Dict[str, Dict[str, float]] = {
            shape: dict(wmap) for shape, wmap in SHAPE_SAMPLER_WEIGHTS.items()
        }
        if sampler_weights:
            for shape, wmap in sampler_weights.items():
                if shape in self.sampler_weights and isinstance(wmap, dict):
                    for key, val in wmap.items():
                        self.sampler_weights[shape][key] = float(val)

        self.irreg_mult = float(irregularity_weight_mult if irregularity_weight_mult is not None
                                else IRREGULARITY_WEIGHT_MULT)

        self.poly_extras_range = tuple(poly_extras_range) if poly_extras_range is not None else POLY_EXTRAS_RANGE
        self.poly_protrusion_bias = float(poly_protrusion_bias if poly_protrusion_bias is not None
                                          else POLY_PROTRUSION_BIAS)

    def _compute_complexity(self, count: int) -> Dict[str, Any]:
        """Normalize counted shapes to [0,1] and provide a categorical level."""
        min_count = int(getattr(self, "cmin", COUNT_MIN_DEFAULT))
        max_count = int(getattr(self, "cmax", COUNT_MAX_DEFAULT))
        span = max(1, max_count - min_count)
        normalized = (int(count) - min_count) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "shape-count-answer-v1",
            "range": {"min_count": min_count, "max_count": max_count},
            "count": int(count),
        }

    # ------------------- small helper: bias extras toward protruding -------------------
    def _sample_poly_extras(self, rng: random.Random) -> int:
        lo, hi = self.poly_extras_range
        if lo > hi: lo, hi = hi, lo
        k = rng.randint(int(lo), int(hi))
        # extra nudge toward more attachments if possible
        if k < hi and rng.random() < max(0.0, min(1.0, self.poly_protrusion_bias)):
            k += 1
        return int(min(k, hi))

    # ------------------- new connected samplers -------------------
    def _sample_polyomino(self, rng: random.Random):
        """Axis-aligned polyomino (base rectangle with 1–3+ extras). May yield squares or rectangles."""
        for _ in range(64):
            base_r = rng.randint(2, 5)
            base_c = rng.randint(2, 5)
            extras = self._sample_poly_extras(rng)
            cells = _build_polyomino_cells(rng, base_r, base_c, extras)
            occ = _cells_to_occ(cells)

            sq = _count_squares_occ(occ)
            rc = _count_rectangles_occ(occ)

            choices: List[Tuple[str, int]] = []
            if self.cmin <= sq <= self.cmax and sq > 0:
                choices.append(("squares", sq))
            if self.cmin <= rc <= self.cmax and rc > 0:
                choices.append(("rectangles", rc))
            if not choices:
                continue

            ask, ans = rng.choice(choices)
            fill = _pick_soft_rgba(rng, 1)[0]
            img = _render_occupancy(self.W, self.H, occ, fill_rgba=fill)
            return img, ask, int(ans), {
                "family": "polyomino",
                "base": (base_r, base_c),
                "extras": int(extras),
                "occ_rows": len(occ), "occ_cols": len(occ[0]),
                "irregular": True,
            }, "polyomino"
        return None

    def _sample_polyparallelogram(self, rng: random.Random):
        """Skew polyomino (parallelogram grid) with extras; ask parallelograms."""
        for _ in range(64):
            base_r = rng.randint(2, 5)   # rows along V
            base_c = rng.randint(2, 5)   # cols along U
            extras = self._sample_poly_extras(rng)
            cells = _build_polyomino_cells(rng, base_r, base_c, extras)
            occ = _cells_to_occ(cells)

            # In (u,v): counting rectangles == counting parallelograms in XY.
            para = _count_rectangles_occ(occ)
            if not (self.cmin <= para <= self.cmax and para > 0):
                continue

            shear = rng.uniform(0.18, 0.32)
            fill = _pick_soft_rgba(rng, 1)[0]
            img = _render_skew_occupancy(self.W, self.H, occ, shear=shear, fill_rgba=fill)
            return img, "parallelograms", int(para), {
                "family": "poly_skew",
                "base": (base_r, base_c), "extras": int(extras),
                "shear": float(shear),
                "occ_rows": len(occ), "occ_cols": len(occ[0]),
                "irregular": True,
            }, "poly_skew"
        return None

    # ------------------- irregular variants (line drop on perfect grids) -------------------
    def _sample_grid_rectangles_irregular(self, rng: random.Random):
        cands = [(r, c) for r in range(2, 8) for c in range(3, 9)]
        rng.shuffle(cands)
        for rows, cols in cands:
            p_drop = rng.uniform(*GRID_LINE_DROP_PROB_RANGE)
            keep_rows = [True] * (rows + 1)
            keep_cols = [True] * (cols + 1)
            for i in range(1, rows):
                keep_rows[i] = rng.random() >= p_drop
            for j in range(1, cols):
                keep_cols[j] = rng.random() >= p_drop
            if all(keep_rows[1:-1]) and rows > 2: keep_rows[rng.randrange(1, rows)] = False
            if all(keep_cols[1:-1]) and cols > 2: keep_cols[rng.randrange(1, cols)] = False
            if sum(keep_rows) < 2 or sum(keep_cols) < 2:
                continue
            fill = _pick_soft_rgba(rng, 1)[0]
            img, meta = render_axis_grid_square_cells(self.W, self.H, rows, cols,
                                                      transparent=False, fill_rgba=fill,
                                                      keep_row_lines=keep_rows, keep_col_lines=keep_cols)
            ans = int(meta["rectangles_irregular"])
            if self.cmin <= ans <= self.cmax and ans > 0:
                return img, "rectangles", ans, {
                    "rows": rows, "cols": cols,
                    "drop_prob": p_drop,
                    "kept_rows": meta["kept_rows"], "kept_cols": meta["kept_cols"],
                    "irregular": True,
                }, "grid"
        return None

    def _sample_grid_squares_irregular(self, rng: random.Random):
        cands = [n for n in range(3, 9)]
        rng.shuffle(cands)
        for n in cands:
            p_drop = rng.uniform(*GRID_LINE_DROP_PROB_RANGE)
            keep_rows = [True] * (n + 1)
            keep_cols = [True] * (n + 1)
            for i in range(1, n):
                keep_rows[i] = rng.random() >= p_drop
                keep_cols[i] = rng.random() >= p_drop
            if all(keep_rows[1:-1]): keep_rows[rng.randrange(1, n)] = False
            if all(keep_cols[1:-1]): keep_cols[rng.randrange(1, n)] = False
            if sum(keep_rows) < 2 or sum(keep_cols) < 2:
                continue
            fill = _pick_soft_rgba(rng, 1)[0]
            img, meta = render_axis_grid_square_cells(self.W, self.H, n, n,
                                                      transparent=False, fill_rgba=fill,
                                                      keep_row_lines=keep_rows, keep_col_lines=keep_cols)
            ans = meta["squares_irregular"]
            if ans is None:
                continue
            ans = int(ans)
            if self.cmin <= ans <= self.cmax and ans > 0:
                return img, "squares", ans, {
                    "n": n,
                    "drop_prob": p_drop,
                    "kept_rows": meta["kept_rows"], "kept_cols": meta["kept_cols"],
                    "irregular": True,
                }, "grid"
        return None

    def _sample_skew_parallelograms_irregular(self, rng: random.Random):
        for nu in rng.sample(list(range(3, 8)), k=len(range(3,8))):
            for nv in rng.sample(list(range(3, 8)), k=len(range(3,8))):
                p_drop = rng.uniform(*GRID_LINE_DROP_PROB_RANGE)
                keep_u = [True] * nu
                keep_v = [True] * nv
                for i in range(1, nu - 1):
                    keep_u[i] = rng.random() >= p_drop
                for j in range(1, nv - 1):
                    keep_v[j] = rng.random() >= p_drop
                if all(keep_u[1:-1]) and nu > 2: keep_u[rng.randrange(1, nu-1)] = False
                if all(keep_v[1:-1]) and nv > 2: keep_v[rng.randrange(1, nv-1)] = False
                if sum(keep_u) < 2 or sum(keep_v) < 2:
                    continue
                shear = rng.uniform(0.15, 0.35)
                fill = _pick_soft_rgba(rng, 1)[0]
                img, meta = render_parallelogram_grid(self.W, self.H, nu, nv, shear,
                                                      transparent=False, fill_rgba=fill,
                                                      keep_u_lines=keep_u, keep_v_lines=keep_v)
                ans = int(meta["parallelograms_irregular"])
                if self.cmin <= ans <= self.cmax and ans > 0:
                    return img, "parallelograms", ans, {
                        "nu": nu, "nv": nv, "shear": shear,
                        "drop_prob": p_drop,
                        "kept_u": meta["kept_u"], "kept_v": meta["kept_v"],
                        "irregular": True,
                    }, "skew"
        return None

    # ------------------- regular fallbacks (connected) -------------------
    def _sample_grid_squares(self, rng: random.Random):
        cands = [(n, count_squares_in_grid(n, n)) for n in range(2, 10) if self.cmin <= count_squares_in_grid(n,n) <= self.cmax]
        if not cands: return None
        n, ans = rng.choice(cands)
        fill = _pick_soft_rgba(rng, 1)[0]
        img, _ = render_axis_grid_square_cells(self.W, self.H, n, n, transparent=False, fill_rgba=fill)
        return img, "squares", ans, {"rows": n, "cols": n, "fill": True, "irregular": False}, "grid"

    def _sample_grid_rectangles(self, rng: random.Random):
        cands = []
        for r in range(2, 7):
            for c in range(3, 9):
                ans = count_rectangles_in_grid(r, c)
                if self.cmin <= ans <= self.cmax:
                    cands.append((r, c, ans))
        if not cands: return None
        r, c, ans = rng.choice(cands)
        fill = _pick_soft_rgba(rng, 1)[0]
        img, _ = render_axis_grid_square_cells(self.W, self.H, r, c, transparent=False, fill_rgba=fill)
        return img, "rectangles", ans, {"rows": r, "cols": c, "fill": True, "irregular": False}, "grid"

    def _sample_skew_parallelograms(self, rng: random.Random):
        cands = []
        for nu in range(3, 8):
            for nv in range(3, 8):
                ans = count_parallelograms_in_skew(nu, nv)
                if self.cmin <= ans <= self.cmax:
                    cands.append((nu, nv, ans))
        if not cands: return None
        nu, nv, ans = rng.choice(cands)
        shear = rng.uniform(0.15, 0.35)
        fill = _pick_soft_rgba(rng, 1)[0]
        img, _ = render_parallelogram_grid(self.W, self.H, nu, nv, shear, transparent=False, fill_rgba=fill)
        return img, "parallelograms", ans, {"nu": nu, "nv": nv, "shear": shear, "fill": True, "irregular": False}, "skew"

    def _sample_staircase(self, rng: random.Random):
        cands = [(n, count_squares_in_staircase(n)) for n in range(2, 8) if self.cmin <= count_squares_in_staircase(n) <= self.cmax]
        if not cands: return None
        n, ans = rng.choice(cands)
        fill = _pick_soft_rgba(rng, 1)[0]
        img = render_staircase_squares(self.W, self.H, n, transparent=False, fill_rgba=fill)
        return img, "squares", ans, {"rows": n, "fill": True, "irregular": False}, "staircase"

    def _sample_triangular(self, rng: random.Random):
        """Proper triangular lattice."""
        cands = [(n, count_triangles_in_triangular_lattice(n)) for n in range(2, 8)
                 if self.cmin <= count_triangles_in_triangular_lattice(n) <= self.cmax]
        if not cands: return None
        n, ans = rng.choice(cands)
        fill = _pick_soft_rgba(rng, 1)[0]
        img = render_triangular_lattice(self.W, self.H, n, transparent=False, fill_rgba=fill)
        return img, "triangles", ans, {"n": n, "fill": True, "irregular": False}, "tri_lattice"

    def _sample_inscribed(self, rng: random.Random) -> Optional[Tuple[Image.Image, str, int, Dict[str, Any], str]]:
        """Inscribed square↔circle; ask about squares only; circles are decorative."""
        start = rng.choice(["square", "circle"])
        k_target = rng.randint(max(5, self.cmin), min(24, self.cmax))
        _, _, _, k_eff = render_inscribed_alt_square_circle(self.W, self.H, k_target, start=start,
                                                            min_inner_px=MIN_INNER_PX, transparent=True, keep_mask=None)
        if k_eff == 0:
            return None
        for _try in range(32):
            p_drop = rng.uniform(*INSCRIBE_DROP_PROB_RANGE)
            keep_mask = [rng.random() >= p_drop for _ in range(k_eff)]
            if not any(keep_mask):
                keep_mask[rng.randrange(k_eff)] = True
            cur = start; sq_vis = 0
            for i in range(k_eff):
                if keep_mask[i] and cur == "square":
                    sq_vis += 1
                cur = "circle" if cur == "square" else "square"
            if not (self.cmin <= sq_vis <= self.cmax and sq_vis > 0):
                continue
            img, n_sq, n_ci, k_eff2 = render_inscribed_alt_square_circle(
                self.W, self.H, k_target, start=start, min_inner_px=MIN_INNER_PX,
                transparent=False, keep_mask=keep_mask
            )
            if n_sq == sq_vis:
                return img, "squares", int(n_sq), {
                    "k_target": int(k_target), "k_effective": int(k_eff2),
                    "start": start, "drop_prob": float(p_drop),
                    "mask_bits": "".join("1" if b else "0" for b in keep_mask),
                    "n_circles_visible": int(n_ci), "irregular": True
                }, "inscribed"
        return None

    # ----------------------------- weighted routing -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        """
        1) Sample *which shape to ask* using ASK_SHAPE_WEIGHTS.
        2) For that shape, sample a sampler family using SHAPE_SAMPLER_WEIGHTS,
           with an extra boost for irregular families via IRREGULARITY_WEIGHT_MULT.
        3) Run the sampler until we get an instance that matches the chosen shape and bounds.
        """
        # Registry per asked-shape: (key, is_irregular, callable)
        sampler_registry: Dict[str, List[Tuple[str, bool, Callable[[random.Random], Optional[Tuple]]]]] = {
            "rectangles": [
                ("grid_irregular", True,  self._sample_grid_rectangles_irregular),
                ("grid_regular",   False, self._sample_grid_rectangles),
                ("polyomino_axis", True,  self._sample_polyomino),
            ],
            "squares": [
                ("grid_irregular", True,  self._sample_grid_squares_irregular),
                ("grid_regular",   False, self._sample_grid_squares),
                ("staircase",      False, self._sample_staircase),
                ("inscribed",      True,  self._sample_inscribed),
                ("polyomino_axis", True,  self._sample_polyomino),
            ],
            "parallelograms": [
                ("skew_irregular", True,  self._sample_skew_parallelograms_irregular),
                ("skew_regular",   False, self._sample_skew_parallelograms),
                ("polyomino_skew", True,  self._sample_polyparallelogram),
            ],
            "triangles": [
                ("tri_lattice",    False, self._sample_triangular),
            ],
        }

        shape_names = list(self.ask_weights.keys())
        shape_w = [self.ask_weights[s] for s in shape_names]

        for _ in range(self.max_retries):
            seed = rng.randrange(2**31 - 1)
            lrng = random.Random(seed)

            ask_target = choice_weighted(lrng, shape_names, shape_w)

            # Build weighted list of candidate samplers for this shape
            entries = sampler_registry.get(ask_target, [])
            if not entries:
                continue
            keys = [k for (k, _, _) in entries]
            base = [self.sampler_weights.get(ask_target, {}).get(k, 1.0) for k in keys]
            irr  = [ir for (_, ir, _) in entries]
            weights = [w * (self.irreg_mult if ir else 1.0) for w, ir in zip(base, irr)]

            # Try a handful of attempts drawing samplers by weight
            attempts = max(6, 2 * len(entries))
            for _try in range(attempts):
                ksel = choice_weighted(lrng, keys, weights)
                sm = dict((k, fn) for (k, _, fn) in entries)[ksel]
                out = sm(lrng)
                if out is None:
                    continue
                img, ask, ans, params, fam = out
                # Enforce selected shape (polyomino can swap between rectangles/squares)
                if ask != ask_target:
                    continue
                if not (self.cmin <= int(ans) <= self.cmax):
                    continue

                # Build spec + meta (shape-specific prompt)
                spec = ShapeCountSpec(
                    seed=seed,
                    canvas=(img.width, img.height),
                    family=fam,
                    ask_shape=ask,
                    params=dict(params),
                    answer=int(ans),
                )
                complexity = self._compute_complexity(int(ans))
                question = lrng.choice(PROMPTS_BY_SHAPE[ask])
                meta = {
                    "pattern_kind": "geometry",
                    "pattern": self.name,
                    "question": question,
                    "answer": str(int(ans)),
                    "dims": (img.width, img.height),
                    "family": fam,
                    "params": dict(params),
                    "count_bounds": [self.cmin, self.cmax],
                    "composite_ready": True,
                    "routing": {
                        "ask_target": ask_target,
                        "sampler_key": ksel,
                        "irregularity_boost": float(self.irreg_mult),
                    },
                }
                meta["complexity"] = complexity
                meta["complexity_score"] = complexity["score"]
                meta["complexity_level"] = complexity["level"]
                meta["complexity_version"] = complexity["version"]
                return img.convert("RGB"), [spec], meta

        raise RuntimeError(
            f"{self.name}: failed to build a valid sample in [{self.cmin},{self.cmax}] after {self.max_retries} attempts."
        )
