# sphinx/geometry/common.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# Constants & utilities
# ------------------------------------------------------------

MIN_RELATIVE_GAP = 0.10  # |vi - vj| / max(vi, vj) >= 10%

# Robust font loader patterned after charts/rendering.py
try:
    from sphinx.utils.drawing import load_font as _load_font_project  # type: ignore
except Exception:
    _load_font_project = None

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf", "NotoSans-Regular.ttf",
    "arial.ttf", "Arial.ttf", "calibri.ttf", "Tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/tahoma.ttf",
]

def _try_truetype(sz: int) -> Optional[ImageFont.ImageFont]:
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size=max(4, int(sz)))
        except Exception:
            continue
    return None

def _get_font(px: int) -> ImageFont.ImageFont:
    px = max(4, int(px))
    f = _try_truetype(px)
    if f is not None:
        return f
    if _load_font_project is not None:
        try:
            f2 = _load_font_project(px)
        except TypeError:
            try:
                f2 = _load_font_project()
            except Exception:
                f2 = None
        except Exception:
            f2 = None
        if f2 is not None and isinstance(f2, getattr(ImageFont, "FreeTypeFont", ImageFont.ImageFont)):
            return f2
    return ImageFont.load_default()

def _text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    x0, y0, x1, y1 = font.getbbox(text)
    return x1 - x0, y1 - y0


# ------------------------------------------------------------
# Graph-paper background
# ------------------------------------------------------------

def graph_paper_rgb(
    width: int, height: int, *,
    minor: int = 32, major: int = 128,
    minor_rgb: Tuple[int, int, int] = (232, 232, 232),
    major_rgb: Tuple[int, int, int] = (192, 192, 192),
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    im = Image.new("RGB", (width, height), bg_rgb)
    dr = ImageDraw.Draw(im)

    # verticals
    x = 0
    while x < width:
        color = major_rgb if (x % major) == 0 else minor_rgb
        dr.line([(x, 0), (x, height)], fill=color, width=1)
        x += minor

    # horizontals
    y = 0
    while y < height:
        color = major_rgb if (y % major) == 0 else minor_rgb
        dr.line([(0, y), (width, y)], fill=color, width=1)
        y += minor

    # thin outer border
    b = max(2, width // 320)
    dr.rectangle([1, 1, width - 2, height - 2], outline=(0, 0, 0), width=b)
    return im


# ------------------------------------------------------------
# Labels (letters)
# ------------------------------------------------------------

def sample_labels(rng: random.Random, k: int) -> List[str]:
    import string
    letters = list(string.ascii_uppercase)
    rng.shuffle(letters)
    return letters[:k]


# ------------------------------------------------------------
# Relative-gap sampler
# ------------------------------------------------------------

def _valid_relative_gap(values: List[float], rel_gap: float) -> bool:
    if len(set(round(v, 9) for v in values)) != len(values):
        return False
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = abs(values[i]), abs(values[j])
            m = max(a, b)
            if m == 0:
                return False
            if abs(a - b) / m < rel_gap:
                return False
    return True

def sample_values_with_relative_gap(
    rng: random.Random, k: int,
    vmin: float, vmax: float,
    rel_gap: float = MIN_RELATIVE_GAP,
) -> List[float]:
    """
    Return k values that satisfy  |a-b| / max(a,b) >= rel_gap  for every pair.
    We build a clean geometric sequence with ratio r < 1, which guarantees
    the constraint for all pairs. If the requested [vmin, vmax] range is
    mathematically infeasible given (k, rel_gap), we *minimally relax vmin*
    so a valid sequence exists. This removes dead-ends and assertion failures.
    """
    vmin = float(vmin); vmax = float(vmax)
    assert vmax > 0 and vmin > 0 and vmax > vmin and k >= 2

    # The largest allowable adjacent ratio to honor the gap (with a safety margin).
    EPS  = 1e-4
    r_hi = min(0.995, 1.0 - float(rel_gap) - EPS)   # ensure r_hi < 1

    # To fit k values in [vmin, vmax] with ratio <= r_hi, we must have:
    #   vmax * (r_hi)^(k-1) >= vmin.
    # If not, lower the effective vmin just enough to be feasible.
    vmin_eff = min(vmin, vmax * (r_hi ** (k - 1)))
    r_low = (vmin_eff / vmax) ** (1.0 / (k - 1))    # feasible lower bound on r

    # Choose an r comfortably between [r_low, r_hi]; midpoint avoids border cases.
    r = 0.5 * (max(0.10, r_low) + r_hi)

    # Construct strictly decreasing values (no jitter => no accidental violations).
    vals = [vmax * (r ** i) for i in range(k)]

    # Final safety: validate and, if needed (extremely unlikely), tighten r a hair.
    if not _valid_relative_gap(vals, rel_gap):
        r = max(0.10, r_hi * (1.0 - 1e-3))
        vals = [vmax * (r ** i) for i in range(k)]
        # At this point, the construction is guaranteed feasible.

    return vals



# ------------------------------------------------------------
# Rendered object
# ------------------------------------------------------------

@dataclass
class RenderedShape:
    label: str
    prop_kind: str      # 'length' | 'angle_deg' | 'area' | 'perimeter'
    prop_value: float
    patch: Image.Image  # RGBA
    size: Tuple[int, int]
    meta: Dict[str, float]


def _paste_letter(draw: ImageDraw.ImageDraw, where: Tuple[int, int], text: str, font_px: int):
    f = _get_font(font_px)
    tw, th = _text_size(f, text)
    x, y = where
    draw.text((int(x - tw // 2), int(y - th // 2)), text, fill="black", font=f)

# ------------------------------------------------------------
# Line (colored stroke; label under lower endpoint)
# ------------------------------------------------------------

def render_line_patch(
    length_px: float,
    orientation_deg: float,
    label: str,
    *,
    stroke_px: int = 4,
    stroke_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
    margin_px: int = 16,
    label_offset_px: int = 10,
    draw_label: bool = True,
    label_font_px: Optional[int] = None,
) -> RenderedShape:
    L = float(length_px)
    theta = math.radians(float(orientation_deg))
    dx = L * math.cos(theta)
    dy = L * math.sin(theta)

    xs = [0.0, dx]; ys = [0.0, dy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = int(math.ceil(max_x - min_x)) + 2 * margin_px + stroke_px
    H = int(math.ceil(max_y - min_y)) + 2 * margin_px + stroke_px

    tx = margin_px - min_x
    ty = margin_px - min_y
    x0, y0 = int(round(0 + tx)), int(round(0 + ty))
    x1, y1 = int(round(dx + tx)), int(round(dy + ty))

    im = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.line([(x0, y0), (x1, y1)], fill=stroke_rgba, width=max(1, int(stroke_px)))

    bx, by = (x0, y0) if y0 > y1 else (x1, y1)  # bottom endpoint
    ly = min(H - 6, by + label_offset_px)
    font_px_suggest = max(10, int(0.22 * max(W, H)))
    if draw_label:
        _paste_letter(dr, (bx, ly), label, font_px=label_font_px or font_px_suggest)

    meta = {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "length": L, "theta_deg": float(orientation_deg),
        "label_x": float(bx), "label_y": float(ly),
        "label_font_px_suggest": float(font_px_suggest),
    }
    return RenderedShape(label=label, prop_kind="length", prop_value=L, patch=im, size=(im.width, im.height), meta=meta)

# ------------------------------------------------------------
# Angle (colored rays + arc; label above/below vertex, not inside)
# ------------------------------------------------------------

def render_angle_patch(
    angle_deg: float,
    radius_px: float,
    label: str,
    *,
    bisector_deg: Optional[float] = None,
    stroke_px: int = 4,
    stroke_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
    margin_px: int = 18,
    show_arc: bool = True,
    draw_label: bool = True,
    label_font_px: Optional[int] = None,
    label_offset_px: Optional[int] = None,  # vertical distance from vertex
) -> RenderedShape:
    ang = max(5.0, min(175.0, float(angle_deg)))
    R = float(radius_px)
    phi = math.radians(float(bisector_deg) if bisector_deg is not None else random.uniform(0, 360))
    half = math.radians(ang / 2.0)
    a1 = phi - half
    a2 = phi + half

    p1 = (R * math.cos(a1), R * math.sin(a1))
    p2 = (R * math.cos(a2), R * math.sin(a2))
    xs = [0.0, p1[0], p2[0]]; ys = [0.0, p1[1], p2[1]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = int(math.ceil(max_x - min_x)) + 2 * margin_px + stroke_px
    H = int(math.ceil(max_y - min_y)) + 2 * margin_px + stroke_px
    tx = margin_px - min_x
    ty = margin_px - min_y

    Vx, Vy = int(round(0 + tx)), int(round(0 + ty))
    x1, y1 = int(round(p1[0] + tx)), int(round(p1[1] + ty))
    x2, y2 = int(round(p2[0] + tx)), int(round(p2[1] + ty))

    im = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.line([(Vx, Vy), (x1, y1)], fill=stroke_rgba, width=max(2, int(stroke_px)))
    dr.line([(Vx, Vy), (x2, y2)], fill=stroke_rgba, width=max(2, int(stroke_px)))

    if show_arc:
        rr = max(10, int(0.35 * R))
        start = math.degrees(a1) % 360
        end   = math.degrees(a2) % 360
        bbox = [Vx - rr, Vy - rr, Vx + rr, Vy + rr]
        dr.arc(bbox, start=start, end=end, fill=stroke_rgba, width=max(1, int(stroke_px * 0.75)))

    # letter ABOVE or BELOW the vertex (never inside)
    off = int(label_offset_px) if label_offset_px is not None else max(12, int(0.30 * R))
    # choose the side with more room within the patch
    room_up = Vx >= 0 and (Vy - off) >= (margin_px // 2)
    room_down = (Vy + off) <= (H - margin_px // 2)
    if room_up and (not room_down or (Vy - off) > (H - Vy - off)):
        lx, ly = Vx, max(3, Vy - off)
    else:
        lx, ly = Vx, min(H - 6, Vy + off)

    font_px_suggest = max(10, int(0.22 * max(W, H)))
    if draw_label:
        _paste_letter(dr, (lx, ly), label, font_px=label_font_px or font_px_suggest)

    meta = {
        "vertex_x": Vx, "vertex_y": Vy,
        "angle_deg": float(ang), "radius_px": R,
        "label_x": float(lx), "label_y": float(ly),
        "label_font_px_suggest": float(font_px_suggest),
    }
    return RenderedShape(label=label, prop_kind="angle_deg", prop_value=float(ang),
                         patch=im, size=(im.width, im.height), meta=meta)

# ------------------------------------------------------------
# Regular polygon (transparent fill; colored outline; letter at center)
# ------------------------------------------------------------

def _regular_polygon_vertices(n: int, R: float, rot_deg: float = 0.0) -> List[Tuple[float, float]]:
    verts = []
    rot = math.radians(rot_deg)
    for i in range(n):
        a = rot + 2 * math.pi * i / n
        verts.append((R * math.cos(a), R * math.sin(a)))
    return verts

def polygon_area_from_R(n: int, R: float) -> float:
    return 0.5 * n * (R ** 2) * math.sin(2 * math.pi / n)

def polygon_perimeter_from_R(n: int, R: float) -> float:
    return 2.0 * n * R * math.sin(math.pi / n)

def render_regular_polygon_patch(
    n_sides: int,
    R_px: float,
    label: str,
    *,
    rotation_deg: float = 0.0,
    fill_rgba: Tuple[int, int, int, int] = (0, 0, 0, 0),          # transparent
    outline_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
    outline_px: int = 3,
    draw_label: bool = True,
    label_font_px: Optional[int] = None,
) -> RenderedShape:
    n = max(3, int(n_sides))
    R = float(R_px)
    verts = _regular_polygon_vertices(n, R, rotation_deg)

    xs = [x for (x, y) in verts]; ys = [y for (x, y) in verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(8, int(max(4, outline_px) * 2))
    W = int(math.ceil(max_x - min_x)) + 2 * pad
    H = int(math.ceil(max_y - min_y)) + 2 * pad
    tx = pad - min_x
    ty = pad - min_y

    im = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    poly = [(x + tx, y + ty) for (x, y) in verts]
    dr.polygon(poly, fill=fill_rgba, outline=outline_rgba)
    if outline_px > 1:
        dr.line(poly + [poly[0]], fill=outline_rgba, width=outline_px)

    cx, cy = W // 2, H // 2
    font_px_suggest = max(12, int(0.38 * min(W, H)))
    if draw_label:
        _paste_letter(dr, (cx, cy), label, font_px=label_font_px or font_px_suggest)

    area = polygon_area_from_R(n, R)
    peri = polygon_perimeter_from_R(n, R)
    meta = {
        "n": n, "R": R, "rotation_deg": rotation_deg,
        "area": area, "perimeter": peri,
        "label_x": float(cx), "label_y": float(cy),
        "label_font_px_suggest": float(font_px_suggest),
    }
    return RenderedShape(label=label, prop_kind="area", prop_value=area,
                         patch=im, size=(im.width, im.height), meta=meta)

# ------------------------------------------------------------
# Ellipse (transparent fill; colored outline; letter at center)
# ------------------------------------------------------------

def ellipse_circumference_ramanujan_k(e_ratio: float) -> float:
    return math.pi * (3.0 * (1.0 + e_ratio) - math.sqrt((3.0 + e_ratio) * (1.0 + 3.0 * e_ratio)))

def render_ellipse_patch(
    a_px: float, b_px: float,
    label: str,
    *,
    rotation_deg: float = 0.0,
    fill_rgba: Tuple[int, int, int, int] = (0, 0, 0, 0),          # transparent
    outline_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
    outline_px: int = 3,
    draw_label: bool = True,
    label_font_px: Optional[int] = None,
) -> RenderedShape:
    a = float(a_px); b = float(b_px)
    W0 = int(math.ceil(2 * a)) + 2 * outline_px + 8
    H0 = int(math.ceil(2 * b)) + 2 * outline_px + 8
    im0 = Image.new("RGBA", (max(1, W0), max(1, H0)), (0, 0, 0, 0))
    dr0 = ImageDraw.Draw(im0)
    bbox = [outline_px + 4, outline_px + 4, W0 - outline_px - 4, H0 - outline_px - 4]
    dr0.ellipse(bbox, fill=fill_rgba, outline=outline_rgba, width=max(1, int(outline_px)))

    if abs(rotation_deg) > 1e-3:
        im = im0.rotate(rotation_deg, expand=True, resample=Image.BICUBIC)
    else:
        im = im0

    dr = ImageDraw.Draw(im)
    cx, cy = im.width // 2, im.height // 2
    font_px_suggest = max(12, int(0.36 * min(im.width, im.height)))
    if draw_label:
        _paste_letter(dr, (cx, cy), label, font_px=label_font_px or font_px_suggest)

    # perimeter approximation (Ramanujan I)
    C = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
    A = math.pi * a * b
    meta = {
        "a": a, "b": b, "rotation_deg": rotation_deg,
        "area": A, "perimeter": C,
        "label_x": float(cx), "label_y": float(cy),
        "label_font_px_suggest": float(font_px_suggest),
    }
    return RenderedShape(label=label, prop_kind="area", prop_value=A,
                         patch=im, size=(im.width, im.height), meta=meta)
