# sphinx/charts/rendering.py
from __future__ import annotations
import math
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

try:
    from sphinx.utils.drawing import load_font as _load_font_project  # type: ignore
except Exception:
    _load_font_project = None

from .common import ChartSpec

# ------------------------------ knobs ------------------------------
MIN_FONT_PX        = 4
INSIDE_LABEL_PX    = 20    # letter in slice (used only if show_values=True)
INSIDE_PERCENT_PX  = 16    # percent in slice (used only if show_values=True)
INSIDE_LINE_GAP_PX = 6     # vertical gap between letter and percent
SMALL_WEDGE_DEG    = 1.0   # show single line in ultra-thin wedges
SMALL_K_FOR_INSIDE = 6     # inside text allowed only when show_values=True and k <= this
MAX_LEGEND_FRAC    = 0.80  # legend may take up to 80% of canvas width

# Fine alignment: move legend text up a hair to match chip midlines
LEGEND_TEXT_Y_NUDGE = -1   # tweak to taste: -2, -1, 0

# ------------------------------ font utils ------------------------------

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf", "NotoSans-Regular.ttf",
    "arial.ttf", "Arial.ttf", "calibri.ttf", "Tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/tahoma.ttf",
]

def _try_truetype(size: int) -> Optional["ImageFont.ImageFont"]:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return None

def _is_scalable_font(f: "ImageFont.ImageFont") -> bool:
    return isinstance(f, getattr(ImageFont, "FreeTypeFont", ImageFont.ImageFont))

def _get_font(sz: int) -> "ImageFont.ImageFont":
    px = max(MIN_FONT_PX, int(sz))
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
        if f2 is not None and _is_scalable_font(f2):
            return f2
    return ImageFont.load_default()

def _text_size(font: "ImageFont.ImageFont", text: str) -> Tuple[int, int]:
    x0, y0, x1, y1 = font.getbbox(text)
    return (x1 - x0), (y1 - y0)

# Text helpers: vertically center with anchor; fallback to bbox if anchors unsupported
def _draw_left_centered(draw: ImageDraw.ImageDraw, x: int, y_center: int, text: str,
                        font: ImageFont.ImageFont, fill="black", y_nudge: int = 0):
    try:
        draw.text((x, y_center + y_nudge), text, font=font, fill=fill, anchor="lm")  # left, middle
    except TypeError:
        tw, th = _text_size(font, text)
        draw.text((x, y_center - th // 2 + y_nudge), text, font=font, fill=fill)

def _draw_right_centered(draw: ImageDraw.ImageDraw, x_right: int, y_center: int, text: str,
                         font: ImageFont.ImageFont, fill="black", y_nudge: int = 0):
    try:
        draw.text((x_right, y_center + y_nudge), text, font=font, fill=fill, anchor="rm")  # right, middle
    except TypeError:
        tw, th = _text_size(font, text)
        draw.text((x_right - tw, y_center - th // 2 + y_nudge), text, font=font, fill=fill)

# ------------------------------ legend layout (right side) ------------------------------

def _legend_layout(
    W: int, H: int, pad: int, labels: List[str], perc: List[int],
    start_frac: float, show_values: bool
) -> Tuple[int, "ImageFont.ImageFont", int, int, int, int, int, int]:
    """
    Compute legend width & font so nothing clips. Legend is ALWAYS on the RIGHT.
    If show_values=False, legend shows [chip] [label] only.
    Returns:
      legend_w, f_legend, em_h, chip, line_h, col_w_label, col_w_dash, col_w_pct
    """
    rows = len(labels)
    legend_w = max(160, int(start_frac * W))
    max_legend_w = int(MAX_LEGEND_FRAC * W)

    # fit vertically first
    target_row_h = max(14, int((H - 2 * pad) / max(1, rows)))
    fsize = max(MIN_FONT_PX, int(target_row_h * 0.46))
    f_legend = _get_font(fsize)

    def measure_cols() -> Tuple[int, int, int, int]:
        em_h = _text_size(f_legend, "Mg")[1] or fsize
        col_w_label = max(_text_size(f_legend, lab)[0] for lab in labels) if labels else 0
        col_w_dash  = _text_size(f_legend, "—")[0] if show_values else 0
        col_w_pct   = max(_text_size(f_legend, f"{p}%")[0] for p in perc) if (show_values and perc) else 0
        return em_h, col_w_label, col_w_dash, col_w_pct

    def measure_layout() -> Tuple[int, int, int, int, int]:
        em_h, col_w_label, col_w_dash, col_w_pct = measure_cols()
        chip = int(0.64 * em_h)
        line_h = max(chip + 6, em_h + 6)
        return em_h, chip, line_h, col_w_label, col_w_pct, col_w_dash  # (we return dash last but unpack below)

    em_h, chip, line_h, col_w_label, col_w_pct, col_w_dash = measure_layout()

    # vertical fit
    while (line_h * rows + 2 * pad) > H and fsize > MIN_FONT_PX:
        fsize -= 1
        f_legend = _get_font(fsize)
        em_h, chip, line_h, col_w_label, col_w_pct, col_w_dash = measure_layout()

    # horizontal fit
    gap1 = 8; gap2 = 8; gap3 = 8; slack = 12
    def needed_width() -> int:
        w = pad + chip + gap1 + col_w_label + pad + slack
        if show_values:
            w = pad + chip + gap1 + col_w_label + gap2 + col_w_dash + gap3 + col_w_pct + pad + slack
        return w

    while True:
        need_w = needed_width()
        if need_w <= legend_w:
            break
        if legend_w < max_legend_w:
            legend_w = min(max_legend_w, int(need_w))
            continue
        if fsize <= MIN_FONT_PX:
            break
        fsize -= 1
        f_legend = _get_font(fsize)
        em_h, chip, line_h, col_w_label, col_w_pct, col_w_dash = measure_layout()

    return legend_w, f_legend, em_h, chip, line_h, col_w_label, col_w_dash, col_w_pct

# ------------------------------ PIE --------------------------------------

# --- REPLACE the whole function ---
def render_pie_chart(spec: ChartSpec, *, show_values: bool = True, show_legend: bool = True) -> Image.Image:
    """
    Colored pie chart with distinct slice colors.

    Legend is on the RIGHT when show_legend=True (vertical, centered, column aligned).
    If show_values=False:
      • No text is drawn inside the pie (no letters, no numbers).
      • If show_legend=True, legend shows [chip] [label] only (no percents).
    If show_values=True:
      • For k <= SMALL_K_FOR_INSIDE, letters (and percents) may be drawn inside.
      • If show_legend=True, legend shows [chip] [label — percent].
    """
    W, H = spec.width_px, spec.height_px
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)

    k = len(spec.labels)
    base = min(W, H)
    pad = int(0.08 * base)

    # Legend (optional)
    if show_legend:
        (legend_w, f_legend, em_h, chip, line_h,
         col_w_label, col_w_dash, col_w_pct) = _legend_layout(
            W, H, pad, spec.labels, spec.percentages_int, start_frac=0.42, show_values=show_values
        )
    else:
        legend_w = 0
        # Safe dummies so code below can branch cleanly
        f_legend = _get_font(max(MIN_FONT_PX, 12))
        em_h = chip = line_h = col_w_label = col_w_dash = col_w_pct = 0

    # Pie geometry (left/center)
    cx = (W - legend_w) // 2
    cy = H // 2
    r = max(1, min(cx - pad, cy - pad))

    start_deg = -90.0
    angles: List[Tuple[float, float]] = []
    for p in spec.percentages_int:
        sweep = 360.0 * (p / 100.0)
        angles.append((start_deg, start_deg + sweep))
        start_deg += sweep
    if angles:
        a0, _ = angles[-1]
        angles[-1] = (a0, 270.0)

    # Draw slices
    box = [cx - r, cy - r, cx + r, cy + r]
    for (i, (a0, a1)) in enumerate(angles):
        draw.pieslice(box, start=a0, end=a1, fill=spec.colors[i], outline="black")
    draw.ellipse(box, outline="black", width=2)
    for (a0, _a1) in angles[:-1]:
        th = math.radians(a0)
        draw.line([(cx, cy), (cx + int(r * math.cos(th)), cy + int(r * math.sin(th)))], fill="black", width=1)

    # Optional inside text (only if show_values=True and small-k)
    draw_inside = (show_values and k <= SMALL_K_FOR_INSIDE)
    if draw_inside:
        fL = _get_font(max(MIN_FONT_PX, INSIDE_LABEL_PX))
        fP = _get_font(max(MIN_FONT_PX, INSIDE_PERCENT_PX))
        r_text = r * 0.52
        gap = max(1, int(INSIDE_LINE_GAP_PX))
        for i, ((a0, a1), p) in enumerate(zip(angles, spec.percentages_int)):
            angle_deg = (a1 - a0) % 360.0
            mid = math.radians((a0 + a1) * 0.5)
            tx = int(cx + r_text * math.cos(mid))
            ty = int(cy + r_text * math.sin(mid))
            lab = spec.labels[i]
            lw, lh = _text_size(fL, lab)
            pct_text = f"{int(p)}%"
            if angle_deg < SMALL_WEDGE_DEG:
                pw, ph = _text_size(fP, pct_text)
                draw.text((tx - pw // 2, ty - ph // 2), pct_text, fill="black", font=fP)
            else:
                pw, ph = _text_size(fP, pct_text)
                w = max(lw, pw); h = lh + gap + ph
                draw.text((tx - w // 2 + (w - lw) // 2, ty - h // 2), lab, fill="black", font=fL)
                draw.text((tx - w // 2 + (w - pw) // 2, ty - h // 2 + lh + gap), pct_text, fill="black", font=fP)

    # Legend (RIGHT) only when enabled
    if show_legend:
        rows = len(spec.labels)
        total_legend_h = line_h * rows
        y = pad + max(0, (H - 2 * pad - total_legend_h) // 2)

        x0 = W - legend_w + pad
        sw = int(0.72 * em_h); sh = sw

        gap1 = 8; gap2 = 8; gap3 = 8
        x_chip = x0
        x_label = x_chip + sw + gap1
        x_dash  = x_label + col_w_label + gap2
        x_pct_l = x_dash + col_w_dash + gap3
        x_pct_r = x_pct_l + col_w_pct

        for lab, col, p in zip(spec.labels, spec.colors, spec.percentages_int):
            row_center = y + line_h // 2

            draw.rectangle([x_chip, row_center - sh // 2, x_chip + sw, row_center + sh // 2],
                           fill=col, outline="black")
            _draw_left_centered(draw, x_label, row_center, lab, f_legend, y_nudge=LEGEND_TEXT_Y_NUDGE)

            if show_values:
                _draw_left_centered(draw, x_dash, row_center, "—", f_legend, y_nudge=LEGEND_TEXT_Y_NUDGE)
                _draw_right_centered(draw, x_pct_r, row_center, f"{p}%", f_legend, y_nudge=LEGEND_TEXT_Y_NUDGE)
            y += line_h

    return im


# ------------------------------ BAR --------------------------------------

# --- REPLACE the whole function ---
def render_bar_chart(spec: ChartSpec, *, show_values: bool = False, show_labels: bool = True) -> Image.Image:
    W, H = spec.width_px, spec.height_px
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)

    # Extra top pad so labels drawn above bars don't clip
    pad = int(0.10 * min(W, H))
    pad_top = int(0.12 * H)

    axis_y = H - pad
    axis_x0 = pad
    axis_x1 = W - pad

    # X-axis
    draw.line([axis_x0, axis_y, axis_x1, axis_y], fill="black", width=2)

    k = len(spec.labels)
    gap = 10
    avail_w = axis_x1 - axis_x0
    bar_w = max(18, int((avail_w - (k + 1) * gap) / max(1, k)))

    base = min(W, H)
    f_label = _get_font(max(MIN_FONT_PX, int(0.024 * base)))
    f_value = _get_font(max(MIN_FONT_PX, int(0.020 * base)))

    for i, (lab, col, p) in enumerate(zip(spec.labels, spec.colors, spec.percentages_int)):
        x = axis_x0 + gap + i * (bar_w + gap)
        max_h = axis_y - pad_top
        h = int(round(max_h * (p / 100.0)))
        y0 = axis_y - h
        y1 = axis_y

        # Bar (can be distinct-colored or mono; depends on spec.colors)
        draw.rectangle([x, y0, x + bar_w, y1], fill=col, outline="black", width=2)

        # Optional label above the bar (letters only)
        if show_labels:
            tw, th = _text_size(f_label, lab)
            ty_lab = max(2, y0 - th - 4)
            draw.text((x + (bar_w - tw) // 2, ty_lab), lab, fill="black", font=f_label)

        # Optional numeric values just above the label
        if show_values:
            vs = f"{p}%"
            vw, vh = _text_size(f_value, vs)
            # If labels are hidden, pin value near the top of the bar
            if show_labels:
                ty_val = max(2, (y0 - vh - 2) - (_text_size(f_label, lab)[1] + 2))
            else:
                ty_val = max(2, y0 - vh - 4)
            draw.text((x + (bar_w - vw) // 2, ty_val), vs, fill="black", font=f_value)

    return im
