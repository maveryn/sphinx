# sphinx/utils/drawing.py
import random
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageChops
from typing import List, Optional, Dict, Tuple, Union
from collections import Counter
from ..config import (
    OUT_CELL, BORDER_PX, SEP_PX, LABEL_PAD, BORDER_COLOR, SEP_COLOR, SUPERSAMPLE,
    LABEL_COLOR, FONT_SIZE, SS_CELL, CANVAS_BG
)


Color = Union[str, int, Tuple[int, int, int], Tuple[int, int, int, int]]


def paste_rgba(canvas: Image.Image, patch: Image.Image, xy: Tuple[int, int]):
    if patch.mode == "RGBA":
        canvas.alpha_composite(patch, xy)
    else:
        canvas.paste(patch, xy)


def _down_on_background(rgba: Image.Image, out_size: int = OUT_CELL, bg: Optional[Color] = None) -> Image.Image:
    """Downscale ``rgba`` to ``out_size`` and composite onto ``bg``."""
    down = rgba.convert("RGBA").resize((out_size, out_size), Image.LANCZOS)

    if bg is None:
        canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
        canvas.alpha_composite(down)
        return canvas

    if isinstance(bg, tuple) and len(bg) == 4:
        canvas = Image.new("RGBA", (out_size, out_size), bg)
        canvas.alpha_composite(down)
        return canvas

    canvas = Image.new("RGB", (out_size, out_size), bg)
    canvas.paste(down, mask=down.split()[-1])
    return canvas

def load_font():
    for ft in ("DejaVuSans-Bold.ttf", "arial.ttf"):
        try: return ImageFont.truetype(ft, FONT_SIZE)
        except Exception: pass
    return ImageFont.load_default()

def crisp_option_tile(img: Image.Image, lbl: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    # Keep the tile size and white label strip exactly as before
    tile = Image.new("RGB", (OUT_CELL, OUT_CELL + LABEL_PAD), "white")
    tile.paste(img, (0, 0))

    d = ImageDraw.Draw(tile)

    # Draw the label centered in the bottom strip
    bbox = d.textbbox((0, 0), lbl, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((OUT_CELL - w) // 2, OUT_CELL + (LABEL_PAD - h) // 2),
           lbl, fill=LABEL_COLOR, font=font)

    return tile

def strip_outer_frame(img: Image.Image, strip_px: int | None = None,
                      fill=(255, 255, 255)) -> Image.Image:
    """Overpaint the outer border area with `fill` to remove any baked-in frame."""
    r = img.convert("RGB").copy()
    w, h = r.size
    s = int(strip_px if strip_px is not None else max(2, max(w, h) // 160))
    d = ImageDraw.Draw(r)
    d.rectangle([0,      0,      w-1,   s-1], fill=fill)  # top
    d.rectangle([0,      h-s,    w-1,   h-1], fill=fill)  # bottom
    d.rectangle([0,      0,      s-1,   h-1], fill=fill)  # left
    d.rectangle([w-s,    0,      w-1,   h-1], fill=fill)  # right
    return r


def _blank_cell_same_style() -> Image.Image:
    """Blank tile that matches the exact border look of real cells."""
    big = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
    border_ss = max(3, SS_CELL // 160)
    ImageDraw.Draw(big).rectangle([0, 0, SS_CELL - 1, SS_CELL - 1],
                                  outline=(0, 0, 0, 255), width=border_ss)
    white = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 255))
    return Image.alpha_composite(white, big).resize((OUT_CELL, OUT_CELL), Image.LANCZOS).convert("RGBA")

def compose_2x2_grid(cell_imgs: List[Image.Image], mask_idx: int) -> Image.Image:
    grid = Image.new("RGB", (OUT_CELL*2, OUT_CELL*2), "white")
    for i, img in enumerate(cell_imgs):
        grid.paste(img, ((i % 2) * OUT_CELL, (i // 2) * OUT_CELL))
    # mask one cell
    mx, my = (mask_idx % 2) * OUT_CELL, (mask_idx // 2) * OUT_CELL
    ImageDraw.Draw(grid).rectangle([mx, my, mx + OUT_CELL, my + OUT_CELL], fill="black")
    # grid lines + outer border
    draw = ImageDraw.Draw(grid)

    b = int(BORDER_PX)
    # vertical separator (center)
    draw.rectangle([OUT_CELL - b, 0, OUT_CELL - 1, OUT_CELL * 2 - 1], fill=BORDER_COLOR)
    # horizontal separator (center)
    draw.rectangle([0, OUT_CELL - b, OUT_CELL * 2 - 1, OUT_CELL - 1], fill=BORDER_COLOR)

    grid = ImageOps.expand(grid, border=BORDER_PX, fill=BORDER_COLOR)
    return grid

def compose_options_2x2(option_tiles):
    """
    Lay out 4 distinct option tiles as:
        0 1
        2 3
    (a) (b) on the top row, (c) (d) on the bottom row.
    """
    assert len(option_tiles) == 4, "compose_options_2x2 expects exactly 4 tiles"
    w, h = option_tiles[0].size
    # sanity: all same size
    for t in option_tiles[1:]:
        assert t.size == (w, h), "All option tiles must have equal size"

    # background with separators
    W = 2 * w + SEP_PX
    H = 2 * h + SEP_PX
    bg = Image.new("RGBA", (W, H), SEP_COLOR + (255,) if len(SEP_COLOR) == 3 else SEP_COLOR)

    # paste each tile in its correct slot
    bg.paste(option_tiles[0], (0,          0))
    bg.paste(option_tiles[1], (w + SEP_PX, 0))
    bg.paste(option_tiles[2], (0,          h + SEP_PX))
    bg.paste(option_tiles[3], (w + SEP_PX, h + SEP_PX))
    return bg

def compose_left_right(grid: Image.Image, options: Image.Image) -> Image.Image:
    comp_w = grid.width + SEP_PX + options.width
    comp_h = max(grid.height, options.height)
    comp = Image.new("RGB", (comp_w, comp_h), CANVAS_BG)
    comp.paste(grid, (0, (comp_h - grid.height) // 2))
    comp.paste(options, (grid.width + SEP_PX, 0))
    ImageDraw.Draw(comp).rectangle([grid.width, 0, grid.width + SEP_PX - 1, comp_h], fill=SEP_COLOR)
    return comp

def _draw_aa_arrow_row(base: Image.Image, x_pairs, y_mid,
                       head_w, head_h, shaft_w, color=(0,0,0,220), ss=3):
    """
    Anti-aliased (by super-sampling) arrows for each (x0, x1) pair at y=y_mid.
    Drawn on a transparent overlay and composited onto base.
    """
    W, H = base.size
    OW, OH = W*ss, H*ss
    overlay = Image.new("RGBA", (OW, OH), (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    def S(x, y):  # scale to oversampled coords
        return (int(round(x*ss)), int(round(y*ss)))

    for (x0, x1) in x_pairs:
        y = y_mid
        # shrink a touch so arrows don't collide with frames
        inset = max(2.0, (x1 - x0) * 0.05)
        ax0 = x0 + inset
        ax1 = x1 - inset

        # shaft
        d.line([S(ax0, y), S(ax1 - head_w, y)], fill=color, width=max(1, int(shaft_w*ss)))

        # head
        tip    = S(ax1, y)
        left   = S(ax1 - head_w, y - head_h)
        right  = S(ax1 - head_w, y + head_h)
        d.polygon([tip, left, right], fill=color)

    # downsample to base and composite
    overlay_ds = overlay.resize((W, H), Image.LANCZOS)
    base.alpha_composite(overlay_ds)
    return base

def _blank_cell_tile(w: int, h: int, border_px: int = 3) -> Image.Image:
    """OUT_CELL-sized white tile with a crisp black border."""
    tile = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    d = ImageDraw.Draw(tile)
    d.rectangle([1, 1, w - 2, h - 2], outline=(0, 0, 0), width=max(1, border_px))
    return tile


def _pad_width_center(im: Image.Image, target_w: int) -> Image.Image:
    if not target_w or target_w <= im.width:
        return im
    canvas = Image.new("RGB", (target_w, im.height), CANVAS_BG)
    x = (target_w - im.width) // 2
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    canvas.paste(im, (x, 0), im)  # use alpha mask
    return canvas


def tight_crop_rgba(img: Image.Image, white_threshold: int = 6, pad: int = 0) -> Image.Image:
    """Tightly crop to visible content, preserving alpha."""
    r = img.convert("RGBA")
    w, h = r.size

    a = r.getchannel("A")
    a_min, a_max = a.getextrema()
    bbox = None
    if a_min < 255:
        mask_a = a.point(lambda p: 255 if p > 0 else 0)
        bbox = mask_a.getbbox()

    if bbox is None:
        rgb = r.convert("RGB")
        corners = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((w - 1, 0)),
            rgb.getpixel((0, h - 1)),
            rgb.getpixel((w - 1, h - 1)),
        ]
        bg = Image.new("RGB", (w, h), (
            sum(c[0] for c in corners) // 4,
            sum(c[1] for c in corners) // 4,
            sum(c[2] for c in corners) // 4,
        ))
        diff = ImageChops.difference(rgb, bg)
        ch_r, ch_g, ch_b = diff.split()
        mx = ImageChops.lighter(ImageChops.lighter(ch_r, ch_g), ch_b)
        for thr in (white_threshold, max(1, white_threshold - 2), max(1, white_threshold - 4)):
            mask = mx.point(lambda p, t=thr: 255 if p > t else 0)
            bbox = mask.getbbox()
            if bbox:
                break

    if bbox is None:
        return r

    if pad:
        x0, y0, x1, y1 = bbox
        bbox = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))

    return r.crop(bbox)


def ensure_transparent(img: Image.Image, bg: Optional[Tuple[int, int, int]] = None, tol: int = 3) -> Image.Image:
    """Convert opaque backgrounds to transparency, preserving edges."""
    rgba = img.convert("RGBA")
    alpha = rgba.split()[-1]
    if alpha.getextrema() != (255, 255):
        return rgba

    if bg is None:
        rgb = rgba.convert("RGB")
        W, H = rgb.size
        corners = [rgb.getpixel((0, 0)), rgb.getpixel((W - 1, 0)),
                   rgb.getpixel((0, H - 1)), rgb.getpixel((W - 1, H - 1))]
        bg = Counter(corners).most_common(1)[0][0]

    bg_img = Image.new("RGB", rgba.size, bg)
    diff = ImageChops.difference(rgba.convert("RGB"), bg_img)
    gray = ImageOps.grayscale(diff)
    new_alpha = gray.point(lambda p, t=tol: 0 if p <= t else p)
    rgba.putalpha(new_alpha)
    return rgba


def ensure_cell_rgba_outcell(im: Image.Image) -> Image.Image:
    """Return a RGBA image scaled to ``OUT_CELL`` square dimensions."""
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    if im.size != (OUT_CELL, OUT_CELL):
        im = im.resize((OUT_CELL, OUT_CELL), Image.LANCZOS)
    return im


def graph_paper_bg(
    size: int,
    grid_step: Optional[int] = None,
    grid_color=(200, 200, 200, 255),
    axis_color=(120, 120, 120, 255),
    grid_width: int = 1,
    axis_width: Optional[int] = None,
) -> Image.Image:
    """
    Create a crisp graph-paper tile: white background, 1px gray grid,
    and slightly thicker X/Y axes through the center.

    - grid_width: thickness of grid lines (px)
    - axis_width: thickness of axes (px); default keeps axes just one pixel
      thicker than grid (now slimmer than before).
    """
    W = H = int(size)
    if grid_step is None:
        grid_step = max(12, W // 16)
    if axis_width is None:
        # Reduced relative thickness; used to be ~3.
        axis_width = max(grid_width + 1, 2)

    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    dr = ImageDraw.Draw(img)

    # Grid lines (crisp, no AA): draw as 1px (or grid_width) rectangles
    for x in range(0, W, grid_step):
        dr.rectangle([x, 0, x + grid_width - 1, H - 1], fill=grid_color)
    for y in range(0, H, grid_step):
        dr.rectangle([0, y, W - 1, y + grid_width - 1], fill=grid_color)

    # Center axes (slightly thicker than grid)
    cx = W // 2
    cy = H // 2

    # Vertical axis
    v_left = cx - (axis_width // 2)
    v_right = v_left + axis_width - 1
    dr.rectangle([v_left, 0, v_right, H - 1], fill=axis_color)

    # Horizontal axis
    h_top = cy - (axis_width // 2)
    h_bottom = h_top + axis_width - 1
    dr.rectangle([0, h_top, W - 1, h_bottom], fill=axis_color)

    return img

def compose_row_with_mask(
    cell_imgs: List[Image.Image],
    mask_idx: int,
    target_width: Optional[int] = None,
    draw_arrows: bool = False,
    arrow_style: Optional[Dict] = None,
    gap: int = 16,
    margin: int = 0,
    mask_style: str = "blank",          # "blank" | "black"
) -> Image.Image:
    """
    Arrange a horizontal row of OUT_CELL tiles with one masked position.

    - cell_imgs: list of OUT_CELL-sized tiles (RGBA).
    - mask_idx: index in [0..len(cell_imgs)-1] to hide.
    - target_width: if provided and > row width, center-pad row to this width.
    - draw_arrows: draw small chevrons in the gaps between tiles.
    - arrow_style: {"alpha": 220, "scale": 1.2, "color": (220,220,220,220)} (any keys optional).
    - gap: pixels between tiles.
    - margin: outer left/right margin around the row.
    - mask_style: "blank" → white tile with black border; "black" → solid black block.

    Returns an RGBA image.
    """
    assert len(cell_imgs) >= 1, "compose_row_with_mask: need at least one tile"
    tile_w, tile_h = cell_imgs[0].size
    n = len(cell_imgs)

    # compute row size
    row_w = margin * 2 + n * tile_w + (n - 1) * gap
    row_h = tile_h
    row = Image.new("RGBA", (row_w, row_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(row)

    # arrow defaults
    a = arrow_style or {}
    a_alpha = int(a.get("alpha", 220))
    a_scale = float(a.get("scale", 1.0))
    a_color = a.get("color", (60,60,60, a_alpha))

    x = margin
    for i, im in enumerate(cell_imgs):
        if i == mask_idx:
            if mask_style == "blank":
                ph = _blank_cell_same_style()
                if ph.size != (tile_w, tile_h):
                    ph = ph.resize((tile_w, tile_h), Image.LANCZOS)
                row.alpha_composite(ph, (x, 0))
            else:  # solid black placeholder
                blk = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 255))
                row.alpha_composite(blk, (x, 0))
        else:
            # draw the actual (non-masked) cell
            row.alpha_composite(im, (x, 0))

        x += tile_w

        # draw chevron in the gap (except after last tile)
        if i < n - 1:
            if draw_arrows:
                # chevron size relative to gap & tile height
                aw = max(6, int(gap * 0.55 * a_scale))
                ah = max(6, int(tile_h * 0.18 * a_scale))
                cx = x + gap // 2
                cy = tile_h // 2
                # triangle pointing right
                pts = [(cx - aw // 2, cy - ah // 2),
                       (cx - aw // 2, cy + ah // 2),
                       (cx + aw // 2, cy)]
                d.polygon(pts, fill=a_color)
            x += gap

    # center-pad to target width if requested
    if target_width and target_width > row_w:
        row = _pad_width_center(row, target_width)

    return row

def compose_top_bottom(top_img: Image.Image, bottom_img: Image.Image, sep_px: int = SEP_PX):
    W = max(top_img.width, bottom_img.width)
    H = top_img.height + sep_px + bottom_img.height

    out = Image.new("RGB", (W, H), CANVAS_BG)

    # paste with alpha mask if present
    if top_img.mode != "RGBA":
        top_img = top_img.convert("RGBA")
    out.paste(top_img, (0, 0), top_img)

    d = ImageDraw.Draw(out)
    d.rectangle([0, top_img.height, W - 1, top_img.height + sep_px - 1], fill=SEP_COLOR)

    if bottom_img.mode != "RGBA":
        bottom_img = bottom_img.convert("RGBA")
    out.paste(bottom_img, (0, top_img.height + sep_px), bottom_img)

    return out

def compose_options_row(tiles, sep=SEP_PX):
    """Place exactly 4 option tiles in a single horizontal row."""
    # assert len(tiles) == 4
    w, h = tiles[0].size
    W = w*4 + sep*3
    H = h
    canvas = Image.new("RGB", (W, H), CANVAS_BG)
    x = 0
    for i, t in enumerate(tiles):
        canvas.paste(t, (x, 0))
        if i < 3:
            ImageDraw.Draw(canvas).rectangle([x + w, 0, x + w + sep - 1, H - 1], fill=SEP_COLOR)
        x += w + sep
    return canvas

def add_tile_border(im: Image.Image, width_px: int | None = None) -> Image.Image:
    """Draw a crisp black border just inside the tile bounds."""
    out = im.convert("RGBA").copy()
    w, h = out.size
    b = int(width_px or max(2, max(w, h) // 160))   # similar thickness rule as sequence
    ImageDraw.Draw(out).rectangle([0, 0, w - 1, h - 1], outline=(0, 0, 0, 255), width=b)
    return out


def compose_row_aligned(cells, mask_idx: int, total_width: int, sep_px: int = SEP_PX) -> Image.Image:
    """Top row with the same total width/column positions as the options row."""
    ow, oh = cells[0].size
    W, H = total_width, oh
    row = Image.new("RGB", (W, H), CANVAS_BG)
    d = ImageDraw.Draw(row)

    d.rectangle([0, 0, sep_px - 1, H - 1], fill=SEP_COLOR)
    x = sep_px
    for i, cell in enumerate(cells):
        row.paste(cell, (x, 0))
        x += ow
        d.rectangle([x, 0, x + sep_px - 1, H - 1], fill=SEP_COLOR)
        x += sep_px
    return row

def _arrow_segment(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, gutter: int):
    """Draw a short arrow centered vertically in the separator gutter [x0..x1]."""
    pad   = max(6, gutter // 6)         # left/right pad inside the gutter
    head  = max(6, gutter // 6)         # arrow head length
    linew = max(4, BORDER_PX * 2)

    sx = x0 + pad
    ex = x1 - pad
    draw.line([(sx, y), (ex, y)], fill="white", width=linew)
    draw.polygon([(ex, y), (ex - head, y - head // 2), (ex - head, y + head // 2)], fill="white")


def _bbox(polys: List[List[Tuple[float, float]]]):
    xs, ys = [], []
    for poly in polys:
        for x, y in poly:
            xs.append(x); ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)

def _quant(p: Tuple[float, float], scale: float = 1e6) -> Tuple[int, int]:
    # robust dedup key for edges coming from separate faces that share coordinates
    return (int(round(p[0] * scale)), int(round(p[1] * scale)))

def render_patch_crisp(
    patch,
    cell_colors: List[Tuple[int, int, int] | Tuple[int, int, int, int] | str],
    size_px: int = OUT_CELL,
    background: str | Tuple[int, int, int] | Tuple[int, int, int, int] | None = "white",
    outline_rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
    outline_px: int = 1,
    supersample: int = SUPERSAMPLE,
    anchor: str = "random",  # "top" | "left" | "right" | "down" | "center" | "random"
):
    """
    High-quality renderer for polygonal tilings:
      - fills per-cell colors
      - strokes every unique edge exactly once
      - supersamples then downsamples for anti-aliased diagonals
      - special ellipse path for near-circular faces
    """
    assert supersample >= 1
    polys = patch.cell_polygons()

    # ---- layout mapping (tiling coords -> hi-res pixels)
    minx, miny, maxx, maxy = _bbox(polys)
    W, H = maxx - minx, maxy - miny
    out = size_px
    hi = out * supersample
    pad = max(8 * supersample, hi // 24)

    avail_w = hi - 2 * pad
    avail_h = hi - 2 * pad
    sx = (avail_w) / (W if W > 0 else 1.0)
    sy = (avail_h) / (H if H > 0 else 1.0)
    s = min(sx, sy)
    tile_w = W * s
    tile_h = H * s

    # choose offsets based on anchor
    a = (anchor or "random").lower()
    if a == "center":
        tx = pad + (avail_w - tile_w) / 2.0
        ty = pad + (avail_h - tile_h) / 2.0
    elif a == "left":
        tx = pad; ty = pad + (avail_h - tile_h) / 2.0
    elif a == "right":
        tx = pad + (avail_w - tile_w); ty = pad + (avail_h - tile_h) / 2.0
    elif a == "top":
        tx = pad + (avail_w - tile_w) / 2.0; ty = pad
    elif a == "down":
        tx = pad + (avail_w - tile_w) / 2.0; ty = pad + (avail_h - tile_h)
    else:
        free_w = max(0.0, avail_w - tile_w)
        free_h = max(0.0, avail_h - tile_h)
        tx = pad + (random.random() * free_w if free_w > 0 else 0.0)
        ty = pad + (random.random() * free_h if free_h > 0 else 0.0)

    def x_to_px(x: float) -> int:
        return int(round(tx + (x - minx) * s))
    def y_to_px(y: float) -> int:
        return int(round(ty + (maxy - y) * s))  # flip Y
    def to_px(p: Tuple[float, float]) -> Tuple[int, int]:
        return (x_to_px(p[0]), y_to_px(p[1]))

    # Heuristic: detect near-circles and render as true ellipses
    def _is_near_circle(poly, min_n=24, tol_rel=0.01):
        if len(poly) < min_n:
            return False, None
        cx = sum(x for x,_ in poly) / len(poly)
        cy = sum(y for _,y in poly) / len(poly)
        rsq = [(x-cx)*(x-cx) + (y-cy)*(y-cy) for (x,y) in poly]
        rmean = sum(rsq)/len(rsq)
        if rmean <= 1e-12:
            return False, None
        var = sum((r - rmean)**2 for r in rsq) / len(rsq)
        rel = (var / (rmean*rmean)) ** 0.5
        if rel <= tol_rel:
            xs = [x for x,_ in poly]; ys = [y for _,y in poly]
            return True, (min(xs), min(ys), max(xs), max(ys))
        return False, None

    # ---- background
    if isinstance(background, str) and background.lower() == "graph_paper":
        im = graph_paper_bg(hi).convert("RGBA")
    elif background is None:
        im = Image.new("RGBA", (hi, hi), (0, 0, 0, 0))
    else:
        bg = background
        if isinstance(bg, tuple) and len(bg) == 3:
            bg = (bg[0], bg[1], bg[2], 255)
        im = Image.new("RGBA", (hi, hi), bg)
    dr = ImageDraw.Draw(im, "RGBA")

    # ---- fill polygons
    circle_cells = {}
    for ci, poly in enumerate(polys):
        is_circ, bbox = _is_near_circle(poly)
        if is_circ:
            x0, y0, x1, y1 = bbox
            X0, X1 = x_to_px(x0), x_to_px(x1)
            Yt, Yb = y_to_px(y1), y_to_px(y0)  # top, bottom
            dr.ellipse([X0, Yt, X1, Yb], fill=cell_colors[ci])
            circle_cells[ci] = (X0, Yt, X1, Yb)
        else:
            dr.polygon([to_px(p) for p in poly], fill=cell_colors[ci])

    # ---- dedupe & stroke edges exactly once
    unique_edges: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[Tuple[float, float], Tuple[float, float]]
    ] = {}
    for ci, poly in enumerate(polys):
        if ci in circle_cells:
            continue
        m = len(poly)
        for k in range(m):
            p0, p1 = poly[k], poly[(k+1) % m]
            a0, b0 = _quant(p0), _quant(p1)
            key = (a0, b0) if a0 <= b0 else (b0, a0)
            if key not in unique_edges:
                unique_edges[key] = (p0, p1)

    lw = max(2, int(round(outline_px * supersample)))
    for (p0, p1) in unique_edges.values():
        dr.line([to_px(p0), to_px(p1)], fill=outline_rgba, width=lw)

    # ellipse outlines for circle cells
    for (X0, Yt, X1, Yb) in circle_cells.values():
        dr.ellipse([X0, Yt, X1, Yb], outline=outline_rgba, width=lw)

    # ---- outer frame (flush to edges)
    dr.rectangle([0, 0, hi - 1, lw - 1],               fill=outline_rgba)  # top
    dr.rectangle([0, hi - lw, hi - 1, hi - 1],         fill=outline_rgba)  # bottom
    dr.rectangle([0, 0, lw - 1, hi - 1],               fill=outline_rgba)  # left
    dr.rectangle([hi - lw, 0, hi - 1, hi - 1],         fill=outline_rgba)  # right

    # ---- downsample
    final_img = im.resize((out, out), Image.LANCZOS)
    return final_img if background is None else final_img.convert("RGB")

# Heuristic: treat high-vertex nearly circular polygons as circles and render with ellipse AA.
def _is_near_circle(poly, min_n=24, tol_rel=0.01):
    if len(poly) < min_n:
        return False, None
    # centroid
    cx = sum(x for x,_ in poly) / len(poly)
    cy = sum(y for _,y in poly) / len(poly)
    rsq = [ (x-cx)*(x-cx) + (y-cy)*(y-cy) for (x,y) in poly ]
    rmean = sum(rsq)/len(rsq)
    if rmean <= 1e-12:
        return False, None
    # variance of radius^2 approximates 2*r*dr; relative tolerance on radius
    var = sum((r - rmean)**2 for r in rsq) / len(rsq)
    rel = (var / (rmean*rmean)) ** 0.5
    if rel <= tol_rel:
        # axis-aligned bbox for ellipse
        xs = [x for x,_ in poly]; ys = [y for _,y in poly]
        return True, (min(xs), min(ys), max(xs), max(ys))
    return False, None
    def to_px(p):
        x, y = p
        X = tx + (x - minx) * s
        Y = ty + (maxy - y) * s  # flip Y for image coords
        return (int(round(X)), int(round(Y)))

    # ---- background
    if isinstance(background, str) and background.lower() == "graph_paper":
        im = graph_paper_bg(hi).convert("RGBA")
    elif background is None:
        im = Image.new("RGBA", (hi, hi), (0, 0, 0, 0))  # transparent
    else:
        bg = background
        if isinstance(bg, tuple) and len(bg) == 3:
            bg = (bg[0], bg[1], bg[2], 255)  # promote to RGBA
        im = Image.new("RGBA", (hi, hi), bg)

    # ---- fill polygons (no outline to avoid double-stroking)
    dr = ImageDraw.Draw(im, "RGBA")
    circle_cells = {}
    for ci, poly in enumerate(polys):
        is_circ, bbox = _is_near_circle(poly)
        if is_circ:
            # draw filled ellipse in hi-res space
            x0, y0, x1, y1 = bbox
            X0, Y1 = to_px((x0, y0))
            X1, Y0 = to_px((x1, y1))
            dr.ellipse([X0, Y0, X1, Y1], fill=cell_colors[ci])
            circle_cells[ci] = (X0, Y0, X1, Y1)
        else:
            pxy = [to_px(p) for p in poly]
            dr.polygon(pxy, fill=cell_colors[ci])
# ---- dedupe & stroke edges exactly once
    unique_edges: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[Tuple[float, float], Tuple[float, float]]
    ] = {}
    for ci, poly in enumerate(polys):
        if 'circle_cells' in locals() and ci in circle_cells:
            continue  # skip; ellipse outline will handle it
        m = len(poly)
        for k in range(m):
            p0 = poly[k]
            p1 = poly[(k + 1) % m]
            a0, b0 = _quant(p0), _quant(p1)
            key = (a0, b0) if a0 <= b0 else (b0, a0)
            if key not in unique_edges:
                unique_edges[key] = (p0, p1)

    lw = max(2, int(round(outline_px * supersample)))
    for (p0, p1) in unique_edges.values():
        dr.line([to_px(p0), to_px(p1)], fill=outline_rgba, width=lw)

    # ellipse outlines for circle cells
    if 'circle_cells' in locals():
        for (X0, Y0, X1, Y1) in circle_cells.values():
            dr.ellipse([X0, Y0, X1, Y1], outline=outline_rgba, width=lw)

    # ---- outer frame# ---- outer frame (flush to edges; no outer white rim)
    # ---- outer frame (flush to edges; no outer white rim)
    dr.rectangle([0, 0, hi - 1, lw - 1], fill=outline_rgba)  # top
    dr.rectangle([0, hi - lw, hi - 1, hi - 1], fill=outline_rgba)  # bottom
    dr.rectangle([0, 0, lw - 1, hi - 1], fill=outline_rgba)  # left
    dr.rectangle([hi - lw, 0, hi - 1, hi - 1], fill=outline_rgba)  # right

    # ---- downsample with proper filter for AA diagonals
    final_img = im.resize((out, out), Image.LANCZOS)
    if background is None:
        return final_img  # keep RGBA (transparent)
    return final_img.convert("RGB")


def labels_default():
    return ["(a)", "(b)", "(c)", "(d)"]
