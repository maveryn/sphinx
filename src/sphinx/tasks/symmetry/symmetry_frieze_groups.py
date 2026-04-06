from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFilter

from ...base import Task
from ...registry import register_task
from ...config import OUT_CELL, MAX_BUILD_RETRIES, OPT_HASH_MIN_BITS, OPT_UNIQUENESS_MIN, SS_CELL
from .common import diff_frac, flip_h, flip_v, rot180
from ...utils.specs import _prefer_asym_mode
from ...utils.rng import choice_weighted, weighted_order
from ...utils.drawing import (
    load_font, labels_default,
    tight_crop_rgba,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sampling weights for frieze kinds (Conway nicknames → IUC). Hop is EXCLUDED.
#   step→p11g, sidle→p1m1, jump→p11m, spinning_hop→p2, spinning_sidle→p2mg, spinning_jump→p2mm
FRIEZE_WEIGHTS: Dict[str, float] = {
    "step": 1.0,            # p11g (glide reflection)
    "sidle": 1.0,           # p1m1 (vertical reflection)
    "jump": 1.0,            # p11m (horizontal reflection)
    "spinning_hop": 1.0,    # p2 (180° rotation)
    "spinning_sidle": 1.0,  # p2mg (rotation + vertical reflection)
    "spinning_jump": 1.0,   # p2mm (rotation + vertical & horizontal reflection)
    # "hop": 0.0,           # p1 (translation only)  ← excluded
}

FRIEZE_IUC = {
    "step": "p11g",
    "sidle": "p1m1",
    "jump": "p11m",
    "spinning_hop": "p2",
    "spinning_sidle": "p2mg",
    "spinning_jump": "p2mm",
}

MOTIF_WEIGHTS = {
    "icons": 5,
    "single_arrow": 0.75,
    "clock": 1.0,
    "crescent": 0.15,
    "gear": 0.25,
    "glyph": 1.0,
    "keyhole": 0.25,
    "pictogram": 0.5,
    "pinwheel_triangles": 0.15,
    "polygon": 0.75,
    "polyhex": 0.25,
    "polyiamond": 0.25,
    "polyomino": 0.75,
    "rings": 0.25,
    "star_polygon": 0.15,
}


PROMPT_TEMPLATES = [
    "Three strips follow the same rule between neighbors; one does not. Which strip is different?",
    "One of the four strips uses a different neighbor relation than the others. Pick the odd one out (a–d).",
    "Focus on how each motif transforms into the next. Which strip follows a different rule?",
    "Three strips share the same step‑to‑step transformation. Which strip breaks the pattern?",
    "Compare the neighbor relations in each strip. Which single strip doesn’t match the other three?",
    "Look at the transformation from each motif to the next. Which strip behaves differently?",
    "Only one strip uses a different symmetry between adjacent motifs. Select it (a–d).",
    "The same rule repeats across three strips; one changes it. Which option is the odd one out?",
    "Inspect how shapes relate to their neighbors. Which strip follows a different rule than the rest?",
    "Which strip’s sequence of transformations doesn’t match the other three? Choose (a–d).",
]

# Visual format
NUM_MOTIF_RANGE = (5, 8)

# Size multiplier for strip tiles (already present)
STRIP_TILE_SCALE: float = 3.0

# ---- Label band controls (identical across the 4 tiles) ----
LABEL_FRAC: float   = 0.16   # target label height as a fraction of strip height
LABEL_MIN_PX: int   = 14     # clamp low
LABEL_MAX_PX: int   = 32     # clamp high
LABEL_PAD_FRAC: float = 1/38 # gap between strip and label (fraction of strip height)
LABEL_PAD_MIN: int  = 4

# Spacing/scale: we sample a single left‑to‑left Δx series and reuse it for ALL strips.
STRIP_BASE_SCALE_H_FRAC = 0.42      # target motif height as a fraction of SS_CELL (pre-fit)
STEP_MULT = 1.15                    # base Δx ≈ 1.15 × (reference motif width)
STEP_JITTER_FRAC = 0.22             # per‑step random jitter around base (in width‑multiples)
STEP_MIN_MULT = 1.00                # guard: Δx ≥ 1.00 × width_ref to avoid overlap
GLIDE_AMP_FRAC = 0.14               # vertical phase for STEP (as fraction of motif height)
# Minimum horizontal white gap between neighboring motifs inside a strip.
# Enforced in pixels against the reference motif width at final scale.
MIN_GAP_FRAC: float = 0.08   # 8% of reference width
MIN_GAP_PX:   int   = 2      # absolute safety floor in pixels



# Gates
ADJ_MIN = float(OPT_UNIQUENESS_MIN) * 0.75      # min visible change between neighbors
OPT_MIN_DELTA = float(OPT_UNIQUENESS_MIN)       # pairwise separation across tiles


# ─────────────────────────────────────────────────────────────────────────────
# Basic transforms (neighbor rules)


def _next_motif(kind: str, idx: int, im: Image.Image) -> Image.Image:
    if kind == "sidle":              # vertical mirror
        return flip_h(im)
    if kind == "jump":               # horizontal mirror
        return flip_v(im)
    if kind == "step":               # glide ≈ flip_v (translation shown by placement)
        return flip_v(im)
    if kind == "spinning_hop":       # 180° rotation
        return rot180(im)
    if kind == "spinning_sidle":     # alternate vertical mirror and 180°
        return flip_h(im) if (idx % 2 == 0) else rot180(im)
    if kind == "spinning_jump":      # alternate vertical vs horizontal mirror
        return flip_h(im) if (idx % 2 == 0) else flip_v(im)
    return im  # hop excluded


# ─────────────────────────────────────────────────────────────────────────────
# Utilities

def _pad_to_cell_center(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Return an RGBA image of size (target_w, target_h) with 'im' centered.
    Background is opaque white to match other tasks.
    """
    im = im.convert("RGBA") if im.mode != "RGBA" else im
    out = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    x = (target_w - im.width) // 2
    y = (target_h - im.height) // 2
    out.paste(im, (x, y), im)
    return out

def _scale_rgba(im: Image.Image, s: float) -> Image.Image:
    """Scale an RGBA image by factor s with LANCZOS, no-op if s≈1."""
    if s <= 0:
        return im
    if abs(s - 1.0) < 1e-3:
        return im
    w = max(1, int(round(im.width  * s)))
    h = max(1, int(round(im.height * s)))
    return im.resize((w, h), Image.LANCZOS).convert("RGBA")

def _scale_to_height_crisp(im: Image.Image, target_h: int) -> Image.Image:
    if target_h <= 0 or im.height == target_h:
        return im
    s = target_h / float(im.height)
    w = max(1, int(round(im.width * s)))

    # Big shrink → staged downscale (BOX then LANCZOS) to avoid ringing.
    if s < 0.7:
        # stage 1: coarse area downscale close to target
        inter_h = max(target_h, int(round(im.height * 0.72)))
        inter_w = max(1, int(round(im.width  * (inter_h / im.height))))
        tmp = im.resize((inter_w, inter_h), Image.BOX)
        out = tmp.resize((w, target_h), Image.LANCZOS)
    else:
        out = im.resize((w, target_h), Image.LANCZOS)

    # Light alpha hardening to reduce fringe noise
    r, g, b, a = out.split()
    # clamp very-near-opaque to opaque; drop near-transparent
    a = a.point(lambda t: 255 if t > 230 else (0 if t < 6 else t))
    return Image.merge("RGBA", (r, g, b, a))


# One-pixel alpha thicken to stabilize very thin strokes (safe & subtle).
def _thicken_alpha_1px(im: Image.Image) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    r, g, b, a = im.split()
    a2 = a.filter(ImageFilter.MaxFilter(size=3))  # 1px grow
    return Image.merge("RGBA", (r, g, b, a2))


def _scale_to_height(im: Image.Image, target_h: int) -> Image.Image:
    s = min(1.0, target_h / max(1, im.height))  # avoid upscaling artifacts
    w = max(8, int(round(im.width * s)))
    h = max(8, int(round(im.height * s)))
    return im.resize((w, h), Image.LANCZOS)

def _frieze_degenerate_for(motif_im: Image.Image, kind: str) -> bool:
    """
    Reject motifs whose required transforms are too close to the base.
    """
    base = _scale_to_height(motif_im, max(10, int(SS_CELL * 0.25)))
    checks: List[Image.Image] = []
    if kind in ("sidle", "spinning_sidle", "spinning_jump"):
        checks.append(flip_h(motif_im))
    if kind in ("jump", "step", "spinning_jump"):
        checks.append(flip_v(motif_im))
    if kind in ("spinning_hop", "spinning_sidle", "spinning_jump"):
        checks.append(rot180(motif_im))
    for im2 in checks:
        ref = _scale_to_height(im2, base.height)
        if diff_frac(base, ref, thresh=8) < max(0.006, 0.5 * OPT_MIN_DELTA):
            return True
    return False

@dataclass
class StripRecipe:
    base_rgba: Image.Image
    frieze_kind: str

def _build_dx_series(num_motifs: int, width_ref: int, rng: random.Random) -> List[int]:
    """
    Sample a sequence of left‑to‑left spacings (Δx) in *multiples* of width_ref, then
    convert to pixel units. The same series is reused for all four strips.
    """
    steps = []
    for _ in range(num_motifs - 1):
        mult = STEP_MULT + rng.uniform(-STEP_JITTER_FRAC, +STEP_JITTER_FRAC)
        mult = max(STEP_MIN_MULT, mult)
        steps.append(mult)
    return [max(1, int(round(m * width_ref))) for m in steps]

def _build_dx_series_with_gap(num_motifs: int, width_ref_px: int, rng: random.Random) -> List[int]:
    """
    Build a common left-to-left Δx series from a *reference* motif width (pixels),
    applying jitter but guaranteeing a minimum gap so motifs never touch.
    The same Δx is reused for all strips.
    """
    dxs: List[int] = []
    base = max(1, int(round(width_ref_px * STEP_MULT)))
    jitter = max(1, int(round(base * STEP_JITTER_FRAC)))
    min_dx = max(int(round(width_ref_px * (1.0 + MIN_GAP_FRAC))), width_ref_px + MIN_GAP_PX)

    for _ in range(num_motifs - 1):
        dx = base + rng.randint(-jitter, +jitter)
        dx = max(min_dx, dx)  # never less than motif width + gap
        dxs.append(dx)
    return dxs

def _fit_dx_series(
    num_motifs: int,
    width_ref_px: int,
    max_strip_w: int,
    rng: random.Random,
) -> Optional[List[int]]:
    """
    Build a left-to-left Δx series that:
      • guarantees at least (motif width + MIN_GAP) between neighbors
      • respects jitter bounds
      • fits inside max_strip_w exactly or below (never overflows)
    Returns None if even the minimum layout cannot fit (caller must shrink heights).
    """
    steps = num_motifs - 1
    if steps <= 0:
        return []

    # bounds
    base   = max(1, int(round(width_ref_px * STEP_MULT)))
    jitter = max(1, int(round(base * STEP_JITTER_FRAC)))
    min_dx = max(int(round(width_ref_px * (1.0 + MIN_GAP_FRAC))), width_ref_px + MIN_GAP_PX)
    hi_dx  = max(min_dx, base + jitter)

    # minimal possible width with guaranteed gap
    min_sum = steps * min_dx + width_ref_px
    if min_sum > max_strip_w:
        return None  # impossible without shrinking heights

    # start with all minimums, then distribute remaining budget up to hi_dx per step
    dx = [min_dx] * steps
    budget = max_strip_w - (width_ref_px + sum(dx))
    if budget <= 0:
        return dx

    room = [hi_dx - min_dx] * steps
    total_room = sum(room)

    if total_room <= 0:
        return dx

    # proportional random allocation, then top-up any rounding leftovers
    r = [rng.random() for _ in range(steps)]
    s = sum(r) or 1.0
    for i in range(steps):
        add = min(room[i], int(round(budget * (r[i] / s))))
        dx[i] += add

    # in case of rounding, top-up 1px at a time without exceeding hi bounds
    while width_ref_px + sum(dx) < max_strip_w:
        # choose a step that still has room
        candidates = [i for i in range(steps) if dx[i] < hi_dx]
        if not candidates:
            break
        i = rng.choice(candidates)
        dx[i] += 1

    return dx


def _crop_with_pad(alpha_canvas: Image.Image, pad: int) -> Image.Image:
    """Crop to non-empty alpha with a small symmetric pad."""
    bbox = alpha_canvas.split()[-1].getbbox()
    if not bbox:
        return alpha_canvas
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(alpha_canvas.width,  x1 + pad)
    y1 = min(alpha_canvas.height, y1 + pad)
    return alpha_canvas.crop((x0, y0, x1, y1))

def _scale_width(im: Image.Image, target_w: int) -> Image.Image:
    if im.width == target_w:
        return im
    h = max(8, int(round(im.height * (target_w / im.width))))
    return im.resize((target_w, h), Image.LANCZOS)

def _pad_height_center(im: Image.Image, target_h: int) -> Image.Image:
    if im.height >= target_h:
        return im
    out = Image.new("RGBA", (im.width, target_h), (255, 255, 255, 255))
    y = (target_h - im.height) // 2
    out.paste(im, (0, y), im)
    return out


def _label_tile_crisp(tile: Image.Image, label: str, base_font,
                      target_h_px: int, pad_y_px: int) -> Image.Image:
    """Render option labels with the same crisp style as sequence tasks."""
    if tile.mode != "RGBA":
        tile = tile.convert("RGBA")

    try:
        x0, y0, x1, y1 = base_font.getbbox(label)
        base_h = max(1, y1 - y0)
        font = base_font.font_variant(
            size=max(6, int(round(getattr(base_font, "size", 16) * (target_h_px / base_h))))
        )
    except Exception:
        font = base_font

    # Measure the label precisely using a temporary draw context
    tmp_img = Image.new("L", (1, 1), 0)
    tmp_draw = ImageDraw.Draw(tmp_img)
    bbox = tmp_draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    W = max(tile.width, tw)
    H = tile.height + pad_y_px + target_h_px
    out = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    out.alpha_composite(tile, ((W - tile.width) // 2, 0))

    draw = ImageDraw.Draw(out)
    x_text = (W - tw) // 2 - bbox[0]
    y_text = tile.height + pad_y_px + (target_h_px - th) // 2 - bbox[1]
    draw.text((x_text, y_text), label, fill=(0, 0, 0), font=font)
    return out


def _render_strip_tile(
    base_item: Image.Image,
    kind: str,
    *,
    target_h: int,
    dx_leftleft: List[int],
    glide_amp_px: int,
    band_inner_w: int,
    band_inner_h: int,
    edge: int,
    border: int,
) -> Tuple[Image.Image, List[Image.Image]]:

    A0 = _scale_to_height_crisp(base_item, target_h)

    # ▶▶ Use the spacing list length (len(dx_leftleft)) to determine count
    seq: List[Image.Image] = [A0]
    for i in range(len(dx_leftleft)):          # ← instead of range(NUM_MOTIFS_STRIP - 1)
        seq.append(_next_motif(kind, i, seq[-1]))

    Hm = A0.height

    band_w = band_inner_w + 2 * edge
    band_h = band_inner_h + 2 * edge
    band   = Image.new("RGBA", (band_w, band_h), (255, 255, 255, 0))

    seq_w = sum(dx_leftleft) + seq[-1].width
    x = edge + max(0, (band_inner_w - seq_w) // 2)
    y_base = edge + max(0, (band_inner_h - Hm) // 2)

    for i, bmp in enumerate(seq):
        y = y_base
        if kind == "step" and glide_amp_px > 0 and (i % 2 == 1):
            y = max(edge, min(y_base + glide_amp_px, edge + band_inner_h - bmp.height))
        band.alpha_composite(bmp, (x, y))
        if i < len(seq) - 1:
            x += dx_leftleft[i]

    ImageDraw.Draw(band).rectangle([0, 0, band_w - 1, band_h - 1], outline=(0, 0, 0), width=border)
    white = Image.new("RGBA", (band_w, band_h), (255, 255, 255, 255))
    tile = Image.alpha_composite(white, band).convert("RGBA")
    return tile, seq



def _compose_rows_4(labeled_tiles: List[Image.Image]) -> Image.Image:
    """
    Stack four labeled tiles vertically (a–d), centering each tile horizontally
    so columns align even if widths differ slightly.
    """
    assert len(labeled_tiles) == 4
    tiles = [im.convert("RGBA") if im.mode != "RGBA" else im for im in labeled_tiles]

    # Cell size = max of provided tiles
    cell_w = max(im.width  for im in tiles)
    cell_h = max(im.height for im in tiles)

    # Compact margins/gaps relative to cell size
    gap_y    = max(4,  min(10, cell_h // 36))
    margin_x = max(6,  min(12, cell_w // 40))
    margin_y = max(6,  min(12, cell_h // 40))

    W = margin_x + cell_w + margin_x
    H = margin_y + cell_h * 4 + gap_y * 3 + margin_y
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # Top-left corner for each row
    y = margin_y
    for im in tiles:
        x = margin_x + (cell_w - im.width) // 2
        canvas.paste(im, (x, y), im)
        y += cell_h + gap_y

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
@register_task
class SymmetryFriezeGroupsTask(Task):
    """
    Four frieze‑style strips (same motif family).
    Exactly three follow the same neighbor rule; the fourth uses a different rule.
    All four strips use the *same sampled spacing* (the left‑to‑left Δx series),
    so translation/spacing is consistent across the grid.

    Display format: four labeled rows (a–d) stacked vertically. User selects the
    *different* strip.
    """
    name = "symmetry_frieze_groups"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # ── weighted motif family selection (same pattern as other sequence tasks) ──
        allowed_motifs = [k for k in motif_impls.keys() if MOTIF_WEIGHTS.get(k, 0) > 0] or list(motif_impls.keys())
        if not allowed_motifs:
            raise RuntimeError(f"{self.name}: no motifs available.")

        motif_order = weighted_order(rng, allowed_motifs, MOTIF_WEIGHTS)

        # Weighted frieze kinds (hop excluded)
        kinds_all = [k for k, w in FRIEZE_WEIGHTS.items() if w > 0.0]
        if not kinds_all:
            raise RuntimeError(f"{self.name}: all frieze kind weights are zero.")
        frieze_order = weighted_order(rng, kinds_all, FRIEZE_WEIGHTS)

        labels = labels_default()  # ["a","b","c","d"]
        # Sample the number of motifs (same for all 4 strips)
        num_motifs = rng.randint(NUM_MOTIF_RANGE[0], NUM_MOTIF_RANGE[1])

        # Try motifs in weighted order
        for mk in motif_order:
            motif = motif_impls[mk]
            for _ in range(self.max_retries):
                # Sample 4 specs (same family, different instances)
                try:
                    specs = [_prefer_asym_mode(motif, motif.sample_spec(rng)) for _ in range(4)]
                    raws = [tight_crop_rgba(motif.render(s)) for s in specs]
                except Exception:
                    continue
                if any(im.width < 8 or im.height < 8 for im in raws):
                    continue

                # Pick the common rule + the odd rule
                common_kind = choice_weighted(rng, frieze_order, [FRIEZE_WEIGHTS[k] for k in frieze_order])
                odd_kind_choices = [k for k in kinds_all if k != common_kind]
                odd_kind = choice_weighted(rng, odd_kind_choices, [FRIEZE_WEIGHTS[k] for k in odd_kind_choices])

                # Degeneracy guards: each raw must support BOTH rules
                if any(_frieze_degenerate_for(im, common_kind) for im in raws):
                    continue
                if any(_frieze_degenerate_for(im, odd_kind) for im in raws):
                    continue

                # ── Global scale & spacing shared by ALL strips ──
                H0 = max(10, int(round(SS_CELL * STRIP_BASE_SCALE_H_FRAC)))

                # --- Assign which labeled slot is odd (a,b,c,d in fixed positions) ---
                odd_slot = rng.randrange(4)  # 0:a, 1:b, 2:c, 3:d
                kinds_for_slot = [common_kind, common_kind, common_kind, common_kind]
                kinds_for_slot[odd_slot] = odd_kind

                # --- Per-slot target heights so final content heights match across strips ---
                # --- Per-slot target heights so final content heights match across strips ---
                vfac = [GLIDE_AMP_FRAC if k == "step" else 0.0 for k in kinds_for_slot]
                border_px = max(3, SS_CELL // 160)
                edge = border_px + 1
                max_strip_w = SS_CELL - 2 * edge  # must fit inside this in _render_strip_tile()

                # Start from a content height anchored to H0 (but cap to inner height)
                content_h_target = int(min(max_strip_w, H0))

                def _heights_for_content(Hc: int) -> Tuple[List[int], List[int], int]:
                    th_list = [max(8, int(round(Hc / (1.0 + vf)))) for vf in vfac]
                    ga_list = [(int(round(th * GLIDE_AMP_FRAC)) if vf > 0.0 else 0) for th, vf in zip(th_list, vfac)]
                    widths = [_scale_to_height(raws[i], th_list[i]).width for i in range(4)]
                    return th_list, ga_list, max(widths)

                # --- Shrink heights until even the *minimum* layout can fit
                for _ in range(16):
                    target_h_list, glide_amp_list, Wref_final = _heights_for_content(content_h_target)
                    min_dx = max(int(round(Wref_final * (1.0 + MIN_GAP_FRAC))), Wref_final + MIN_GAP_PX)
                    min_needed = (num_motifs - 1) * min_dx + Wref_final  # ◀ changed
                    if min_needed <= max_strip_w:
                        break
                    shrink = max(0.60, min(0.98, max_strip_w / float(min_needed)))
                    content_h_target = max(8, int(round(content_h_target * shrink)))

                # Build a Δx series that *fits*
                dx_px = _fit_dx_series(num_motifs, Wref_final, max_strip_w, rng)  # ◀ changed
                if dx_px is None:
                    content_h_target = max(8, int(round(content_h_target * 0.9)))
                    target_h_list, glide_amp_list, Wref_final = _heights_for_content(content_h_target)
                    dx_px = _fit_dx_series(num_motifs, Wref_final, max_strip_w, rng)  # ◀ changed
                    if dx_px is None:
                        continue

                strip_w = sum(dx_px) + Wref_final
                if strip_w > max_strip_w:
                    # scale Δx but never below (Wref_final + min gap)
                    scale = max_strip_w / float(strip_w)
                    min_dx = max(int(round(Wref_final * (1.0 + MIN_GAP_FRAC))), Wref_final + MIN_GAP_PX)
                    dx_px = [max(int(round(d * scale)), min_dx) for d in dx_px]

                # --- Build the four strips in fixed label order (a,b,c,d) ---
                tiles_raw: List[Image.Image] = []
                payloads: List[Dict[str, Any]] = []
                ok_build = True

                # Inner band width: use the *reference* last width so all strips share the same span
                band_inner_w = sum(dx_px) + Wref_final
                # Inner band height: the common content height you already computed
                band_inner_h = content_h_target
                # edge = border + 1; we already have 'edge' where we built max_strip_w
                # (if you don't have it in scope here, use: edge = max(2, SS_CELL // 160) + 1)

                # --- Render at final output resolution: push scale BEFORE rasterization
                HR = max(1.0, float(STRIP_TILE_SCALE))

                border_hr = border_px
                edge_hr = border_hr + 1
                dx_px_hr = [max(1, int(round(d * HR))) for d in dx_px]
                band_inner_w_hr = max(1, int(round(band_inner_w * HR)))
                band_inner_h_hr = max(1, int(round(band_inner_h * HR)))
                target_h_list_hr = [max(8, int(round(h * HR))) for h in target_h_list]
                glide_amp_hr = [int(round(g * HR)) for g in glide_amp_list]

                for i in range(4):
                    tile_rgba, seq = _render_strip_tile(
                        raws[i],
                        kinds_for_slot[i],
                        target_h=target_h_list_hr[i],
                        dx_leftleft=dx_px_hr,
                        glide_amp_px=glide_amp_hr[i],
                        band_inner_w=band_inner_w_hr,
                        band_inner_h=band_inner_h_hr,
                        edge=edge_hr,
                        border=border_hr,
                    )

                    # neighbor visibility inside strip
                    diffs = [diff_frac(seq[j], seq[j + 1], thresh=8) for j in range(len(seq) - 1)]
                    if min(diffs) < ADJ_MIN:
                        ok_build = False
                        break

                    tiles_raw.append(tile_rgba)
                    payloads.append({
                        "slot": i,
                        "is_odd": (i == odd_slot),
                        "frieze_kind": kinds_for_slot[i],
                        "frieze_iuc": FRIEZE_IUC.get(kinds_for_slot[i], ""),
                        "target_h_px": int(target_h_list[i]),
                        "glide_amp_px": int(glide_amp_list[i]),
                    })

                if not ok_build:
                    continue

                # pairwise separation across tiles (avoid near duplicates)
                for i in range(4):
                    for j in range(i + 1, 4):
                        if diff_frac(tiles_raw[i], tiles_raw[j], thresh=8) < OPT_MIN_DELTA:
                            ok_build = False
                            break
                    if not ok_build:
                        break
                if not ok_build:
                    continue

                # --- Rectangular standardization (no width rescale; pad all to the same content size) ---
                content_w = max(im.width for im in tiles_raw)  # now reflects scaled size
                content_h = max(im.height for im in tiles_raw)
                equalized = [_pad_to_cell_center(im, content_w, content_h) for im in tiles_raw]
                common_h = content_h

                # ---- Compute UNIFORM label metrics (once), derived from the *scaled* height ----
                common_h = max(im.height for im in tiles_raw)  # after HR rendering
                label_max_px_hr = max(LABEL_MAX_PX, int(round(LABEL_MAX_PX * HR)))
                target_label_h = max(LABEL_MIN_PX, min(int(round(common_h * LABEL_FRAC)), label_max_px_hr))
                label_pad_y = max(LABEL_PAD_MIN, int(round(common_h * LABEL_PAD_FRAC)))

                base_font = load_font()

                # ---- Label with EXACT metrics for all tiles ----
                labeled = [
                    _label_tile_crisp(im, lab, base_font, target_label_h, label_pad_y)
                    for im, lab in zip(equalized, labels)
                ]

                # ---- Force identical labeled tile size & center content (usually already true) ----
                cell_w = max(im.width for im in labeled)
                cell_h = max(im.height for im in labeled)
                labeled = [_pad_to_cell_center(im, cell_w, cell_h) for im in labeled]

                composite = _compose_rows_4(labeled)

                question = random.choice(PROMPT_TEMPLATES)
                answer_label = labels[odd_slot]  # fixed mapping to grid positions

                meta = {
                    "pattern_kind": "sequence",
                    "pattern": self.name,
                    "grid": (4, 1),
                    "motif_kind": mk,
                    "labels": labels,
                    "answer": answer_label,
                    "question": question,
                    "composite_ready": True,
                    "common_kind": common_kind,
                    "common_iuc": FRIEZE_IUC.get(common_kind, ""),
                    "odd_kind": odd_kind,
                    "odd_iuc": FRIEZE_IUC.get(odd_kind, ""),
                    "odd_index": int(odd_slot),
                    "num_motifs": int(num_motifs),
                    "spacing": {
                        "target_h": int(max(target_h_list)),  # per-slot max actually used
                        "glide_amp_px": int(max(glide_amp_list)),  # per-slot max actually used
                        "dx_leftleft": [int(d) for d in dx_px],  # shared Δx series
                        "ref_width_px": int(Wref_final),  # reference width at final heights
                    },
                }
                return composite, payloads, meta

        raise RuntimeError(f"{self.name}: failed to produce an instance after {self.max_retries} attempts.")
