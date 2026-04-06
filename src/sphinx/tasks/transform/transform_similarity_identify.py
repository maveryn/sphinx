
import random
import math
from typing import Any, Dict, List, Optional, Tuple, Callable
from PIL import Image, ImageDraw, ImageOps

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import (
    OUT_CELL,
    IMG_DIFF_MIN,
    OPT_UNIQUENESS_MIN,
    OPT_HASH_MIN_BITS,
    IMG_EQUAL_TOL,
    EQUAL_HASH_MAX_BITS,
    MAX_BUILD_RETRIES,
)
from sphinx.tasks.symmetry.common import sig, strong_same, strong_distinct, pairwise_unique
from sphinx.utils.specs import _prefer_asym_mode
from sphinx.utils.drawing import (
    load_font, crisp_option_tile, tight_crop_rgba, ensure_transparent,
    add_tile_border, paste_rgba
)
from .common import (
    scale_patch, center_xy, bounds_ok, compose_tile_with_patch,
    apply_patch_transform, candidate_translations, weighted_order
)

# ============================================================================
#                         PUBLIC-FACING CONFIG (TOP)
# ============================================================================

# Add near the other top-level toggles
DISSIMILAR_INPUT_PAD_FRAC = 0.12  # transparent padding before dissimilar warps

# ----- Prompts (parallel to other transform tasks) -----
PROMPT_TEMPLATES_SIMILAR = [
    "The top image is the original. Exactly one option below shows a similar shape (uniform scale allowed, plus rotation/reflection/translation). Which option {label_span} is similar?",
    "One of the options shows the same shape as the top (up to uniform scaling and rotation/flip/translation). Which option {label_span}?",
    "Find the option that is shape-similar to the top (uniform scale + rotation/reflection/translation). Which one {label_span}?",
    "Which option {label_span} shows a figure that is similar in the Euclidean sense to the top (allow uniform scale and rigid/mirror motion)?",
    "Pick the option {label_span} that has the same shape as the top (possibly scaled, rotated, reflected, and/or translated).",
    "Select the option {label_span} that could be obtained by uniform scaling of the top, plus rotation/reflection/translation.",
    "Which option {label_span} is similar to the top figure under uniform scaling and rigid/mirror motions?",
    "Choose the option {label_span} that is the same shape as the top, up to uniform scale and D4 symmetries (plus translation).",
    "Identify the similar shape: which option {label_span} matches the top up to uniform scaling and rotation/flip/translation?",
    "One of the options {label_span} is similar to the top under uniform scale + rotation/reflection/translation. Which one?",
]

PROMPT_TEMPLATES_DISSIMILAR = [
    "The top image is the original. Exactly one option below is not similar to the top. Which option {label_span} is dissimilar?",
    "Find the dissimilar option {label_span}: it cannot be obtained from the top by any uniform scale + rotation/reflection/translation.",
    "Which option {label_span} is not shape-similar to the top (i.e., no uniform scale + rigid/mirror motion can match it)?",
    "Exactly one option {label_span} breaks Euclidean shape similarity with the top. Which is it?",
    "Choose the option {label_span} that is not obtainable by uniform scaling of the top plus rotation/reflection/translation.",
    "One option {label_span} is dissimilar in shape to the top. Which one?",
    "Identify the dissimilar option {label_span}: similarity under uniform scale and D4 symmetries (plus translation) fails.",
    "Which option {label_span} is not shape-equivalent to the top up to uniform scaling and rigid/mirror motions?",
    "Pick the option {label_span} that violates shape similarity with the top (no uniform scale + rotation/flip/translation works).",
    "Find the lone dissimilar option {label_span}.",
]

# ----- Generation toggles for building 'similar' options -----
# Allowed orientation symmetries (subset of D4). Disable to avoid using them in generation.
SIMILAR_D4_KEYS = ["mirror_v", "mirror_h", "diag_main", "diag_anti", "rot90", "rot180", "rot270"]
SIMILAR_ALLOW_TRANSLATION = True      # if True, we may add a small translation (still similar)
SIMILAR_ENABLE_UNIFORM_SCALE = True   # if False, we won't rescale (s=1.0)

# Prefer asymmetric, varied motifs (same tuning as other transform tasks)
MOTIF_WEIGHTS = {
    "icons": 15,
    "clock": 0.5,
    "crescent": 0.75,
    "glyph": 1.0,
    "keyhole": 0.5,
    "pictogram": 0.5,
    "pinwheel_triangles": 0.5,
    "polygon": 1.0,
    "polyhex": 0.25,
    "polyiamond": 0.25,
    "polyomino": 0.25,
    "rings": 0.25,
    "star_polygon": 0.25,
}


# ----- Dissimilar breakers registry & enable switches -----
# You can turn any breaker off here by setting its name to False.
DISSIMILAR_BREAKERS_ENABLED = {
    "anisotropic_scale": True,
    "shear_x": True,
    "shear_y": True,
    "perspective_keystone": True,
    "mesh_warp_wave": False,
    "radial_barrel_pincushion": False,
    "mild_swirl": False,
    "partial_mirror": False,
}

# ----- Verification scope (conceptual notion of similarity) -----
# These keys are used by the canonical checker. By default, we verify across full D4;
# edit this if you want to *redefine* similarity to exclude some symmetries.
VERIFICATION_D4_KEYS = ["id", "mirror_v", "mirror_h", "diag_main", "diag_anti", "rot90", "rot180", "rot270"]

# ----- Options & variant sampling -----
OPTIONS_K_CHOICES = (4, 4)  # selectable counts of options
P_ONE_SIMILAR = 0.5            # probability of the 'one_similar' variant
SHEAR_ABS_RANGE = (0.2, 0.5)
# When composing options, keep an inner margin to avoid clipping against tile edges/border.
FIT_MARGIN_FRAC = 0.06  # 6% of tile size on each side

# ============================================================================
#                               IMPLEMENTATION
# ============================================================================

def _labels_for_n(n: int) -> List[str]:
    base = list("abcdefghijklmnopqrstuvwxyz")
    return [f"({base[i]})" for i in range(n)]

def _format_prompt(variant: str, labels: List[str], rng: random.Random) -> str:
    span = f"{labels[0]}–{labels[-1]}" if len(labels) >= 2 else labels[0]
    if variant == "one_similar":
        return rng.choice(PROMPT_TEMPLATES_SIMILAR).format(label_span=span)
    else:
        return rng.choice(PROMPT_TEMPLATES_DISSIMILAR).format(label_span=span)

def _xform_rgba(p: Image.Image, size, method, data, resample=Image.BICUBIC) -> Image.Image:
    """
    Safe wrapper for Image.transform that prefers transparent fill.
    Falls back on older Pillow versions that don't support 'fillcolor'.
    """
    p = p.convert("RGBA")
    try:
        return p.transform(size, method, data, resample=resample, fillcolor=(0, 0, 0, 0))
    except TypeError:
        return p.transform(size, method, data, resample=resample)


def _compose_top_bottom_variable(top: Image.Image, option_tiles: List[Image.Image], tile_px: int) -> Image.Image:
    pad = max(8, tile_px // 16)

    top_w, top_h = top.width, top.height
    opt_w, opt_h = option_tiles[0].width, option_tiles[0].height  # includes label band

    n = len(option_tiles)
    W = max(top_w + 2 * pad, n * opt_w + (n + 1) * pad)
    H = top_h + opt_h + 3 * pad

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # top centered
    x_top = (W - top_w) // 2
    y_top = pad
    paste_rgba(canvas, top, (x_top, y_top))

    # bottom row
    y_bot = top_h + 2 * pad
    for i, tile in enumerate(option_tiles):
        x = pad + i * (opt_w + pad)
        paste_rgba(canvas, tile, (x, y_bot))

    # frame around the top tile
    dr = ImageDraw.Draw(canvas)
    tile_border_w = max(3, tile_px // 48)
    dr.rectangle([x_top, y_top, x_top + top_w - 1, y_top + top_h - 1],
                 outline=(0, 0, 0), width=tile_border_w)

    # thin outer border
    border_w = max(2, W // 320)
    dr.rectangle([1, 1, W - 2, H - 2], outline=(0, 0, 0), width=border_w)
    return canvas

def _to_canonical(patch: Image.Image, canon_px: int) -> Image.Image:
    """Scale/crop/center to a canonical square for similarity checks (remove translation & uniform scale)."""
    p = ensure_transparent(patch)
    p = tight_crop_rgba(p)

    w, h = p.size
    w = max(1, w); h = max(1, h)
    scale = (canon_px - 8) / max(w, h)
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    p2 = p.resize((tw, th), Image.LANCZOS)

    canvas = Image.new("RGBA", (canon_px, canon_px), (0, 0, 0, 0))
    x = (canon_px - tw) // 2
    y = (canon_px - th) // 2
    canvas.alpha_composite(p2, (x, y))
    return canvas

def _apply_d4(patch: Image.Image, key: str) -> Image.Image:
    return patch if key == "id" else apply_patch_transform(patch, key)

def _is_similar_under_keys(a_patch: Image.Image, b_patch: Image.Image,
                           canon_px: int, tol: float, eq_bits: int,
                           verify_keys: List[str]) -> bool:
    """Return True if b is similar to a under uniform scale + verify_keys + translation (neutralized by canonicalization)."""
    a_can = _to_canonical(a_patch, canon_px)
    for key in verify_keys:
        tb = _apply_d4(b_patch, key)
        b_can = _to_canonical(tb, canon_px)
        if strong_same(a_can, b_can, tol, eq_bits):
            return True
    return False


def _scale_to_fit_with_margin(patch: Image.Image, tile_px: int, margin_frac: float = FIT_MARGIN_FRAC) -> Image.Image:
    """
    Downscale (never upscale) so that the patch fits within the tile with an inner margin.
    Returns the (possibly) resized patch.
    """
    p = patch
    w, h = max(1, p.width), max(1, p.height)
    margin_px = max(1, int(round(tile_px * float(margin_frac))))
    max_w = max(1, tile_px - 2 * margin_px)
    max_h = max(1, tile_px - 2 * margin_px)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 0.999:
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        p = p.resize((tw, th), Image.LANCZOS)
    return p

def _clamp_xy_into_tile(x: int, y: int, w: int, h: int, tile_px: int, margin_frac: float = FIT_MARGIN_FRAC) -> tuple[int, int]:
    """Clamp top-left (x,y) so the patch is fully inside the tile with the margin."""
    margin_px = max(1, int(round(tile_px * float(margin_frac))))
    min_x = margin_px
    min_y = margin_px
    max_x = max(margin_px, tile_px - margin_px - w)
    max_y = max(margin_px, tile_px - margin_px - h)
    if max_x < min_x:  # in rare extreme cases, allow at least (0,0)
        min_x, max_x = 0, max(0, tile_px - w)
    if max_y < min_y:
        min_y, max_y = 0, max(0, tile_px - h)
    x = min(max_x, max(min_x, x))
    y = min(max_y, max(min_y, y))
    return x, y
# ------------------ "similar" generators (uniform scale + selected D4 + optional translation) ------------------
def _random_uniform_scale(patch: Image.Image, rng: random.Random, min_s=0.55, max_s=1.35) -> Image.Image:
    s = rng.uniform(min_s, max_s)
    w = max(1, int(round(patch.width * s)))
    h = max(1, int(round(patch.height * s)))
    return patch.resize((w, h), Image.LANCZOS)

def _make_similar_tile(
    patch: Image.Image,
    tile_px: int,
    centered_xy: Tuple[int, int],
    rng: random.Random,
    enable_uniform_scale: bool,
    allow_translation: bool,
    similar_d4_keys: List[str],
) -> Optional[Tuple[Image.Image, Image.Image]]:
    # optional uniform scaling
    tpatch = _random_uniform_scale(patch, rng) if enable_uniform_scale else patch

    # apply a short random sequence of allowed D4 keys (excluding id)
    if similar_d4_keys:
        k = rng.randint(0, 3)
        keys = similar_d4_keys[:]
        rng.shuffle(keys)
        for i in range(k):
            tpatch = apply_patch_transform(tpatch, keys[i])

    # Ensure it fits the tile with margin
    tpatch = _scale_to_fit_with_margin(tpatch, tile_px)

    # optionally add a legal translation (still similar)
    cx, cy = centered_xy
    if allow_translation and rng.random() < 0.6:
        cands = candidate_translations(tile_px, tpatch.width, tpatch.height, center_xy(tile_px, tpatch.width, tpatch.height))
        if cands:
            dx, dy = rng.choice(cands)
            x = center_xy(tile_px, tpatch.width, tpatch.height)[0] + dx
            y = center_xy(tile_px, tpatch.width, tpatch.height)[1] + dy
        else:
            x, y = center_xy(tile_px, tpatch.width, tpatch.height)
    else:
        x, y = center_xy(tile_px, tpatch.width, tpatch.height)

    # clamp into tile to guarantee fit (extra safety)
    x, y = _clamp_xy_into_tile(x, y, tpatch.width, tpatch.height, tile_px)

    if not bounds_ok(tile_px, x, y, tpatch.width, tpatch.height):
        # final fallback: downscale a bit and recenter
        tpatch = _scale_to_fit_with_margin(tpatch, tile_px)
        x, y = center_xy(tile_px, tpatch.width, tpatch.height)
        x, y = _clamp_xy_into_tile(x, y, tpatch.width, tpatch.height, tile_px)
        if not bounds_ok(tile_px, x, y, tpatch.width, tpatch.height):
            return None
    return compose_tile_with_patch(tile_px, tpatch, (x, y)), tpatch

# ------------------ "breaker" generators (violate similarity) ------------------
def _anisotropic_scale(patch: Image.Image, rng: random.Random, min_s=0.6, max_s=1.4, min_gap=0.12) -> Image.Image:
    sx = rng.uniform(min_s, max_s)
    while True:
        sy = rng.uniform(min_s, max_s)
        if abs(sx - sy) >= min_gap:
            break
    w = max(1, int(round(patch.width * sx)))
    h = max(1, int(round(patch.height * sy)))
    return patch.resize((w, h), Image.LANCZOS)

def _shear(patch: Image.Image, rng: random.Random, axis: str = "x") -> Image.Image:
    """
    Shear with canvas growth + compensating translation so no content is clipped.
    Works for both positive and negative shears. Uses transparent fill.
    """
    p = patch.convert("RGBA")
    w, h = p.size
    if w < 2 or h < 2:
        return p

    min_abs, max_abs = SHEAR_ABS_RANGE  # e.g., (0.2, 0.5)
    k = rng.choice([-1.0, 1.0]) * rng.uniform(min_abs, max_abs)

    if axis == "x":
        # Output width large enough to hold the entire sheared rectangle.
        # Reverse-mapping form: x_in = x_out + k*y_out + tx.
        # To ensure every source pixel has an output location, we must shift
        # by -k*(h-1) when k>0; for k<0, no shift is needed.
        nw = int(math.ceil(w + abs(k) * (h - 1)))
        tx = -k * (h - 1) if k > 0 else 0.0
        return _xform_rgba(p, (nw, h), Image.AFFINE, (1, k, tx, 0, 1, 0))
    else:
        # Reverse-mapping form: y_in = k*x_out + y_out + ty.
        nh = int(math.ceil(h + abs(k) * (w - 1)))
        ty = -k * (w - 1) if k > 0 else 0.0
        return _xform_rgba(p, (w, nh), Image.AFFINE, (1, 0, 0, k, 1, ty))


def _partial_mirror(patch: Image.Image, rng: random.Random) -> Image.Image:
    """Mirror one half while leaving the other intact (breaks global similarity)."""
    p = patch.convert("RGBA")
    w, h = p.size
    if rng.random() < 0.5:
        left = p.crop((0, 0, w // 2, h))
        right = p.crop((w // 2, 0, w, h))
        left_m = ImageOps.mirror(left)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.alpha_composite(left_m, (0, 0))
        canvas.alpha_composite(right, (w // 2, 0))
        return canvas
    else:
        top = p.crop((0, 0, w, h // 2))
        bot = p.crop((0, h // 2, w, h))
        bot_f = ImageOps.flip(bot)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.alpha_composite(top, (0, 0))
        canvas.alpha_composite(bot_f, (0, h // 2))
        return canvas

def _perspective_keystone(patch: Image.Image, rng: random.Random, skew_frac: Tuple[float, float] = (0.09, 0.18)) -> Image.Image:
    """Subtle perspective (keystone) warp: compress one side to create a trapezoid-like distortion."""
    p = patch.convert("RGBA")
    w, h = p.size
    if w < 3 or h < 3:
        return p
    mode = rng.choice(["top", "bottom", "left", "right"])
    frac = rng.uniform(skew_frac[0], skew_frac[1])
    dx = int(round(frac * w))
    dy = int(round(frac * h))
    if mode == "top":
        quad = (dx, 0, w - dx, 0, w, h, 0, h)
    elif mode == "bottom":
        quad = (0, 0, w, 0, w - dx, h, dx, h)
    elif mode == "left":
        quad = (0, 0, w, dy, w, h - dy, 0, h)
    else:  # right
        quad = (0, dy, w, 0, w, h, 0, h - dy)
    return _xform_rgba(p, (w, h), Image.QUAD, quad, resample=Image.BICUBIC)


def _mesh_warp_wave(patch: Image.Image, rng: random.Random) -> Image.Image:
    """Sine-wave bend using a small mesh. Subtle, smooth non-linear warp."""
    p = patch.convert("RGBA")
    w, h = p.size
    if w < 4 or h < 4:
        return p
    grid = rng.choice([3, 4, 5])
    amp = rng.uniform(0.05, 0.12) * min(w, h)
    freq = rng.choice([1, 2])
    phase = rng.uniform(0.0, 2 * math.pi)
    axis = rng.choice(["x", "y"])

    def wave_dx(x, y):
        if axis == "x":
            v = y / max(1.0, h - 1)
            return amp * math.sin(2 * math.pi * freq * v + phase), 0.0
        else:
            u = x / max(1.0, w - 1)
            return 0.0, amp * math.sin(2 * math.pi * freq * u + phase)

    xs = [int(round(i * w / grid)) for i in range(grid)] + [w]
    ys = [int(round(j * h / grid)) for j in range(grid)] + [h]
    mesh = []
    for j in range(grid):
        for i in range(grid):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[j], ys[j+1]
            bbox = (x0, y0, x1, y1)
            quad = []
            for (xx, yy) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                dx, dy = wave_dx(xx, yy)
                quad.extend([xx + dx, yy + dy])
            mesh.append((bbox, tuple(quad)))
    return _xform_rgba(p, (w, h), Image.MESH, mesh, resample=Image.BICUBIC)


def _radial_barrel_pincushion(patch: Image.Image, rng: random.Random) -> Image.Image:
    """Radial barrel/pincushion distortion via mesh; breaks similarity while remaining subtle."""
    p = patch.convert("RGBA")
    w, h = p.size
    if w < 4 or h < 4:
        return p
    grid = rng.choice([4, 5, 6])
    k = rng.uniform(0.06, 0.14) * rng.choice([-1.0, 1.0])
    cx, cy = w / 2.0, h / 2.0
    R = max(w, h) / 2.0

    xs = [int(round(i * w / grid)) for i in range(grid)] + [w]
    ys = [int(round(j * h / grid)) for j in range(grid)] + [h]
    mesh = []
    for j in range(grid):
        for i in range(grid):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[j], ys[j+1]
            bbox = (x0, y0, x1, y1)
            quad = []
            for (xx, yy) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                dx = xx - cx
                dy = yy - cy
                r = (dx*dx + dy*dy) ** 0.5
                t = (r / R) if R > 1e-6 else 1.0
                factor = 1.0 + k * (t * t)
                sx = cx + dx * factor
                sy = cy + dy * factor
                quad.extend([sx, sy])
            mesh.append((bbox, tuple(quad)))
    return _xform_rgba(p, (w, h), Image.MESH, mesh, resample=Image.BICUBIC)


def _mild_swirl(patch: Image.Image, rng: random.Random) -> Image.Image:
    """Centered swirl via mesh; angle decays with radius to keep it subtle and smooth."""
    p = patch.convert("RGBA")
    w, h = p.size
    if w < 4 or h < 4:
        return p
    grid = rng.choice([5, 6])
    max_deg = rng.uniform(10, 22)
    max_ang = math.radians(max_deg) * rng.choice([-1.0, 1.0])
    cx, cy = w / 2.0, h / 2.0
    R = max(w, h) / 2.0
    falloff_p = rng.uniform(0.8, 1.6)

    xs = [int(round(i * w / grid)) for i in range(grid)] + [w]
    ys = [int(round(j * h / grid)) for j in range(grid)] + [h]
    mesh = []
    for j in range(grid):
        for i in range(grid):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[j], ys[j+1]
            bbox = (x0, y0, x1, y1)
            quad = []
            for (xx, yy) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                dx = xx - cx
                dy = yy - cy
                r = (dx*dx + dy*dy) ** 0.5
                t = min(1.0, (r / R) if R > 1e-6 else 1.0)
                ang = max_ang * (1.0 - t ** falloff_p)
                ca = math.cos(ang); sa = math.sin(ang)
                sx = cx + dx * ca - dy * sa
                sy = cy + dx * sa + dy * ca
                quad.extend([sx, sy])
            mesh.append((bbox, tuple(quad)))
    return _xform_rgba(p, (w, h), Image.MESH, mesh, resample=Image.BICUBIC)


# Breaker registry
BREAKER_REGISTRY: Dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "anisotropic_scale": _anisotropic_scale,
    "shear_x":           lambda p, r: _shear(p, r, "x"),
    "shear_y":           lambda p, r: _shear(p, r, "y"),
    "perspective_keystone": _perspective_keystone,
    "mesh_warp_wave": _mesh_warp_wave,
    "radial_barrel_pincushion": _radial_barrel_pincushion,
    "mild_swirl": _mild_swirl,
    "partial_mirror": _partial_mirror,
}

def _make_dissimilar_tile(
    patch: Image.Image,
    tile_px: int,
    rng: random.Random,
    enabled_breakers: Dict[str, bool],
) -> Optional[Tuple[Image.Image, Image.Image, str]]:
    # Choose from enabled breakers, apply, then center on the tile
    active = [name for name, on in enabled_breakers.items() if on and name in BREAKER_REGISTRY]
    if not active:
        return None
    for _ in range(8):  # a few attempts if bounds fail
        name = rng.choice(active)
        # Pad input before applying any breaker to avoid clipping during warps
        pad_px = max(2, int(round(DISSIMILAR_INPUT_PAD_FRAC * max(patch.size))))
        padded = ImageOps.expand(ensure_transparent(patch), border=pad_px, fill=(0, 0, 0, 0))

        b = BREAKER_REGISTRY[name](padded, rng)
        b = ensure_transparent(b)
        b = tight_crop_rgba(b)
        b = _scale_to_fit_with_margin(b, tile_px)
        x = (tile_px - b.width) // 2
        y = (tile_px - b.height) // 2
        x, y = _clamp_xy_into_tile(x, y, b.width, b.height, tile_px)
        if bounds_ok(tile_px, x, y, b.width, b.height):
            return compose_tile_with_patch(tile_px, b, (x, y)), b, name
    return None


# ----------------------------- task -----------------------------
@register_task
class TransformSimilarityIdentifyTask(Task):
    """
    Show the original motif (graph paper) on top. On the bottom row, show N options.
    Two variants (default 50/50):

      • Variant "one_similar": exactly 1 option is *similar* to the top
        (uniform scaling allowed, plus rotation/reflection/translation).

      • Variant "one_dissimilar": exactly 1 option is *not similar* to the top;
        the rest are similar.

    Prompts and transform toggles are defined at the top of this file.
    """
    name = "transform_similarity_identify"

    def __init__(self,
                 p_one_similar: float = P_ONE_SIMILAR,
                 options_k_choices: Tuple[int, ...] = OPTIONS_K_CHOICES,
                 # generation toggles (default to top-of-file switches)
                 similar_d4_keys: Optional[List[str]] = None,
                 allow_translation_in_similar: Optional[bool] = None,
                 enable_uniform_scale: Optional[bool] = None,
                 enabled_breakers: Optional[Dict[str, bool]] = None,
                 # verification keys (conceptual similarity)
                 verification_d4_keys: Optional[List[str]] = None,
                 canon_px: Optional[int] = None):
        self.min_delta = float(IMG_DIFF_MIN)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.eq_tol = float(IMG_EQUAL_TOL)
        self.eq_bits = int(EQUAL_HASH_MAX_BITS)
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.tile_px = int(OUT_CELL)

        # knobs
        self.p_one_similar = float(p_one_similar)
        self.options_k_choices = tuple(int(k) for k in options_k_choices)
        self.similar_d4_keys = list(SIMILAR_D4_KEYS if similar_d4_keys is None else similar_d4_keys)
        self.allow_translation_in_similar = SIMILAR_ALLOW_TRANSLATION if allow_translation_in_similar is None else bool(allow_translation_in_similar)
        self.enable_uniform_scale = SIMILAR_ENABLE_UNIFORM_SCALE if enable_uniform_scale is None else bool(enable_uniform_scale)
        self.enabled_breakers = dict(DISSIMILAR_BREAKERS_ENABLED if enabled_breakers is None else enabled_breakers)
        self.verify_keys = list(VERIFICATION_D4_KEYS if verification_d4_keys is None else verification_d4_keys)
        self.canon_px = int(canon_px) if canon_px is not None else max(96, self.tile_px // 2)

    # ----- build top/original -----
    def _build_top(self, motif, rng: random.Random) -> Tuple[Image.Image, Image.Image, Tuple[int, int], Any]:
        spec = _prefer_asym_mode(motif, motif.sample_spec(rng))
        raw = motif.render(spec)
        raw = ensure_transparent(raw)
        patch = tight_crop_rgba(raw)
        patch = scale_patch(patch, self.tile_px, rng)
        patch = _scale_to_fit_with_margin(patch, self.tile_px)
        cx, cy = center_xy(self.tile_px, patch.width, patch.height)
        top = compose_tile_with_patch(self.tile_px, patch, (cx, cy))
        return top, patch, (cx, cy), spec

    # Public API ------------------------------------------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # weighted motif order (without replacement), mirroring other tasks
        weights_map = globals().get("MOTIF_WEIGHTS", {})
        if weights_map:
            kinds_all = [k for k in motif_impls.keys() if float(weights_map.get(k, 0.0)) > 0.0]
            if not kinds_all:
                raise RuntimeError(f"{self.name}: no allowed motifs available.")
            motif_order = weighted_order(rng, kinds_all, weights_map)
        else:
            # Fallback to uniform if weights are missing
            kinds_all = list(motif_impls.keys())
            if not kinds_all:
                raise RuntimeError(f"{self.name}: no allowed motifs available.")
            motif_order = weighted_order(rng, kinds_all, {k: 1.0 for k in kinds_all})

        font = load_font()

        failures: List[str] = []
        for mk in motif_order:
            motif = motif_impls[mk]
            for _ in range(self.max_retries):
                try:
                    top_tile, patch, centered_xy, spec = self._build_top(motif, rng)
                except Exception:
                    continue

                # Variant & number of options
                variant = "one_similar" if rng.random() < self.p_one_similar else "one_dissimilar"
                num_options = int(rng.choice(self.options_k_choices))
                labels = _labels_for_n(num_options)

                # Build options
                picks: List[Tuple[str, Image.Image, Image.Image, Optional[str]]] = []  # (kind, tile, patch_used, breaker_name)
                used_hashes = set()

                def add_if_unique(kind: str, tile: Image.Image, patch_used: Image.Image, breaker_name: Optional[str] = None) -> bool:
                    h = sig(tile)
                    if h in used_hashes:
                        return False
                    # ensure options are pairwise distinct visually
                    if any(not strong_distinct(tile, t, self.opt_min_delta, self.opt_hash_min_bits) for _, t, _, _ in picks):
                        return False
                    used_hashes.add(h)
                    picks.append((kind, tile, patch_used, breaker_name))
                    return True

                # Target counts
                target_sim = 1 if variant == "one_similar" else (num_options - 1)

                # 1) Build required similar options
                tries = 0
                while sum(1 for k,_,_,_ in picks if k == "similar") < target_sim and tries < 40:
                    built = _make_similar_tile(
                        patch, self.tile_px, centered_xy, rng,
                        enable_uniform_scale=self.enable_uniform_scale,
                        allow_translation=self.allow_translation_in_similar,
                        similar_d4_keys=self.similar_d4_keys,
                    )
                    tries += 1
                    if built is None:
                        continue
                    tri_tile, tri_patch = built
                    if not _is_similar_under_keys(patch, tri_patch, self.canon_px, self.eq_tol, self.eq_bits, self.verify_keys):
                        continue
                    add_if_unique("similar", tri_tile, tri_patch, None)

                if sum(1 for k,_,_,_ in picks if k == "similar") < target_sim:
                    continue  # retry motif/spec

                # 2) Fill remaining with dissimilar options
                tries = 0
                while len(picks) < num_options and tries < 80:
                    built = _make_dissimilar_tile(patch, self.tile_px, rng, self.enabled_breakers)
                    tries += 1
                    if built is None:
                        continue
                    tri_tile, tri_patch, breaker_name = built
                    if _is_similar_under_keys(patch, tri_patch, self.canon_px, self.eq_tol, self.eq_bits, self.verify_keys):
                        continue
                    add_if_unique("dissimilar", tri_tile, tri_patch, breaker_name)

                if len(picks) < num_options:
                    continue

                # 3) Ensure the variant has exactly one target
                if variant == "one_dissimilar":
                    num_dis = sum(1 for k,_,_,_ in picks if k == "dissimilar")
                    if num_dis != 1:
                        # Try to enforce exactly one dissimilar
                        if num_dis == 0:
                            got = False
                            for _ in range(30):
                                built = _make_dissimilar_tile(patch, self.tile_px, rng, self.enabled_breakers)
                                if built is None:
                                    continue
                                tri_tile, tri_patch, breaker_name = built
                                if _is_similar_under_keys(patch, tri_patch, self.canon_px, self.eq_tol, self.eq_bits, self.verify_keys):
                                    continue
                                idx = rng.randrange(len(picks))
                                picks[idx] = ("dissimilar", tri_tile, tri_patch, breaker_name)
                                got = True
                                break
                            if not got:
                                continue
                        else:  # >1 dissimilar
                            keep = rng.choice([i for i,(k,_,_,_) in enumerate(picks) if k == "dissimilar"])
                            for i,(k,_,_,_) in enumerate(picks):
                                if k == "dissimilar" and i != keep:
                                    ok = False
                                    for _ in range(30):
                                        built = _make_similar_tile(
                                            patch, self.tile_px, centered_xy, rng,
                                            enable_uniform_scale=self.enable_uniform_scale,
                                            allow_translation=self.allow_translation_in_similar,
                                            similar_d4_keys=self.similar_d4_keys,
                                        )
                                        if built is None:
                                            continue
                                        tri_tile, tri_patch = built
                                        if not _is_similar_under_keys(patch, tri_patch, self.canon_px, self.eq_tol, self.eq_bits, self.verify_keys):
                                            continue
                                        picks[i] = ("similar", tri_tile, tri_patch, None)
                                        ok = True
                                        break
                                    if not ok:
                                        break
                            if sum(1 for k,_,_,_ in picks if k == "dissimilar") != 1:
                                continue
                else:  # variant == "one_similar"
                    num_sim = sum(1 for k,_,_,_ in picks if k == "similar")
                    if num_sim != 1:
                        while num_sim > 1:
                            indices = [i for i,(k,_,_,_) in enumerate(picks) if k == "similar"]
                            if not indices:
                                break
                            idx = rng.choice(indices)
                            ok = False
                            for _ in range(30):
                                built = _make_dissimilar_tile(patch, self.tile_px, rng, self.enabled_breakers)
                                if built is None:
                                    continue
                                tri_tile, tri_patch, breaker_name = built
                                if _is_similar_under_keys(patch, tri_patch, self.canon_px, self.eq_tol, self.eq_bits, self.verify_keys):
                                    continue
                                picks[idx] = ("dissimilar", tri_tile, tri_patch, breaker_name)
                                ok = True
                                break
                            if not ok:
                                break
                            num_sim = sum(1 for k,_,_,_ in picks if k == "similar")
                        if sum(1 for k,_,_,_ in picks if k == "similar") != 1:
                            continue

                # 4) Final uniqueness check
                if not pairwise_unique([im for _,im,_,_ in picks], self.opt_min_delta, self.opt_hash_min_bits):
                    continue

                rng.shuffle(picks)

                tile_border_w = max(3, self.tile_px // 48)
                option_tiles = [
                    crisp_option_tile(add_tile_border(img, width_px=tile_border_w), lab, font)
                    for (_, img, _, _), lab in zip(picks, labels)
                ]

                # Identify the correct label
                if variant == "one_similar":
                    correct_index = next(i for i,(k,_,_,_) in enumerate(picks) if k == "similar")
                else:
                    correct_index = next(i for i,(k,_,_,_) in enumerate(picks) if k == "dissimilar")
                correct_label = labels[correct_index]

                # Compose
                composite = _compose_top_bottom_variable(top_tile, option_tiles, self.tile_px)

                # Build prompt
                question = _format_prompt(variant, labels, rng)

                meta = {
                    "pattern_kind": "transform",
                    "pattern": self.name,
                    "grid": (1, len(picks)),
                    "mask_idx": -1,
                    "variant": variant,
                    "motif_kind": mk,
                    "labels": labels,
                    "answer": correct_label,
                    "option_strategy": "image_similarity_mixed",
                    "option_descs": [k for (k,_,_,_) in picks],  # "similar" / "dissimilar" flags per option (for debugging)
                    "option_payload": {
                        "num_options": len(picks),
                        "canon_px": self.canon_px,
                        "p_one_similar": self.p_one_similar,
                        "allow_translation_in_similar": self.allow_translation_in_similar,
                        "enable_uniform_scale": self.enable_uniform_scale,
                        "similar_d4_keys": self.similar_d4_keys,
                        "verification_d4_keys": self.verify_keys,
                        "enabled_breakers": self.enabled_breakers,
                    },
                    "question": question,
                    "composite_ready": True,
                }
                return composite, [spec], meta

            failures.append(mk)

        tried = ", ".join(failures) if failures else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a verifiable sample after {self.max_retries} attempts per motif. "
            f"Motifs tried (in order): {tried}"
        )
