from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple
from PIL import Image
import math
import numpy as np

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES, OUT_CELL
from sphinx.utils.drawing import tight_crop_rgba

from sphinx.charts import (
    ChartSpec,
    CHART_MIN_K, CHART_MAX_K,
    compute_chart_complexity,
    sample_category_labels,
    sample_percentages_int,
    choose_colors,
    render_pie_chart, render_bar_chart,
)

# composition utilities (same as sequence tasks)
from sphinx.utils.drawing import (
    load_font, crisp_option_tile, labels_default,
    compose_top_bottom, compose_options_row, _pad_width_center
)

# ------------------------------- knobs ---------------------------------
# At least one distractor must differ from the correct color→share mapping
# by either a RELATIVE or ABSOLUTE amount (per color).
DISTRACTOR_MIN_REL = 0.25    # e.g., 25% relative change  (|q-p| / max(p,1))
DISTRACTOR_MIN_ABS = 12       # e.g., 12 percentage points absolute change

# --- height harmonization knobs for the TOP chart --------------------------
# Target the top chart’s content height to be similar to the options row height.
TOP_HEIGHT_MIN_FRAC = 0.90   # try to be at least 90% of options-row height
TOP_HEIGHT_MAX_FRAC = 1.08   # but no more than 108%
TOP_WIDTH_FRAC_RELAX = 0.92  # if height target can't be met due to width cap, relax width cap up to 92%

# Layout tuning for the TOP chart footprint (keeps pies smaller, trims bar whitespace)
TOP_CONTENT_MAX_FRAC = 0.78  # scale top content to ≤ this fraction of the options-row width
TOP_CROP_WHITE_THR   = 6     # white-threshold for tight cropping before scaling
TOP_SEP_PX           = 40    # pixels between top chart and options row


# --- option tile tuning: make bars taller in the square ----------------
OPTION_SUPERRES = 2            # keep 2 for crispness; set 1 if you want speed
OPTION_CONTENT_FRAC = 0.96     # was 0.92: bars/pies can occupy more of the tile
OPTION_BAR_BASE_W_FRAC = 0.72  # was 0.78: render on a narrower canvas → taller bars


# (no change to your DISTRACTOR_* or TOP_* knobs)


# --- stronger numeric separation among distractors ---
PAIRWISE_L1_MIN = 18           # total-variation floor between any two distractors
PAIRWISE_LARGE_COORDS_MIN = 2  # at least this many coordinates must be "large-diff"

# --- image-level distinctness among distractors (after rendering) ---
VIS_MIN_DIFF_FRAC = 0.02      # ≥2% of non-white pixels must dif
# -----------------------------------------------------------------------


# ---------------------------------------------------------------------
# Prompts (explicitly name which chart is on top)
PROMPTS_PIE_TOP = [
    "The top image shows a pie chart. Below it are four bar charts labeled (a)–(d). Which bar chart has the same color proportions as the pie chart?",
    "You are given a pie chart on top and four bar charts underneath. Which one of the options (a)–(d) exactly preserves the pie’s color distribution?",
    "At the top is a pie chart, and at the bottom are four bar charts. Which bar chart represents the same proportions by color as the pie?",
    "A pie chart is shown on top, followed by four bar charts. Which bar chart (a–d) matches the proportions of the slices in the pie?",
    "The top figure is a pie chart. The four options below are bar charts. Which one reflects the same color proportions as the pie?",
    "On top you see a pie chart. Underneath are four bar charts. Which option (a–d) encodes the identical distribution of colors as the pie?",
    "The top diagram is a pie chart. The bottom row contains four bar charts. Which bar chart matches the same proportional breakdown by color as the pie?",
    "You see a pie chart above and four bar charts below. Which bar chart (a–d) reproduces the same proportions as the pie?",
    "The top image is a pie chart, and the bottom row shows four bar charts. Which one corresponds to the same color distribution as the pie?",
    "A pie chart is displayed on top, with four bar charts below. Which option (a–d) encodes the same proportions as the top pie chart?",
]

PROMPTS_BAR_TOP = [
    "The top image shows a bar chart. Below it are four pie charts labeled (a)–(d). Which pie chart has the same color proportions as the bar chart?",
    "You are given a bar chart on top and four pie charts underneath. Which one of the options (a)–(d) exactly preserves the bar chart’s color distribution?",
    "At the top is a bar chart, and at the bottom are four pie charts. Which pie chart represents the same proportions by color as the bar chart?",
    "A bar chart is shown on top, followed by four pie charts. Which pie chart (a–d) matches the proportions of the bars in the chart above?",
    "The top figure is a bar chart. The four options below are pie charts. Which one reflects the same color proportions as the bar chart?",
    "On top you see a bar chart. Underneath are four pie charts. Which option (a–d) encodes the identical distribution of colors as the bar chart?",
    "The top diagram is a bar chart. The bottom row contains four pie charts. Which pie chart matches the same proportional breakdown by color as the bar chart?",
    "You see a bar chart above and four pie charts below. Which pie chart (a–d) reproduces the same proportions as the bar chart?",
    "The top image is a bar chart, and the bottom row shows four pie charts. Which one corresponds to the same color distribution as the bar chart?",
    "A bar chart is displayed on top, with four pie charts below. Which option (a–d) encodes the same proportions as the top bar chart?",
]

# ---------------------------------------------------------------------

def _sample_distinct_percentages(rng: random.Random, k: int, max_tries: int = 5000) -> List[int]:
    """
    Sample integer percentages that sum to 100 with all values distinct (>=1).
    Mirrors the distinctness goal used in bar sorting to avoid ambiguity.
    """
    for _ in range(max_tries):
        p = sample_percentages_int(rng, k, enforce_min1=True)
        if len(set(p)) == k:
            return p
    # Robust fallback: strictly-increasing base + randomized remainder (as in charts_bar).
    base = list(range(1, k + 1))
    R = max(0, 100 - sum(base))
    if k > 1:
        positions = sorted(rng.sample(range(R + k - 1), k - 1)) if (R + k - 1) > 0 else list(range(k - 1))
        inc, prev = [], -1
        for pos in positions + [R + k - 1]:
            inc.append(pos - prev - 1); prev = pos
        inc.sort()
        p_sorted = [b + d for b, d in zip(base, inc)]
    else:
        p_sorted = [100]
    rng.shuffle(p_sorted)
    diff = 100 - sum(p_sorted)
    if diff != 0:
        i = rng.randrange(k)
        p_sorted[i] = max(1, p_sorted[i] + diff)
    return p_sorted

def _permute(p: List[int], rng: random.Random) -> List[int]:
    k = len(p)
    idx = list(range(k))
    rng.shuffle(idx)
    if all(i == j for i, j in enumerate(idx)):  # avoid identity
        idx = idx[-1:] + idx[:-1]
    return [p[j] for j in idx]

def _jitter(p: List[int], rng: random.Random, *, delta_max: int = 3) -> List[int]:
    """Move a few percent points between two categories, keep sum=100 and each >=1."""
    k = len(p); a, b = rng.sample(range(k), 2)
    take = rng.randint(1, min(delta_max, max(1, p[a] - 1)))
    q = p[:]
    q[a] -= take; q[b] += take
    return q

def _resize_to_square_rgba(im: Image.Image, side: int) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im.resize((side, side), Image.LANCZOS)

from sphinx.utils.drawing import tight_crop_rgba  # already used in sequence tasks :contentReference[oaicite:2]{index=2}

def _meets_diff_threshold(p: List[int], q: List[int]) -> bool:
    """Return True if ∃ color i with either rel ≥ DISTRACTOR_MIN_REL or abs ≥ DISTRACTOR_MIN_ABS."""
    for a, b in zip(p, q):
        absd = abs(b - a)
        reld = absd / float(max(a, 1))
        if absd >= DISTRACTOR_MIN_ABS or reld >= DISTRACTOR_MIN_REL:
            return True
    return False

def _enforce_strong_distractor(
    rng: random.Random, p: List[int], wrongs: List[List[int]],
    max_tries: int = 500
) -> List[List[int]]:
    """
    Ensure at least ONE wrong candidate satisfies the strong-difference rule.
    If none do, synthesize a jittered one that does.
    """
    if any(_meets_diff_threshold(p, w) for w in wrongs):
        return wrongs

    k = len(p)
    q = p[:]
    for _ in range(max_tries):
        i, j = rng.sample(range(k), 2)
        need_abs = max(1, DISTRACTOR_MIN_ABS)
        need_rel = int(round(DISTRACTOR_MIN_REL * max(p[i], 1)))
        need = max(need_abs, need_rel)

        take = rng.randint(need, min(max(need, 5), max(1, q[j] - 1))) if q[j] > 1 else 1
        q2 = q[:]
        # push away from p[i]
        if q2[i] >= p[i]:
            # reducing i and increasing j
            if q2[i] - take >= 1:
                q2[i] -= take; q2[j] += take
            else:
                continue
        else:
            # increasing i and decreasing j
            if q2[j] - take >= 1:
                q2[i] += take; q2[j] -= take
            else:
                continue

        if len(set(q2)) == k and sum(q2) == 100 and _meets_diff_threshold(p, q2):
            wrongs = wrongs[:-1] + [q2] if wrongs else [q2]
            return wrongs

    # Hard fallback: force a single index to hit absolute threshold, repair sum via another index.
    q2 = p[:]
    i = rng.randrange(k)
    j = (i + 1) % k
    need_abs = max(DISTRACTOR_MIN_ABS, 1)
    delta = need_abs if q2[i] <= p[i] else -need_abs
    if q2[i] + delta >= 1 and q2[j] - delta >= 1:
        q2[i] += delta; q2[j] -= delta
    wrongs = wrongs[:-1] + [q2] if wrongs else [q2]
    return wrongs

def _render_pie_no_legend(spec: ChartSpec) -> Image.Image:
    """Try to hide legend if supported; else fall back gracefully."""
    try:
        return render_pie_chart(spec, show_values=False, show_legend=False)
    except TypeError:
        return render_pie_chart(spec, show_values=False)

def _render_bar_no_labels(spec: ChartSpec) -> Image.Image:
    """Try to hide per-bar labels if supported; else fall back gracefully."""
    try:
        return render_bar_chart(spec, show_values=False, show_labels=False)
    except TypeError:
        return render_bar_chart(spec, show_values=False)

def _prepare_top_canvas(
    top_img: Image.Image,
    target_width: int,
    ref_height: int,
    *,
    prefer_height: bool = False,
    relax_w_frac: float = TOP_WIDTH_FRAC_RELAX
) -> Image.Image:
    """
    Tight-crop, then scale to meet the height band [TOP_HEIGHT_MIN_FRAC, TOP_HEIGHT_MAX_FRAC]·ref_height.
    If prefer_height=True (bars), we allow *anisotropic* scaling so height hits the band
    while width stays within the (possibly relaxed) width cap.
    """
    rgba = top_img.convert("RGBA")
    cropped = tight_crop_rgba(rgba, white_threshold=TOP_CROP_WHITE_THR, pad=2) or rgba

    cw, ch = cropped.width, cropped.height
    if cw <= 0 or ch <= 0:
        cropped = rgba; cw, ch = cropped.width, cropped.height

    max_w_strict = int(TOP_CONTENT_MAX_FRAC * target_width)
    h_min = int(TOP_HEIGHT_MIN_FRAC * ref_height)
    h_max = int(TOP_HEIGHT_MAX_FRAC * ref_height)

    s_w = max_w_strict / float(cw)
    s_h_min = h_min / float(ch)
    s_h_max = h_max / float(ch)

    if prefer_height:
        # Try to hit the height band; if width would overflow, allow slight width relaxation.
        max_w_cap = max_w_strict
        if s_h_min > s_w:
            max_w_cap = int(relax_w_frac * target_width)
        sy = min(s_h_max, max(s_h_min, 0.01))
        sx = min(sy, max_w_cap / float(cw))
        nw, nh = max(1, int(round(cw * sx))), max(1, int(round(ch * sy)))
    else:
        # Isotropic as before.
        if s_h_min > s_w:
            s_w = (relax_w_frac * target_width) / float(cw)
        s = min(s_w, max(s_h_min, 0.01), s_h_max)
        if ch * s < h_min:
            s = min(s_w, s_h_max)
        nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))

    scaled = cropped.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (target_width, nh), (255, 255, 255, 255))
    canvas.alpha_composite(scaled, ((target_width - nw) // 2, 0))
    return canvas


def _l1(u: List[int], v: List[int]) -> int:
    return sum(abs(a - b) for a, b in zip(u, v))

def _count_large_coords(u: List[int], v: List[int]) -> int:
    cnt = 0
    for a, b in zip(u, v):
        absd = abs(a - b)
        reld = absd / float(max(max(a, b), 1))
        if absd >= DISTRACTOR_MIN_ABS or reld >= DISTRACTOR_MIN_REL:
            cnt += 1
    return cnt

def _pairwise_threshold_ok(u: List[int], v: List[int]) -> bool:
    """
    Two distractors are 'different enough' iff BOTH:
      • at least PAIRWISE_LARGE_COORDS_MIN coordinates are large-diff, and
      • L1(u, v) ≥ PAIRWISE_L1_MIN.
    """
    return (_count_large_coords(u, v) >= PAIRWISE_LARGE_COORDS_MIN) and (_l1(u, v) >= PAIRWISE_L1_MIN)


def _img_diff_frac(a: Image.Image, b: Image.Image, *, white_thr: int = 245, pix_thr: int = 18) -> float:
    A = np.asarray(a.convert("RGB"), dtype=np.int16)
    B = np.asarray(b.convert("RGB"), dtype=np.int16)
    mask = ((A < white_thr).any(axis=-1)) | ((B < white_thr).any(axis=-1))
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    diff = np.abs(A - B).sum(axis=-1)  # [0..765]
    changed = (diff > pix_thr) & mask
    return float(changed.sum()) / float(denom)

def _visual_pairwise_ok(images: List[Image.Image]) -> bool:
    n = len(images)
    for i in range(n):
        for j in range(i + 1, n):
            if _img_diff_frac(images[i], images[j]) < VIS_MIN_DIFF_FRAC:
                return False
    return True

def _fit_into_square(im: Image.Image, side: int, frac: float, *, prefer_height: bool = False) -> Image.Image:
    """
    Tight-crop white margins, then fit the content into a side×side square.

    If prefer_height=True (bars), we scale the Y dimension to hit the target height
    (≈ frac*side) and compress X if necessary so the width stays within the square.
    For pies (prefer_height=False), we keep isotropic scaling.
    """
    rgba = im.convert("RGBA")
    cropped = tight_crop_rgba(rgba, white_threshold=TOP_CROP_WHITE_THR, pad=2) or rgba
    cw, ch = cropped.size
    if cw <= 0 or ch <= 0:
        cropped = rgba; cw, ch = cropped.size

    max_w = int(side * frac)
    max_h = int(side * frac)

    if prefer_height:
        # Hit the target height; compress width only if it would overflow.
        sy = max_h / float(ch)
        sx = min(sy, max_w / float(cw))
        nw, nh = max(1, int(round(cw * sx))), max(1, int(round(ch * sy)))
    else:
        # Usual isotropic fit
        s = min(max_w / float(cw), max_h / float(ch))
        nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))

    scaled = cropped.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.alpha_composite(scaled, ((side - nw) // 2, (side - nh) // 2))
    return canvas

def _build_wrongs_pairwise_and_strong(
    rng: random.Random, p: List[int], *, max_tries: int = 200
) -> List[List[int]]:
    """
    Build 3 distractors deterministically (+ light randomness) by moving delta
    from a donor slice j to a target slice i, guaranteeing:
      • _meets_diff_threshold(p, q) for each q (strong vs. correct),
      • _pairwise_threshold_ok for every pair of q's.
    """
    k = len(p)
    idx_small = sorted(range(k), key=lambda i: p[i])           # asc by size
    idx_large = sorted(range(k), key=lambda i: p[i], reverse=True)  # desc by size

    # Helper: make one q by moving delta from j -> i
    def make_shift(i: int, j: int, base_bump: int) -> List[int] | None:
        if i == j:
            return None
        # Need delta big enough to be 'strong' on i (we increase i)
        need_i = max(DISTRACTOR_MIN_ABS, int(math.ceil(DISTRACTOR_MIN_REL * max(p[i], 1))))
        max_take = p[j] - 1
        if max_take < need_i:
            return None
        delta = min(max_take, need_i + base_bump)
        q = p[:]
        q[i] += delta
        q[j] -= delta
        return q

    wrongs: List[List[int]] = []
    bumps = [0, 2, 4]  # distinct magnitudes → helps pairwise separation

    # Prefer using distinct targets/donors across the three q's
    cand_pairs: List[Tuple[int, int, int]] = []
    # Try to pair the three smallest with the three largest (cyclic if needed)
    for t in range(3):
        i = idx_small[t % k]
        # choose donor j with enough mass (scan larges)
        j = None
        need_i = max(DISTRACTOR_MIN_ABS, int(math.ceil(DISTRACTOR_MIN_REL * max(p[i], 1))))
        for jl in idx_large:
            if jl != i and (p[jl] - 1) >= need_i:
                j = jl; break
        if j is None:
            # fallback: pick any donor with enough mass
            cand = [jj for jj in range(k) if jj != i and (p[jj] - 1) >= need_i]
            if not cand:
                continue
            j = rng.choice(cand)
        cand_pairs.append((i, j, bumps[t]))

    # Build q's from the candidate pairs; enforce pairwise separation
    for (i, j, bump) in cand_pairs:
        q = make_shift(i, j, bump)
        if q is None:
            continue
        if not _meets_diff_threshold(p, q):
            continue
        if all(_pairwise_threshold_ok(q, w) for w in wrongs):
            wrongs.append(q)
        if len(wrongs) == 3:
            break

    # If we still have < 3, escalate with randomized 2-pair shifts.
    tries = 0
    while len(wrongs) < 3 and tries < max_tries:
        tries += 1
        i1, j1 = rng.sample(range(k), 2)
        i2, j2 = rng.sample(range(k), 2)
        if len({i1, j1, i2, j2}) < 3 or i1 == j1 or i2 == j2:
            continue

        def need_for(i):  # make i 'strong'
            return max(DISTRACTOR_MIN_ABS, int(math.ceil(DISTRACTOR_MIN_REL * max(p[i], 1))))

        d1 = min(max(p[j1] - 1, 0), need_for(i1) + rng.randint(0, 3))
        d2 = min(max(p[j2] - 1, 0), need_for(i2) + rng.randint(1, 4))
        if d1 < need_for(i1) or d2 < need_for(i2):
            continue

        q = p[:]
        q[i1] += d1; q[j1] -= d1
        q[i2] += d2; q[j2] -= d2

        if not _meets_diff_threshold(p, q):
            continue
        if all(_pairwise_threshold_ok(q, w) for w in wrongs):
            wrongs.append(q)

    if len(wrongs) < 3:
        # Final fallback: jitter until pairwise passes (should be rare with the constructive path)
        while len(wrongs) < 3:
            q = _jitter(p, rng, delta_max=8)
            if _meets_diff_threshold(p, q) and all(_pairwise_threshold_ok(q, w) for w in wrongs):
                wrongs.append(q)

    return wrongs


def _render_option_superres(s: ChartSpec) -> Image.Image:
    """
    Render an option chart at super-res, FIT that high-res bitmap into a square
    (height-biased for bars), then do a single anti-aliased downsample to OUT_CELL.
    This avoids the double-resize that was making bars look ugly for large k.
    """
    # High-res base canvas; slightly narrower for bars so width never dominates
    base_w = OUT_CELL * OPTION_SUPERRES
    base_h = OUT_CELL * OPTION_SUPERRES
    if s.chart_type == "bar":
        base_w = int(round(base_w * OPTION_BAR_BASE_W_FRAC))

    s_big = ChartSpec(
        seed=s.seed, chart_type=s.chart_type,
        labels=s.labels, value_kind=s.value_kind,
        counts=s.counts[:], percentages_int=s.percentages_int[:],
        colors=s.colors[:], color_mode=s.color_mode,
        width_px=base_w, height_px=base_h, render_mode=s.render_mode,
    )

    # Render at super-res
    base = _render_pie_no_legend(s_big) if s.chart_type == "pie" else _render_bar_no_labels(s_big)

    # IMPORTANT: fit the high-res image into a high-res square first (height-biased for bars)
    fitted_hr = _fit_into_square(
        base,
        side=OUT_CELL * OPTION_SUPERRES,
        frac=OPTION_CONTENT_FRAC,
        prefer_height=(s.chart_type == "bar")
    )

    # Single clean downsample to final tile size
    return fitted_hr.resize((OUT_CELL, OUT_CELL), Image.LANCZOS)


@register_task
class ChartsMatchProportionsTask(Task):
    """
    Cross-type proportion matching:
      • Top: either a PIE or a BAR (randomly chosen).
      • Bottom: 4 options of the OTHER type; colors are the legend between charts.
      • Exactly one option preserves the SAME color→proportion mapping as the top.
    Distinct integer percentages ensure unambiguous matching.
    """
    name = "charts_match_proportions"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _build_specs(self, rng: random.Random) -> Tuple[ChartSpec, ChartSpec, List[ChartSpec]]:
        seed = rng.randint(0, 2 ** 31 - 1)
        lrng = random.Random(seed)

        k = lrng.randint(CHART_MIN_K, CHART_MAX_K)
        labels = sample_category_labels(lrng, k)
        perc = _sample_distinct_percentages(lrng, k)  # all distinct (>=1), sum to 100
        colors, color_mode = choose_colors(lrng, k, mode="distinct")

        top_is_pie = bool(lrng.random() < 0.5)
        top_type = "pie" if top_is_pie else "bar"
        opt_type = "bar" if top_is_pie else "pie"

        # initial sizes (we'll crop+scale top later)
        top_w, top_h = 1024, 768

        top_spec = ChartSpec(
            seed=seed, chart_type=top_type,
            labels=labels, value_kind="percentage",
            counts=perc[:], percentages_int=perc,
            colors=colors, color_mode=color_mode,
            width_px=top_w, height_px=top_h, render_mode="color",
        )

        corr_spec = ChartSpec(
            seed=seed ^ 0xA5A5, chart_type=opt_type,
            labels=labels, value_kind="percentage",
            counts=perc[:], percentages_int=perc[:],
            colors=colors[:], color_mode=color_mode,
            width_px=OUT_CELL, height_px=OUT_CELL, render_mode="color",
        )

        # --- Build 3 distractor specs  ---
        wrongs: List[List[int]] = _build_wrongs_pairwise_and_strong(lrng, perc)

        wrong_specs: List[ChartSpec] = []
        for q in wrongs[:3]:
            wrong_specs.append(ChartSpec(
                seed=seed ^ (0x1111 * (len(wrong_specs) + 1)), chart_type=opt_type,
                labels=labels, value_kind="percentage",
                counts=q[:], percentages_int=q[:],
                colors=colors[:], color_mode=color_mode,
                width_px=OUT_CELL, height_px=OUT_CELL, render_mode="color",
            ))

        return top_spec, corr_spec, wrong_specs

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        for _ in range(self.max_retries):
            top_spec, corr_spec, wrong_specs = self._build_specs(rng)

            # crisper options + bars/pies same visible height inside tiles
            correct_img = _render_option_superres(corr_spec)
            wrong_imgs = [_render_option_superres(ws) for ws in wrong_specs[:3]]

            # Image-level distinctness (rarely triggers; catches near-duplicates)
            if not _visual_pairwise_ok(wrong_imgs):
                continue  # build a new instance

            # Pack options with labels (a)–(d)
            options = [(True, correct_img)] + [(False, im) for im in wrong_imgs]
            random.shuffle(options)
            font = load_font()
            label_strs = labels_default()
            option_tiles = [crisp_option_tile(im, lab, font) for (_, im), lab in zip(options, label_strs)]
            correct_idx = [i for i, (is_corr, _) in enumerate(options) if is_corr][0]
            correct_label = label_strs[correct_idx]

            # Compose the options row to get its exact width
            opts = compose_options_row(option_tiles)

            # 2) Now render the TOP chart and normalize its footprint to the options width
            if top_spec.chart_type == "pie":
                top_raw = _render_pie_no_legend(top_spec)
                prompt = random.choice(PROMPTS_PIE_TOP)
            else:
                top_raw = _render_bar_no_labels(top_spec)
                prompt = random.choice(PROMPTS_BAR_TOP)

            top_norm = _prepare_top_canvas(
                top_raw, opts.width, opts.height,
                prefer_height=(top_spec.chart_type == "bar")
            )
            composite = compose_top_bottom(top_norm, opts, sep_px=TOP_SEP_PX)

            k = len(top_spec.labels)
            color_hex = list(top_spec.colors)
            perc = list(top_spec.percentages_int)
            p_by_color = {c: int(p) for c, p in zip(color_hex, perc)}

            complexity = compute_chart_complexity(k)
            range_info = complexity.get("range")
            chart_complexity = {
                "score": complexity["score"],
                "level": complexity["level"],
            }
            if isinstance(range_info, dict):
                chart_complexity["range"] = dict(range_info)

            variant = {
                "kind": "proportion_match",
                "top_type": top_spec.chart_type,
                "option_type": corr_spec.chart_type,
                "k": k,
                "numbers_shown": False,
                "legend_shown": False,
                "colors_as_legend": True,
                "distinct_percentages": True,
                "distractor_kinds": ["permute", "jitter"],
                "distractor_min_rel": float(DISTRACTOR_MIN_REL),
                "distractor_min_abs": int(DISTRACTOR_MIN_ABS),
            }

            chart_top_info = {
                "type": top_spec.chart_type,
                "k": k,
                "value_kind": top_spec.value_kind,
                "percentages_by_color": p_by_color,
                "color_mode": top_spec.color_mode,
                "complexity": chart_complexity,
            }

            meta = {
                "pattern_kind": "charts",
                "pattern": self.name,
                "variant": variant,
                "question": prompt,
                "answer": correct_label,
                "chart_top": chart_top_info,
                "dims": (max(top_norm.width, opts.width), top_norm.height + TOP_SEP_PX + opts.height),
                "composite_ready": True,
            }
            meta["complexity"] = complexity
            meta["complexity_score"] = complexity["score"]
            meta["complexity_level"] = complexity["level"]
            meta["complexity_version"] = complexity["version"]
            return composite, [top_spec, corr_spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
