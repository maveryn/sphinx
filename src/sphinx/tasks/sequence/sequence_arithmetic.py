# sphinx/tasks/sequence/sequence_arithmetic.py
import math, random
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image, ImageDraw

from ...base import Task
from ...registry import register_task
from ...config import (
    OUT_CELL, IMG_DIFF_MIN, MAX_BUILD_RETRIES, OPT_HASH_MIN_BITS,
    OPT_UNIQUENESS_MIN, SS_CELL
)
from ...utils.image_compare import sig, diff_frac
from ...utils.drawing import _pad_width_center, tight_crop_rgba, ensure_cell_rgba_outcell
from ...utils.specs import _prefer_sym_mode
from .config import SEQ_CONFIG  # limits & min_delta
from ...utils.rng import weighted_order

from ...utils.drawing import (
    load_font, crisp_option_tile, labels_default,
    compose_row_with_mask, compose_top_bottom, compose_options_row
)

# Generation constants -------------------------------------------------------
NUM_PANELS = 4      # number of panels in the sequence
NUM_OPTIONS = 4     # number of answer choices
STEP_ATTEMPTS = 12  # retries for finding a valid step
STEP_CHOICES = [0, 1, 2, 3]            # candidate step sizes
STEP_WEIGHTS = [0.10, 0.50, 0.30, 0.10]  # probabilities for step sizes

# Complexity normalization bounds (max count among top-row panels)
MAX_COUNT_MIN = 3
MAX_COUNT_MAX = 18

MOTIF_WEIGHTS = {
    "icons": 5,
    "arc": 0.25,
    "bars": 0.25,
    "crescent": 0.75,
    "dot": 1.25,
    "gear": 0.125,
    "glyph": 0.25,
    "keyhole": 0.25,
    "ladder": 0.25,
    "pinwheel_triangles": 0.75,
    "polygon": 0.75,
    "rings": 0.25,
    "star_polygon": 0.125,
    "pictogram": 0.25,
}



PROMPT_TEMPLATES = {
    "general": [
        "The top row shows a sequence with one tile missing; options (a)–(d) are below. Which option completes the pattern?",
        "There is a blank cell in the top row; from the options (a)–(d) below, which tile completes the pattern?",
        "Which option (a)–(d) below best fills the blank in the top-row sequence?",
        "Which option (a)–(d) below should replace the blank to complete the pattern shown in the top row?",
        "Which tile from the options (a)–(d) below completes the top-row pattern?"
    ],
    "explicit": [
        "In the top row, only the number of shapes changes, and by a constant step. Which option (a)–(d) below completes the row?",
        "The top-row counts follow an arithmetic rule (same amount added or subtracted each step). Which option (a)–(d) fills the blank?",
        "Infer the fixed count difference from the visible tiles in the top row. Which option (a)–(d) belongs in the blank?",
        "Exactly one option (a)–(d) keeps the number of shapes changing by an equal step across the top row. Which is it?",
        "Focus on counts: the top row uses a constant difference between neighboring tiles. Which option (a)–(d) below completes the pattern?"
    ],
}

@register_task
class SequenceArithmeticTask(Task):
    """
    Sequence task where the rule is an arithmetic change in *how many* shapes appear.
    If a motif’s 'count' is effective, we vary it directly; otherwise we
    repeat the same rendered motif inside each cell with a fixed layout (row/col/grid/pyramid).
    """
    name = "sequence_arithmetic"

    def __init__(self):
        self.min_step_delta = float(SEQ_CONFIG.get("min_delta", IMG_DIFF_MIN))
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)
        self.count_lo, self.count_hi = SEQ_CONFIG["limits"]["count_lo_hi"]

    def _compute_complexity(self, peak_count: int) -> Dict[str, Any]:
        """Normalize the maximum number of shapes across the top row and map to EASY/HARD complexity."""
        min_total = MAX_COUNT_MIN
        max_total = MAX_COUNT_MAX
        span = max(1, max_total - min_total)
        normalized = (int(peak_count) - min_total) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "sequence-arithmetic-max-count-v1",
            "range": {"min_total": min_total, "max_total": max_total},
            "max_count": int(peak_count),
        }

    # --------- helpers ---------
    def _effective_count_bounds(self, motif) -> Tuple[int, int]:
        """Intersection of global count limits and motif's intrinsic count range (if any)."""
        lo_g, hi_g = self.count_lo, self.count_hi
        lo_m, hi_m = lo_g, hi_g
        ar = getattr(motif, "attr_ranges", None)
        if isinstance(ar, dict) and "count" in ar and isinstance(ar["count"], (tuple, list)) and len(ar["count"]) == 2:
            lo_m, hi_m = int(ar["count"][0]), int(ar["count"][1])
        return max(lo_g, lo_m), min(hi_g, hi_m)


    def _repeat_layout(self, n: int, layout: str) -> Tuple[int, int]:
        """
        Compute placement grid for 'n' items.
          - 'row'   : single row, wrap as needed
          - 'col'   : single column, wrap as needed
          - 'grid'  : near-square wrap
          - 'pyr_u' : pyramid 1,2,3,... rows until ≥ n
        """
        if layout == "row":
            cols = min(max(1, n), 10)
            rows = (n + cols - 1) // cols
            return cols, rows
        if layout == "col":
            rows = min(max(1, n), 10)
            cols = (n + rows - 1) // rows
            return cols, rows
        if layout == "grid":
            side = max(1, int(math.ceil(math.sqrt(n))))
            cols = side
            rows = (n + cols - 1) // cols
            return cols, rows
        if layout == "pyr_u":
            r = 1
            total = 0
            while total < n:
                total += r
                r += 1
            rows = r - 1
            cols = rows
            return cols, rows
        # fallback
        return self._repeat_layout(n, "grid")

    def _render_direct_anchored(
            self, motif, spec, anchor: str = "top"
    ) -> Image.Image:
        """
        Anchor a single rendered motif (direct-count path) to the chosen wall/center,
        using the same border/background treatment as the repeated renderer.
        """
        # Render raw item and crop to its non-empty alpha region
        raw = motif.render(motif.clamp_spec(spec))
        if raw.mode != "RGBA":
            raw = raw.convert("RGBA")

        # Compose canvas
        big = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        margin_soft = max(10, SS_CELL // 20)
        border_ss = max(3, SS_CELL // 160)
        edge_pad = border_ss + 1

        # Tight crop by alpha; if fully opaque, fallback to full image
        bbox = raw.split()[-1].getbbox()
        content = raw.crop(bbox) if bbox else raw

        # Scale content to fit inside the inner box while preserving aspect
        max_w = SS_CELL - 2 * margin_soft
        max_h = SS_CELL - 2 * margin_soft
        sx = max_w / max(1, content.width)
        sy = max_h / max(1, content.height)
        s = min(1.0, sx, sy)  # avoid scaling up; only downscale if needed
        w_scaled = max(8, int(round(content.width * s)))
        h_scaled = max(8, int(round(content.height * s)))
        content = content.resize((w_scaled, h_scaled), Image.LANCZOS)

        # Anchor position
        if anchor == "left":
            x0 = edge_pad
            y0 = (SS_CELL - h_scaled) // 2
        elif anchor == "right":
            x0 = SS_CELL - edge_pad - w_scaled
            y0 = (SS_CELL - h_scaled) // 2
        elif anchor == "top":
            x0 = (SS_CELL - w_scaled) // 2
            y0 = edge_pad
        elif anchor == "bottom":
            x0 = (SS_CELL - w_scaled) // 2
            y0 = SS_CELL - edge_pad - h_scaled
        else:  # "center"
            x0 = (SS_CELL - w_scaled) // 2
            y0 = (SS_CELL - h_scaled) // 2

        big.alpha_composite(content, (x0, y0))

        # Draw border and composite to white
        ImageDraw.Draw(big).rectangle([0, 0, SS_CELL - 1, SS_CELL - 1], outline=(0, 0, 0), width=border_ss)
        white = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 255))
        composed = Image.alpha_composite(white, big)
        return composed.resize((OUT_CELL, OUT_CELL), Image.LANCZOS).convert("RGBA")

    def _render_repeated(
            self, motif, base_spec, n: int, layout: str, rng: random.Random,
            item_size: Optional[Tuple[int, int]] = None,
            anchor: str = "top",
    ) -> Image.Image:
        """
        Render 'n' repeats of the same motif into a single tile.

        If item_size is provided, treat it as the per-item SLOT size (w_slot, h_slot)
        and scale each rendered item to fit within that slot while preserving aspect ratio.
        Items are centered inside their slots; row centering is performed using slot widths,
        so layout is stable regardless of the actual bitmap aspect ratio.

        Anchoring:
          - "top":    block is flush to top, rows grow downward (horizontally centered)
          - "bottom": block is flush to bottom (horizontally centered)
          - "left":   block is flush to left (vertically centered)
          - "right":  block is flush to right (vertically centered)
          - "center": block is centered both horizontally and vertically
        """
        big = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))

        # Keep a visible gap on non-anchored sides, but hug the chosen wall tightly
        gap = max(6, SS_CELL // 40)
        margin_soft = max(10, SS_CELL // 20)  # used for the non-anchored sides
        border_ss = max(3, SS_CELL // 160)
        edge_pad = border_ss   # minimal pad from the drawn border on the anchored wall

        cols, rows = self._repeat_layout(n, layout)

        # Determine per-item slot size
        if item_size is None:
            avail_w = SS_CELL - 2 * margin_soft - (cols - 1) * gap
            avail_h = SS_CELL - 2 * margin_soft - (rows - 1) * gap
            w_slot = max(8, avail_w // max(1, cols))
            h_slot = max(8, avail_h // max(1, rows))
        else:
            w_slot, h_slot = item_size

        # Render one base item and scale to fit within a slot (preserve aspect)
        item_raw = tight_crop_rgba(motif.render(base_spec))

        # Cap slot by the tile’s inner bounds as a final guard
        max_w = min(w_slot, SS_CELL - 2 * margin_soft)
        max_h = min(h_slot, SS_CELL - 2 * margin_soft)

        # Preserve aspect ratio: scale to fit inside (max_w, max_h)
        sx = max_w / max(1, item_raw.width)
        sy = max_h / max(1, item_raw.height)
        s = min(sx, sy)
        w_scaled = max(8, int(round(item_raw.width * s)))
        h_scaled = max(8, int(round(item_raw.height * s)))
        item = item_raw.resize((w_scaled, h_scaled), Image.LANCZOS)

        # Compute vertical start based on anchor (hug the selected wall)
        total_h = rows * h_slot + (rows - 1) * gap
        if anchor == "top":
            y0 = edge_pad
        elif anchor == "bottom":
            y0 = SS_CELL - edge_pad - total_h
        else:  # "left", "right", or "center" → vertically center the block
            y0 = (SS_CELL - total_h) // 2

        placed = 0
        for r in range(rows):
            want = min(cols, n - placed)
            if want <= 0:
                break

            # Row width in SLOT units (not actual bitmap width)
            row_w = want * w_slot + (want - 1) * gap
            if anchor == "left":
                start_x = edge_pad
            elif anchor == "right":
                start_x = SS_CELL - edge_pad - row_w
            else:  # "top", "bottom", or "center" → horizontally center this row
                start_x = (SS_CELL - row_w) // 2
            y_slot = y0 + r * (h_slot + gap)

            for c in range(want):
                x_slot = start_x + c * (w_slot + gap)

                # Center the bitmap within its slot
                x = x_slot + (w_slot - item.width) // 2
                y = y_slot + (h_slot - item.height) // 2

                if 0 <= x < SS_CELL and 0 <= y < SS_CELL:
                    big.alpha_composite(item, (x, y))

                placed += 1
                if placed >= n:
                    break

        # Draw border and composite to white
        ImageDraw.Draw(big).rectangle([0, 0, SS_CELL - 1, SS_CELL - 1], outline=(0, 0, 0), width=border_ss)
        white = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 255))
        composed = Image.alpha_composite(white, big)
        return composed.resize((OUT_CELL, OUT_CELL), Image.LANCZOS).convert("RGBA")

    def _item_size_for(self, layout: str, max_n: int) -> Tuple[int, int]:
        """
        Slot (w,h) that guarantees max_n can be placed in 'layout' inside one SS_CELL tile.
        This returns the per-item SLOT size (not necessarily the exact bitmap size),
        which _render_repeated will treat as the cell each item is centered within.
        """
        gap = max(6, SS_CELL // 40)
        margin = max(10, SS_CELL // 20)
        cols, rows = self._repeat_layout(max_n, layout)
        # Reuse the same math as _scale_into but keep the semantics clear: this is the slot.
        avail_w = SS_CELL - 2 * margin - (cols - 1) * gap
        avail_h = SS_CELL - 2 * margin - (rows - 1) * gap
        slot_w = max(8, avail_w // max(1, cols))
        slot_h = max(8, avail_h // max(1, rows))
        return slot_w, slot_h

    def _supports_direct_count(self, motif, base_spec) -> bool:
        if not hasattr(base_spec, "count"):
            return False
        lo_eff, hi_eff = self._effective_count_bounds(motif)
        c0 = int(getattr(base_spec, "count", max(lo_eff, 1)))
        # Pick a neighbor strictly inside motif bounds
        if c0 < hi_eff:
            c1 = c0 + 1
        elif c0 > lo_eff:
            c1 = c0 - 1
        else:
            return False  # no room to vary
        try:
            im0 = motif.render(motif.clamp_spec(base_spec.clone(count=c0)))
            im1 = motif.render(motif.clamp_spec(base_spec.clone(count=c1)))
        except Exception:
            return False
        return diff_frac(im0, im1, thresh=8) >= 0.005

    def _build_sequence(self, motif, rng):
        # lock look-and-feel FIRST so we can decide direct vs repeat correctly
        base_spec = _prefer_sym_mode(motif, motif.sample_spec(rng))
        has_direct = self._supports_direct_count(motif, base_spec)

        # Bounds for the thing we're varying:
        if has_direct:
            lo, hi = self._effective_count_bounds(motif)
        else:
            lo, hi = self.count_lo, self.count_hi
            if (hi - lo + 1) < NUM_PANELS:
                raise ValueError("Repeat-count range too narrow for 3 distractors")

        # Direction and weighted step; allow step=0 but make it rare
        ok = False
        for _ in range(STEP_ATTEMPTS):
            step = rng.choices(STEP_CHOICES, weights=STEP_WEIGHTS, k=1)[0]
            dir_sign = rng.choice([1, -1])

            if dir_sign == 1:
                start_lo, start_hi = lo, hi - (NUM_PANELS - 1) * step
            else:
                start_lo, start_hi = lo + (NUM_PANELS - 1) * step, hi

            if start_lo <= start_hi:
                start = rng.randint(start_lo, start_hi)
                counts = [start + dir_sign * i * step for i in range(NUM_PANELS)]
                ok = True
                break

        if not ok:
            step = 0
            dir_sign = 1
            start = rng.randint(lo, hi)
            counts = [start] * NUM_PANELS

        # Choose a single anchor for the entire instance (top/bottom/left/right/center)
        anchor = rng.choices(["top", "bottom", "left", "right", "center"], weights=[0.16, 0.16, 0.16, 0.16, 0.36], k=1)[0]

        # Render top row & payloads
        imgs, payloads = [], []
        if has_direct:
            layout = "count_attr"
            for n in counts:
                s = motif.clamp_spec(base_spec.clone(count=int(n)))
                imgs.append(self._render_direct_anchored(motif, s, anchor=anchor))
                payloads.append(s)  # spec per cell (count differs)
        else:
            layout = rng.choice(["row", "col", "grid", "pyr_u"])
            for n in counts:
                imgs.append(self._render_repeated(motif, base_spec, int(n), layout, rng, anchor=anchor))
                payloads.append(base_spec)  # SAME spec in all NUM_PANELS cells

        meta = {
            "variant": "count_arith_taskstyle",
            "pattern_kind": "sequence",
            "grid": (1, NUM_PANELS),
            "motif_kind": getattr(motif, "name", "unknown"),
            "has_direct_count": has_direct,
            "repeat_layout": layout,
            "repeat_anchor": anchor,  # now recorded for BOTH paths
            "step": step,
            "counts": counts,
            "rep_base_spec": base_spec if not has_direct else None,
            "dir_sign": int(dir_sign),
            "order_human": "increasing L→R" if dir_sign == 1 else "increasing R→L",
            "count_bounds_eff": (int(lo), int(hi)),
        }
        return imgs, payloads, meta

    def _format_question(self, rng: random.Random, labels: List[str]) -> Tuple[str, str]:
        """
        Return (question, prompt_style), where prompt_style ∈ {"general", "explicit"}.
        We sample the bucket first so adding/removing prompts won’t break logic.
        """
        # If PROMPT_TEMPLATES is a dict with buckets
        if isinstance(PROMPT_TEMPLATES, dict):
            bucket = rng.choice(list(PROMPT_TEMPLATES.keys())) or "general"
            choices = PROMPT_TEMPLATES.get(bucket, []) or ["Which option (a)–(d) completes the pattern?"]
            return rng.choice(choices), bucket
        # Fallback: treat as a flat list
        q = rng.choice(PROMPT_TEMPLATES)
        return q, "general"


    # ------------- options -------------
    def _build_options(
            self, motif, rng, corr_payload, corr_count: int,
            has_direct_count: bool, repeat_layout: str, labels: List[str],
            *, fixed_item_size: Optional[Tuple[int, int]] = None,
            preset_wrong_counts: Optional[List[int]] = None,
            max_n_bound: Optional[int] = None,
            anchor: str = "top",
    ):
        # Choose effective bounds based on the path
        if has_direct_count:
            lo_eff, hi_eff = self._effective_count_bounds(motif)
        else:
            lo_eff, hi_eff = self.count_lo, self.count_hi  # repeat path = global bounds

        if max_n_bound is not None:
            hi_eff = min(hi_eff, int(max_n_bound))

        c0 = int(getattr(corr_payload, "count")) if has_direct_count else int(corr_count)

        def render_for_count(n: int) -> Image.Image:
            """Render the motif repeated ``n`` times using the chosen path."""
            if has_direct_count:
                s = motif.clamp_spec(corr_payload.clone(count=n))
                # Anchor direct-count tiles exactly like the top row
                return self._render_direct_anchored(motif, s, anchor=anchor)
            else:
                base_spec = corr_payload
                return self._render_repeated(
                    motif, base_spec, n, repeat_layout, rng,
                    item_size=fixed_item_size, anchor=anchor
                )

        # Always render the correct image first
        correct_img = render_for_count(c0)
        correct_sig = sig(correct_img)

        # If we have a preset list (second pass), try to use it as-is
        if preset_wrong_counts is not None:
            picks = [v for v in (preset_wrong_counts or []) if lo_eff <= v <= hi_eff and v != c0][:3]
            if len(picks) < 3:
                return None
            options: List[Tuple[int, Image.Image]] = [(c0, correct_img)]
            used = {correct_sig}
            for v in picks:
                im = render_for_count(v)
                h = sig(im)
                if h in used:
                    return None
                used.add(h)
                options.append((v, im))
            if len(options) < NUM_OPTIONS:
                return None
            # Pairwise uniqueness check
            imgs = [im for (_c, im) in options]
            for i in range(NUM_OPTIONS):
                for j in range(i + 1, NUM_OPTIONS):
                    if sig(imgs[i]) == sig(imgs[j]):
                        return None
                    if diff_frac(imgs[i], imgs[j], thresh=8) < self.opt_min_delta:
                        return None

            # Pack tiles
            random.shuffle(options)
            counts = [c for (c, _im) in options]
            font = load_font()
            label_strs = labels_default()
            option_tiles = [crisp_option_tile(im, lab, font) for (_c, im), lab in zip(options, label_strs)]
            correct_label = label_strs[counts.index(c0)]
            descs = [f"{c}" for c in counts]
            return option_tiles, correct_label, descs, picks

        # --- No preset: score all feasible counts by actual visual difference ---
        candidates = [v for v in range(lo_eff, hi_eff + 1) if v != c0]
        rng.shuffle(candidates)  # avoid bias when diffs tie

        scored: List[Tuple[float, int, bytes, Image.Image]] = []  # (diff, v, signature, img)
        used_hashes = {correct_sig}
        for v in candidates:
            im = render_for_count(v)
            h = sig(im)
            if h in used_hashes:
                continue
            diff = diff_frac(correct_img, im, thresh=8)
            scored.append((diff, v, h, im))

        # Prefer large visual changes
        scored.sort(key=lambda t: t[0], reverse=True)

        def greedy_select(threshold: float) -> Tuple[List[int], List[Image.Image]]:
            """Select wrong options differing by at least ``threshold``."""
            picks_: List[int] = []
            imgs_: List[Image.Image] = []
            hashes_ = set(used_hashes)
            for diff, v, h, im in scored:
                if diff < max(threshold, 0.005):
                    continue
                if h in hashes_:
                    continue
                ok = True
                for im2 in imgs_:
                    # ensure pairwise separation among wrongs
                    if sig(im2) == h or diff_frac(im, im2, thresh=8) < threshold:
                        ok = False
                        break
                if ok:
                    picks_.append(v)
                    imgs_.append(im)
                    hashes_.add(h)
                    if len(picks_) == 3:
                        return picks_, imgs_
            return picks_, imgs_

        # Try strict threshold, then relax slightly if needed
        for relax in (1.0, 0.85, 0.7):
            thr = self.opt_min_delta * relax
            picks, pick_imgs = greedy_select(thr)
            if len(picks) == 3:
                break
        else:
            return None  # could not find 3 visually distinct wrongs

        options: List[Tuple[int, Image.Image]] = [(c0, correct_img)] + list(zip(picks, pick_imgs))
        random.shuffle(options)

        counts = [c for (c, _im) in options]
        imgs = [_im for (_c, _im) in options]
        # Final sanity (usually redundant now)
        for i in range(NUM_OPTIONS):
            for j in range(i + 1, NUM_OPTIONS):
                if sig(imgs[i]) == sig(imgs[j]):
                    return None
                if diff_frac(imgs[i], imgs[j], thresh=8) < self.opt_min_delta:
                    return None

        font = load_font()
        label_strs = labels_default()
        option_tiles = [crisp_option_tile(im, lab, font) for im, lab in zip(imgs, label_strs)]
        correct_label = label_strs[counts.index(c0)]
        descs = [f"{c}" for c in counts]
        return option_tiles, correct_label, descs, picks

    # ------------- public -------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # ---- Collect allowed motifs and build a weighted order (without replacement) ----
        kinds_all = [k for k in motif_impls.keys() if k in MOTIF_WEIGHTS and MOTIF_WEIGHTS[k] > 0]
        if not kinds_all:
            raise RuntimeError(f"{self.name}: no allowed motifs available.")

        motif_order = weighted_order(rng, kinds_all, MOTIF_WEIGHTS)

        tried_failures: List[str] = []

        # ---- Try each motif up to max_retries before moving on ----
        for mk in motif_order:
            motif = motif_impls[mk]

            for _ in range(self.max_retries):
                try:
                    cell_imgs, cell_payloads, meta = self._build_sequence(motif, rng)
                except ValueError:
                    # sequence builder couldn't produce a valid sample this attempt
                    continue

                mask_idx = rng.randint(0, 3)
                has_direct = bool(meta["has_direct_count"])
                anchor = str(meta.get("repeat_anchor") or "top")

                if has_direct:
                    corr_payload = cell_payloads[mask_idx]
                    corr_count = int(getattr(corr_payload, "count"))
                    built = self._build_options(
                        motif, rng, corr_payload, corr_count,
                        has_direct_count=True,
                        repeat_layout="count_attr",
                        labels=labels_default(),
                        anchor=anchor,
                    )
                    if built is None:
                        continue
                    option_tiles, correct_label, option_descs, picked_wrongs = built
                    final_cell_imgs = [ensure_cell_rgba_outcell(im) for im in cell_imgs]
                else:
                    base_spec = meta["rep_base_spec"]
                    corr_payload = base_spec
                    corr_count = int(meta["counts"][mask_idx])

                    # First probe to pin wrong counts (helps avoid collisions later)
                    probe = self._build_options(
                        motif, rng, corr_payload, corr_count,
                        has_direct_count=False,
                        repeat_layout=str(meta["repeat_layout"]),
                        labels=labels_default(),
                        anchor=anchor,
                    )
                    if probe is None:
                        continue
                    _, _, option_descs_first, picked_wrongs = probe
                    option_counts = [int(s) for s in option_descs_first]
                    max_n = max(max(meta["counts"]), max(option_counts))

                    fixed_size = self._item_size_for(str(meta["repeat_layout"]), max_n)

                    final_cell_imgs = []
                    for n in meta["counts"]:
                        final_cell_imgs.append(
                            self._render_repeated(
                                motif, base_spec, int(n), str(meta["repeat_layout"]), rng,
                                item_size=fixed_size, anchor=anchor
                            )
                        )
                    final_cell_imgs = [ensure_cell_rgba_outcell(im) for im in final_cell_imgs]

                    built = self._build_options(
                        motif, rng, corr_payload, corr_count,
                        has_direct_count=False,
                        repeat_layout=str(meta["repeat_layout"]),
                        labels=labels_default(),
                        fixed_item_size=fixed_size,
                        preset_wrong_counts=picked_wrongs,
                        max_n_bound=max_n,
                        anchor=anchor,
                    )
                    if built is None:
                        # fallback: let builder choose wrong counts anew
                        built = self._build_options(
                            motif, rng, corr_payload, corr_count,
                            has_direct_count=False,
                            repeat_layout=str(meta["repeat_layout"]),
                            labels=labels_default(),
                            fixed_item_size=fixed_size,
                            preset_wrong_counts=None,
                            max_n_bound=max_n,
                            anchor=anchor,
                        )
                        if built is None:
                            continue

                    option_tiles, correct_label, option_descs, _ = built

                # ---- assemble & return on first success ----
                opts = compose_options_row(option_tiles)
                top = compose_row_with_mask(
                    final_cell_imgs, mask_idx,
                    target_width=opts.width,
                    draw_arrows=True,
                    arrow_style={"ss": 3, "alpha": 220, "scale": 1.2},
                    gap=OUT_CELL // 8,
                    margin=0,
                )
                top = _pad_width_center(top, opts.width)
                composite = compose_top_bottom(top, opts, sep_px=40)

                question, prompt_style = self._format_question(rng, labels_default())

                meta_out = {
                    "pattern_kind": "sequence",
                    "pattern": self.name,
                    "grid": (1, NUM_PANELS),
                    "mask_idx": mask_idx,
                    "variant": "count_arith_taskstyle",
                    "motif_kind": mk,
                    "labels": labels_default(),
                    "answer": correct_label,
                    "option_strategy": "image_counts",
                    "option_descs": option_descs,
                    "option_counts": [int(x) for x in option_descs],
                    "repeat_layout": str(meta["repeat_layout"]),
                    "repeat_anchor": anchor,
                    "has_direct_count": has_direct,
                    "counts_top_row": [int(x) for x in meta["counts"]],
                    "step": int(meta.get("step", 0)),
                    "question": question,
                    "prompt_style": str(prompt_style),
                    "composite_ready": True,
                }
        if meta.get("rep_base_spec") is not None:
            bs = meta["rep_base_spec"]
            meta_out["motif_spec"] = bs.to_dict() if hasattr(bs, "to_dict") else dict(bs.__dict__)

            max_count_top = max(int(x) for x in meta_out["counts_top_row"])
            meta_out["max_count_top_row"] = int(max_count_top)
            complexity = self._compute_complexity(max_count_top)
            meta_out["complexity"] = complexity
            meta_out["complexity_score"] = complexity["score"]
            meta_out["complexity_level"] = complexity["level"]
            meta_out["complexity_version"] = complexity["version"]
            return composite, cell_payloads, meta_out

        # if we reach here, this motif failed all attempts
        tried_failures.append(mk)

        # ---- All motifs failed ----
        order_str = ", ".join(motif_order) if motif_order else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a verifiable sample after {self.max_retries} attempts per motif. "
            f"Motifs tried (in order): {order_str}"
        )

