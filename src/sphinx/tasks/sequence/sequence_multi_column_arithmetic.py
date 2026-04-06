# sphinx/tasks/sequence/sequence_multi_column_arithmetic.py
import random
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw

from ...base import Task
from ...registry import register_task
from ...config import (
    OUT_CELL, SS_CELL, IMG_DIFF_MIN,
    MAX_BUILD_RETRIES, OPT_HASH_MIN_BITS, OPT_UNIQUENESS_MIN,
)
from ...utils.image_compare import sig, diff_frac
from ...utils.drawing import (
    load_font, crisp_option_tile, labels_default,
    compose_row_with_mask, compose_top_bottom, compose_options_row,
    tight_crop_rgba,
)
from ...utils.specs import _prefer_sym_mode

COLUMNS_MIN_DEFAULT = 2
COLUMNS_MAX_DEFAULT = 6

# -------------------------------------------------------------------
# Motif weights (you can tune freely; same spirit as sequence_arithmetic)
# -------------------------------------------------------------------
MOTIF_WEIGHTS = {
    "icons": 15,
    "arc": 0.25,
    "single_arrow": 0.125,
    "clock": 0.5,
    "crescent": 0.75,
    "dot": 1.25,
    "glyph": 0.75,
    "keyhole": 0.25,
    "pictogram": 0.25,
    "pinwheel_triangles": 0.5,
    "polygon": 0.75,
    "star_polygon": 0.5,
}



PROMPT_TEMPLATES = {
    "general": [
        "The top row shows three panels with a blank for the fourth. Which option (a)–(d) completes the sequence?",
        "There is a missing fourth panel in the top row. Which option (a)–(d) belongs there?",
        "Which option (a)–(d) below should replace the blank to complete the top-row sequence?",
        "Which panel (a)–(d) completes the sequence shown in the top row?",
        "Which option (a)–(d) continues the sequence of panels in the top row?"
    ],
    "explicit": [
        "Each vertical column follows its own arithmetic step in the number of shapes. Which option (a)–(d) completes the fourth panel?",
        "Every column changes by a fixed count each step. Which option (a)–(d) correctly fills the blank panel?",
        "Columns evolve independently: in each one the count increases or decreases by the same amount. Which option (a)–(d) continues the pattern?",
        "Focus on counts per column: each follows an arithmetic progression (possibly reaching zero). Which option (a)–(d) completes the row?",
        "Infer the per-column differences from the first three panels. Which option (a)–(d) is the correct next panel?"
    ]
}


@register_task
class SequenceMultiColumnArithmeticTask(Task):
    """
    Multi-column sequence: show 3 time steps, each panel contains C columns (2..6).
    Within each column, the count of an identical motif changes by a column-specific
    arithmetic step. Ask for panel t=3 (the next one). Options differ by minimal
    edits to the final per-column counts.
    """
    name = "sequence_multi_column_arithmetic"

    def __init__(self):
        # visual uniqueness knobs
        self.min_delta = float(IMG_DIFF_MIN)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.max_retries = int(MAX_BUILD_RETRIES)

        # global numeric bounds; we adapt hi to #columns for readability
        self.count_lo_default = 0
        self.count_hi_default = 10

        # columns
        self.cols_lo = COLUMNS_MIN_DEFAULT
        self.cols_hi = COLUMNS_MAX_DEFAULT

    def _compute_complexity(self, num_columns: int) -> Dict[str, Any]:
        """Normalize column count and derive EASY/HARD complexity."""
        min_cols = int(getattr(self, "cols_lo", COLUMNS_MIN_DEFAULT))
        max_cols = int(getattr(self, "cols_hi", COLUMNS_MAX_DEFAULT))
        span = max(1, max_cols - min_cols)
        normalized = (int(num_columns) - min_cols) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "sequence-multi-column-c-v1",
            "range": {"min_columns": min_cols, "max_columns": max_cols},
            "num_columns": int(num_columns),
        }

    # -------------------------- core sampling --------------------------

    def _count_hi_for_columns(self, C: int) -> int:
        """
        Keep stacks legible as C grows.
        """
        if C <= 3:
            return 12
        if C == 4:
            return 10
        if C == 5:
            return 8
        return 6  # C == 6

    def _pick_columns(self, motif_impls: Dict[str, Any], rng: random.Random, C: int):
        allowed = [k for k in motif_impls if MOTIF_WEIGHTS.get(k, 0.0) > 0.0]
        if not allowed:
            raise RuntimeError(f"{self.name}: no allowed motif kinds found.")
        weights = [max(1e-9, float(MOTIF_WEIGHTS.get(k, 0.0))) for k in allowed]
        chosen_kinds = rng.choices(allowed, weights=weights, k=C)  # with replacement, by weight

        motifs = [motif_impls[k] for k in chosen_kinds]
        # sample a base spec per column (fixed across time/options)
        base_specs = []
        for m in motifs:
            s = _prefer_sym_mode(m, m.sample_spec(rng))
            base_specs.append(s)
        return chosen_kinds, motifs, base_specs

    def _sample_sequences(
        self, rng: random.Random, C: int, count_lo: int, count_hi: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Sample per-column arithmetic sequences for t=0..3 (we will hide t=3).
        Returns:
          counts[t][c] : counts at time t for column c
          steps[c]: per-column step
        """
        steps: List[int] = []
        counts: List[List[int]] = [[0] * C for _ in range(4)]

        # allow small steps, including 0; bias toward ±1 for clean visuals
        step_choices = [-2, -1, 0, +1, +2]
        step_weights = [0.15, 0.35, 0.05, 0.35, 0.10]

        for c in range(C):
            for _ in range(100):
                s = rng.choices(step_choices, weights=step_weights, k=1)[0]
                # choose start so all 4 steps remain in bounds
                lo, hi = count_lo, count_hi
                if s >= 0:
                    start_lo, start_hi = lo, hi - 3 * s
                else:
                    start_lo, start_hi = lo - 3 * s, hi  # s negative
                if start_lo > start_hi:
                    continue
                start = rng.randint(start_lo, start_hi)
                seq = [start + t * s for t in range(4)]
                # permit zeros (columns can go empty), but avoid all zeros across the whole panel set
                if any(v > 0 for v in seq):
                    counts[0][c], counts[1][c], counts[2][c], counts[3][c] = seq
                    steps.append(s)
                    break
            else:
                # fallback: flat small positive sequence
                s = 0
                start = rng.randint(max(count_lo, 1), max(count_lo, 1) + 1)
                steps.append(s)
                for t in range(4):
                    counts[t][c] = start

        # global sanity: at each of the shown times (t=0..2), avoid all-zero columns simultaneously
        for t in range(3):
            if sum(counts[t]) == 0:
                # bump a random column minimally
                c = rng.randrange(C)
                counts[t][c] = min(count_hi, 1)

        return counts, steps

    # -------------------------- rendering helpers --------------------------

    def _render_item_scaled(self, motif, spec, slot_w: int, slot_h: int) -> Image.Image:
        """
        Render a single motif item and scale to fit within a slot (preserve aspect).
        Transparent RGBA; no border.
        """
        raw = motif.render(motif.clamp_spec(spec))
        if raw.mode != "RGBA":
            raw = raw.convert("RGBA")
        item = tight_crop_rgba(raw)
        sx = slot_w / max(1, item.width)
        sy = slot_h / max(1, item.height)
        s = min(sx, sy)
        w = max(6, int(round(item.width * s)))
        h = max(6, int(round(item.height * s)))
        return item.resize((w, h), Image.LANCZOS)

    def _column_slot_size(self, C: int, max_n: int) -> Tuple[int, int, int, int, int, int]:
        """
        Compute per-column bounding box & per-item slot sizes for a panel with C columns.
        Returns (col_w, col_h, margin_x, margin_y, col_gap, vgap_per_item).
        """
        # global panel margins and gaps (SS space)
        margin_x = max(12, SS_CELL // 16)
        margin_y = max(12, SS_CELL // 16)
        col_gap = max(6, SS_CELL // 40)

        inner_w = SS_CELL - 2 * margin_x - (C - 1) * col_gap
        col_w = max(8, inner_w // C)
        col_h = SS_CELL - 2 * margin_y

        # vertical spacing between stacked items (small)
        vgap = max(4, SS_CELL // 80)
        # NOTE: per-item slot height is derived later from max_n per column
        return col_w, col_h, margin_x, margin_y, col_gap, vgap

    def _column_boxes_out(self, C: int) -> List[Tuple[int, int, int, int]]:
        """
        Bounding boxes for each column in OUT_CELL coordinates.
        Matches the layout used in _render_panel (computed in SS then resized).
        """
        col_w, col_h, margin_x, margin_y, col_gap, _ = self._column_slot_size(C, 1)
        sx = OUT_CELL / float(SS_CELL)
        sy = OUT_CELL / float(SS_CELL)
        boxes = []
        for c in range(C):
            x0_ss = margin_x + c * (col_w + col_gap)
            y0_ss = margin_y
            x1_ss = x0_ss + col_w
            y1_ss = y0_ss + col_h
            boxes.append((
                int(round(x0_ss * sx)),
                int(round(y0_ss * sy)),
                int(round(x1_ss * sx)),
                int(round(y1_ss * sy)),
            ))
        return boxes

    def _render_panel(
        self,
        C: int,
        col_kinds: List[str],
        motifs: List[Any],
        base_specs: List[Any],
        counts_for_cols: List[int],
        max_counts_per_col: List[int],
    ) -> Image.Image:
        """
        Render one OUT_CELL×OUT_CELL panel that contains C columns.
        Each column shows 'counts_for_cols[c]' stacked items of that column's motif.
        Slot sizes are set from 'max_counts_per_col' to keep scale consistent over time/options.
        """
        # SS canvas with white background
        ss = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 255))
        dr = ImageDraw.Draw(ss)

        col_w, col_h, margin_x, margin_y, col_gap, vgap = self._column_slot_size(C, max(1, max(max_counts_per_col)))
        # precompute a scaled item per column (fits within per-item slot)
        # slot sizes per column (w_slot, h_slot) based on its max count
        w_slot = col_w
        slot_heights = []
        for c in range(C):
            max_n = max(1, max_counts_per_col[c])
            # rows = max_n, each slot plus (rows-1) gaps
            avail_h = col_h - (max_n - 1) * vgap
            h_slot = max(6, avail_h // max_n)
            slot_heights.append(h_slot)

        # draw faint column frames (helps at high C)
        frame_w = max(1, SS_CELL // 200)
        for c in range(C):
            x0 = margin_x + c * (col_w + col_gap)
            y0 = margin_y
            dr.rectangle([x0, y0, x0 + col_w - 1, y0 + col_h - 1], outline=(200, 200, 200), width=frame_w)

        # paste stacks (bottom-anchored)
        for c in range(C):
            n = max(0, int(counts_for_cols[c]))
            x0 = margin_x + c * (col_w + col_gap)
            y0 = margin_y
            h_slot = slot_heights[c]

            # render a single item for this column at the slot size
            item = self._render_item_scaled(motifs[c], base_specs[c], w_slot, h_slot)

            # bottom anchor
            total_h = n * h_slot + max(0, n - 1) * vgap
            y_start = y0 + (col_h - total_h)

            for k in range(n):
                # centered in the column width, and within its slot
                y_slot = y_start + k * (h_slot + vgap)
                x = x0 + (col_w - item.width) // 2
                y = y_slot + (h_slot - item.height)
                ss.alpha_composite(item, (x, y))

        # draw outer border (like other tiles)
        border_ss = max(3, SS_CELL // 160)
        ImageDraw.Draw(ss).rectangle([0, 0, SS_CELL - 1, SS_CELL - 1], outline=(0, 0, 0), width=border_ss)
        return ss.resize((OUT_CELL, OUT_CELL), Image.LANCZOS).convert("RGBA")

    # -------------------------- options --------------------------

    def _build_options(
            self,
            rng: random.Random,
            C: int,
            col_kinds: List[str],
            motifs: List[Any],
            base_specs: List[Any],
            counts_t0: List[int],
            counts_t1: List[int],
            counts_t2: List[int],
            counts_true_t3: List[int],
            count_lo: int,
            count_hi: int,
    ):
        """
        Build three minimal-change wrong panels for t=3 by editing exactly one column.
        Use column-local difference to judge uniqueness and adapt delta (±1, ±2, ±3, …)
        until the change is visually sufficient.
        """
        # For scale consistency across time/options
        max_counts_base = [max(counts_t0[c], counts_t1[c], counts_t2[c], counts_true_t3[c]) for c in range(C)]

        # Correct panel first
        correct_img = self._render_panel(C, col_kinds, motifs, base_specs, counts_true_t3, max_counts_base)
        correct_sig = sig(correct_img)

        # Local uniqueness threshold: smaller when there are more columns
        local_thr = max(0.012, self.opt_min_delta / max(2, C))

        # Column bounding boxes for local diff checks (in OUT_CELL coords)
        col_boxes = self._column_boxes_out(C)

        wrongs: List[Tuple[List[int], Image.Image]] = []
        used_hashes = {correct_sig}

        # Try to add one wrong on a given column using smallest delta that passes local_thr
        def try_add_wrong_for(col_index: int) -> bool:
            """Attempt to inject a wrong count in column ``col_index``."""
            # Prefer ±1, then ±2, ±3, ±4 ...
            delta_order = [1, -1, 2, -2, 3, -3, 4, -4]
            for d in delta_order:
                v = counts_true_t3[col_index] + d
                if v < count_lo or v > count_hi or v == counts_true_t3[col_index]:
                    continue
                counts_bad = list(counts_true_t3)
                counts_bad[col_index] = v
                # include the new v when scaling that column
                max_counts = list(max_counts_base)
                max_counts[col_index] = max(max_counts[col_index], v)

                img = self._render_panel(C, col_kinds, motifs, base_specs, counts_bad, max_counts)

                # Local difference inside the edited column’s box
                bx = col_boxes[col_index]
                dloc = diff_frac(correct_img.crop(bx), img.crop(bx), thresh=8)
                if dloc < local_thr:
                    continue  # too similar; try a larger |delta|

                h = sig(img)
                if h in used_hashes:
                    continue

                # Ensure pairwise separation among wrongs (use a slightly softer global check)
                ok = True
                for _cb, im_prev in wrongs:
                    # Global check but with a relaxed threshold proportional to local_thr
                    if diff_frac(img, im_prev, thresh=8) < (0.6 * local_thr):
                        ok = False
                        break
                if not ok:
                    continue

                used_hashes.add(h)
                wrongs.append((counts_bad, img))
                return True
            # Fallback: choose the minimal delta that was closest to passing local_thr
            # so we always make progress even with tiny items.
            best = None
            best_d = None
            for d in [1, -1, 2, -2, 3, -3, 4, -4]:
                v = counts_true_t3[col_index] + d
                if v < count_lo or v > count_hi or v == counts_true_t3[col_index]:
                    continue
                counts_bad = list(counts_true_t3)
                counts_bad[col_index] = v
                max_counts = list(max_counts_base)
                max_counts[col_index] = max(max_counts[col_index], v)
                img = self._render_panel(C, col_kinds, motifs, base_specs, counts_bad, max_counts)
                bx = col_boxes[col_index]
                dloc = diff_frac(correct_img.crop(bx), img.crop(bx), thresh=8)
                h = sig(img)
                if (best is None) or (dloc > best_d):
                    best = (counts_bad, img, h)
                    best_d = dloc
            if best is not None and best[2] not in used_hashes:
                used_hashes.add(best[2])
                wrongs.append((best[0], best[1]))
                return True
            return False

        # First, aim for distinct columns when possible
        cols_order = list(range(C))
        rng.shuffle(cols_order)
        for c in cols_order:
            if len(wrongs) == 3:
                break
            try_add_wrong_for(c)

        # If we still need more, revisit columns
        if len(wrongs) < 3:
            for c in cols_order:
                if len(wrongs) == 3:
                    break
                try_add_wrong_for(c)

        if len(wrongs) < 3:
            return None  # let caller retry

        # Assemble options (correct + wrongs), shuffle & label
        options: List[Tuple[List[int], Image.Image]] = [(counts_true_t3, correct_img)] + wrongs[:3]
        rng.shuffle(options)

        imgs = [im for (_cnts, im) in options]
        counts_per_opt = [cnts for (cnts, _im) in options]

        # ---- Final sanity: pairwise DISTINCT using *local* regions that differ ----
        def pairwise_locally_distinct(i: int, j: int) -> bool:
            """Return True if options ``i`` and ``j`` differ in at least one column."""
            cols_diff = [c for c in range(C) if counts_per_opt[i][c] != counts_per_opt[j][c]]
            # If counts vectors are equal (should never happen), reject
            if not cols_diff:
                return False
            # Require that at least one differing column shows a visible local change
            for c in cols_diff:
                bx = col_boxes[c]
                dloc = diff_frac(imgs[i].crop(bx), imgs[j].crop(bx), thresh=8)
                if dloc >= 0.6 * local_thr:  # relaxed local check for pairwise
                    return True
            return False

        for i in range(4):
            for j in range(i + 1, 4):
                if not pairwise_locally_distinct(i, j):
                    return None

        labels = labels_default()
        font = load_font()
        option_tiles = [crisp_option_tile(im, lab, font) for im, lab in zip(imgs, labels)]
        # Identify the correct label by counts vector (robust to shuffling)
        correct_label = labels[counts_per_opt.index(counts_true_t3)]
        option_descs = [";".join(map(str, cnts)) for cnts in counts_per_opt]
        counts_list_all = counts_per_opt
        return option_tiles, correct_label, option_descs, counts_list_all

    def _format_question(self, rng: random.Random, labels: List[str]) -> Tuple[str, str]:
        """
        Return (question, prompt_style) where prompt_style ∈ {"general","explicit"}.
        Bucket-first sampling so edits won’t break logic.
        """
        if isinstance(PROMPT_TEMPLATES, dict):
            bucket = rng.choice(list(PROMPT_TEMPLATES.keys())) or "general"
            choices = PROMPT_TEMPLATES.get(bucket, []) or ["Which option (a)–(d) completes the pattern?"]
            return rng.choice(choices), bucket
        q = rng.choice(PROMPT_TEMPLATES)
        return q, "general"
    # -------------------------- public API --------------------------

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        failures: List[str] = []
        for _try in range(self.max_retries):
            try:
                # ---- sample columns & counts ----
                C = rng.randint(self.cols_lo, self.cols_hi)
                count_lo = int(self.count_lo_default)
                count_hi = int(min(self.count_hi_default, self._count_hi_for_columns(C)))

                col_kinds, motifs, base_specs = self._pick_columns(motif_impls, rng, C)
                counts, steps = self._sample_sequences(rng, C, count_lo, count_hi)
                counts_t0, counts_t1, counts_t2, counts_t3 = counts

                # ---- render top row ----
                max_counts_for_scale = [max(counts_t0[c], counts_t1[c], counts_t2[c], counts_t3[c]) for c in range(C)]
                top_imgs = [
                    self._render_panel(C, col_kinds, motifs, base_specs, counts_t0, max_counts_for_scale),
                    self._render_panel(C, col_kinds, motifs, base_specs, counts_t1, max_counts_for_scale),
                    self._render_panel(C, col_kinds, motifs, base_specs, counts_t2, max_counts_for_scale),
                    self._render_panel(C, col_kinds, motifs, base_specs, counts_t3, max_counts_for_scale),
                ]

                # ---- build options ----
                built = self._build_options(
                    rng, C, col_kinds, motifs, base_specs,
                    counts_t0, counts_t1, counts_t2, counts_t3, count_lo, count_hi
                )
                if built is None:
                    failures.append("opt-build")
                    continue

                option_tiles, correct_label, option_descs, counts_list_all = built

                # ---- compose & return ----
                mask_idx = 3
                opts = compose_options_row(option_tiles)
                top = compose_row_with_mask(
                    top_imgs, mask_idx,
                    target_width=opts.width,
                    draw_arrows=True,
                    arrow_style={"ss": 3, "alpha": 220, "scale": 1.2},
                    gap=OUT_CELL // 8,
                    margin=0,
                )
                composite = compose_top_bottom(top, opts, sep_px=40)

                labels = labels_default()
                question_text, prompt_style = self._format_question(rng, labels)

                meta = {
                    "pattern_kind": "sequence",
                    "pattern": self.name,
                    "grid": (1, 4),
                    "mask_idx": mask_idx,
                    "labels": labels,
                    "num_columns": C,
                    "column_kinds": col_kinds,
                    "column_steps": steps,
                    "counts_t0": counts_t0,
                    "counts_t1": counts_t1,
                    "counts_t2": counts_t2,
                    "counts_t3_true": counts_t3,
                    "option_descs": option_descs,
                    "option_counts": counts_list_all,
                    "question": question_text,
                    "answer": correct_label,
                    "prompt_style": prompt_style,
                    "composite_ready": True,
                }
                complexity = self._compute_complexity(C)
                meta["complexity"] = complexity
                meta["complexity_score"] = complexity["score"]
                meta["complexity_level"] = complexity["level"]
                meta["complexity_version"] = complexity["version"]
                return composite, [{"k": k} for k in col_kinds], meta

            except Exception:
                failures.append("exception")
                continue

        # If we exhausted retries:
        raise RuntimeError(
            f"{self.name}: failed after {self.max_retries} attempts. Failures: {', '.join(failures) or '(none)'}")
