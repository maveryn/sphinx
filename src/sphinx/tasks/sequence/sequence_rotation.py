# sphinx/tasks/sequence/sequence_rotation.py
import math, random
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image, ImageDraw

from ...base import Task
from ...registry import register_task
from ...config import (
    OUT_CELL, MAX_BUILD_RETRIES, OPT_HASH_MIN_BITS,
    OPT_UNIQUENESS_MIN, SS_CELL
)
from ...utils.image_compare import sig, diff_frac
from .config import SEQ_CONFIG
from ...utils.specs import _prefer_asym_mode
from ...utils.rng import choice_weighted, weighted_order

from ...utils.drawing import (
    load_font, crisp_option_tile, labels_default,
    compose_row_with_mask, compose_top_bottom, compose_options_row,
    tight_crop_rgba, ensure_cell_rgba_outcell,
)

# ---- steps & prompt ---------------------------------------------------------

STEP_CHOICES = [30, 45, 60, 90]
STEP_WEIGHTS = [0.15, 0.35, 0.15, 0.35]
NUM_PANELS = 4    # number of panels in the sequence
NUM_OPTIONS = 4   # number of answer choices

PROMPT_TEMPLATES = {
    "general": [
        "The top row shows a sequence with one tile missing; options (a)–(d) are below. Which option completes the pattern?",
        "There is a blank cell in the top row. From the options (a)–(d) below, which tile completes the pattern?",
        "Which option (a)–(d) below best fills the blank in the top-row sequence?",
        "Which option (a)–(d) should replace the blank to complete the top-row pattern?",
        "Which tile from the options (a)–(d) below completes the top-row pattern?"
    ],
    "explicit": [
        "In the top row, the motif rotates by a constant angle each step. Which option (a)–(d) below fills the blank?",
        "The top-row sequence is a fixed-angle rotation of the same shape. Which option (a)–(d) completes the row?",
        "Infer the equal rotation step from the visible tiles in the top row. Which option (a)–(d) belongs in the blank?",
        "Only one option (a)–(d) continues the constant-angle rotation in the top row. Which is it?",
        "The motif turns by the same angle between neighboring tiles on the top row. Which option (a)–(d) completes the pattern?"
    ],
}


# ---- motifs to try ----------------------------------------------------------
MOTIF_WEIGHTS = {
    "icons": 15,
    "arc": 0.25,
    "single_arrow": 0.5,
    "clock": 0.5,
    "crescent": 1.0,
    "gear": 0.125,
    "glyph": 0.5,
    "keyhole": 0.25,
    "pictogram": 0.25,
    "pinwheel_triangles": 0.5,
    "polygon": 1.0,
    "polyhex": 0.25,
    'polyiamond': 0.25,
    "polyomino": 0.5,
    "rings": 0.25,
    "star_polygon": 0.25,
}

# -----------------------------------------------------------------------------


@register_task
class SequenceRotationTask(Task):
    """
    Sequence task where the rule is a fixed-angle rotation of a single motif raster.

    We render ONE base motif, then rotate that same bitmap by a constant step
    (30°/45°/60°/90°) L→R. A random cell is masked and the user must pick the
    correct rotation from four options.

    Implementation mirrors SequenceArithmeticTask:
      - two-pass rendering to lock a common scale for both top row and options;
      - careful uniqueness checks via image signatures + pixel-delta;
      - rejection sampling if progression is visually weak or ambiguous.
    """
    name = "sequence_rotation"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)

        # --- NEW: rotation-specific gates (gentler than count-based defaults) ---
        # Adjacent-frame change required for the top row (rotations are subtler).
        self.rot_adjacent_min = float(SEQ_CONFIG.get("rot_adjacent_min", 0.015))
        # Pairwise separation required among options.
        self.rot_option_min = float(SEQ_CONFIG.get("rot_option_min", 0.02))

    def _compute_complexity(self, step_deg: int) -> Dict[str, Any]:
        """Map rotation step to a reversed ordinal and normalize to [0,1]."""
        sorted_steps = sorted(set(STEP_CHOICES), reverse=True)
        reverse_rank = {step: idx for idx, step in enumerate(sorted_steps)}

        rank = reverse_rank.get(int(step_deg))
        if rank is None:
            try:
                rank = sorted_steps.index(int(step_deg))
            except ValueError:
                rank = 0
        min_rank = 0
        max_rank = len(sorted_steps) - 1 if sorted_steps else 0
        span = max(1, max_rank - min_rank)
        normalized = (rank - min_rank) / span

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "sequence-rotation-step-v1",
            "range": {
                "min_rank": min_rank,
                "max_rank": max_rank,
                "step_mapping": reverse_rank,
            },
            "step_deg": int(step_deg),
            "rank": int(rank),
        }

    # --------- helpers ---------

    def _scale_for_angles(
            self, base_w: int, base_h: int, angles: List[float], inner_w: int, inner_h: int
    ) -> float:
        """
        Compute a global scale 's' so that after rotation by ANY angle in 'angles'
        the rotated bounding box fits within (inner_w, inner_h).
        Allows reasonable upscaling so small motifs can fill the tile.
        """
        if base_w <= 0 or base_h <= 0:
            return 1.0

        def rotated_dims(w: float, h: float, deg: float) -> Tuple[float, float]:
            th = math.radians(deg % 360)
            c, s = abs(math.cos(th)), abs(math.sin(th))
            # AABB bounds of rotated rectangle
            return (w * c + h * s, w * s + h * c)

        max_w_need, max_h_need = 0.0, 0.0
        for a in angles:
            rw, rh = rotated_dims(base_w, base_h, a)
            max_w_need = max(max_w_need, rw)
            max_h_need = max(max_h_need, rh)

        sx = inner_w / max(1.0, max_w_need)
        sy = inner_h / max(1.0, max_h_need)

        # Allow upscaling, but cap it to avoid extreme blow-ups.
        cap = float(SEQ_CONFIG.get("rot_upscale_cap", 1.6))
        s = min(sx, sy)
        s = min(s, cap)

        # Guard: never let the longest side drop below ~8 px
        return max(8.0 / max(base_w, base_h), s)

    def _render_rotated_centered(
            self,
            base_rgba: Image.Image,
            angle_deg: float,
            *,
            global_scale: float,
            border_ss: Optional[int] = None,
    ) -> Image.Image:
        """
        Center the rotated content in an SS_CELL tile with a thin border, then
        composite onto white and downscale to OUT_CELL. Uses a GLOBAL scale so
        all tiles/options use the same size.
        """
        big = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        border = border_ss if border_ss is not None else max(3, SS_CELL // 160)
        edge_pad = border + 1

        # scale base before rotation (common scale across the instance)
        w_scaled = max(8, int(round(base_rgba.width * global_scale)))
        h_scaled = max(8, int(round(base_rgba.height * global_scale)))
        content = base_rgba.resize((w_scaled, h_scaled), Image.LANCZOS)

        # rotate with expansion, then center
        rotated = content.rotate(angle_deg % 360, resample=Image.BICUBIC, expand=True)

        # center inside inner bounds (leave a soft margin)
        x0 = (SS_CELL - rotated.width) // 2
        y0 = (SS_CELL - rotated.height) // 2
        # guard against pathological rounding
        x0 = max(edge_pad, min(x0, SS_CELL - edge_pad - rotated.width))
        y0 = max(edge_pad, min(y0, SS_CELL - edge_pad - rotated.height))

        big.alpha_composite(rotated, (x0, y0))
        ImageDraw.Draw(big).rectangle([0, 0, SS_CELL - 1, SS_CELL - 1], outline=(0, 0, 0), width=border)
        white = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 255))
        composed = Image.alpha_composite(white, big)
        return composed.resize((OUT_CELL, OUT_CELL), Image.LANCZOS).convert("RGBA")

    def _angles_for_sequence(self, start: int, step: int, dir_sign: int) -> List[int]:
        return [int((start + dir_sign * i * step) % 360) for i in range(NUM_PANELS)]

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

    # --- NEW: rule-level check to avoid ambiguous options ---
    @staticmethod
    def _mod360(x: int) -> int:
        return int(x) % 360

    def _completes_progression(
        self, angles_full_true: List[int], mask_idx: int, candidate: int, step: int, dir_sign: int
    ) -> bool:
        """
        Returns True iff replacing the masked angle with `candidate` makes the
        entire 4-tile row follow the SAME (step, dir_sign) arithmetic progression mod 360.
        """
        seq = list(angles_full_true)
        seq[mask_idx] = self._mod360(candidate)
        step_m = self._mod360(step)

        # Compare consecutive diffs in the chosen direction.
        for i in range(3):
            a, b = self._mod360(seq[i]), self._mod360(seq[i+1])
            d = (b - a) % 360 if dir_sign == 1 else (a - b) % 360
            if d != step_m:
                return False
        return True


    # ------------- options -------------
    def _candidate_angles(self, corr_angle: int, step: int) -> List[int]:
        """
        Plausible distractors: multiples of the step around the correct angle,
        plus a small set of nearby off-step angles to avoid triviality if symmetry collapses.
        """
        base = set()
        # step-multiples (wrapping mod 360)
        for k in [1, 2, 3, 4, -1, -2, -3]:
            base.add((corr_angle + k * step) % 360)
        # add a few near-misses (±15° around one and two steps)
        near = [15, -15, step + 15, step - 15, 2 * step + 15, 2 * step - 15]
        for d in near:
            base.add((corr_angle + d) % 360)

        # ensure integers 0..359
        cand = [int(a % 360) for a in base if int(a % 360) != int(corr_angle % 360)]
        random.shuffle(cand)
        return cand

    def _build_options(
        self,
        base_rgba: Image.Image,
        rng: random.Random,
        corr_angle: int,
        step: int,
        labels: List[str],
        *,
        fixed_scale: Optional[float] = None,
        preset_wrong_angles: Optional[List[int]] = None,
        angles_seq_full: Optional[List[int]] = None,
        mask_idx: Optional[int] = None,
        dir_sign: Optional[int] = None,
    ):
        """
        Build four image options (one correct) and return:
          option_tiles, correct_label, option_descs, picked_wrongs
        If preset_wrong_angles is supplied, we use them verbatim (validating uniqueness);
        otherwise we pick greedily by visual separation.
        """
        margin_soft = max(10, SS_CELL // 20)
        inner_w = SS_CELL - 2 * margin_soft
        inner_h = SS_CELL - 2 * margin_soft

        def render(angle: int, *, scale: Optional[float]) -> Image.Image:
            """Rotate ``base_rgba`` by ``angle`` using optional ``scale``."""
            if scale is None:
                # single-angle fit (probe path)
                s = self._scale_for_angles(base_rgba.width, base_rgba.height, [angle], inner_w, inner_h)
            else:
                s = scale
            return self._render_rotated_centered(base_rgba, angle, global_scale=s)

        # Correct image first (probe scale or fixed)
        correct_img = render(corr_angle, scale=fixed_scale)
        correct_sig = sig(correct_img)

        # If we already know which wrongs we want (second pass), try to use them.
        if preset_wrong_angles is not None:
            picks = [int(a) % 360 for a in (preset_wrong_angles or []) if (int(a) % 360) != int(corr_angle % 360)]
            if len(picks) < 3:
                return None
            options = [(corr_angle, correct_img)]
            used = {correct_sig}
            for a in picks:
                if angles_seq_full is not None and mask_idx is not None and dir_sign is not None:
                    if self._completes_progression(angles_seq_full, mask_idx, a, step, dir_sign):
                        return None  # would create a second valid answer

                im = render(a, scale=fixed_scale)
                h = sig(im)
                if h in used:
                    return None
                used.add(h)
                options.append((a, im))
            if len(options) < NUM_OPTIONS:
                return None
            # Pairwise uniqueness check
            imgs = [im for (_a, im) in options]
            for i in range(NUM_OPTIONS):
                for j in range(i + 1, NUM_OPTIONS):
                    if sig(imgs[i]) == sig(imgs[j]):
                        return None
                    if diff_frac(imgs[i], imgs[j], thresh=8) < self.rot_option_min:
                        return None

            random.shuffle(options)
            angles = [a for (a, _im) in options]
            font = load_font()
            label_strs = labels_default()
            option_tiles = [crisp_option_tile(im, lab, font) for (_a, im), lab in zip(options, label_strs)]
            correct_label = label_strs[angles.index(corr_angle)]
            descs = [f"{a}°" for a in angles]
            return option_tiles, correct_label, descs, picks

        # --- No preset: score candidates by visual difference ---
        candidates = self._candidate_angles(corr_angle, step)
        rng.shuffle(candidates)

        scored: List[Tuple[float, int, bytes, Image.Image]] = []  # (diff, angle, signature, img)
        used_hashes = {correct_sig}
        for a in candidates:
            # Rule-level filter: do not consider angles that would also complete the sequence.
            if angles_seq_full is not None and mask_idx is not None and dir_sign is not None:
                if self._completes_progression(angles_seq_full, mask_idx, a, step, dir_sign):
                    continue
            im = render(a, scale=None)
            h = sig(im)
            if h in used_hashes:
                continue
            diff = diff_frac(correct_img, im, thresh=8)
            scored.append((diff, a, h, im))

        # Prefer large visual changes
        scored.sort(key=lambda t: t[0], reverse=True)

        def greedy_select(threshold: float) -> Tuple[List[int], List[Image.Image]]:
            """Select wrong angles with at least ``threshold`` difference."""
            picks_: List[int] = []
            imgs_: List[Image.Image] = []
            hashes_ = set(used_hashes)
            for diff, a, h, im in scored:
                if diff < max(threshold, 0.005):
                    continue
                if h in hashes_:
                    continue
                ok = True
                for im2 in imgs_:
                    if sig(im2) == h or diff_frac(im, im2, thresh=8) < threshold:
                        ok = False
                        break
                if ok:
                    picks_.append(a)
                    imgs_.append(im)
                    hashes_.add(h)
                    if len(picks_) == 3:
                        return picks_, imgs_
            return picks_, imgs_

        # Try strict threshold, then relax slightly if needed
        for relax in (1.0, 0.85, 0.7):
            thr = self.rot_option_min * relax
            picks, pick_imgs = greedy_select(thr)
            if len(picks) == 3:
                break
        else:
            return None

        options = [(corr_angle, correct_img)] + list(zip(picks, pick_imgs))
        random.shuffle(options)

        angles = [a for (a, _im) in options]
        imgs = [_im for (_a, _im) in options]
        # Final sanity (usually redundant now)
        for i in range(NUM_OPTIONS):
            for j in range(i + 1, NUM_OPTIONS):
                if sig(imgs[i]) == sig(imgs[j]):
                    return None
                if diff_frac(imgs[i], imgs[j], thresh=8) < self.rot_option_min:
                    return None

        font = load_font()
        label_strs = labels_default()
        option_tiles = [crisp_option_tile(im, lab, font) for im, lab in zip(imgs, label_strs)]
        correct_label = label_strs[angles.index(corr_angle)]
        descs = [f"{a}°" for a in angles]
        return option_tiles, correct_label, descs, picks

    def _degenerate_for_step(
            self, base_rgba: Image.Image, step: int, inner_w: int, inner_h: int
    ) -> bool:
        # Scale once to fit worst-case among 0, step, 2*step, 3*step
        angles = [0, step % 360, (2 * step) % 360, (3 * step) % 360]
        s = self._scale_for_angles(base_rgba.width, base_rgba.height, angles, inner_w, inner_h)
        ims = [self._render_rotated_centered(base_rgba, a, global_scale=s) for a in angles]
        # Require distinct hashes and a minimal pixel separation between successive angles
        sigs = [sig(im) for im in ims]
        if len(set(sigs)) < len(ims):
            return True
        for i in range(len(ims) - 1):
            if diff_frac(ims[i], ims[i + 1], thresh=8) < max(0.006, 0.5 * self.rot_option_min):
                return True
        return False

    # ------------- public -------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # ---- Collect allowed motifs and build a weighted order (without replacement) ----
        weights_map = globals().get("MOTIF_WEIGHTS", {})
        kinds_all = [k for k in motif_impls.keys() if k in weights_map] or list(motif_impls.keys())
        if not kinds_all:
            raise RuntimeError(f"{self.name}: no allowed motifs available.")

        motif_order = weighted_order(rng, kinds_all, weights_map)

        tried_failures: List[str] = []

        # ---- Try each motif up to max_retries before moving on ----
        for mk in motif_order:
            motif = motif_impls[mk]

            for _ in range(self.max_retries):
                # Render one base motif bitmap and crop
                try:
                    base_spec = _prefer_asym_mode(motif, motif.sample_spec(rng))
                    raw = motif.render(base_spec)
                except Exception:
                    continue

                thr = int(SEQ_CONFIG.get("rot_crop_white_threshold", 6))
                pad = int(SEQ_CONFIG.get("rot_crop_pad", 0))
                base_rgba = tight_crop_rgba(raw, white_threshold=thr, pad=pad)
                if base_rgba.width < 8 or base_rgba.height < 8:
                    continue  # too tiny / degenerate

                # --- NEW: pre-filter steps that would degenerate for THIS motif ---
                margin_soft = max(10, SS_CELL // 20)
                inner_w = SS_CELL - 2 * margin_soft
                inner_h = SS_CELL - 2 * margin_soft

                allowed_steps: List[int] = []
                allowed_weights: List[float] = []
                for s, w in zip(STEP_CHOICES, STEP_WEIGHTS):
                    # Reject steps that produce duplicates in [0, s, 2s, 3s]
                    if not self._degenerate_for_step(base_rgba, s, inner_w, inner_h):
                        allowed_steps.append(s)
                        allowed_weights.append(w)

                if not allowed_steps:
                    # All step choices collapse for this motif (e.g., extreme symmetry) → try a new motif/attempt
                    continue

                # Pick step from the filtered set (weights are auto‑renormalized by choices)
                step = choice_weighted(rng, allowed_steps, allowed_weights)
                dir_sign = rng.choice([1, -1])
                step_g = math.gcd(360, step)
                start = rng.randrange(0, 360, step_g)
                angles_seq = self._angles_for_sequence(start, step, dir_sign)

                # Probe scale for the sequence (will be refined after options)
                s_seq = self._scale_for_angles(base_rgba.width, base_rgba.height, angles_seq, inner_w, inner_h)

                # Render sequence with provisional scale and ensure visible progression
                seq_imgs_probe = [
                    self._render_rotated_centered(base_rgba, a, global_scale=s_seq) for a in angles_seq
                ]
                adj_min = min(
                    diff_frac(seq_imgs_probe[i], seq_imgs_probe[i + 1], thresh=8)
                    for i in range(len(seq_imgs_probe) - 1)
                )
                if adj_min < self.rot_adjacent_min:
                    continue

                # Basic uniqueness across the row
                sigs = [sig(im) for im in seq_imgs_probe]
                if len(set(sigs)) < NUM_PANELS:
                    continue

                # Mask a random cell and build options (probe pass)
                mask_idx = rng.randint(0, 3)
                corr_angle = int(angles_seq[mask_idx])

                probe = self._build_options(
                    base_rgba, rng, corr_angle, step, labels_default(),
                    fixed_scale=None, preset_wrong_angles=None,
                    angles_seq_full=angles_seq, mask_idx=mask_idx, dir_sign=dir_sign
                )
                if probe is None:
                    continue
                _, _, option_descs_first, picked_wrongs = probe

                # Lock a global scale for ALL angles (row + options), then re-render
                all_angles = list(angles_seq) + [int(a) for a in picked_wrongs]
                s_fixed = self._scale_for_angles(base_rgba.width, base_rgba.height, all_angles, inner_w, inner_h)

                final_cell_imgs = [
                    self._render_rotated_centered(base_rgba, a, global_scale=s_fixed) for a in angles_seq
                ]
                final_cell_imgs = [ensure_cell_rgba_outcell(im) for im in final_cell_imgs]

                # Rebuild options with fixed scale & the same wrong angles; if that fails, try fresh
                built = self._build_options(
                    base_rgba, rng, corr_angle, step, labels_default(),
                    fixed_scale=s_fixed, preset_wrong_angles=picked_wrongs,
                    angles_seq_full=angles_seq, mask_idx=mask_idx, dir_sign=dir_sign
                )
                if built is None:
                    built = self._build_options(
                        base_rgba, rng, corr_angle, step, labels_default(),
                        fixed_scale=s_fixed, preset_wrong_angles=None,
                        angles_seq_full=angles_seq, mask_idx=mask_idx, dir_sign=dir_sign
                    )
                    if built is None:
                        continue

                option_tiles, correct_label, option_descs, _ = built

                # Assemble composite
                opts = compose_options_row(option_tiles)
                top = compose_row_with_mask(
                    final_cell_imgs, mask_idx,
                    target_width=opts.width,
                    draw_arrows=True,
                    arrow_style={"ss": 3, "alpha": 220, "scale": 1.2},
                    gap=OUT_CELL // 8,
                    margin=0,
                )
                composite = compose_top_bottom(top, opts, sep_px=40)

                question, prompt_style = self._format_question(rng, labels_default())

                meta_out = {
                    "pattern_kind": "sequence",
                    "pattern": self.name,
                    "grid": (1, NUM_PANELS),
                    "mask_idx": mask_idx,
                    "variant": "rotation_fixed_step",
                    "motif_kind": mk,
                    "motif_spec": base_spec.to_dict() if hasattr(base_spec, "to_dict") else dict(base_spec.__dict__),
                    "labels": labels_default(),
                    "answer": correct_label,
                    "option_strategy": "image_rotations",
                    "option_descs": option_descs,
                    "option_angles": [int(str(a).rstrip("°")) for a in option_descs],
                    "angles_top_row": [int(a) for a in angles_seq],
                    "step_deg": int(step),
                    "direction": int(dir_sign),
                    "start_angle": int(start),
                    "question": question,
                    "prompt_style": str(prompt_style),
                    "composite_ready": True,
                }
                cell_payloads = [{"angle": int(a)} for a in angles_seq]
                complexity = self._compute_complexity(int(step))
                meta_out["complexity"] = complexity
                meta_out["complexity_score"] = complexity["score"]
                meta_out["complexity_level"] = complexity["level"]
                meta_out["complexity_version"] = complexity["version"]
                return composite, cell_payloads, meta_out

            # this motif failed all retries; record and move to the next
            tried_failures.append(mk)

        # ---- All motifs failed ----
        order_str = ", ".join(motif_order) if motif_order else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a verifiable sample after {self.max_retries} attempts per motif. "
            f"Motifs tried (in order): {order_str}"
        )
