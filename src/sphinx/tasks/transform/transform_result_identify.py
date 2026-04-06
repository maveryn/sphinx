# sphinx/tasks/transform/transform_result_identify.py
import random
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import (
    OUT_CELL, IMG_DIFF_MIN, OPT_UNIQUENESS_MIN, MAX_BUILD_RETRIES, OPT_HASH_MIN_BITS
)
from sphinx.tasks.symmetry.common import sig, strong_distinct, pairwise_unique  # reuse shared transform utils
from sphinx.utils.specs import _prefer_asym_mode
from sphinx.utils.drawing import (
    load_font, crisp_option_tile, labels_default,
    tight_crop_rgba, ensure_transparent, add_tile_border, paste_rgba
)
from .common import (
    scale_patch, center_xy, bounds_ok, compose_tile_with_patch,
    apply_patch_transform, sample_translation, candidate_translations,
    weighted_order
)

# ----------------------------- public-facing names -----------------------------
TF_RULES = [
    "mirror_v", "mirror_h", "diag_main", "diag_anti",
    "rot90", "rot180", "rot270",
    "translate",  # single 'pure translation' family; vector sampled per-instance
]

TF_DISPLAY = {
    "mirror_v":  "reflect across a vertical line",
    "mirror_h":  "reflect across a horizontal line",
    "diag_main": "reflect across the main diagonal (↘︎)",
    "diag_anti": "reflect across the anti‑diagonal (↙︎)",
    "rot90":     "rotate 90° counterclockwise",
    "rot180":    "rotate 180°",
    "rot270":    "rotate 90° clockwise",
    "translate": "translate (shift) the shape",
}

PROMPT_TEMPLATES = [
    "The top image is the original; the bottom shows options (a)–(d). Which option is the result of applying {tf_name}?",
    "Apply {tf_name} to the top image. Which option (a)–(d) matches the transformed result?",
    "After performing {tf_name} on the top figure, which option (a)–(d) shows the correct outcome?",
    "Which option (a)–(d) corresponds to the top image after {tf_name} is applied?",
    "Look at the top figure. If you apply {tf_name}, which option (a)–(d) is the correct result?",
    "The top panel shows the original. Which option (a)–(d) below is the result of {tf_name}?",
    "After {tf_name} is applied to the top shape, which option (a)–(d) matches the outcome?",
    "Which option (a)–(d) represents the result of applying {tf_name} to the top figure?",
    "If the top image undergoes {tf_name}, which option (a)–(d) shows the correct transformed version?",
    "The top image is the original, and the bottom shows options (a)–(d). After applying {tf_name}, which option is correct?"
]


MOTIF_WEIGHTS = {
    "icons": 10,
    "arc": 0.5,
    "arrow": 0.5,
    "bars": 0.25,
    "clock": 0.75,
    "crescent": 0.75,
    "gear": 0.25,
    "glyph": 1.0,
    "keyhole": 0.5,
    "pictogram": 0.5,
    "pinwheel_triangles": 0.5,
    "polygon": 1.0,
    "polyhex": 0.25,
    "polyiamond": 0.25,
    "polyline": 0.5,
    "polyomino": 0.25,
    "rings": 0.25,
    "star_polygon": 0.25,
}

# Generation constants -------------------------------------------------------
NUM_OPTIONS = 4  # number of transformed options
def _format_prompt(tf_key: str, rng: random.Random) -> str:
    return rng.choice(PROMPT_TEMPLATES).format(tf_name=TF_DISPLAY[tf_key])


# ----------------------------- task -----------------------------
@register_task
class TransformResultIdentifyTask(Task):
    """
    Show the original motif (over graph paper) on top. On the bottom row, show four
    transformed options (also over graph paper). Ask: which option equals applying
    the sampled transformation to the original?
    """
    name = "transform_result_identify"

    def __init__(self):
        self.min_delta = float(IMG_DIFF_MIN)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.tile_px = int(OUT_CELL)

    def _build_original_tile(
            self, motif, rng: random.Random
    ) -> Tuple[Image.Image, Image.Image, Tuple[int, int], Any]:
        """
        Returns:
          - orig_tile: RGBA tile with graph background + centered motif
          - patch: the centered motif patch (RGBA), pre-scaled/cropped
          - center_xy: top-left of the centered patch within tile
          - spec: the motif spec used
        """
        spec = _prefer_asym_mode(motif, motif.sample_spec(rng))
        raw = motif.render(spec)  # may be opaque
        raw = ensure_transparent(raw)  # <<< make motif transparent
        patch = tight_crop_rgba(raw)  # tight crop after alpha
        patch = scale_patch(patch, self.tile_px, rng)

        cx, cy = center_xy(self.tile_px, patch.width, patch.height)
        orig_tile = compose_tile_with_patch(self.tile_px, patch, (cx, cy))
        return orig_tile, patch, (cx, cy), spec

    def _build_options(
        self,
        patch: Image.Image,
        centered_xy: Tuple[int, int],
        rule_key: str,
        rng: random.Random,
        labels: List[str],
    ):
        """
        Build four unique option images; return (picks, correct_label, tf_names, extra_payload)
        where picks = [(tf_key, tile_img), ...].
        """
        picks: List[Tuple[str, Image.Image]] = []
        tf_order = TF_RULES[:]
        rng.shuffle(tf_order)

        # Prepare vector for the "correct" translation (if needed)
        correct_tvec = (0, 0)
        if rule_key == "translate":
            dx, dy = sample_translation(rng, self.tile_px, patch.width, patch.height)
            if (dx, dy) == (0, 0):
                return None
            correct_tvec = (dx, dy)

        # Helper to make a tile for a given transform
        def make_tile(tf_key: str, tvec_override: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
            transformed = apply_patch_transform(patch, tf_key)
            # Where to place?
            cx, cy = centered_xy
            if tf_key == "translate":
                if tvec_override is None:
                    dx, dy = correct_tvec
                else:
                    dx, dy = tvec_override
                x = cx + dx; y = cy + dy
            else:
                # keep at the same center; patch dims may change after transpose/rot90
                x = self.tile_px // 2 - transformed.width // 2
                y = self.tile_px // 2 - transformed.height // 2
            if not bounds_ok(self.tile_px, x, y, transformed.width, transformed.height):
                return None
            return compose_tile_with_patch(self.tile_px, transformed, (x, y))

        # 1) Start with the correct option
        correct_img = make_tile(rule_key)
        if correct_img is None:
            return None
        picks.append((rule_key, correct_img))
        used_hashes = {sig(correct_img)}

        # 2) Build a pool for distractors (other transforms)
        pool: List[Tuple[str, Image.Image]] = []

        # For translate as a distractor, use a different vector than the correct one
        def alt_translate() -> Optional[Tuple[int, int]]:
            """Return an alternative translation vector for distractors."""
            cands = candidate_translations(
                self.tile_px, patch.width, patch.height, centered_xy
            )
            rng.shuffle(cands)
            for dx, dy in cands:
                if (rule_key == "translate") and (dx, dy) == correct_tvec:
                    continue
                return (dx, dy)
            return None

        for key in TF_RULES:
            if key == rule_key:
                continue
            if key == "translate":
                tvec = alt_translate()
                if tvec is None:
                    continue
                img = make_tile("translate", tvec_override=tvec)
            else:
                img = make_tile(key)
            if img is None:
                continue
            h = sig(img)
            if h in used_hashes:
                continue
            used_hashes.add(h)
            pool.append((key, img))

        rng.shuffle(pool)

        # 3) Admit distractors that remain pairwise distinct
        for key, im in pool:
            if all(strong_distinct(im, pj[1], self.opt_min_delta, self.opt_hash_min_bits) for pj in picks):
                picks.append((key, im))
            if len(picks) == NUM_OPTIONS:
                break

        if len(picks) < NUM_OPTIONS or not pairwise_unique([p[1] for p in picks], self.opt_min_delta, self.opt_hash_min_bits):
            return None

        rng.shuffle(picks)
        tf_names = [p[0] for p in picks]
        correct_label = labels[tf_names.index(rule_key)]
        payload = {
            "tf_keys": tf_names,
            "correct_tf": rule_key,
        }
        if rule_key == "translate":
            payload["translate_vec"] = {"dx": correct_tvec[0], "dy": correct_tvec[1]}
        return picks, correct_label, tf_names, payload

    def _compose_top_bottom(self, top: Image.Image, option_tiles: List[Image.Image]) -> Image.Image:
        # use actual image sizes, not OUT_CELL
        pad = max(8, self.tile_px // 16)

        top_w, top_h = top.width, top.height
        opt_w, opt_h = option_tiles[0].width, option_tiles[0].height  # includes label band

        # canvas sized to fit the tallest/largest tiles
        W = max(top_w + 2 * pad, NUM_OPTIONS * opt_w + (NUM_OPTIONS + 1) * pad)
        H = top_h + opt_h + 3 * pad

        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

        # top centered
        x_top = (W - top_w) // 2
        y_top = pad
        paste_rgba(canvas, top, (x_top, y_top))

        # bottom row (use opt_w/opt_h so labels aren’t clipped)
        y_bot = top_h + 2 * pad
        for i, tile in enumerate(option_tiles):
            x = pad + i * (opt_w + pad)
            paste_rgba(canvas, tile, (x, y_bot))

        # frame around the top tile
        dr = ImageDraw.Draw(canvas)
        tile_border_w = max(3, min(top_w, top_h) // 48)
        dr.rectangle([x_top, y_top, x_top + top_w - 1, y_top + top_h - 1],
                     outline=(0, 0, 0), width=tile_border_w)

        # thin outer border
        border_w = max(2, W // 320)
        dr.rectangle([1, 1, W - 2, H - 2], outline=(0, 0, 0), width=border_w)
        return canvas

    # ----------------------------- public API -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # 1) choose motif order by weights
        kinds_all = [k for k in motif_impls.keys() if float(MOTIF_WEIGHTS.get(k, 0.0)) > 0.0]
        if not kinds_all:
            raise RuntimeError(f"{self.name}: no allowed motifs available.")
        motif_order = weighted_order(rng, kinds_all, MOTIF_WEIGHTS)

        font = load_font()
        labels = labels_default()

        failures: List[str] = []
        for mk in motif_order:
            motif = motif_impls[mk]
            for _ in range(self.max_retries):
                try:
                    top_tile, patch, centered_xy, spec = self._build_original_tile(motif, rng)
                except Exception:
                    continue

                rule_key = rng.choice(TF_RULES)
                built = self._build_options(patch, centered_xy, rule_key, rng, labels)
                if built is None:
                    continue

                picks, correct_label, tf_names, payload = built

                tile_border_w = max(3, self.tile_px // 48)

                # label the options (a)–(d)
                option_tiles = [
                    crisp_option_tile(add_tile_border(img, width_px=tile_border_w), lab, font)
                    for (_, img), lab in zip(picks, labels)
                ]

                composite = self._compose_top_bottom(top_tile, option_tiles)

                question = _format_prompt(rule_key, rng)
                meta = {
                    "pattern_kind": "transform",
                    "pattern": self.name,
                    "grid": (1, NUM_OPTIONS),
                    "mask_idx": -1,
                    "variant": rule_key,
                    "motif_kind": mk,
                    "labels": labels,
                    "answer": correct_label,
                    "option_strategy": "image_transforms",
                    "option_descs": [TF_DISPLAY[k] for k in tf_names],
                    "option_payload": payload,
                    "question": question,
                    "composite_ready": True,
                }
                # Return: composite image, optional [specs] for provenance (not used by solver)
                return composite, [spec], meta

            failures.append(mk)

        tried = ", ".join(failures) if failures else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a verifiable sample after {self.max_retries} attempts per motif. "
            f"Motifs tried (in order): {tried}"
        )
