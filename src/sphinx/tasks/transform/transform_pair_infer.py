# sphinx/tasks/transform/transform_pair_infer.py
import random
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import (
    OUT_CELL,
    IMG_EQUAL_TOL,
    EQUAL_HASH_MAX_BITS,
    MAX_BUILD_RETRIES,
)
from sphinx.tasks.symmetry.common import strong_same
from sphinx.utils.specs import _prefer_asym_mode
from sphinx.utils.drawing import (
    tight_crop_rgba, ensure_transparent, paste_rgba,
)
from .common import (
    scale_patch, center_xy, bounds_ok, compose_tile_with_patch,
    apply_patch_transform, sample_translation, candidate_translations,
    weighted_order
)

# ----------------------------- transforms & labels -----------------------------
TF_RULES = [
    "mirror_v", "mirror_h", "diag_main", "diag_anti",
    "rot90", "rot180", "rot270",
    "translate",
]
OPT_KEYS = TF_RULES + ["none"]

# Synonymized display strings (randomly chosen per option)
DISPLAY_SYNONYMS = {
    "mirror_v":  ["reflect across a vertical line", "vertical mirror", "vertical line symmetry"],
    "mirror_h":  ["reflect across a horizontal line", "horizontal mirror", "horizontal line symmetry"],
    "diag_main": ["reflect across the main diagonal (↘)", "main-diagonal mirror", "main-diagonal reflection"],
    "diag_anti": ["reflect across the anti-diagonal (↙)", "anti-diagonal mirror", "anti-diagonal reflection"],
    "rot90":     ["rotate 90° counterclockwise", "quarter-turn CCW (90°)", "rotate 270° clockwise (≡ 90° CCW)"],
    "rot180":    ["rotate 180°", "half-turn (180°)", "rotate by 180°"],
    "rot270":    ["rotate 270° counterclockwise", "rotate 90° clockwise (≡ 270° CCW)", "quarter-turn clockwise (90°)"],
    "translate": ["translate (shift) the shape", "move without rotation/flip", "pure translation (shift)"],
    "none":      ["none of the above", "none of these", "none of the options"],
}


PROMPT_TEMPLATES = [
    "The left tile transforms into the right tile. Which single transformation was applied? Choose one: {choices}",
    "Look at the change from left to right. What single transformation produces the right image? Select one: {choices}",
    "Which transformation maps the left image to the right image? Pick one option: {choices}",
    "From the left tile to the right tile, which operation was performed? Choose one: {choices}",
    "Exactly one transformation changes the left figure into the right. Which one is it? {choices}",
    "What single operation turns the left tile into the right tile? {choices}",
    "Identify the transformation applied between the left and right images. {choices}",
    "Which option best describes the transformation from left to right? {choices}",
    "The left image becomes the right image after one step. What is that step? {choices}",
    "Which transformation produces the right image from the left image? {choices}"
]


# Motif sampling weights (tuned for asymmetry + variety)
MOTIF_WEIGHTS = {
    "icons": 10,
    "arc": 0.5,
    "arrow": 0.5,
    "bars": 0.25,
    "clock": 0.5,
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

NUM_OPTIONS = 6           # a–f
LABELS6 = list("abcdef")  # labels for the six options
NONE_OMIT_PROB = 1.0 / 6  # probability that we omit the true transform so "none" is correct

# ----------------------------- helpers -----------------------------
# candidate_translations is provided by .common

def _format_prompt(labels: List[str], descs: List[str], rng: random.Random) -> str:
    choices = ", ".join([f"({l}) {d}" for l, d in zip(labels, descs)])
    return rng.choice(PROMPT_TEMPLATES).format(choices=choices)

def _move_none_last(keys: List[str]) -> List[str]:
    """If 'none' is present, move it to the last position."""
    if "none" in keys and keys[-1] != "none":
        keys = [k for k in keys if k != "none"] + ["none"]
    return keys

def _compose_pair(canvas_w: int, canvas_h: int, left: Image.Image, right: Image.Image, pad: int) -> Image.Image:
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    # place tiles
    xL = pad; yL = (canvas_h - left.height) // 2
    xR = canvas_w - pad - right.width; yR = (canvas_h - right.height) // 2
    paste_rgba(canvas, left, (xL, yL))
    paste_rgba(canvas, right, (xR, yR))

    # borders
    dr = ImageDraw.Draw(canvas)
    bw = max(2, left.width // 48)
    dr.rectangle([xL, yL, xL + left.width - 1, yL + left.height - 1], outline=(0, 0, 0), width=bw)
    dr.rectangle([xR, yR, xR + right.width - 1, yR + right.height - 1], outline=(0, 0, 0), width=bw)

    # arrow (→) between tiles
    gap_left = xL + left.width
    gap_right = xR
    mid_y = canvas_h // 2
    margin = max(4, pad // 6)
    x0 = gap_left + margin
    x1 = gap_right - margin
    shaft_w = max(2, left.width // 80)
    head_len = max(8, min((x1 - x0) // 3, left.width // 8))
    # shaft
    dr.line([(x0, mid_y), (x1 - head_len, mid_y)], fill=(0, 0, 0), width=shaft_w)
    # triangular head
    head_h = max(6, shaft_w * 4)
    dr.polygon([
        (x1 - head_len, mid_y - head_h // 2),
        (x1,            mid_y),
        (x1 - head_len, mid_y + head_h // 2),
    ], fill=(0, 0, 0))
    # outer border (optional)
    ob = max(2, canvas_w // 320)
    dr.rectangle([1, 1, canvas_w - 2, canvas_h - 2], outline=(0, 0, 0), width=ob)
    return canvas

# ----------------------------- task -----------------------------
@register_task
class TransformPairInferTask(Task):
    """
    Show a pair of tiles (left=source, right=target) over graph paper with an arrow.
    Ask: which single transformation maps left → right?
    6 text options (a–f). With probability 1/6, the true transform is omitted so
    'none of the above' is the correct answer.
    """
    name = "transform_pair_infer"

    def __init__(self):
        self.tol = float(IMG_EQUAL_TOL)
        self.eq_bits = int(EQUAL_HASH_MAX_BITS)
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.tile_px = int(OUT_CELL)

    # ----- build source patch & left tile -----
    def _build_left(self, motif, rng: random.Random) -> Tuple[Image.Image, Image.Image, Tuple[int, int], Any]:
        spec = _prefer_asym_mode(motif, motif.sample_spec(rng))
        raw = motif.render(spec)
        raw = ensure_transparent(raw)
        patch = tight_crop_rgba(raw)
        patch = scale_patch(patch, self.tile_px, rng)
        cx, cy = center_xy(self.tile_px, patch.width, patch.height)
        left = compose_tile_with_patch(self.tile_px, patch, (cx, cy))
        return left, patch, (cx, cy), spec

    # ----- create right tile for a given rule -----
    def _make_right(self, patch: Image.Image, centered_xy: Tuple[int, int], rule_key: str, rng: random.Random):
        if rule_key == "translate":
            dx, dy = sample_translation(rng, self.tile_px, patch.width, patch.height)
            if (dx, dy) == (0, 0):
                return None
            x, y = centered_xy[0] + dx, centered_xy[1] + dy
            right = compose_tile_with_patch(self.tile_px, patch, (x, y))
            return right, {"translate_vec": {"dx": dx, "dy": dy}}
        else:
            tpatch = apply_patch_transform(patch, rule_key)
            x = self.tile_px // 2 - tpatch.width // 2
            y = self.tile_px // 2 - tpatch.height // 2
            if not bounds_ok(self.tile_px, x, y, tpatch.width, tpatch.height):
                return None
            right = compose_tile_with_patch(self.tile_px, tpatch, (x, y))
            return right, {}

    # ----- exact-match checker over all allowed single transforms -----
    def _compose_left_right(self, left: Image.Image, right: Image.Image) -> Image.Image:
        pad = max(10, self.tile_px // 10)
        W = self.tile_px * 2 + 3 * pad
        H = self.tile_px + 2 * pad
        return _compose_pair(W, H, left, right, pad)

    # ----------------------------- public -----------------------------
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        labels = LABELS6  # 6 options

        kinds_all = [k for k in motif_impls.keys() if float(MOTIF_WEIGHTS.get(k, 0.0)) > 0.0]
        if not kinds_all:
            raise RuntimeError(f"{self.name}: no allowed motifs available.")
        motif_order = weighted_order(rng, kinds_all, MOTIF_WEIGHTS)

        tried_failures: List[str] = []
        for mk in motif_order:
            motif = motif_impls[mk]
            for _ in range(self.max_retries):
                # Build left/source
                try:
                    left, patch, centered_xy, spec = self._build_left(motif, rng)
                except Exception:
                    continue

                # Choose a *true* transform to generate the right image (not 'none')
                correct_tf = rng.choice(TF_RULES)

                # Build right/target and verify uniqueness of mapping
                built = self._make_right(patch, centered_xy, correct_tf, rng)
                if built is None:
                    continue
                right, extra_payload = built

                # ensure exactly one transform matches (the chosen one)
                matches = []
                for tf in TF_RULES:
                    if tf == "translate":
                        for dx, dy in candidate_translations(self.tile_px, patch.width, patch.height, centered_xy):
                            x = centered_xy[0] + dx; y = centered_xy[1] + dy
                            cand = compose_tile_with_patch(self.tile_px, patch, (x, y))
                            if strong_same(cand, right, self.tol, self.eq_bits):
                                matches.append(("translate", (dx, dy)))
                    else:
                        tpatch = apply_patch_transform(patch, tf)
                        x = self.tile_px // 2 - tpatch.width // 2
                        y = self.tile_px // 2 - tpatch.height // 2
                        if not bounds_ok(self.tile_px, x, y, tpatch.width, tpatch.height):
                            continue
                        cand = compose_tile_with_patch(self.tile_px, tpatch, (x, y))
                        if strong_same(cand, right, self.tol, self.eq_bits):
                            matches.append((tf, None))

                # Must be at least one match for the chosen tf
                if not any(m[0] == correct_tf for m in matches):
                    continue
                # Heuristic: avoid cases where multiple *different* TF_RULES also match (ambiguous motif)
                # (Allow multiple vectors for translation.)
                other_tfs = {m[0] for m in matches if m[0] != "translate"}
                if correct_tf != "translate":
                    other_tfs.discard(correct_tf)
                    if other_tfs:
                        continue

                # Compose visual pair
                composite = self._compose_left_right(left, right)

                # ----- Build 6-option text MCQ with synonyms -----
                none_by_omission = rng.random() < NONE_OMIT_PROB

                if none_by_omission:
                    # Omit the real transform; force 'none' to be present and correct
                    pool = [k for k in TF_RULES if k != correct_tf]
                    rng.shuffle(pool)
                    keys = ["none"] + pool[:NUM_OPTIONS - 1]  # 1 ('none') + 5 others (no correct_tf)
                    rng.shuffle(keys)
                    keys = _move_none_last(keys)  # ensure 'none' is last
                    correct_key = "none"
                else:
                    # Include the real transform; fill the rest from all others (including possibly 'none')
                    pool = [k for k in OPT_KEYS if k != correct_tf]
                    rng.shuffle(pool)
                    keys = [correct_tf] + pool[:NUM_OPTIONS - 1]
                    rng.shuffle(keys)
                    keys = _move_none_last(keys)  # ensure 'none' (if present) is last
                    correct_key = correct_tf

                # Pick a random synonym string per key
                descs = [rng.choice(DISPLAY_SYNONYMS[k]) for k in keys]
                correct_label = labels[keys.index(correct_key)]

                # Compose question text
                choices_str = ", ".join([f"({l}) {d}" for l, d in zip(labels, descs)])
                question = rng.choice(PROMPT_TEMPLATES).format(choices=choices_str)

                meta = {
                    "pattern_kind": "transform",
                    "pattern": self.name,
                    "grid": (1, 2),
                    "mask_idx": -1,
                    "variant": correct_key,
                    "motif_kind": mk,
                    "labels": labels,
                    "answer": correct_label,
                    "option_strategy": "text_transform_synonyms",
                    "option_descs": descs,
                    "option_payload": {
                        "option_keys": keys,
                        "none_by_omission": none_by_omission,
                    },
                    "question": question,
                    "composite_ready": True,
                }
                if correct_tf == "translate":
                    meta["option_payload"]["translate_vec"] = extra_payload.get("translate_vec", {})

                return composite.convert("RGB"), [spec], meta

            tried_failures.append(mk)

        tried = ", ".join(tried_failures) if tried_failures else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a unique-answer sample after {self.max_retries} attempts per motif. "
            f"Motifs tried (in order): {tried}"
        )
