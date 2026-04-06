# sphinx/tasks/symmetry/symmetry_grid_mirror_fill.py
import random
from typing import Any, Dict, List, Tuple
from PIL import Image

from ...base import Task
from ...registry import register_task

from ...config import (
    IMG_DIFF_MIN, OPT_UNIQUENESS_MIN, MAX_BUILD_RETRIES,
    OPT_HASH_MIN_BITS, GRID_HASH_MIN_BITS, BORDER_PX
)
from .common import apply_tf, sig, BASE_RULES_NO_ROT, ALL_TFS_ORDER, strong_distinct, pairwise_unique
from ...utils.specs import _prefer_asym_mode
from ...utils.rng import weighted_order

from ...utils.drawing import (
    load_font, crisp_option_tile, labels_default,
    compose_2x2_grid, compose_options_2x2, compose_left_right,
    ensure_transparent, add_tile_border, strip_outer_frame
)

# Generation constants -------------------------------------------------------
NUM_OPTIONS = 4             # number of choice tiles
GRID_SHAPE = (2, 2)         # layout for the options

SYMMETRY_NAMES = {
    "mirror_v":  "vertical mirror symmetry",
    "mirror_h":  "horizontal mirror symmetry",
    "mirror_hv": "vertical + horizontal mirror symmetry",
    "diag_main": "main‑diagonal line symmetry",
    "diag_anti": "anti‑diagonal line symmetry",
}

PROMPT_TEMPLATES = [
    "The left panel shows a 2×2 grid with one tile hidden (black square). Which option (a)-(d) completes the grid so the full pattern has {sym_name}?",
    "In the left panel, one tile of the 2×2 grid is blank (black). Which option (a)-(d) restores {sym_name}?",
    "Look at the 2×2 grid on the left. Which option (a)-(d) should replace the black square so the pattern exhibits {sym_name}?",
    "Which option (a)-(d) fills the missing tile in the 2×2 grid on the left to make the pattern obey {sym_name}?",
    "Which candidate (a)-(d) completes the left-panel 2×2 grid so that it has {sym_name}?",
    "Which tile (a)-(d), when placed in the black square on the left, yields {sym_name}?",
    "To make the 2×2 grid on the left display {sym_name}, which option (a)-(d) should fill the blank tile?",
    "Which option (a)-(d) should be inserted into the black square of the left panel so the finished grid has {sym_name}?",
    "Which option (a)-(d) completes the left 2×2 grid with {sym_name}?",
    "Which option (a)-(d) completes the 2×2 grid on the left so that it satisfies {sym_name}?",
]



MOTIF_WEIGHTS = {
    "icons": 15,
    "arc": 0.5,
    "arrow": 0.25,
    "bars": 0.25,
    "bitgrid": 0.25,
    "clock": 0.75,
    "concentric_polygon": 0.1,
    "crescent": 1.0,
    "fractal": 0.25,
    "gear": 0.25,
    "glyph": 1.25,
    "keyhole": 0.5,
    "pinwheel_triangles": 0.5,
    "polygon": 1.25,
    "polyhex": 0.3,
    "polyiamond": 0.3,
    "polyline": 0.25,
    "polyomino": 0.3,
    "rings": 0.25,
    "segment": 0.25,
    "star_polygon": 0.25,
    "stripes": 0.25,
    "pictogram": 0.5,
}


@register_task
class SymmetryGridMirrorFillTask(Task):
    name = "symmetry_grid_mirror_fill"
    RULE_KEYS = ["mirror_v", "mirror_h", "mirror_hv", "diag_main", "diag_anti"]

    def __init__(self):
        self.min_delta = float(IMG_DIFF_MIN)
        self.opt_min_delta = float(OPT_UNIQUENESS_MIN)
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.opt_hash_min_bits = int(OPT_HASH_MIN_BITS)
        self.grid_hash_min_bits = int(GRID_HASH_MIN_BITS)

    @staticmethod
    def _format_prompt(rule_key: str, rng: random.Random) -> str:
        return rng.choice(PROMPT_TEMPLATES).format(sym_name=SYMMETRY_NAMES[rule_key])

    def _verify_grid_rule(self, base: Image.Image, tfs: List[str]) -> bool:
        # 1) Every distinct transform (other than "original") must actually change the image.
        for tf in set(tfs):
            if tf == "original":
                continue
            im = apply_tf(base, tf)
            if not strong_distinct(im, base, self.min_delta, self.grid_hash_min_bits):
                return False
        # 2) The grid must have at least as many distinct *looks* as the number of
        #    distinct transform names in the rule (2 for v/h/diag, 4 for hv).
        imgs = [apply_tf(base, tf).convert("RGB") for tf in tfs]
        distinct_looks = {sig(im) for im in imgs}
        return len(distinct_looks) >= len(set(tfs))

    def _build_options(self, base: Image.Image, correct_tf: str, rng: random.Random, labels: List[str]):
        """Return four option images and the correct label.

        The pool is built by applying spatial transforms to ``base`` and keeping
        only visually distinct results. The correct transform is included along
        with three distractors.
        """

        # 1) correct first
        correct_img = apply_tf(base, correct_tf).convert("RGB")
        picks: List[Tuple[str, Image.Image]] = [(correct_tf, correct_img)]

        # 2) build candidate pool without exact duplicates
        used = {sig(correct_img)}
        pool: List[Tuple[str, Image.Image]] = []
        for tf in ALL_TFS_ORDER:
            if tf == correct_tf:
                continue
            im = apply_tf(base, tf).convert("RGB")
            h = sig(im)
            if h in used:
                continue
            used.add(h)
            pool.append((tf, im))

        rng.shuffle(pool)

        # 3) admit only candidates that are distinct from all current picks
        for tf, im in pool:
            if all(strong_distinct(im, pj[1], self.opt_min_delta, self.opt_hash_min_bits) for pj in picks):
                picks.append((tf, im))
            if len(picks) == NUM_OPTIONS:
                break

        if len(picks) < NUM_OPTIONS or not pairwise_unique([p[1] for p in picks], self.opt_min_delta, self.opt_hash_min_bits):
            return None

        # 4) shuffle and compute labels
        rng.shuffle(picks)
        tf_names = [p[0] for p in picks]
        correct_label = labels[tf_names.index(correct_tf)]

        imgs = [p[1] for p in picks]
        # 1) TF names themselves must be unique
        tf_names = [p[0] for p in picks]
        assert len(set(tf_names)) == NUM_OPTIONS, f"TF names duplicated: {tf_names}"

        # 2) Pairwise uniqueness on the raw bitmaps
        for i in range(NUM_OPTIONS):
            for j in range(i + 1, NUM_OPTIONS):
                assert sig(imgs[i]) != sig(imgs[j]) and strong_distinct(imgs[i], imgs[j], self.opt_min_delta, self.opt_hash_min_bits), \
                    f"Option bitmaps not unique at ({i},{j})"

        return picks, correct_label

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        # Filter allowed motifs present in the registry (skip zero/negative weights)
        kinds_all = [k for k in motif_impls.keys() if k in MOTIF_WEIGHTS and MOTIF_WEIGHTS[k] > 0]
        if not kinds_all:
            raise RuntimeError("SymmetryGridMirrorFillTask: no allowed motifs available.")

        motif_order = weighted_order(rng, kinds_all, MOTIF_WEIGHTS)

        font = load_font()
        labels = labels_default()

        # Try each motif up to max_retries before moving on
        failures: List[str] = []
        for mk in motif_order:
            motif = motif_impls[mk]

            for _ in range(self.max_retries):
                spec = _prefer_asym_mode(motif, motif.sample_spec(rng))
                base = motif.render(spec)
                # Ensure motif is composited onto a solid white background
                base = ensure_transparent(base)
                white_bg = Image.new("RGBA", base.size, (255, 255, 255, 255))
                base = Image.alpha_composite(white_bg, base).convert("RGB")
                base = strip_outer_frame(base, strip_px=BORDER_PX)

                rule_key = rng.choice(self.RULE_KEYS)
                tfs = BASE_RULES_NO_ROT[rule_key]
                if not self._verify_grid_rule(base, tfs):
                    continue

                cells = [apply_tf(base, tf).convert("RGB") for tf in tfs]
                mask_idx = rng.randint(0, NUM_OPTIONS - 1)
                correct_tf = tfs[mask_idx]

                built = self._build_options(base, correct_tf, rng, labels)
                if built is None:
                    continue

                picks, correct_label = built

                option_tiles = [
                    crisp_option_tile(add_tile_border(img, width_px=BORDER_PX), lab, font)
                    for (_, img), lab in zip(picks, labels)
                ]
                left = compose_2x2_grid(cells, mask_idx)
                right = compose_options_2x2(option_tiles)
                composite = compose_left_right(left, right)

                question = self._format_prompt(rule_key, rng)
                meta = {
                    "pattern_kind": "symmetry",
                    "pattern": self.name,
                    "grid": GRID_SHAPE,
                    "mask_idx": mask_idx,
                    "variant": rule_key,
                    "grid_spatial_tfs": tfs,
                    "motif_kind": mk,
                    "labels": labels,
                    "answer": correct_label,
                    "option_strategy": "image_spatial_tfs",
                    "option_descs": [p[0] for p in picks],
                    "option_payload": {"correct_tf": correct_tf},
                    "question": question,
                    "composite_ready": True,
                }
                return composite, [spec] * NUM_OPTIONS, meta

            # If we got here, this motif failed all retries; record and continue
            failures.append(mk)

        # All motifs failed
        tried = ", ".join(failures) if failures else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a verifiable sample after {self.max_retries} attempts "
            f"per motif. Motifs tried (in order): {tried}"
        )
