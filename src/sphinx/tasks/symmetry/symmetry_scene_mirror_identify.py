# sphinx/tasks/symmetry/symmetry_scene_mirror_identify.py
import random
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image, ImageDraw

from ...base import Task
from ...registry import register_task
from ...config import SS_CELL, IMG_EQUAL_TOL, MAX_BUILD_RETRIES, EQUAL_HASH_MAX_BITS
from .common import apply_tf, strong_same
from ...utils.specs import _prefer_asym_mode
from ...utils.drawing import tight_crop_rgba, paste_rgba
from ...utils.rng import choice_weighted, weighted_order

OPT_KEYS = ["mirror_v", "mirror_h", "diag_main", "diag_anti", "mirror_hv", "none"]

DISPLAY_SYNONYMS = {
    "mirror_v":  ["vertical mirror symmetry", "vertical line symmetry", "reflection across a vertical line"],
    "mirror_h":  ["horizontal mirror symmetry", "horizontal line symmetry", "reflection across a horizontal line"],
    "diag_main": ["main-diagonal symmetry", "main-diagonal (↘︎) mirror", "line symmetry along the main diagonal"],
    "diag_anti": ["anti-diagonal symmetry", "anti-diagonal (↙︎) mirror", "line symmetry along the anti-diagonal"],
    "mirror_hv": ["vertical + horizontal symmetry", "two-axis mirror symmetry (V+H)", "both vertical and horizontal"],
    "none":      ["no mirror symmetry", "no line symmetry", "no reflection symmetry"],
}


PROMPT_TEMPLATES = [
    "Which type of mirror (line) symmetry does the figure have? Choose one: {choices}",
    "Identify the figure’s mirror symmetry. Select one: {choices}",
    "Pick the best description of the figure’s line symmetry. Choose one: {choices}",
    "Exactly one option describes the figure’s mirror symmetry. Which is it? {choices}",
    "Determine the figure’s line symmetry. Options: {choices}",
    "Select the correct mirror-symmetry category for the figure: {choices}",
    "Which option correctly names the figure’s mirror symmetry? Pick one: {choices}",
    "What is the figure’s mirror-symmetry class? Select exactly one: {choices}",
    "Which mirror-symmetry type best matches the figure? Select one: {choices}",
    "Choose the correct mirror-symmetry category for the figure: {choices}",
]


MOTIF_WEIGHTS = {
    "icons": 10,
    "arc": 1.25,
    "single_arrow": 0.5,
    "clock": 0.25,
    "crescent": 0.75,
    "fractal": 0.25,
    "glyph": 1.5,
    "keyhole": 0.25,
    "pinwheel_triangles": 0.25,
    "polygon": 1.0,
    "polyhex": 0.125,
    "polyiamond": 0.125,
    "polyline": 0.25,
    "polyomino": 0.125,
    "rings": 0.25,
    "segment": 1.0,
    "star_polygon": 0.125,
    "pictogram": 0.75,
}


# ------------ geometry / utility helpers ------------
def _rects_intersect(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int], pad: int = 0) -> bool:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or
                ay + ah + pad <= by or by + bh + pad <= ay)

def _pairwise_no_overlap(rects: List[Tuple[int,int,int,int]], pad: int) -> bool:
    n = len(rects)
    for i in range(n):
        for j in range(i+1, n):
            if _rects_intersect(rects[i], rects[j], pad):
                return False
    return True

def _mirror_rect(r: Tuple[int,int,int,int], key: str, W: int, H: int) -> List[Tuple[int,int,int,int]]:
    """Return the rectangles produced by mirroring 'r' under the key (excluding r itself)."""
    x, y, w, h = r
    out: List[Tuple[int,int,int,int]] = []
    if key == "mirror_v":
        out.append((W - x - w, y, w, h))
    elif key == "mirror_h":
        out.append((x, H - y - h, w, h))
    elif key == "mirror_hv":
        out.append((W - x - w, y, w, h))           # vertical
        out.append((x, H - y - h, w, h))           # horizontal
        out.append((W - x - w, H - y - h, w, h))   # both (rot180)
    elif key == "diag_main":
        # transpose: (x,y,w,h) -> (y,x,h,w)
        out.append((y, x, h, w))
    elif key == "diag_anti":
        # anti-transpose: (x,y,w,h) -> (H - (y+h), W - (x+w), h, w)
        out.append((H - (y + h), W - (x + w), h, w))
    return out

def _scale_bounds_for_count(final_count: int) -> Tuple[float, float]:
    """
    Target *shape width* as a fraction of CANVAS width (after tight-crop).
    """
    if final_count <= 3:
        return (0.20, 0.60)   # big
    elif final_count <= 5:
        return (0.15, 0.45)   # medium
    else:  # 6..8
        return (0.10, 0.30)   # smaller but clear

def _canvas_size_for_count(final_count: int, base: int = SS_CELL) -> int:
    """
    Base size for <=4. For larger totals, scale linear dimension so
    area ≈ (final_count / 3) times base area (looser than /4).
    Cap at 2.2× to avoid huge images.
    """
    if final_count <= 4:
        return int(base)
    from math import sqrt
    f = sqrt(final_count / 3.0)
    f = min(2.2, max(1.0, f))
    return int(round(base * f))

def _sample_top_left_triangle_center(rng: random.Random, W: int, H: int,
                                     w: int, h: int, inner: int, pad: int) -> Optional[Tuple[int,int]]:
    """
    Sample a center in the upper-left triangle y <= x - pad and ensure
    the entire axis-aligned rectangle stays in that triangle:
      max(y - x) over rect = (y1 - x0) <= -pad  =>  y1 <= x0 - pad
    """
    xmin = inner + w//2; xmax = W - inner - w//2
    ymin = inner + h//2; ymax = H - inner - h//2
    if xmin >= xmax or ymin >= ymax:
        return None
    for _ in range(100):
        cx = rng.randint(xmin, xmax)
        cy = rng.randint(ymin, ymax)
        x0 = cx - w//2; y1 = cy + h//2
        if y1 <= x0 - pad:
            return (cx, cy)
    return None

def _sample_top_right_triangle_center(rng: random.Random, W: int, H: int,
                                      w: int, h: int, inner: int, pad: int) -> Optional[Tuple[int,int]]:
    """
    Sample a center in the upper-right triangle y + x <= H - pad and ensure
    the entire rectangle stays inside:
      max(y + x) over rect = (y1 + x1) <= H - pad
    """
    xmin = inner + w//2; xmax = W - inner - w//2
    ymin = inner + h//2; ymax = H - inner - h//2
    if xmin >= xmax or ymin >= ymax:
        return None
    for _ in range(100):
        cx = rng.randint(xmin, xmax)
        cy = rng.randint(ymin, ymax)
        x1 = cx + w//2; y1 = cy + h//2
        if y1 + x1 <= H - pad:
            return (cx, cy)
    return None

@register_task
class SymmetrySceneMirrorIdentifyTask(Task):
    name = "symmetry_scene_mirror_identify"

    def __init__(self):
        self.tol = float(IMG_EQUAL_TOL)
        self.max_retries = int(MAX_BUILD_RETRIES)

    # ----- symmetry checks (color-aware) -----
    def _holds_basic(self, img):
        return {
            "mirror_v":  strong_same(img, apply_tf(img, "mirror"),     self.tol, EQUAL_HASH_MAX_BITS),
            "mirror_h":  strong_same(img, apply_tf(img, "flip"),       self.tol, EQUAL_HASH_MAX_BITS),
            "diag_main": strong_same(img, apply_tf(img, "transpose"),  self.tol, EQUAL_HASH_MAX_BITS),
            "diag_anti": strong_same(img, apply_tf(img, "transverse"), self.tol, EQUAL_HASH_MAX_BITS),
        }

    def _holds_category(self, img: Image.Image) -> Dict[str, bool]:
        b = self._holds_basic(img)
        hv = b["mirror_v"] and b["mirror_h"]
        return {
            "mirror_v":  b["mirror_v"] and not b["mirror_h"],
            "mirror_h":  b["mirror_h"] and not b["mirror_v"],
            "diag_main": b["diag_main"] and not (b["diag_anti"] or hv or b["mirror_v"] or b["mirror_h"]),
            "diag_anti": b["diag_anti"] and not (b["diag_main"] or hv or b["mirror_v"] or b["mirror_h"]),
            "mirror_hv": hv,
            "none":      not (b["mirror_v"] or b["mirror_h"] or b["diag_main"] or b["diag_anti"]),
        }

    def _layout_seed_layer(
            self, motif_impls: Dict[str, Any], rng: random.Random, key: str,
            seeds_needed: int, final_target: int, canvas_px: int
    ) -> Optional[Tuple[Image.Image, str, int, List[Dict[str, Any]]]]:
        kinds = [k for k in motif_impls.keys() if k in MOTIF_WEIGHTS]
        weights = [MOTIF_WEIGHTS[k] for k in kinds]
        if not kinds:
            return None
        mk = choice_weighted(rng, kinds, weights)
        motif = motif_impls[mk]

        W = H = int(canvas_px)
        inner_margin = int(0.05 * W)
        min_gap = max(8, W // 70)

        scale_min, scale_max = _scale_bounds_for_count(final_target)

        placed: List[Tuple[int, int, int, int, Image.Image, Any]] = []
        tries = 0
        MAX_TRIES = 1200
        shrink = 1.0  # adaptive shrink if we struggle

        cx, cy = W // 2, H // 2

        while len(placed) < seeds_needed and tries < MAX_TRIES:
            tries += 1

            # render + crop to painted area
            spec = _prefer_asym_mode(motif, motif.sample_spec(rng))
            tile = tight_crop_rgba(motif.render(spec))

            # scale relative to CANVAS size (shrink if needed)
            scale = rng.uniform(scale_min, scale_max) * shrink
            w = max(8, int(W * scale))
            h = max(8, int(tile.height * (w / max(1, tile.width))))
            tile = tile.resize((w, h), Image.LANCZOS)

            # sample a center obeying the fundamental region
            cxmin = inner_margin + w // 2;
            cxmax = W - inner_margin - w // 2
            cymin = inner_margin + h // 2;
            cymax = H - inner_margin - h // 2
            if key == "mirror_v":
                cxmax = min(cxmax, cx - min_gap - w // 2)
            elif key == "mirror_h":
                cymax = min(cymax, cy - min_gap - h // 2)
            elif key == "mirror_hv":
                cxmax = min(cxmax, cx - min_gap - w // 2)
                cymax = min(cymax, cy - min_gap - h // 2)

            if key in ("diag_main", "diag_anti"):
                if key == "diag_main":
                    c = _sample_top_left_triangle_center(rng, W, H, w, h, inner_margin, min_gap)
                else:
                    c = _sample_top_right_triangle_center(rng, W, H, w, h, inner_margin, min_gap)
                if not c:
                    if tries % 60 == 0: shrink *= 0.95
                    continue
                cx_seed, cy_seed = c
            else:
                if cxmin >= cxmax or cymin >= cymax:
                    if tries % 60 == 0: shrink *= 0.95
                    continue
                cx_seed = rng.randint(cxmin, cxmax)
                cy_seed = rng.randint(cymin, cymax)

            x = cx_seed - w // 2
            y = cy_seed - h // 2
            rect = (x, y, w, h)

            if any(_rects_intersect(rect, (px, py, pw, ph), min_gap) for (px, py, pw, ph, _t, _s) in placed):
                if tries % 60 == 0: shrink *= 0.95
                continue

            placed.append((x, y, w, h, tile, spec))

            # If still under target and it's been a while, shrink slightly to make room
            if tries % 150 == 0 and len(placed) < seeds_needed:
                shrink *= 0.95

        if len(placed) < seeds_needed:
            return None  # failed to place enough seeds cleanly

        # ---- FULL-SCENE overlap check ----
        base_rects = [(x, y, w, h) for (x, y, w, h, _t, _s) in placed]
        all_rects: List[Tuple[int, int, int, int]] = []
        for r in base_rects:
            all_rects.append(r)
            all_rects.extend(_mirror_rect(r, key, W, H))

        if not _pairwise_no_overlap(all_rects, min_gap):
            return None

        # Build seed layer image
        S = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        for (x, y, w, h, t, _s) in placed:
            paste_rgba(S, t, (x, y))

        mult = 1 if key == "none" else (4 if key == "mirror_hv" else 2)
        final_total = len(placed) * mult
        placements_meta = []
        for (x, y, w, h, _t, s) in placed:
            placements_meta.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "spec": s,
            })
        return S, mk, final_total, placements_meta

    def _compose_scene(self, seed: Image.Image, key: str) -> Image.Image:
        # White canvas + black border
        scene = Image.new("RGBA", seed.size, (255,255,255,255))
        draw = ImageDraw.Draw(scene)
        border_w = max(3, scene.width // 160)
        scene.alpha_composite(seed)
        if key == "mirror_v":
            scene.alpha_composite(apply_tf(seed, "mirror"))
        elif key == "mirror_h":
            scene.alpha_composite(apply_tf(seed, "flip"))
        elif key == "diag_main":
            scene.alpha_composite(apply_tf(seed, "transpose"))
        elif key == "diag_anti":
            scene.alpha_composite(apply_tf(seed, "transverse"))
        elif key == "mirror_hv":
            scene.alpha_composite(apply_tf(seed, "mirror"))
            scene.alpha_composite(apply_tf(seed, "flip"))
            scene.alpha_composite(apply_tf(seed, "rot180"))
        draw.rectangle([0,0,scene.width-1, scene.height-1], outline=(0,0,0), width=border_w)
        return scene

    def _format_question(self, rng: random.Random, labels: List[str], descs: List[str]) -> str:
        choices = ", ".join([f"({l}) {d}" for l, d in zip(labels, descs)])
        return rng.choice(PROMPT_TEMPLATES).format(choices=choices)

    # ----- public -----
    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        labels_plain = list("abcdef")
        labels = [f"({l})" for l in labels_plain]

        # Choose a symmetry ONCE and a valid final count ONCE; keep across motifs.
        key = rng.choice(OPT_KEYS)
        mult = 1 if key == "none" else (4 if key == "mirror_hv" else 2)
        valid_totals = [m for k in range(1, 8 // mult + 1) for m in [mult * k] if m >= 2]
        final_target = rng.choice(valid_totals)
        canvas_px = _canvas_size_for_count(final_target, SS_CELL)
        seeds_needed = final_target // mult

        # ---- Build weighted motif order (without replacement) ----
        weights_map = globals().get("MOTIF_WEIGHTS", {})  # fallback to uniform if missing
        kinds_all = [k for k in motif_impls.keys() if float(weights_map.get(k, 1.0)) > 0]
        if not kinds_all:
            raise RuntimeError(f"{self.name}: no allowed motifs available.")

        motif_order = weighted_order(rng, kinds_all, weights_map)

        # ---- Try each motif up to max_retries before moving on ----
        tried_failures: List[str] = []

        for mk in motif_order:
            motif_single = {mk: motif_impls[mk]}  # restrict candidate set to this motif

            for _ in range(self.max_retries):
                layout = self._layout_seed_layer(
                    motif_single, rng, key, seeds_needed, final_target, canvas_px
                )
                if layout is None:
                    continue

                seed, mk_found, final_total, placements = layout
                scene = self._compose_scene(seed, key)

                # verify uniqueness of answer
                cats = self._holds_category(scene)
                if sum(1 for v in cats.values() if v) != 1 or not cats.get(key, False):
                    continue

                # options in the question
                items = [(rng.choice(DISPLAY_SYNONYMS[k]), k) for k in OPT_KEYS]
                rng.shuffle(items)
                descs = [t for (t, _k) in items]
                keys = [_k for (t, _k) in items]
                correct_label = labels[keys.index(key)]
                question = self._format_question(rng, labels_plain, descs)

                meta = {
                    "pattern_kind": "symmetry",
                    "pattern": self.name,
                    "grid": (1, 1),
                    "mask_idx": -1,
                    "variant": key,
                    "grid_spatial_tfs": None,
                    "motif_kind": mk_found,
                    "labels": labels,
                    "answer": correct_label,
                    "option_strategy": "text_symmetry",
                    "option_descs": descs,
                    "option_payload": {
                        "option_keys": keys,
                        "final_total": final_total,
                        "canvas_px": canvas_px,
                    },
                    "question": question,
                    "composite_ready": True,
                }
                # Return RGB image to ensure a solid white background
                return scene.convert("RGB"), placements, meta

            # this motif failed all retries; keep note and continue
            tried_failures.append(mk)

        # ---- All motifs failed ----
        order_str = ", ".join(motif_order) if motif_order else "(none)"
        raise RuntimeError(
            f"{self.name}: failed to build a unique-answer sample after {self.max_retries} attempts per motif. "
            f"Motifs tried (in order): {order_str}"
        )
