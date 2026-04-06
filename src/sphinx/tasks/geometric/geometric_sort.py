# sphinx/tasks/geometric/geometric_sort.py
from __future__ import annotations
import math, random
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

from PIL import Image, ImageDraw

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES

# Weighted choice helper (pattern used in tiles_missing_tiles)
try:
    from sphinx.utils.rng import choice_weighted  # preferred
except Exception:
    choice_weighted = None  # fallback

# compose helper (does correct alpha blending to avoid fringe/halo)
try:
    from sphinx.utils.drawing import paste_rgba  # type: ignore
except Exception:
    paste_rgba = None

from sphinx.charts import choose_colors  # colored edges like charts palette
from sphinx.geometry import (
    MIN_RELATIVE_GAP,
    graph_paper_rgb,
    sample_labels,
    sample_values_with_relative_gap,
    LineSegment,
    AngleShape,
    RegularPolygonShape,
    EllipseShape,
)

# ----------------------------- config -----------------------------
K_MIN_DEFAULT = 3
K_MAX_DEFAULT = 10

# Enforce at least this relative gap for AREA/ANGLE values (pairwise).
# This is applied on top of MIN_RELATIVE_GAP by taking max(...).
AREA_ANGLE_MIN_REL_GAP_DEFAULT = 0.1

SHAPE_FAMILY_WEIGHTS_DEFAULT: Dict[str, float] = {
    "line": 0.0,
    "angle": 1.0,
    "polygon": 3.0,
    "ellipse": 1.0,
}

PROP_WEIGHTS_DEFAULT: Dict[str, Dict[str, float]] = {
    "polygon": {"area": 1.0},
    "ellipse": {"area": 1.0},
}

# Wider ranges (so 10% gaps are easy)
LENGTH_LOW_FRAC      = 0.35
ANGLE_DEG_RANGE      = (15.0, 170.0)
POLY_AREA_LOW_FRAC   = 0.15
POLY_PERIM_LOW_FRAC  = 0.25
ELL_AREA_LOW_FRAC    = 0.15
ELL_PERIM_LOW_FRAC   = 0.25
ELL_RATIO_RANGE      = (0.60, 0.90)

# Random-pack layout
MIN_SEP_PX_DEFAULT   = 2    # small but positive
PACK_TRIES_PER_SHAPE = 600
GLOBAL_SHRINK_STEPS  = 10
GLOBAL_SHRINK_FACTOR = 0.92
CANVAS_PAD_PX        = 16   # pad around tight bbox

# Composite AA (NEW): render whole scene at 2x and downsample
AA_COMPOSITE_SCALE_DEFAULT = 2
DOWNSAMPLE_FILTER = getattr(Image, "LANCZOS", Image.BICUBIC)

def _shape_px_max_for_k(k: int) -> int:
    # Make shapes large; the canvas grows to fit, so we can be generous.
    if k <= 5:  return 320
    if k <= 8:  return 250
    if k <= 10: return 210
    return 190

def _hex_to_rgba(h: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = h.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (r, g, b, alpha)

# ----------------------------- MCQ prompts -----------------------------
# MCQ-style prompts per property/direction. We inline (a)–(d) options below the question.
PROMPTS_MCQ_DESC = {
    "length": [
        "Which option (a)–(d) orders the line segments from longest to shortest?",
        "Select the option (a)–(d) listing segments in decreasing length (longest → shortest).",
        "Pick the ordering (a)–(d) with segments ranked from largest length to smallest.",
        "Choose the descending-length ordering of the segments (a)–(d).",
        "Which option lists the segments by length in descending order?",
        "Among the options (a)–(d), which shows the line segments from longest down to shortest?",
        "Choose the option that arranges the segments in decreasing length (a)–(d).",
        "Select the ordering where line segments go from greatest to least length (a)–(d).",
        "Which option (a)–(d) displays the correct longest-to-shortest sequence of segments?",
        "Pick the descending order of line segment lengths (a)–(d).",
    ],
    "angle_deg": [
        "Which option (a)–(d) orders the angles from largest to smallest?",
        "Select the descending angle-measure ordering (a)–(d).",
        "Pick the option that ranks angles from widest to narrowest (a)–(d).",
        "Choose the angles listed in decreasing degree measure (a)–(d).",
        "Which option lists the angles by size in descending order?",
        "Among the options, which lists the angles from biggest to smallest (a)–(d)?",
        "Select the ordering of angles going from largest opening to smallest (a)–(d).",
        "Which option arranges the angles from greatest degree to least (a)–(d)?",
        "Choose the descending sequence of angle measures (a)–(d).",
        "Pick the option (a)–(d) where angles go from widest down to narrowest.",
    ],
    "area": [
        "Which option (a)–(d) orders the shapes by area from largest to smallest?",
        "Select the descending-area ordering of the shapes (a)–(d).",
        "Pick the option that ranks shapes by area (largest → smallest).",
        "Choose the shapes listed in decreasing area (a)–(d).",
        "Which option lists shapes by area in descending order?",
        "Among the options, which shows the shapes from biggest area down to smallest (a)–(d)?",
        "Pick the sequence that arranges shapes by area from greatest to least (a)–(d).",
        "Which option gives the correct largest-to-smallest area ordering (a)–(d)?",
        "Choose the option (a)–(d) where shapes are ranked by area from largest to smallest.",
        "Select the descending order of shape areas (a)–(d).",
    ],
}

PROMPTS_MCQ_ASC = {
    "length": [
        "Which option (a)–(d) orders the line segments from shortest to longest?",
        "Select the option (a)–(d) listing segments in increasing length (shortest → longest).",
        "Pick the ordering with segments ranked from smallest length to largest (a)–(d).",
        "Choose the ascending-length ordering of the segments (a)–(d).",
        "Which option lists the segments by length in ascending order?",
        "Among the options (a)–(d), which shows the line segments from shortest up to longest?",
        "Select the option that arranges the segments in increasing length (a)–(d).",
        "Which ordering lists line segments from least to greatest length (a)–(d)?",
        "Pick the option (a)–(d) where segments go from shortest to longest in order.",
        "Choose the ascending sequence of line segment lengths (a)–(d).",
    ],
    "angle_deg": [
        "Which option (a)–(d) orders the angles from smallest to largest?",
        "Select the ascending angle-measure ordering (a)–(d).",
        "Pick the option that ranks angles from narrowest to widest (a)–(d).",
        "Choose the angles listed in increasing degree measure (a)–(d).",
        "Which option lists the angles by size in ascending order?",
        "Among the options, which lists the angles from smallest opening to largest (a)–(d)?",
        "Select the ordering of angles going from least degree to most (a)–(d).",
        "Which option arranges the angles from narrowest to widest (a)–(d)?",
        "Pick the ascending sequence of angle measures (a)–(d).",
        "Choose the option (a)–(d) where angles go from smallest to largest.",
    ],
    "area": [
        "Which option (a)–(d) orders the shapes by area from smallest to largest?",
        "Select the ascending-area ordering of the shapes (a)–(d).",
        "Pick the option that ranks shapes by area (smallest → largest).",
        "Choose the shapes listed in increasing area (a)–(d).",
        "Which option lists shapes by area in ascending order?",
        "Among the options, which shows the shapes from smallest area up to largest (a)–(d)?",
        "Pick the sequence that arranges shapes by area from least to greatest (a)–(d).",
        "Which option gives the correct smallest-to-largest area ordering (a)–(d)?",
        "Choose the option (a)–(d) where shapes are ranked by area from smallest to largest.",
        "Select the ascending order of shape areas (a)–(d).",
    ],
}


# ----------------------------- spec -----------------------------

@dataclass
class GeoSortSpec:
    seed: int
    canvas: Tuple[int, int]
    shape_type: str
    prop_kind: str
    k: int
    labels: List[str]
    k_min: int
    k_max: int
    shape_weights: Dict[str, float]
    prop_weights: Dict[str, Dict[str, float]]
    layout: str
    min_sep_px: int
    scale_factor: float
    polygon_n: Optional[int] = None
    ellipse_ratio: Optional[float] = None
    ellipse_rotation_deg: Optional[float] = None
    line_orientation_deg: Optional[float] = None
    angle_radius_px: Optional[int] = None

# ----------------------------- task -----------------------------

@register_task
class GeometricSortTask(Task):
    """
    Geometric sorting with random-pack layout and colored edges (no fill).
    Now MCQ: we present 4 text options (a–d) for the ordering.
    """
    name = "geometric_sort"

    def __init__(
        self,
        k_min: int = K_MIN_DEFAULT,
        k_max: int = K_MAX_DEFAULT,
        shape_weights: Optional[Dict[str, float]] = None,
        prop_weights: Optional[Dict[str, Dict[str, float]]] = None,
        min_sep_px: int = MIN_SEP_PX_DEFAULT,
        aa_composite_scale: int = AA_COMPOSITE_SCALE_DEFAULT,
        area_angle_min_relative_gap: float = AREA_ANGLE_MIN_REL_GAP_DEFAULT,  # NEW
    ):
        self.max_retries = int(MAX_BUILD_RETRIES)
        self.W, self.H = 1024, 768  # not used for tight canvas; kept for metadata
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        if self.k_min < 2 or self.k_max < self.k_min:
            raise ValueError("Invalid (k_min, k_max) in GeometricSortTask")

        self.shape_weights = dict(SHAPE_FAMILY_WEIGHTS_DEFAULT)
        if shape_weights:
            self.shape_weights.update({k: float(v) for k, v in shape_weights.items() if k in self.shape_weights})

        self.prop_weights = {
            "polygon": dict(PROP_WEIGHTS_DEFAULT["polygon"]),
            "ellipse": dict(PROP_WEIGHTS_DEFAULT["ellipse"]),
        }
        if prop_weights:
            for fam in ("polygon", "ellipse"):
                if fam in prop_weights:
                    self.prop_weights[fam].update({k: float(v) for k, v in prop_weights[fam].items()
                                                   if k in self.prop_weights[fam]})
        self.min_sep_px = int(min_sep_px)
        self.aa_composite_scale = max(1, int(aa_composite_scale))
        self.area_angle_min_relative_gap = float(area_angle_min_relative_gap)

    def _compute_complexity(self, k: int) -> Dict[str, Any]:
        """Normalize number of items to sort and derive a complexity label."""
        min_k = int(getattr(self, "k_min", K_MIN_DEFAULT))
        max_k = int(getattr(self, "k_max", K_MAX_DEFAULT))
        span = max(1, max_k - min_k)
        normalized = (int(k) - min_k) / span
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.75:
            level = "EASY"
        else:
            level = "HARD"

        return {
            "score": normalized,
            "level": level,
            "version": "geometric-sort-k-v1",
            "range": {"min_k": min_k, "max_k": max_k},
            "k": int(k),
        }

    # ------------------- weighted picks -------------------

    @staticmethod
    def _weighted_pick(rng: random.Random, items: List[str], weights: List[float]) -> str:
        if choice_weighted is not None:
            return choice_weighted(rng, items, weights)
        tot = sum(max(0.0, float(w)) for w in weights)
        if tot <= 0:
            return rng.choice(items)
        x = rng.random() * tot
        s = 0.0
        for it, w in zip(items, weights):
            s += max(0.0, float(w))
            if x <= s:
                return it
        return items[-1]

    def _choose_family(self, rng: random.Random) -> str:
        fams = list(self.shape_weights.keys())
        w = [self.shape_weights[f] for f in fams]
        return self._weighted_pick(rng, fams, w)

    def _choose_prop_for(self, rng: random.Random, family: str) -> str:
        if family == "polygon":
            keys = list(self.prop_weights["polygon"].keys())
            w = [self.prop_weights["polygon"][k] for k in keys]
            return self._weighted_pick(rng, keys, w)
        if family == "ellipse":
            keys = list(self.prop_weights["ellipse"].keys())
            w = [self.prop_weights["ellipse"][k] for k in keys]
            return self._weighted_pick(rng, keys, w)
        if family == "line": return "length"
        if family == "angle": return "angle_deg"
        raise ValueError(f"Unknown family '{family}'")

    # ------------------- family sampling (wide ranges) -------------------

    def _sample_family_shapes(self, rng: random.Random, k: int, family: str, prop_kind: str, px_max: int,
                              edge_colors_rgba: List[Tuple[int,int,int,int]]
                              ) -> Tuple[List[Any], Dict[str, Any]]:
        shapes: List[Any] = []
        extra: Dict[str, Any] = {}

        if family == "line":
            orientation = rng.uniform(10, 170)
            L_max = float(px_max)
            L_min = max(20.0, LENGTH_LOW_FRAC * L_max)
            vals = sample_values_with_relative_gap(rng, k, L_min, L_max, MIN_RELATIVE_GAP)
            for v, col in zip(vals, edge_colors_rgba):
                shapes.append(LineSegment(length_px=v, orientation_deg=orientation, edge_rgba=col))
            extra["line_orientation_deg"] = orientation

        elif family == "angle":
            R = max(40, int(0.50 * px_max))
            amin, amax = ANGLE_DEG_RANGE
            gap = max(MIN_RELATIVE_GAP, self.area_angle_min_relative_gap)
            vals = sample_values_with_relative_gap(rng, k, amin, amax, gap)
            for v, col in zip(vals, edge_colors_rgba):
                shapes.append(AngleShape(angle_deg=v, radius_px=R, bisector_deg=rng.uniform(0, 360), edge_rgba=col))
            extra["angle_radius_px"] = R

        elif family == "polygon":
            n = rng.randint(3, 8)
            R_max = 0.55 * px_max
            cA = 0.5 * n * math.sin(2 * math.pi / n)   # A = cA R^2
            A_max = cA * (R_max ** 2)
            A_min = max(40.0, POLY_AREA_LOW_FRAC * A_max)
            gap = max(MIN_RELATIVE_GAP, self.area_angle_min_relative_gap)
            vals = sample_values_with_relative_gap(rng, k, A_min, A_max, gap)
            R_vals = [math.sqrt(A / cA) for A in vals]
            for Rv, col in zip(R_vals, edge_colors_rgba):
                shapes.append(RegularPolygonShape(n_sides=n, R_px=Rv, rotation_deg=rng.uniform(0, 360),
                                                edge_rgba=col))
            extra["polygon_n"] = n

        else:  # ellipse
            e = rng.uniform(*ELL_RATIO_RANGE)
            rot = rng.uniform(0, 360)
            cr = math.radians(rot)
            def bbox_dims(a: float) -> Tuple[float, float]:
                b = e * a
                W = 2.0 * math.sqrt((a * math.cos(cr)) ** 2 + (b * math.sin(cr)) ** 2)
                H = 2.0 * math.sqrt((a * math.sin(cr)) ** 2 + (b * math.cos(cr)) ** 2)
                return W, H
            a_max = 0.65 * px_max
            for _ in range(40):
                Wb, Hb = bbox_dims(a_max)
                if Wb <= 0.98 * px_max and Hb <= 0.98 * px_max:
                    break
                a_max *= 0.92
            a_max = max(20.0, a_max)
            # area-only path
            A_max = math.pi * e * (a_max ** 2)
            A_min = max(30.0, ELL_AREA_LOW_FRAC * A_max)
            gap = max(MIN_RELATIVE_GAP, self.area_angle_min_relative_gap)
            vals = sample_values_with_relative_gap(rng, k, A_min, A_max, gap)
            a_vals = [math.sqrt(A / (math.pi * e)) for A in vals]
            for a, col in zip(a_vals, edge_colors_rgba):
                shapes.append(EllipseShape(a_px=a, b_px=e * a, rotation_deg=rot, edge_rgba=col))
            extra["ellipse_ratio"] = e
            extra["ellipse_rotation_deg"] = rot

        return shapes, extra

    # ------------------- packing -------------------

    @staticmethod
    def _pack_positions(rng: random.Random, patches: List[Tuple[int, int]], min_sep_px: int
                        ) -> Optional[List[Tuple[float, float]]]:
        n = len(patches)
        if n == 0:
            return []
        radii = [0.5 * (w * w + h * h) ** 0.5 for (w, h) in patches]
        centers: List[Tuple[float, float]] = [(0.0, 0.0)]
        for i in range(1, n):
            ri = radii[i]
            placed = False
            for _ in range(PACK_TRIES_PER_SHAPE):
                j = rng.randrange(i)
                rj = radii[j]
                cx, cy = centers[j]
                phi = rng.uniform(0, 2 * math.pi)
                base = ri + rj + min_sep_px
                jitter = 0.25 * min(ri, rj)
                d = base + rng.uniform(0.0, jitter)
                x = cx + d * math.cos(phi)
                y = cy + d * math.sin(phi)
                ok = True
                for m in range(i):
                    dx = x - centers[m][0]
                    dy = y - centers[m][1]
                    if (dx * dx + dy * dy) ** 0.5 < (ri + radii[m] + min_sep_px):
                        ok = False; break
                if ok:
                    centers.append((x, y)); placed = True; break
            if not placed:
                return None
        return centers

    def _compose_on_tight_canvas(self, rendered, centers, pad_px: int, aa_scale: int) -> Image.Image:
        """
        Compose patches onto a graph-paper RGBA background with a tight bounding box.
        Ensures final W,H are multiples of aa_scale to enable clean LANCZOS downsampling.
        """
        xs_min, ys_min, xs_max, ys_max = [], [], [], []
        for (rs, (cx, cy)) in zip(rendered, centers):
            w, h = rs.size
            xs_min.append(cx - w / 2); xs_max.append(cx + w / 2)
            ys_min.append(cy - h / 2); ys_max.append(cy + h / 2)
        min_x = min(xs_min); max_x = max(xs_max)
        min_y = min(ys_min); max_y = max(ys_max)

        W = int(math.ceil(max_x - min_x)) + 2 * pad_px
        H = int(math.ceil(max_y - min_y)) + 2 * pad_px
        W = max(W, 256 * aa_scale); H = max(H, 256 * aa_scale)

        # Round up to a multiple of aa_scale for clean downsample
        if aa_scale > 1:
            remW = W % aa_scale
            remH = H % aa_scale
            if remW: W += (aa_scale - remW)
            if remH: H += (aa_scale - remH)

        bg = graph_paper_rgb(W, H).convert("RGBA")
        for (rs, (cx, cy)) in zip(rendered, centers):
            w, h = rs.size
            x = int(round((cx - w / 2) - min_x + pad_px))
            y = int(round((cy - h / 2) - min_y + pad_px))
            if paste_rgba is not None:
                paste_rgba(bg, rs.patch, (x, y))
            else:
                bg.paste(rs.patch, (x, y), rs.patch)
        return bg

    # ------------------- ranking helpers -------------------

    @staticmethod
    def _rank_answer(labels: List[str], values: List[float], direction: str) -> str:
        if direction == "desc":
            order = sorted(range(len(labels)), key=lambda i: (-values[i], labels[i]))
        else:
            order = sorted(range(len(labels)), key=lambda i: (values[i], labels[i]))
        return ",".join(labels[i] for i in order)

    @staticmethod
    def _build_mcq_options(rng: random.Random, correct: str, k: int) -> Tuple[List[str], int]:
        """
        Build 4 unique options: correct ordering and 3 distractors formed by swapping
        two positions in the correct list.
        Returns (options_list, correct_index).
        """
        correct_list = correct.split(",")
        seen = {correct}
        opts = [correct]
        # keep trying swaps until we have 4 unique options or give up after some attempts
        attempts = 0
        while len(opts) < 4 and attempts < 50:
            attempts += 1
            i, j = rng.sample(range(k), 2)
            cand = correct_list[:]
            cand[i], cand[j] = cand[j], cand[i]
            s = ",".join(cand)
            if s not in seen:
                seen.add(s)
                opts.append(s)
        # ensure we have 4 (fallback: random shuffles)
        while len(opts) < 4:
            cand = correct_list[:]
            rng.shuffle(cand)
            s = ",".join(cand)
            if s not in seen:
                seen.add(s)
                opts.append(s)
        rng.shuffle(opts)
        correct_idx = opts.index(correct)
        return opts, correct_idx

    # ------------------- public API -------------------

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        for _ in range(self.max_retries):
            seed = rng.randint(0, 2**31 - 1)
            lrng = random.Random(seed)

            family = self._choose_family(lrng)
            prop_kind = self._choose_prop_for(lrng, family)
            k = lrng.randint(int(self.k_min), int(self.k_max))
            labels = sample_labels(lrng, k)

            # distinct edge colors (charts palette)
            hex_cols, _ = choose_colors(lrng, k)
            edge_cols = [_hex_to_rgba(c, 255) for c in hex_cols]

            px_max = _shape_px_max_for_k(k)
            base_shapes, extra = self._sample_family_shapes(lrng, k, family, prop_kind, px_max, edge_cols)

            # --- High-res composite path ------------------------------------
            aa = self.aa_composite_scale

            # First preview at AA scale (no labels) to get patch sizes & suggested font
            preview = [s.render(lab, prop_kind=prop_kind, draw_label=False)
                       for s, lab in zip(base_shapes, labels)]
            font_min = min(int(max(8, pr.meta.get("label_font_px_suggest", 12))) for pr in preview)

            # Try to pack; shrink if needed. All sizes/separations in AA space.
            scale = 1.0
            centers = None
            rendered = None
            for _sh in range(GLOBAL_SHRINK_STEPS):
                shapes_scaled = [s.scaled(scale * aa) for s in base_shapes]
                preview = [s.render(lab, prop_kind=prop_kind, draw_label=False)
                           for s, lab in zip(shapes_scaled, labels)]
                patches_wh = [rs.size for rs in preview]
                centers = self._pack_positions(lrng, patches_wh, self.min_sep_px * aa)
                if centers is not None:
                    # final render with UNIFORM font size (in AA space)
                    # Recompute min font suggestion in scaled space for safety
                    font_suggest = min(int(max(8, rs.meta.get("label_font_px_suggest", font_min))) for rs in preview)
                    font_px = max(8, font_suggest)
                    rendered = [s.render(lab, prop_kind=prop_kind, draw_label=True, label_font_px=font_px)
                                for s, lab in zip(shapes_scaled, labels)]
                    break
                scale *= GLOBAL_SHRINK_FACTOR
            if centers is None or rendered is None:
                continue

            # Compose in AA space, then downsample to final canvas
            pad_hi = CANVAS_PAD_PX * aa
            image_hi = self._compose_on_tight_canvas(rendered, centers, pad_px=pad_hi, aa_scale=aa)

            if aa > 1:
                W_final = image_hi.width // aa
                H_final = image_hi.height // aa
                image = image_hi.resize((W_final, H_final), resample=DOWNSAMPLE_FILTER)
            else:
                image = image_hi

            # --- MCQ answer construction ------------------------------------
            values = [rs.prop_value for rs in rendered]
            direction = "desc" if lrng.random() < 0.5 else "asc"
            correct_order = self._rank_answer(labels, values, direction)
            options, correct_idx = self._build_mcq_options(lrng, correct_order, k)

            # MCQ prompt
            if direction == "desc":
                prompt_lead = lrng.choice(PROMPTS_MCQ_DESC[prop_kind])
            else:
                prompt_lead = lrng.choice(PROMPTS_MCQ_ASC[prop_kind])

            # Inline options (a)–(d)
            opt_labels = ["a", "b", "c", "d"]
            lines = [prompt_lead, ""]
            for i, s in enumerate(options):
                lines.append(f"({opt_labels[i]}) {s}")
            lines.append("")
            lines.append("Answer with one letter (a–d).")
            prompt = "\n".join(lines)

            spec = GeoSortSpec(
                seed=seed,
                canvas=(image.width, image.height),
                shape_type=family,
                prop_kind=prop_kind,
                k=k,
                labels=labels,
                k_min=self.k_min,
                k_max=self.k_max,
                shape_weights=dict(self.shape_weights),
                prop_weights={k: dict(v) for k, v in self.prop_weights.items()},
                layout="random_pack",
                min_sep_px=int(self.min_sep_px),
                scale_factor=float(scale),  # effective geometry scale (pre-AA)
                polygon_n=extra.get("polygon_n"),
                ellipse_ratio=extra.get("ellipse_ratio"),
                ellipse_rotation_deg=extra.get("ellipse_rotation_deg"),
                line_orientation_deg=extra.get("line_orientation_deg"),
                angle_radius_px=extra.get("angle_radius_px"),
            )

            meta = {
                "pattern_kind": "geometry",
                "pattern": self.name,
                "variant": {
                    "family": family,
                    "direction": direction,
                    "prop_kind": prop_kind,
                    "min_relative_gap": MIN_RELATIVE_GAP,
                    "k_bounds": [self.k_min, self.k_max],
                    "shape_weights": dict(self.shape_weights),
                    "layout": "random_pack",
                    "min_sep_px": int(self.min_sep_px),
                    "scale_factor": float(scale),
                    "edge_colors": hex_cols,
                    "aa_composite_scale": int(aa),
                    "area_angle_min_relative_gap": float(self.area_angle_min_relative_gap)
                },
                # MCQ fields
                "question": prompt,
                "mcq": True,
                "options": {opt_labels[i]: options[i] for i in range(4)},
                "answer": opt_labels[correct_idx],  # 'a' | 'b' | 'c' | 'd'
                # additional provenance
                "k": k,
                "dims": (image.width, image.height),
                "letters": labels,
                "shapes": [
                    {"label": rs.label, "prop_kind": rs.prop_kind, "prop_value": float(rs.prop_value), "meta": rs.meta}
                    for rs in rendered
                ],
                "option_strategy": "swap_two_positions_from_correct",
                "correct_ordering": correct_order,
                "composite_ready": True,
            }
            complexity = self._compute_complexity(k)
            meta["complexity"] = complexity
            meta["complexity_score"] = complexity["score"]
            meta["complexity_level"] = complexity["level"]
            meta["complexity_version"] = complexity["version"]
            return image.convert("RGB"), [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
