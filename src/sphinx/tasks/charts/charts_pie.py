# sphinx/tasks/charts/charts_pie.py
from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Tuple

from sphinx.base import Task
from sphinx.registry import register_task
from sphinx.config import MAX_BUILD_RETRIES

from sphinx.charts import (
    ChartSpec,
    CHART_MIN_K, CHART_MAX_K,
    compute_chart_complexity,
    sample_category_labels,
    choose_colors,
    render_pie_chart,
    sample_percentages_int,   # exported by charts __init__
)

# ---------------------------------------------------------------------
# Require at least this RELATIVE difference between ANY two slices.
# Example: 0.9 means |pi - pj| / max(pi, pj) >= 0.09 for all i != j.
MIN_RELATIVE_GAP = 0.09
# ---------------------------------------------------------------------

# ---------------------------- MCQ prompts --------------------------------
# Descending (largest → smallest)
PROMPTS_MCQ_DESC = [
    "Which option (a)–(d) orders the categories by pie-slice size from largest to smallest?",
    "Select the option (a)–(d) that lists categories in decreasing share (largest → smallest).",
    "Pick the ordering (a)–(d) with categories ranked from greatest slice to least.",
    "Choose the descending ordering of categories by slice size (a)–(d).",
    "Which option lists the categories by percentage in descending order?",
    "Among the options (a)–(d), which shows the categories from largest slice down to smallest?",
    "Select the option where categories are arranged in decreasing pie-slice size (a)–(d).",
    "Which option (a)–(d) displays the correct largest-to-smallest order of slices?",
    "Pick the descending sequence of categories by share of the pie (a)–(d).",
    "Choose the option that orders categories from greatest proportion to least (a)–(d).",
]

# Ascending (smallest → largest)
PROMPTS_MCQ_ASC = [
    "Which option (a)–(d) orders the categories by pie-slice size from smallest to largest?",
    "Select the option (a)–(d) that lists categories in increasing share (smallest → largest).",
    "Pick the ordering (a)–(d) with categories ranked from least slice to greatest.",
    "Choose the ascending ordering of categories by slice size (a)–(d).",
    "Which option lists the categories by percentage in ascending order?",
    "Among the options (a)–(d), which shows the categories from smallest slice up to largest?",
    "Select the option where categories are arranged in increasing pie-slice size (a)–(d).",
    "Which option (a)–(d) displays the correct smallest-to-largest order of slices?",
    "Pick the ascending sequence of categories by share of the pie (a)–(d).",
    "Choose the option that orders categories from least proportion to greatest (a)–(d).",
]


# ---------------------------- helpers -----------------------------------

def _valid_relative_gap(perc: List[int], rel_gap: float) -> bool:
    """Check |pi - pj| / max(pi, pj) >= rel_gap for all pairs; also ensure distinct & min 1."""
    if len(set(perc)) != len(perc):
        return False
    if any(p < 1 for p in perc):
        return False
    n = len(perc)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = perc[i], perc[j]
            m = max(a, b)
            if m == 0:
                return False
            if abs(a - b) / float(m) < rel_gap:
                return False
    return True


def _sample_percentages_with_relative_gap(rng: random.Random, k: int, rel_gap: float,
                                          max_tries: int = 5000) -> List[int]:
    """
    Sample k integer percentages that sum to 100, are all distinct (>=1),
    and satisfy the relative-gap constraint. Resamples until valid.
    """
    for _ in range(max_tries):
        p = sample_percentages_int(rng, k, enforce_min1=True)
        if _valid_relative_gap(p, rel_gap):
            return p

    # Fallback: construct a geometric-like profile then normalize via largest remainder.
    r = 1.0 - rel_gap
    base = [r ** (k - 1 - i) for i in range(k)]  # descending
    # small jitter to break ties after rounding
    jitter = [1.0 + rng.uniform(-0.04, 0.04) for _ in range(k)]
    w = [b * j for b, j in zip(base, jitter)]
    tot = sum(w) or 1.0
    exact = [100.0 * (x / tot) for x in w]
    floors = [int(x) for x in exact]
    need = 100 - sum(floors)
    rems = [e - f for e, f in zip(exact, floors)]
    order = sorted(range(k), key=lambda i: (-rems[i], i))
    for i in range(need):
        floors[order[i % k]] += 1
    # ensure min-1
    for i in range(k):
        if floors[i] < 1:
            j = max(range(k), key=lambda t: floors[t])
            floors[i] += 1; floors[j] -= 1

    # try a few nudges to reach distinctness + relative condition
    for _ in range(1000):
        if _valid_relative_gap(floors, rel_gap):
            return floors
        # nudge: move 1 from current max to current min if it improves separation
        i_max = max(range(k), key=lambda i: floors[i])
        i_min = min(range(k), key=lambda i: floors[i])
        if floors[i_max] > 1:
            floors[i_max] -= 1
            floors[i_min] += 1
        else:
            break

    # If still not valid, give up (caller will retry with a new RNG seed)
    raise RuntimeError("Failed to sample percentages meeting relative gap")


def _counts_from_percentages(rng: random.Random, perc: List[int],
                             total_min: int = 40, total_max: int = 200) -> List[int]:
    """Given integer percentages, generate integer counts that sum to a random total."""
    T = rng.randint(total_min, total_max)
    exact = [T * (p / 100.0) for p in perc]
    floors = [int(e) for e in exact]
    need = T - sum(floors)
    rem = [e - f for e, f in zip(exact, floors)]
    order = sorted(range(len(perc)), key=lambda i: (-rem[i], i))
    for i in range(need):
        floors[order[i % len(perc)]] += 1
    for i in range(len(floors)):
        if floors[i] < 1:
            j = max(range(len(floors)), key=lambda t: floors[t])
            floors[i] += 1
            floors[j] -= 1
    return floors


def _rank_answer(labels: List[str], perc: List[int], direction: str) -> str:
    if direction == "desc":
        order = sorted(range(len(labels)), key=lambda i: (-perc[i], labels[i]))
    else:
        order = sorted(range(len(labels)), key=lambda i: (perc[i], labels[i]))
    return ",".join(labels[i] for i in order)


def _build_mcq_options(rng: random.Random, correct: str, k: int) -> Tuple[List[str], int]:
    """
    Build 4 unique options: correct ordering and 3 distractors formed by swapping
    two positions in the correct list. Returns (options_list, correct_index).
    """
    correct_list = correct.split(",")
    seen = {correct}
    opts = [correct]
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
    # fallback to random shuffles if needed
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


# ---------------------------- task class --------------------------------

@register_task
class ChartsPieTask(Task):
    """
    Pie chart with K categories.
    MCQ sorting questions (asc/desc, 50/50). Numbers are HIDDEN on the chart.
    Guarantees: all slice percentages are DISTINCT and satisfy the RELATIVE gap:
      |pi - pj| / max(pi, pj) >= MIN_RELATIVE_GAP  for all i != j.
    Returns: (image, [spec], meta).
    """
    name = "charts_pie"

    def __init__(self):
        self.max_retries = int(MAX_BUILD_RETRIES)

    def _sample_spec(self, rng: random.Random) -> ChartSpec:
        seed = rng.randint(0, 2**31 - 1)
        lrng = random.Random(seed)

        # Choose k freely in the configured range (no absolute-gap feasibility cap).
        k = lrng.randint(CHART_MIN_K, CHART_MAX_K)

        labels = sample_category_labels(lrng, k)

        # DISTINCT percentages with RELATIVE separation
        # Resample internally until valid; may raise and be retried by caller.
        perc = _sample_percentages_with_relative_gap(lrng, k, MIN_RELATIVE_GAP)

        # Keep counts for provenance; percentages are the ground truth
        if lrng.random() < 0.5:
            counts = _counts_from_percentages(lrng, perc, total_min=max(40, k), total_max=200)
            value_kind = "count"
        else:
            counts = perc[:]  # mirror for transparency
            value_kind = "percentage"

        cols, color_mode = choose_colors(lrng, k, mode="distinct")

        return ChartSpec(
            seed=seed,
            chart_type="pie",
            labels=labels,
            value_kind=value_kind,
            counts=counts,
            percentages_int=perc,   # DISTINCT + relative separation
            colors=cols,
            color_mode=color_mode,
            width_px=1024,
            height_px=768,
            render_mode="color",
        )

    def generate_instance(self, motif_impls: Dict[str, Any], rng: random.Random):
        """
        Build a pie chart and ask MCQ sorting (asc/desc with 50-50 probability).
        Numbers are hidden on the chart (legend-only; renderer shows legend on right).
        """
        for _ in range(self.max_retries):
            try:
                spec = self._sample_spec(rng)
            except RuntimeError:
                # Try again with a fresh seed if the internal sampler gave up
                continue

            labels = spec.labels
            perc = spec.percentages_int

            # 50/50 asc vs desc
            direction = "desc" if rng.random() < 0.5 else "asc"
            prompt_lead = random.choice(PROMPTS_MCQ_DESC if direction == "desc" else PROMPTS_MCQ_ASC)

            # Correct ordering and MCQ options
            correct_order = _rank_answer(labels, perc, direction)
            options, correct_idx = _build_mcq_options(rng, correct_order, len(labels))
            opt_labels = ["a", "b", "c", "d"]

            # Compose final MCQ prompt (inline options)
            lines = [prompt_lead, ""]
            for i, s in enumerate(options):
                lines.append(f"({opt_labels[i]}) {s}")
            lines.append("")
            lines.append("Answer with one letter (a–d).")
            prompt = "\n".join(lines)

            variant = {
                "kind": "sort_mcq",
                "direction": direction,
                "tie_break": "lexicographic",
                "numbers_shown": False,
                "legend_always_right": True,
                "min_relative_gap": MIN_RELATIVE_GAP,
                "options_strategy": "swap_two_positions_from_correct",
            }

            # Hide all in-pie text; legend (right) handles mapping
            image = render_pie_chart(spec, show_values=False)

            complexity = compute_chart_complexity(len(labels))
            range_info = complexity.get("range")
            chart_complexity = {
                "score": complexity["score"],
                "level": complexity["level"],
            }
            if isinstance(range_info, dict):
                chart_complexity["range"] = dict(range_info)

            chart_info = {
                "type": spec.chart_type,
                "k": len(labels),
                "labels": labels,
                "value_kind": spec.value_kind,
                "percentages": {lab: int(p) for lab, p in zip(labels, perc)},
                "color_mode": spec.color_mode,
                "numbers_shown": False,
                "complexity": chart_complexity,
            }

            meta = {
                "pattern_kind": "charts",
                "pattern": self.name,
                "variant": variant,
                "question": prompt,
                "mcq": True,
                "options": {opt_labels[i]: options[i] for i in range(4)},
                "answer": opt_labels[correct_idx],  # 'a' | 'b' | 'c' | 'd'
                "chart": chart_info,
                "dims": (spec.width_px, spec.height_px),
                "composite_ready": True,
                "correct_ordering": correct_order,
            }
            meta["complexity"] = complexity
            meta["complexity_score"] = complexity["score"]
            meta["complexity_level"] = complexity["level"]
            meta["complexity_version"] = complexity["version"]
            return image, [spec], meta

        raise RuntimeError(f"{self.name}: failed to build a valid sample after {self.max_retries} attempts.")
