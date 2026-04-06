import random

from sphinx.registry import build_motif_registry, build_task_registry


SMOKE_TASKS = [
    "charts_match_proportions",
    "geometric_sort",
    "sequence_multi_column_arithmetic",
    "shape_count",
    "symmetry_grid_mirror_fill",
    "tiles_missing_tiles",
    "transform_result_identify",
]


def test_representative_tasks_generate_one_example():
    motifs = build_motif_registry()
    tasks = build_task_registry()

    for idx, task_name in enumerate(SMOKE_TASKS):
        task = tasks[task_name]
        rng = random.Random(42 + idx)
        image, _cell_specs, meta = task.generate_instance(motifs, rng)
        assert image is not None, task_name
        assert image.width > 0 and image.height > 0, task_name
        assert meta.get("question"), task_name
        assert "answer" in meta, task_name
