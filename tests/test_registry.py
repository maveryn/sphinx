from pathlib import Path

import yaml

from sphinx.registry import build_task_registry


EXPECTED_TASKS = {
    "charts_match_proportions",
    "charts_pie",
    "geometric_position",
    "geometric_sort",
    "geometric_stack_count",
    "rect_venn",
    "sequence_arithmetic",
    "sequence_multi_column_arithmetic",
    "sequence_rotation",
    "shape_count",
    "symmetry_frieze_groups",
    "symmetry_grid_mirror_fill",
    "symmetry_scene_mirror_identify",
    "symmetry_wallpaper_groups",
    "tiles_connected_component",
    "tiles_decompose_compose",
    "tiles_geometry",
    "tiles_line_intersections",
    "tiles_line_length",
    "tiles_missing_tiles",
    "tiles_recoloring",
    "tiles_shortest_path",
    "transform_pair_infer",
    "transform_result_identify",
    "transform_similarity_identify",
}


def test_task_registry_matches_sphinx_surface():
    task_names = set(build_task_registry().keys())
    assert task_names == EXPECTED_TASKS


def test_icon_manifest_files_exist():
    root = Path(__file__).resolve().parents[1] / "src" / "sphinx" / "icons"
    manifest = yaml.safe_load((root / "available-icons.yaml").read_text())
    names = set(manifest["train"]) | set(manifest["test"])
    for name in names:
        assert (root / "svgs" / f"light-{name}.svg").exists()
