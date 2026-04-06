# sphinx/geometry/__init__.py
from .common import (
    MIN_RELATIVE_GAP,
    graph_paper_rgb,
    sample_labels,
    sample_values_with_relative_gap,
    ellipse_circumference_ramanujan_k,
    render_line_patch,
    render_angle_patch,
    render_regular_polygon_patch,
    render_ellipse_patch,
    RenderedShape,
)

from .shapes import (
    LineSegment,
    AngleShape,
    RegularPolygonShape,
    EllipseShape,
)

__all__ = [
    # sampling / drawing utils
    "MIN_RELATIVE_GAP",
    "graph_paper_rgb",
    "sample_labels",
    "sample_values_with_relative_gap",
    "ellipse_circumference_ramanujan_k",
    "render_line_patch",
    "render_angle_patch",
    "render_regular_polygon_patch",
    "render_ellipse_patch",
    "RenderedShape",
    # reusable shape classes
    "LineSegment",
    "AngleShape",
    "RegularPolygonShape",
    "EllipseShape",
]
