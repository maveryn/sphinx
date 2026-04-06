# sphinx/tilings/__init__.py
from ..base import Tiling, Vertex, Edge, Cell, TilingPatch
from ..schema import TilingSpec
from ..registry import register_tiling, create_tiling, get_tiling_names
from .coloring import Colorer
from .graph import build_dual_graph, largest_color_component, reachable

# Three regular tilings
from .regular import TriangularTiling, SquareTiling, HexagonalTiling

# Irregular / non-Archimedean additions
from .irregular import CirclePackingTiling, RhombilleTiling, VoronoiTiling, OrthogonalSplitTiling

__all__ = [
    "Tiling", "Vertex", "Edge", "Cell", "TilingPatch",
    "TilingSpec",
    "register_tiling", "create_tiling", "get_tiling_names",
    "Colorer",
    "build_dual_graph", "largest_color_component", "reachable",
    "TriangularTiling", "SquareTiling", "HexagonalTiling",
    "CirclePackingTiling", "RhombilleTiling", "VoronoiTiling", "OrthogonalSplitTiling",
]
