# sphinx/base.py
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw

from .schema import MotifSpec, TilingSpec
from .utils.geom import clamp
from .config import SUPERSAMPLE, SS_CELL, COLORS_NAMES

# Derive palette list from the color-name mapping to avoid duplicated sources.
COLORS = list(COLORS_NAMES)
from .utils.drawing import _down_on_background

class Motif:
    """Renderable motif (visual primitive)."""
    name = "base"
    attr_ranges: Dict[str, Tuple[float, float]] = {}

    def __init__(self, split: str = "train"):
        """Initialize the motif with a data split.

        Parameters
        ----------
        split: str
            Name of the data split to use. Currently either ``"train"`` or
            ``"test"``.  Sub-classes may use this to specialise their behaviour
            depending on the split, but all motifs default to the ``"train"``
            split to preserve existing behaviour.
        """
        self.split = split

    def sample_spec(self, rng: random.Random) -> MotifSpec:
        raise NotImplementedError

    def render(self, spec: MotifSpec) -> Image.Image:
        raise NotImplementedError

    def clamp_spec(self, spec: MotifSpec) -> MotifSpec:
        for attr, (lo, hi) in self.attr_ranges.items():
            if hasattr(spec, attr):
                val = getattr(spec, attr)
                if isinstance(val, int): val = int(clamp(val, lo, hi))
                else: val = clamp(val, lo, hi)
                spec = spec.clone(**{attr: val})
        return spec

class Task:
    """High-level task that generates its own composite image + metadata + question."""
    name = "task_base"

    def generate_instance(
        self, motif_impls: Dict[str, Motif], rng: random.Random
    ) -> Tuple[Image.Image, List[MotifSpec], dict]:
        """
        Return (composite_image, cell_specs, meta_dict).
        See task implementations for the 'meta_dict' schema.
        """
        raise NotImplementedError

@dataclass
class Vertex:
    id: int
    xy: Tuple[float, float]


@dataclass
class Edge:
    id: int
    a: int
    b: int


@dataclass
class Cell:
    id: int
    verts: List[int]
    kind: str
    coord: Tuple[int, int]
    color: Optional[str] = None
    poly_coords: Optional[List[Tuple[float, float]]] = None


@dataclass
class TilingPatch:
    vertices: List[Vertex]
    edges: List[Edge]
    cells: List[Cell]
    vid_to_idx: Dict[int, int]
    eid_to_idx: Dict[int, int]
    cid_to_idx: Dict[int, int]

    def cell_polygons(self) -> List[List[Tuple[float, float]]]:
        vs = self.vertices
        return [[vs[v].xy for v in c.verts] for c in self.cells]


class Tiling:
    name = "base"
    supports_wythoffian = False

    def sample_spec(self, rng: random.Random) -> TilingSpec:
        raise NotImplementedError

    def clamp_spec(self, spec: TilingSpec) -> TilingSpec:
        w = max(1, int(getattr(spec, "width", 64)))
        h = max(1, int(getattr(spec, "height", 64)))
        margin_frac = float(getattr(spec, "margin_frac", 0.08))
        mode = getattr(spec, "color_mode", "uniform")
        return spec.clone(width=w, height=h, margin_frac=margin_frac, color_mode=mode)

    def generate(self, spec: TilingSpec) -> TilingPatch:
        """Return geometry (without colors)."""
        raise NotImplementedError

    def apply_coloring(self, patch: TilingPatch, spec: TilingSpec):
        from .tilings.coloring import Colorer
        Colorer().apply(self, patch, spec)

    def wythoffian_class_id(self, cell: Cell) -> int:
        return 0

    def render(self, spec: TilingSpec, outline: bool = True) -> Image.Image:
        s = self.clamp_spec(spec)
        patch = self.generate(s)

        polys = patch.cell_polygons()
        for c, poly in zip(patch.cells, polys):
            c.poly_coords = poly

        self.apply_coloring(patch, s)

        xs = [v.xy[0] for v in patch.vertices]
        ys = [v.xy[1] for v in patch.vertices]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        inner = SS_CELL * (1.0 - 2.0 * s.margin_frac)
        gw = max(1e-9, maxx - minx)
        gh = max(1e-9, maxy - miny)
        scale = min(inner / gw, inner / gh)
        ox = (SS_CELL - scale * gw) * 0.5 - scale * minx
        oy = (SS_CELL - scale * gh) * 0.5 - scale * miny

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        ow = max(1, 1 * SUPERSAMPLE)

        for c in patch.cells:
            poly = [
                (ox + scale * patch.vertices[v].xy[0],
                 oy + scale * patch.vertices[v].xy[1]) for v in c.verts
            ]
            col = c.color if (c.color is not None) else COLORS[0]
            if outline:
                d.polygon(poly, fill=col, outline="black", width=ow)
            else:
                d.polygon(poly, fill=col)

        return _down_on_background(img)
