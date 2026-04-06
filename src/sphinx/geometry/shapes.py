# sphinx/geometry/shapes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import math

from .common import (
    RenderedShape,
    render_line_patch,
    render_angle_patch,
    render_regular_polygon_patch,
    render_ellipse_patch,
    polygon_area_from_R,
    polygon_perimeter_from_R,
)

RGBA = Tuple[int, int, int, int]

# -------- Line --------

@dataclass
class LineSegment:
    length_px: float
    orientation_deg: float
    stroke_px: int = 4
    edge_rgba: RGBA = (0, 0, 0, 255)
    margin_px: int = 16
    label_offset_px: int = 10

    def family_name(self) -> str: return "line"
    def property_value(self, kind: str) -> float:
        if kind != "length": raise ValueError("LineSegment supports only 'length'")
        return float(self.length_px)

    def scaled(self, s: float) -> "LineSegment":
        return LineSegment(self.length_px * float(s), self.orientation_deg,
                           self.stroke_px, self.edge_rgba, self.margin_px, self.label_offset_px)

    def render(self, label: str, *, prop_kind: str = "length",
               draw_label: bool = True, label_font_px: Optional[int] = None) -> RenderedShape:
        rs = render_line_patch(self.length_px, self.orientation_deg, label,
                               stroke_px=self.stroke_px, stroke_rgba=self.edge_rgba,
                               margin_px=self.margin_px, label_offset_px=self.label_offset_px,
                               draw_label=draw_label, label_font_px=label_font_px)
        rs.prop_kind = prop_kind
        rs.prop_value = self.property_value(prop_kind)
        return rs

# -------- Angle --------

@dataclass
class AngleShape:
    angle_deg: float
    radius_px: float
    bisector_deg: Optional[float] = None
    stroke_px: int = 4
    edge_rgba: RGBA = (0, 0, 0, 255)
    margin_px: int = 18
    show_arc: bool = True

    def family_name(self) -> str: return "angle"
    def property_value(self, kind: str) -> float:
        if kind != "angle_deg": raise ValueError("AngleShape supports only 'angle_deg'")
        return float(self.angle_deg)

    def scaled(self, s: float) -> "AngleShape":
        return AngleShape(self.angle_deg, self.radius_px * float(s), self.bisector_deg,
                          self.stroke_px, self.edge_rgba, self.margin_px, self.show_arc)

    def render(self, label: str, *, prop_kind: str = "angle_deg",
               draw_label: bool = True, label_font_px: Optional[int] = None) -> RenderedShape:
        rs = render_angle_patch(self.angle_deg, self.radius_px, label,
                                bisector_deg=self.bisector_deg,
                                stroke_px=self.stroke_px, stroke_rgba=self.edge_rgba,
                                margin_px=self.margin_px, show_arc=self.show_arc,
                                draw_label=draw_label, label_font_px=label_font_px)
        rs.prop_kind = prop_kind
        rs.prop_value = self.property_value(prop_kind)
        return rs

# -------- Regular polygon --------

@dataclass
class RegularPolygonShape:
    n_sides: int
    R_px: float
    rotation_deg: float = 0.0
    fill_rgba: RGBA = (0, 0, 0, 0)               # transparent fill
    edge_rgba: RGBA = (0, 0, 0, 255)
    outline_px: int = 3

    def family_name(self) -> str: return "polygon"

    def property_value(self, kind: str) -> float:
        n, R = int(self.n_sides), float(self.R_px)
        if kind == "area": return polygon_area_from_R(n, R)
        if kind == "perimeter": return polygon_perimeter_from_R(n, R)
        raise ValueError("RegularPolygonShape supports 'area' or 'perimeter'")

    def scaled(self, s: float) -> "RegularPolygonShape":
        return RegularPolygonShape(self.n_sides, self.R_px * float(s), self.rotation_deg,
                                   self.fill_rgba, self.edge_rgba, self.outline_px)

    def render(self, label: str, *, prop_kind: str = "area",
               draw_label: bool = True, label_font_px: Optional[int] = None) -> RenderedShape:
        rs = render_regular_polygon_patch(self.n_sides, self.R_px, label,
                                          rotation_deg=self.rotation_deg,
                                          fill_rgba=self.fill_rgba,
                                          outline_rgba=self.edge_rgba,
                                          outline_px=self.outline_px,
                                          draw_label=draw_label, label_font_px=label_font_px)
        rs.prop_kind = prop_kind
        rs.prop_value = self.property_value(prop_kind)
        return rs

# -------- Ellipse --------

@dataclass
class EllipseShape:
    a_px: float
    b_px: float
    rotation_deg: float = 0.0
    fill_rgba: RGBA = (0, 0, 0, 0)               # transparent fill
    edge_rgba: RGBA = (0, 0, 0, 255)
    outline_px: int = 3

    def family_name(self) -> str: return "ellipse"

    def property_value(self, kind: str) -> float:
        a, b = float(self.a_px), float(self.b_px)
        if kind == "area": return math.pi * a * b
        if kind == "perimeter":
            return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
        raise ValueError("EllipseShape supports 'area' or 'perimeter'")

    def scaled(self, s: float) -> "EllipseShape":
        s = float(s)
        return EllipseShape(self.a_px * s, self.b_px * s, self.rotation_deg,
                            self.fill_rgba, self.edge_rgba, self.outline_px)

    def render(self, label: str, *, prop_kind: str = "area",
               draw_label: bool = True, label_font_px: Optional[int] = None) -> RenderedShape:
        rs = render_ellipse_patch(self.a_px, self.b_px, label,
                                  rotation_deg=self.rotation_deg,
                                  fill_rgba=self.fill_rgba,
                                  outline_rgba=self.edge_rgba,
                                  outline_px=self.outline_px,
                                  draw_label=draw_label, label_font_px=label_font_px)
        rs.prop_kind = prop_kind
        rs.prop_value = self.property_value(prop_kind)
        return rs
