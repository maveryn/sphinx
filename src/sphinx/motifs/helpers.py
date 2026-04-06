# sphinx/motifs/helpers.py
import math, random
from typing import List, Tuple
from PIL import Image, ImageColor
from ..config import OUT_CELL
from ..utils.drawing import _down_on_background


def _to_rgba(c):
    if isinstance(c, tuple):
        return c if len(c) == 4 else (c[0], c[1], c[2], 255)
    r, g, b = ImageColor.getrgb(c)
    return (r, g, b, 255)


def _non_self_intersecting_polygon(rng: random.Random, sides: int, bounds: tuple) -> List[tuple]:
    xmin, ymin, xmax, ymax = bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    max_r  = min((xmax - xmin), (ymax - ymin)) / 2 * 0.95
    min_r  = max_r * 0.7
    angles = sorted(rng.random() * 2 * math.pi for _ in range(sides))
    return [(
        cx + rng.uniform(min_r, max_r) * math.cos(a),
        cy + rng.uniform(min_r, max_r) * math.sin(a)
    ) for a in angles]


def _rot2d(x: float, y: float, ang_deg: float) -> Tuple[float, float]:
    """Rotate (x, y) by ang_deg around the origin."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    return (x * ca - y * sa, x * sa + y * ca)


