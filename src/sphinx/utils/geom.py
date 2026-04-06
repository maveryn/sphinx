# sphinx/utils/geom.py
import math
from typing import List, Tuple


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return (1.0, 0.0)
    return (vx / n, vy / n)


def perp(vx: float, vy: float) -> Tuple[float, float]:
    return (-vy, vx)

def polygon_area(verts: List[tuple]) -> float:
    a = 0.0
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0

def rotate_points(pts: List[Tuple[float, float]], cx: float, cy: float, deg: float):
    rad = math.radians(deg)
    s, c = math.sin(rad), math.cos(rad)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        out.append((cx + c*dx - s*dy, cy + s*dx + c*dy))
    return out
