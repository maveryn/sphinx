# sphinx/motifs/polyomino.py
import random
from typing import List, Tuple
from PIL import Image, ImageDraw
from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background


def _make_growth_path(seed: int, max_len: int) -> List[Tuple[int, int]]:
    """
    Deterministic connected growth path of grid cells (4-neighborhood).
    Returns a list of (x,y) cells; prefix of length k is a k-square polyomino.
    """
    rng = random.Random(seed)
    cells = {(0, 0)}
    order = [(0, 0)]
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while len(order) < max_len:
        # frontier = all empty neighbors of current shape
        front = set()
        for (x, y) in cells:
            for dx, dy in dirs:
                nb = (x + dx, y + dy)
                if nb not in cells:
                    front.add(nb)
        if not front:
            # extremely unlikely; but restart from a new seed if it happens
            rng.seed(seed + len(order) * 9973)
            front = {(order[-1][0] + 1, order[-1][1])}
        nb = rng.choice(list(front))
        cells.add(nb)
        order.append(nb)

    # normalize to top-left origin so rendering is centered nicely
    minx = min(x for x, _ in order)
    miny = min(y for _, y in order)
    order = [(x - minx, y - miny) for (x, y) in order]
    return order


@register_motif
class PolyominoMotif(Motif):
    """
    Tetris-like single piece with variable number of unit squares.

    - count: number of squares (>=1)
    - extra.scale: overall size multiplier
    - extra.rotation: {0, 90, 180, 270}
    - extra.path: growth path (list of (x,y) cells) reused across clones
    """
    name = "polyomino"
    attr_ranges = {"count": (3, 15), "thickness": (2, 4)}

    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(*self.attr_ranges["count"])
        thickness = rng.randint(*self.attr_ranges["thickness"])
        # Pre-build a growth path long enough for any count progression
        max_len = max(self.attr_ranges["count"][1], 12)
        path = _make_growth_path(seed, max_len)
        extra = {
            "scale": rng.uniform(0.95, 1.10),
            "rotation": rng.choice([0, 90, 180, 270]),
            "path": path,
            # small seam to make individual squares legible (like Tetris skin)
            "gap_frac": rng.uniform(0.06, 0.12),
        }
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=count,
            thickness=thickness,
            size=1.0,
            extra=extra,
        )

    # Clamp so sequences don’t request more squares than we have in the path
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        path = ex.get("path")
        if not path:
            # build a default path if missing
            path = _make_growth_path(spec.seed, max(self.attr_ranges["count"][1], 12))
            ex["path"] = path
        max_count = len(path)
        c = int(spec.count) if hasattr(spec, "count") else self.attr_ranges["count"][0]
        c = max(1, min(max_count, c))
        # keep ints for PIL widths
        thick = max(1, int(getattr(spec, "thickness", self.attr_ranges["thickness"][0])))
        rot = int(float(ex.get("rotation", 0.0))) % 360
        ex["rotation"] = (rot // 90) * 90  # snap to multiples of 90
        ex["scale"] = float(ex.get("scale", 1.0))
        ex["gap_frac"] = float(ex.get("gap_frac", 0.1))
        return spec.clone(count=c, thickness=thick, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)
        path: List[Tuple[int, int]] = s.extra["path"]
        k = int(s.count)
        cells = path[:k]

        # piece bounds in cells
        w = max(x for x, _ in cells) + 1
        h = max(y for _, y in cells) + 1

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        color = COLORS[s.color_idx]

        # layout
        scale = float(s.extra.get("scale", 1.0))
        rot = int(s.extra.get("rotation", 0)) % 360
        margin = int(SS_CELL * 0.12)
        usable_w = SS_CELL - 2 * margin
        usable_h = SS_CELL - 2 * margin

        gap_frac = float(s.extra.get("gap_frac", 0.1))
        # cell size derived from bounds and gap fraction: width_px = w*c + (w-1)*gap
        # where gap = gap_frac * c  => width_px = c*(w + (w-1)*gap_frac)
        denom_w = (w + (w - 1) * gap_frac)
        denom_h = (h + (h - 1) * gap_frac)
        c_w = usable_w / max(denom_w, 1e-6)
        c_h = usable_h / max(denom_h, 1e-6)
        cell = int(min(c_w, c_h) * scale)

        # ensure it fits after scaling
        gap = max(0, int(round(cell * gap_frac)))
        while True:
            piece_w = w * cell + (w - 1) * gap
            piece_h = h * cell + (h - 1) * gap
            if piece_w <= usable_w and piece_h <= usable_h:
                break
            cell -= 1
            gap = max(0, int(round(cell * gap_frac)))
            if cell <= 2 * SUPERSAMPLE:
                break

        ox = margin + (usable_w - piece_w) // 2
        oy = margin + (usable_h - piece_h) // 2

        # draw squares
        outline_w = max(1, int(s.thickness) * SUPERSAMPLE)
        for (cx, cy) in cells:
            x0 = ox + cx * (cell + gap)
            y0 = oy + cy * (cell + gap)
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=color, outline="black", width=outline_w)

        if rot:
            img = img.rotate(rot, resample=Image.NEAREST, expand=False)

        return _down_on_background(img)
