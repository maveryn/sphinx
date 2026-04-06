# sphinx/motifs/concentric_polygon.py
import math
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageChops

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


def _poly_centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    A = Cx = Cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross; Cx += (x0 + x1) * cross; Cy += (y0 + y1) * cross
    if abs(A) < 1e-9:
        return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    A *= 0.5
    return Cx / (6.0 * A), Cy / (6.0 * A)


def _unit_regular_ngon(n: int, phase_rad: float) -> List[Tuple[float, float]]:
    return [(math.cos(phase_rad + 2.0 * math.pi * i / n),
             math.sin(phase_rad + 2.0 * math.pi * i / n)) for i in range(n)]


def _unit_irregular_ngon(
    n: int, phase_rad: float, jitter_ang: float = 0.18, rmin: float = 0.72, rmax: float = 1.0, rng_seed: int = 0
) -> List[Tuple[float, float]]:
    rng = math.floor(abs(rng_seed))
    def _rand():
        nonlocal rng
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        return (rng / 0x7FFFFFFF)
    pts = []
    for i in range(n):
        base = phase_rad + 2.0 * math.pi * i / n
        a = base + (2.0 * _rand() - 1.0) * jitter_ang
        r = rmin + (rmax - rmin) * _rand()
        pts.append((r * math.cos(a), r * math.sin(a)))
    cx, cy = _poly_centroid(pts)
    pts = [(x - cx, y - cy) for (x, y) in pts]
    maxr = max((x * x + y * y) ** 0.5 for (x, y) in pts) or 1.0
    return [(x / maxr, y / maxr) for (x, y) in pts]




@register_motif
class ConcentricPolygonMotif(Motif):
    """
    Concentric outline polygons (regular or irregular).

    Modes (spec.extra["mode"]):
      - "sym"  : regular n-gons, equally inset (default).
      - "asym" : start with one irregular n-gon; draw larger scaled copies around it.

    Extras:
      - scale     (float): global size multiplier (affects margin).
      - rotation  (float): degrees; applied after drawing.
      - sides     (int)  : number of polygon sides (3..10).
      - mode      (str)  : "sym" or "asym" (default "sym").
      - aa        (int)  : optional extra supersample factor (1..4), default=max(2,SUPERSAMPLE).
    """
    name = "concentric_polygon"
    attr_ranges = {"count": (3, 8), "thickness": (2, 4), "sides": (3, 10)}

    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(*self.attr_ranges["count"])
        thickness = rng.randint(*self.attr_ranges["thickness"])
        sides = rng.randint(*self.attr_ranges["sides"])
        extra = {
            "scale": rng.uniform(0.90, 1.15),
            "rotation": rng.choice([0.0, 15.0, -15.0]),
            "sides": sides,
            "mode": rng.choice(["sym", "asym"]),
            "seed": seed,
            # 'aa' omitted → decided in clamp_spec
        }
        return MotifSpec(self.name, seed, rng.randrange(len(COLORS)),
                         count=count, thickness=thickness, size=1.0, extra=extra)

    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        cmin, cmax = self.attr_ranges["count"]
        count = max(int(cmin), min(int(cmax), int(getattr(spec, "count", 5))))
        thick = max(1, int(getattr(spec, "thickness", 3)))

        scale = max(0.75, min(1.30, float(ex.get("scale", 1.0))))
        rot   = float(ex.get("rotation", 0.0))
        smin, smax = self.attr_ranges["sides"]
        sides = max(int(smin), min(int(smax), int(ex.get("sides", 4))))
        mode  = ex.get("mode", "sym")
        if mode not in ("sym", "asym"): mode = "sym"
        seed  = int(ex.get("seed", spec.seed))

        aa_raw = ex.get("aa", None)
        try:
            aa = int(aa_raw)
        except Exception:
            aa = max(2, SUPERSAMPLE)
        aa = max(1, min(4, aa))

        ex.update({"scale": scale, "rotation": rot, "sides": sides, "mode": mode, "seed": seed, "aa": aa})
        return spec.clone(count=count, thickness=thick, size=1.0, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)

        # High-res canvas for AA
        AA = int(s.extra.get("aa", max(2, SUPERSAMPLE)))
        W = H = SS_CELL * AA
        cx = cy = W // 2

        color_rgba = _to_rgba(COLORS[s.color_idx])

        # Outline width in AA pixels
        ow = max(AA, int(s.thickness) * AA)

        # Layout
        m = int(SS_CELL * 0.12 * float(s.extra["scale"]) * AA)  # margin scaled like original
        Wdraw = W - 2 * m
        if Wdraw <= 2 * ow:
            Wdraw = 2 * ow + 2  # guard
        R_outer = Wdraw / 2.0

        # Step so innermost polygon stays visible
        step = max(2.0 * AA, Wdraw / (2.0 * max(1, int(s.count))))

        # Base polygon (unit radius) → regular or irregular
        n     = int(s.extra["sides"])
        mode  = s.extra["mode"]
        seed  = int(s.extra["seed"])
        unit_pts = _unit_regular_ngon(n, 0.0) if mode == "sym" else _unit_irregular_ngon(n, 0.0, rng_seed=seed)

        # Normalize unit polygon radius
        maxr = max((x * x + y * y) ** 0.5 for (x, y) in unit_pts) or 1.0

        # Accumulator
        img_big = Image.new("RGBA", (W, H), (255, 255, 255, 0))

        def scaled_pts(radius: float):
            sc = radius / maxr
            return [(int(round(cx + sc * x)), int(round(cy + sc * y))) for (x, y) in unit_pts]

        for i in range(int(s.count)):
            Rout = R_outer - i * step
            Rin  = Rout - ow
            if Rin <= 0 or Rout <= 0:
                break

            # Outer/inner filled masks
            mask_out = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask_out).polygon(scaled_pts(Rout), fill=255)
            mask_in = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask_in).polygon(scaled_pts(Rin), fill=255)

            ring = ImageChops.subtract(mask_out, mask_in)
            ring_rgba = Image.new("RGBA", (W, H), color_rgba)
            ring_rgba.putalpha(ring)
            img_big = Image.alpha_composite(img_big, ring_rgba)

        # Apply global rotation at high-res, then downsample with LANCZOS
        rot = float(s.extra["rotation"])
        if abs(rot) > 1e-3:
            img_big = img_big.rotate(rot, resample=Image.BICUBIC, center=(cx, cy), expand=False)

        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)
        return _down_on_background(img)
