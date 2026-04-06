# sphinx/motifs/fractal.py
# sphinx/motifs/fractal.py
import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


@register_motif
class FractalMotif(Motif):
    """
    Multi-fractal motif (single symmetric mode).

    Families (spec.extra["family"]):
      - "sierpinski"         : Sierpinski triangle (order 2–4), canonical gasket.
      - "sierpinski_carpet"  : Sierpinski carpet (order 2–3) bounded in a square.
      - "hilbert"            : Hilbert curve (order 2–5) with rounded joins.
      - "tree"               : Symmetric fractal tree (equal angles/length scales).

    """
    name = "fractal"
    FAMS = ("sierpinski", "sierpinski_carpet", "hilbert", "tree")

    attr_ranges = {
        "size": (0.90, 1.15),
        "thickness": (3, 6),
        "count": (1, 1),
    }

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        fam = rng.choice(self.FAMS)

        extra = {
            "family": fam,
            "rotation": rng.uniform(0, 360),
            # "aa" optional; set in clamp_spec
        }

        if fam == "sierpinski":
            extra.update({"order": rng.choice([2, 3, 4])})
        elif fam == "sierpinski_carpet":
            extra.update({"order": rng.choice([2, 3])})
        elif fam == "hilbert":
            extra.update({
                "order": rng.choice([2, 3, 4]),
                "round_caps": True,
            })
        else:  # tree
            base_ang = rng.uniform(24, 32)
            extra.update({
                "depth": rng.randint(4, 5),
                "angle": base_ang,          # same on both sides
                "scale": rng.uniform(0.68, 0.74),
            })

        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,
            size=rng.uniform(*self.attr_ranges["size"]),
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    # --- normalization ---
    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        smin, smax = self.attr_ranges["size"]
        tmin, tmax = self.attr_ranges["thickness"]
        size = max(smin, min(smax, float(getattr(spec, "size", 1.0))))
        t = max(tmin, min(tmax, int(getattr(spec, "thickness", 2))))

        fam = ex.get("family", "sierpinski")
        if fam not in self.FAMS:
            fam = "sierpinski"

        rotation = float(ex.get("rotation", 0.0)) % 360.0
        aa = int(ex.get("aa", max(3, SUPERSAMPLE)))        # extra AA for crisp edges
        aa = max(1, min(4, aa))

        if fam == "sierpinski":
            order = max(1, min(4, int(ex.get("order", 2))))
            ex.update({"order": order})
        elif fam == "sierpinski_carpet":
            order = max(1, min(3, int(ex.get("order", 2))))
            ex.update({"order": order})
        elif fam == "hilbert":
            order = max(1, min(5, int(ex.get("order", 3))))
            round_caps = bool(ex.get("round_caps", True))
            ex.update({"order": order, "round_caps": round_caps})
        else:  # tree
            depth = max(2, min(7, int(ex.get("depth", 5))))
            angle = max(10.0, min(45.0, float(ex.get("angle", 28.0))))
            scale = max(0.50, min(0.90, float(ex.get("scale", 0.70))))
            ex.update({"depth": depth, "angle": angle, "scale": scale})

        ex.update({"family": fam, "rotation": rotation, "aa": aa})
        return spec.clone(count=1, size=size, thickness=t, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        col = COLORS[s.color_idx]
        AA = int(s.extra["aa"])
        W = H = SS_CELL * AA
        cx = cy = W // 2
        ow = max(AA, int(s.thickness) * AA)  # stroke width at AA scale

        img_big = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        d = ImageDraw.Draw(img_big)

        fam = s.extra["family"]
        rot = float(s.extra["rotation"])

        if fam == "sierpinski":
            self._draw_sierpinski_triangle(d, W, H, cx, cy, col, ow, float(s.size), s.extra)
        elif fam == "sierpinski_carpet":
            self._draw_sierpinski_carpet(d, W, H, cx, cy, col, ow, float(s.size), s.extra)
        elif fam == "hilbert":
            self._draw_hilbert(d, W, H, cx, cy, col, ow, float(s.size), s.extra)
        else:
            self._draw_tree(d, W, H, cx, cy, col, ow, float(s.size), s.extra)

        if abs(rot) > 1e-6:
            img_big = img_big.rotate(rot, resample=Image.BICUBIC, center=(cx, cy), expand=False)

        img = img_big.resize((SS_CELL, SS_CELL), Image.LANCZOS)
        return _down_on_background(img)

    # ---------- Family renderers ----------

    # Sierpinski triangle (canonical gasket)
    def _draw_sierpinski_triangle(self, d, W, H, cx, cy, col, ow, size, ex):
        side = int(0.78 * W * size)
        h = int(side * math.sqrt(3) / 2.0)
        p_top = (cx, cy - h // 2)
        p_left = (cx - side // 2, cy + h // 2)
        p_right = (cx + side // 2, cy + h // 2)

        order = int(ex["order"])
        leaves: List[Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int]]] = []

        def subdiv(p0, p1, p2, depth):
            if depth == 0:
                leaves.append((p0, p1, p2))
                return
            m01 = ((p0[0]+p1[0])//2, (p0[1]+p1[1])//2)
            m12 = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            m20 = ((p2[0]+p0[0])//2, (p2[1]+p0[1])//2)
            # three corner triangles
            subdiv(p0, m01, m20, depth-1)
            subdiv(m01, p1, m12, depth-1)
            subdiv(m20, m12, p2, depth-1)

        subdiv(p_top, p_left, p_right, order)

        # draw filled corner triangles (holes are implicit)
        for tri in leaves:
            d.polygon(tri, fill=col, outline=None)
        # outer boundary for clarity
        d.polygon([p_top, p_left, p_right], outline="black", width=ow)

    # Sierpinski carpet (bounded in a square)
    def _draw_sierpinski_carpet(self, d, W, H, cx, cy, col, ow, size, ex):
        side = int(0.80 * W * size)
        half = side // 2
        x0, y0 = cx - half, cy - half
        order = int(ex["order"])

        # Draw filled squares; the “hole” (center) remains unfilled
        def carpet(x, y, s, depth):
            if depth == 0:
                d.rectangle((x, y, x + s, y + s), fill=col, outline=None)
                return
            t = s // 3
            for dy in range(3):
                for dx in range(3):
                    if dx == 1 and dy == 1:
                        continue  # center hole
                    carpet(x + dx * t, y + dy * t, t, depth - 1)

        carpet(x0, y0, side, order)
        # outer boundary
        d.rectangle((x0, y0, x0 + side, y0 + side), outline="black", width=ow)

    # Hilbert curve with rounded joints
    def _draw_hilbert(self, d, W, H, cx, cy, col, ow, size, ex):
        order = int(ex["order"])
        n = 1 << order
        margin = int(0.10 * W)
        usable = W - 2 * margin
        step = usable / (n - 1) if n > 1 else usable
        round_caps = bool(ex.get("round_caps", True))

        def rot(n, x, y, rx, ry):
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                x, y = y, x
            return x, y

        def d2xy(n, dval):
            x = y = 0
            t = dval
            s = 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                x, y = rot(s, x, y, rx, ry)
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            return x, y

        pts: List[Tuple[int,int]] = []
        total = n * n
        for i in range(total):
            gx, gy = d2xy(n, i)
            x = int(margin + gx * step)
            y = int(margin + gy * step)
            pts.append((x, y))

        cap_r = ow // 2
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            d.line((x0, y0, x1, y1), fill=col, width=ow)
            if round_caps:
                d.ellipse((x0 - cap_r, y0 - cap_r, x0 + cap_r, y0 + cap_r), fill=col)
                d.ellipse((x1 - cap_r, y1 - cap_r, x1 + cap_r, y1 + cap_r), fill=col)

    # Symmetric fractal tree (bounded within the tile)
    def _draw_tree(self, d, W, H, cx, cy, col, ow, size, ex):
        depth = int(ex["depth"])
        angle = float(ex["angle"])
        scale = float(ex["scale"])

        trunk_len = 0.26 * W * size
        base_x, base_y = cx, int(H * 0.78)
        cap_r = ow // 2

        def branch(x, y, angle_deg, length, width, k):
            ang = math.radians(angle_deg)
            x2 = int(x + length * math.cos(ang))
            y2 = int(y - length * math.sin(ang))
            d.line((x, y, x2, y2), fill=col, width=max(1, int(width)))
            # rounded caps for smooth joints
            d.ellipse((x - cap_r, y - cap_r, x + cap_r, y + cap_r), fill=col)
            d.ellipse((x2 - cap_r, y2 - cap_r, x2 + cap_r, y2 + cap_r), fill=col)
            if k == 0:
                return
            wchild = max(1.0, width * 0.72)
            branch(x2, y2, angle_deg + angle, length * scale, wchild, k - 1)
            branch(x2, y2, angle_deg - angle, length * scale, wchild, k - 1)

        branch(base_x, base_y, 90.0, trunk_len, float(ow), depth)
