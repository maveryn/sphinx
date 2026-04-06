# sphinx/motifs/clock.py
import math
from PIL import Image, ImageDraw
from ..base import Motif
from ..schema import MotifSpec
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background

@register_motif
class ClockMotif(Motif):
    """
    Clock face with ticks and a center-connected hour hand.

    Extras:
      - position (int): tick index for the hour hand (0..count-1)
      - rotation (deg): global face rotation
      - scale (float): radius multiplier
      - tick_len (px@1x): radial length of ticks
      - aa (int): extra supersample factor (optional; default max(2, SUPERSAMPLE))
      - round_caps (bool): draw circular end-caps on ticks/hand for smoother ends
    """
    name = "clock"
    attr_ranges = {"count": (4, 16), "thickness": (8, 20)}

    def sample_spec(self, rng):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(*self.attr_ranges["count"])
        thickness = rng.randint(*self.attr_ranges["thickness"])
        extra = {
            "scale": rng.uniform(0.9, 1.15),
            "rotation": rng.choice([0, 10, 15, 20, 30]),
            "tick_len": rng.randint(25, 40),
            "position": rng.randrange(count),
            "round_caps": rng.choice([True, False]),
            # omit "aa" so default logic picks max(2, SUPERSAMPLE)
        }
        return MotifSpec(
            self.name, seed, rng.randrange(len(COLORS)),
            count=count, thickness=thickness, size=1.0, extra=extra
        )

    def clamp_spec(self, spec):
        ex = dict(spec.extra or {})
        cmin, cmax = self.attr_ranges["count"]
        count = max(int(cmin), min(int(cmax), int(getattr(spec, "count", cmin))))
        tmin, tmax = self.attr_ranges["thickness"]
        thickness = max(int(tmin), min(int(tmax), int(getattr(spec, "thickness", tmin))))

        scale = max(0.80, min(1.30, float(ex.get("scale", 1.0))))
        rotation = float(ex.get("rotation", 0.0)) % 360.0
        tick_len = max(8, min(30, int(ex.get("tick_len", 18))))
        pos = (int(ex.get("position", 0)) % count) if count > 0 else 0
        round_caps = bool(ex.get("round_caps", True))

        # Extra supersampling factor for AA; clamped to keep things fast
        aa = int(ex.get("aa", max(2, SUPERSAMPLE)))
        aa = max(1, min(4, aa))

        ex.update({
            "scale": scale, "rotation": rotation, "tick_len": tick_len,
            "position": pos, "round_caps": round_caps, "aa": aa
        })
        return spec.clone(count=count, thickness=thickness, extra=ex)

    def render(self, spec):
        s = self.clamp_spec(spec)
        color = COLORS[s.color_idx]

        AA = int(s.extra["aa"])                           # supersample factor for AA
        big = SS_CELL * AA
        temp = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(temp)

        Cx = Cy = big // 2
        scale = float(s.extra["scale"])
        R = int((SS_CELL * 0.40) * scale * AA)

        tick_len = int(s.extra["tick_len"]) * AA
        inner = max(2 * AA, R - tick_len)
        outer = R

        base_thick = max(1, int(s.thickness) * AA)
        tick_w = max(3 * AA, base_thick)
        ring_w = max(3 * AA, base_thick)
        hand_w = max(tick_w, int(1.5 * base_thick))
        hub_r  = max(2 * AA, int(0.7 * base_thick))

        n = max(1, int(s.count))
        rot = float(s.extra["rotation"])
        round_caps = bool(s.extra["round_caps"])
        cap_r = tick_w // 2

        # ticks (with optional round end-caps)
        for i in range(n):
            ang = math.radians(rot + 360.0 * i / n)
            x0 = int(Cx + inner * math.cos(ang))
            y0 = int(Cy + inner * math.sin(ang))
            x1 = int(Cx + outer * math.cos(ang))
            y1 = int(Cy + outer * math.sin(ang))
            d.line((x0, y0, x1, y1), fill=color, width=tick_w)
            if round_caps:
                d.ellipse((x0 - cap_r, y0 - cap_r, x0 + cap_r, y0 + cap_r), fill=color)
                d.ellipse((x1 - cap_r, y1 - cap_r, x1 + cap_r, y1 + cap_r), fill=color)

        # outer rim
        d.ellipse((Cx - outer, Cy - outer, Cx + outer, Cy + outer), outline=color, width=ring_w)

        # hour hand to selected tick
        step_deg = 360.0 / n if n else 0.0
        A = math.radians(rot + int(s.extra["position"]) * step_deg)
        hand_end = int(outer - max(2 * AA, tick_len // 2))
        hx = int(Cx + hand_end * math.cos(A))
        hy = int(Cy + hand_end * math.sin(A))
        d.line((Cx, Cy, hx, hy), fill=color, width=hand_w)
        if round_caps:
            d.ellipse((hx - hand_w//2, hy - hand_w//2, hx + hand_w//2, hy + hand_w//2), fill=color)

        # center hub
        d.ellipse((Cx - hub_r, Cy - hub_r, Cx + hub_r, Cy + hub_r), fill=color, outline=None)

        # High-quality downsample for anti-aliasing
        img = temp.resize((SS_CELL, SS_CELL), resample=Image.LANCZOS)
        return _down_on_background(img)
