# sphinx/motifs/bars.py
import random
from typing import List
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..utils.geom import clamp
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background


@register_motif
class BarsMotif(Motif):
    """
    Vertical bars motif with controllable symmetry.

    Modes (spec.extra["mode"]):
      - "sym"  (default): all bars share the same height; uses extra["height"] ∈ (0,1)
      - "asym": per-column heights; uses extra["heights"] = [h_i], enforced non-palindromic

    """
    name = "bars"
    attr_ranges = {"count": (2, 10), "size": (0.9, 1.2), "thickness": (0, 2)}

    # --- sampling (DEFAULT = symmetric) ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        count = rng.randint(int(self.attr_ranges["count"][0]), int(self.attr_ranges["count"][1]))
        # default to symmetric mode (tasks can override by cloning with extra={"mode":"asym"})
        extra = {"mode": "sym", "height": rng.uniform(0.30, 0.85)}
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=count,
            size=rng.uniform(*self.attr_ranges["size"]),
            angle=0.0,
            thickness=rng.randint(*self.attr_ranges["thickness"]),
            extra=extra,
        )

    @staticmethod
    def _ensure_asym_heights(rng: random.Random, hs: List[float]) -> List[float]:
        if not hs:
            return [0.5]
        for _ in range(16):
            if hs != list(reversed(hs)) and len(set(round(h, 3) for h in hs)) > 1:
                break
            j = rng.randrange(len(hs))
            hs[j] = min(0.95, max(0.05, hs[j] + rng.uniform(-0.15, 0.15)))
        return hs

    # --- normalization (DEFAULT mode fallback = "sym") ---
    def clamp_spec(self, spec):
        rng = random.Random(spec.seed ^ 0xDA7A5EED)

        cmin, cmax = int(self.attr_ranges["count"][0]), int(self.attr_ranges["count"][1])
        count = max(cmin, min(cmax, int(getattr(spec, "count", 5))))
        size = float(getattr(spec, "size", 1.0))
        thick = max(1, int(getattr(spec, "thickness", 3)))

        ex = dict(spec.extra or {})
        mode = ex.get("mode", "sym")  # <-- default to symmetric if unspecified

        if mode == "sym":
            if "height" in ex and isinstance(ex["height"], (int, float)):
                h = float(ex["height"])
            else:
                hs = ex.get("heights")
                h = (sum(float(v) for v in hs) / len(hs)) if isinstance(hs, list) and hs else 0.5
            ex["height"] = clamp(h, 0.05, 0.95)
            ex.pop("heights", None)
        else:
            hs = ex.get("heights")
            if not isinstance(hs, list) or len(hs) != count:
                base = float(ex.get("height", 0.5))
                hs = [base + rng.uniform(-0.2, 0.2) for _ in range(count)]
            hs = [clamp(float(v), 0.05, 0.95) for v in hs]
            hs = self._ensure_asym_heights(rng, hs)
            ex["heights"] = hs
            ex.pop("height", None)

        ex["mode"] = mode
        return spec.clone(count=count, size=size, thickness=thick, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        rng = random.Random(s.seed)

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        pad = int(16 * SUPERSAMPLE)
        k = max(1, int(s.count))
        mode = s.extra.get("mode", "sym")

        if mode == "sym":
            h_norm = clamp(float(s.extra.get("height", 0.5)), 0.05, 0.95)
            heights_norm = [h_norm] * k
        else:
            heights_norm = s.extra.get("heights") or [rng.uniform(0.30, 0.85) for _ in range(k)]
            if len(heights_norm) != k:
                base = heights_norm[0] if heights_norm else 0.5
                heights_norm = [base] * k

        usable_h = clamp((SS_CELL - 2 * pad) * float(s.size), SS_CELL * 0.20, SS_CELL - 2 * pad)
        heights_px = [clamp(h, 0.05, 0.95) * usable_h for h in heights_norm]

        gap = max(2 * SUPERSAMPLE, int(SS_CELL * 0.025))
        total_gap = (k + 1) * gap
        bar_w = max(3 * SUPERSAMPLE, int((SS_CELL - 2 * pad - total_gap) / max(k, 1)))

        x = pad + gap
        base_y = SS_CELL - pad
        color = COLORS[s.color_idx]
        ow = max(1, int(s.thickness) * SUPERSAMPLE)

        for hpx in heights_px:
            top = base_y - hpx
            draw.rectangle([x, top, x + bar_w, base_y], fill=color, outline="black", width=ow)
            x += bar_w + gap

        return _down_on_background(img)

    # --- perturbations (unchanged; now respect default 'sym') ---
    def attr_perturbations(self, spec, pattern_name=None):
        perts = []
        rng_local = random.Random(spec.seed ^ 0xBADC0DE)

        # count ±d
        MAXC = int(self.attr_ranges["count"][1])
        MINC = int(self.attr_ranges["count"][0])
        for d in (-3, -2, -1, 1, 2, 3):
            nc = int(spec.count) + d
            if MINC <= nc <= MAXC:
                perts.append(lambda s, d=d: (self.clamp_spec(s.clone(count=int(s.count)+d)), f"count{d:+d}"))

        s = self.clamp_spec(spec)
        mode = s.extra.get("mode", "sym")

        if mode == "sym":
            base_h = float(s.extra.get("height", 0.5))
            def sym_delta(delta):
                def _p(sp):
                    ex = dict(sp.extra or {})
                    ex["mode"] = "sym"
                    ex["height"] = clamp(base_h + delta, 0.05, 0.95)
                    return self.clamp_spec(sp.clone(extra=ex)), f"height{delta:+.2f}"
                return _p
            for delta in (-0.15, -0.10, -0.06, 0.06, 0.10, 0.15):
                perts.append(sym_delta(delta))
        else:
            base_hs = s.extra.get("heights") or [rng_local.uniform(0.30, 0.85) for _ in range(int(s.count))]
            k = len(base_hs)
            def all_delta(delta):
                def _p(sp):
                    hs = [clamp(v + delta, 0.05, 0.95) for v in base_hs]
                    ex = dict(sp.extra or {}); ex["mode"] = "asym"; ex["heights"] = self._ensure_asym_heights(rng_local, hs)
                    return self.clamp_spec(sp.clone(extra=ex)), f"alls{delta:+.2f}"
                return _p
            def one_bump(mag):
                def _p(sp):
                    hs = list(base_hs); idx = rng_local.randrange(k)
                    hs[idx] = clamp(hs[idx] + mag, 0.05, 0.95)
                    ex = dict(sp.extra or {}); ex["mode"] = "asym"; ex["heights"] = self._ensure_asym_heights(rng_local, hs)
                    return self.clamp_spec(sp.clone(extra=ex)), f"col{idx}_d{mag:+.2f}"
                return _p
            for delta in (-0.12, -0.08, 0.08, 0.12):
                perts.append(all_delta(delta))
            for mag in (-0.20, -0.12, 0.12, 0.20):
                perts.append(one_bump(mag))

        return perts
