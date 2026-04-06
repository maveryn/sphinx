# sphinx/motifs/bitgrid.py
import random
from typing import List
from PIL import Image, ImageDraw

from ..base import Motif
from ..schema import MotifSpec
from ..utils.geom import clamp
from ..utils.bits import rand_bits, xor_bits
from ..config import SUPERSAMPLE, SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background


@register_motif
class BitGridMotif(Motif):
    """
    n×n binary grid motif (single mode).

    Notes:
      * clamp_spec sanitizes extras, fixes bit length (= n*n), and ignores any stray 'mode'.
      * sample_spec avoids degenerate all-0 and all-1 patterns.
    """
    name = "bitgrid"
    attr_ranges = {
        "size": (0.9, 1.2),
        "thickness": (3, 6),
        "count": (1, 1),
        "n": (3, 6),  # <-- grid size is now an attribute (externally tunable)
    }

    # --- sampling ---
    def sample_spec(self, rng: random.Random):
        seed = rng.randint(0, 2**31 - 1)
        n = rng.randint(*self.attr_ranges["n"])  # ← was hardcoded; now attribute-driven
        bits = rand_bits(random.Random(seed), n)

        # avoid fully empty/full patterns
        if sum(bits) in (0, n * n):
            i = rng.randrange(n * n)
            bits[i] = 1 - bits[i]

        extra = {"n": n, "bits": bits}
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
        ex.pop("mode", None)  # ignore any symmetry-mode flag

        nmin, nmax = self.attr_ranges["n"]
        n = int(ex.get("n", nmin))
        n = max(int(nmin), min(int(nmax), n))

        L = n * n
        bits = list(ex.get("bits", [0] * L))
        if len(bits) < L:
            bits = bits + [0] * (L - len(bits))
        elif len(bits) > L:
            bits = bits[:L]

        # avoid trivial all-0/all-1 patterns
        s = sum(bits)
        if s == 0:
            bits[0] = 1
        elif s == L:
            bits[0] = 0

        size = float(getattr(spec, "size", 1.0))
        thick = max(1, int(getattr(spec, "thickness", 3)))

        ex.update({"n": n, "bits": bits})
        return spec.clone(count=1, size=size, thickness=thick, extra=ex)

    # --- rendering ---
    def render(self, spec):
        s = self.clamp_spec(spec)
        n = int(s.extra["n"])
        bits: List[int] = list(s.extra["bits"])

        img = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        pad = int(16 * SUPERSAMPLE)
        cell = (SS_CELL - 2 * pad) / n
        fill_frac = clamp(0.82 * float(s.size), 0.50, 0.95)
        inner = cell * fill_frac
        margin = (cell - inner) / 2
        color = COLORS[s.color_idx]

        for r in range(n):
            for c in range(n):
                if bits[r * n + c]:
                    x0 = pad + c * cell + margin
                    y0 = pad + r * cell + margin
                    x1 = x0 + inner
                    y1 = y0 + inner
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline=None)

        # grid lines
        g = max(2 * SUPERSAMPLE, int(SUPERSAMPLE * 1))
        for i in range(n + 1):
            draw.line([(pad, pad + i * cell), (pad + n * cell, pad + i * cell)], fill="black", width=g)
            draw.line([(pad + i * cell, pad), (pad + i * cell, pad + n * cell)], fill="black", width=g)

        return _down_on_background(img)

    # --- perturbations ---
    def attr_perturbations(self, spec, pattern_name=None):
        perts = []
        s = self.clamp_spec(spec)
        n = int(s.extra["n"])
        L = n * n
        rng_local = random.Random(spec.seed ^ 0xC0FFEE)

        for k in (1, 2, 3, 4):
            def p(sp, kk=k):
                idxs = rng_local.sample(range(L), kk)
                bits = list((sp.extra or {}).get("bits", s.extra["bits"]))
                new_bits = xor_bits(bits, idxs)
                ex = dict(sp.extra or {})
                ex["bits"] = new_bits
                ex["n"] = n
                return self.clamp_spec(sp.clone(extra=ex)), f"xor({kk})"
            perts.append(p)

        return perts
