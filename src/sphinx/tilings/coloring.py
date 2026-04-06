# sphinx/tilings/coloring.py
from __future__ import annotations
import random
from typing import List, Tuple
from ..config import COLORS  # palette of hex strings
from ..schema import TilingSpec
from ..base import Tiling, TilingPatch

def _pick(colors_idx: List[int], k: int) -> str:
    idxs = colors_idx or list(range(len(COLORS)))
    return COLORS[idxs[k % len(idxs)]]

def _choose_idxs(rng: random.Random, k: int, given: List[int] | None = None) -> List[int]:
    """Return k distinct color indices (or all shuffled if palette smaller)."""
    if given:
        return list(given)
    n = len(COLORS)
    if n <= k:
        idxs = list(range(n))
        rng.shuffle(idxs)
        return idxs  # fewer than k available; _pick will modulo as needed
    return rng.sample(range(n), k)

class Colorer:
    def apply(self, tiling: Tiling, patch: TilingPatch, spec: TilingSpec):
        rng = random.Random(spec.seed)

        if spec.color_mode == "non_uniform":
            conf = spec.non_uniform or {}
            idxs = list(conf.get("colors_idx") or range(len(COLORS)))
            weights = conf.get("p")
            if weights:
                # normalize
                s = sum(float(w) for w in weights) or 1.0
                ps = [float(w)/s for w in weights]
            for c in patch.cells:
                if weights:
                    r = rng.random()
                    acc = 0.0; k = 0
                    for i,p in enumerate(ps):
                        acc += p
                        if r <= acc: k = i; break
                else:
                    k = rng.randrange(1 << 30)
                c.color = _pick(idxs, k)
            return

        # uniform modes
        uni = spec.uniform or {}
        scheme = (uni.get("scheme") or "same").lower()
        provided_idxs = uni.get("colors_idx")

        if scheme == "same":
            # pick ONE random color if not provided
            idxs = _choose_idxs(rng, 1, given=provided_idxs)
            col = _pick(idxs, 0)
            for c in patch.cells:
                c.color = col
            return

        if scheme == "wythoffian" and getattr(tiling, "supports_wythoffian", False):
            # tri/hex = 3 classes, square = 4 classes; choose random palette if not provided
            class_count = 4 if tiling.__class__.__name__ == "SquareTiling" else 3
            idxs = _choose_idxs(rng, class_count, given=provided_idxs)
            for c in patch.cells:
                cid = int(tiling.wythoffian_class_id(c))
                c.color = _pick(idxs, cid)
            return

        # nonwythoffian: stock patterns (default to 2 random colors if not provided)
        variant = (uni.get("variant") or "parity").lower()
        idxs = _choose_idxs(rng, 2, given=provided_idxs)

        if variant == "ring":
            cx = sum(c.coord[0] for c in patch.cells)/len(patch.cells)
            cy = sum(c.coord[1] for c in patch.cells)/len(patch.cells)
            for c in patch.cells:
                r = int(round(abs(c.coord[0]-cx) + abs(c.coord[1]-cy)))
                c.color = _pick(idxs, r)
        else:
            for c in patch.cells:
                i,j = c.coord
                c.color = _pick(idxs, (i + j) & 1)
