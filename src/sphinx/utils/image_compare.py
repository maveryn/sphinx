# sphinx/utils/image_compare.py
"""Image similarity and hashing utilities shared across tasks."""

import hashlib
from typing import Sequence, Tuple

from PIL import Image, ImageChops, ImageFilter

from ..config import (
    DHASH_SIZE,
    HASH_PRE_BLUR,
    HASH_PRE_DOWNSAMPLE,
)


def sig(img: Image.Image) -> bytes:
    """Stable digest of exact pixels (RGBA)."""
    return hashlib.sha1(img.convert("RGBA").tobytes()).digest()


def _prep_rgb(img: Image.Image, out: int = HASH_PRE_DOWNSAMPLE, blur: float = HASH_PRE_BLUR) -> Image.Image:
    """RGB pre-averaging to suppress anti-aliasing noise while preserving color."""
    im = img.convert("RGB")
    if blur and blur > 0:
        im = im.filter(ImageFilter.BoxBlur(radius=blur))
    if out and (im.width != out or im.height != out):
        im = im.resize((out, out), Image.BOX)
    return im


def _dhash_bits_1ch(grey: Image.Image, hash_size: int) -> Tuple[int, int]:
    gh = grey.resize((hash_size + 1, hash_size), Image.BOX)
    px = list(gh.getdata())
    w = hash_size + 1
    code_h = 0
    for y in range(hash_size):
        row = y * w
        for x in range(hash_size):
            code_h = (code_h << 1) | (1 if px[row + x] < px[row + x + 1] else 0)
    gv = grey.resize((hash_size, hash_size + 1), Image.BOX)
    pv = list(gv.getdata())
    wv = hash_size
    code_v = 0
    for y in range(hash_size):
        for x in range(hash_size):
            a = pv[y * wv + x]
            b = pv[(y + 1) * wv + x]
            code_v = (code_v << 1) | (1 if a < b else 0)
    return code_h, code_v


def _ham(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dhash_hamming_pair_rgb(a: Image.Image, b: Image.Image, hash_size: int = DHASH_SIZE) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
    """Return ((horiz R,G,B), (vert R,G,B), bits_per_channel)."""
    A = _prep_rgb(a)
    B = _prep_rgb(b)
    Ar, Ag, Ab = A.split()
    Br, Bg, Bb = B.split()

    arh, arv = _dhash_bits_1ch(Ar, hash_size)
    agh, agv = _dhash_bits_1ch(Ag, hash_size)
    abh, abv = _dhash_bits_1ch(Ab, hash_size)

    brh, brv = _dhash_bits_1ch(Br, hash_size)
    bgh, bgv = _dhash_bits_1ch(Bg, hash_size)
    bbh, bbv = _dhash_bits_1ch(Bb, hash_size)

    hh = (_ham(arh, brh), _ham(agh, bgh), _ham(abh, bbh))
    hv = (_ham(arv, brv), _ham(agv, bgv), _ham(abv, bbv))
    bits = hash_size * hash_size
    return hh, hv, bits


def diff_frac(a: Image.Image, b: Image.Image, thresh: int = 8) -> float:
    """Fraction of pixels differing more than `thresh` in luminance."""
    d = ImageChops.difference(a.convert("RGB"), b.convert("RGB")).convert("L")
    hist = d.histogram()
    changed = sum(hist[thresh + 1 :])
    return changed / max(1, d.size[0] * d.size[1])


def strong_distinct(a: Image.Image, b: Image.Image, pix_min: float, hash_min_bits: int) -> bool:
    """Distinct if pixel diff ≥ `pix_min` or hash distance ≥ `hash_min_bits`."""
    if diff_frac(a, b) >= pix_min:
        return True
    (hhR, hhG, hhB), (hvR, hvG, hvB), _ = dhash_hamming_pair_rgb(a, b)
    return max(hhR, hhG, hhB, hvR, hvG, hvB) >= hash_min_bits


def strong_same(a: Image.Image, b: Image.Image, pix_tol: float, hash_max_bits: int) -> bool:
    """Same if pixel diff ≤ `pix_tol` or all hash distances ≤ `hash_max_bits`."""
    if diff_frac(a, b) <= pix_tol:
        return True
    (hhR, hhG, hhB), (hvR, hvG, hvB), _ = dhash_hamming_pair_rgb(a, b)
    return max(hhR, hhG, hhB, hvR, hvG, hvB) <= hash_max_bits


def pairwise_unique(
    imgs: Sequence[Image.Image],
    pix_min: float | None = None,
    hash_min_bits: int | None = None,
) -> bool:
    """Return ``True`` if all images in ``imgs`` are distinct.

    The check first ensures that every image has a unique pixel signature via
    :func:`sig`.  If ``pix_min`` and ``hash_min_bits`` are provided, the function
    additionally verifies that each pair of images differs by at least the given
    pixel fraction or perceptual-hash distance using :func:`strong_distinct`.
    """

    hashes = [sig(im) for im in imgs]
    if len(set(hashes)) != len(imgs):
        return False
    if pix_min is None or hash_min_bits is None:
        return True
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            if not strong_distinct(imgs[i], imgs[j], pix_min, hash_min_bits):
                return False
    return True


__all__ = [
    "sig",
    "dhash_hamming_pair_rgb",
    "diff_frac",
    "strong_distinct",
    "strong_same",
    "pairwise_unique",
]

