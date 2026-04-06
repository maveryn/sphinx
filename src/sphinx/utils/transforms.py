"""Utility helpers for discrete image transforms."""

from PIL import Image, ImageOps


TRANSFORMS = [
    "original",
    "rot90",
    "rot180",
    "rot270",
    "mirror",
    "flip",
    "transpose",
    "transverse",
]


def apply_transform(
    img: Image.Image,
    tf: str,
    *,
    resample: int = Image.BICUBIC,
    expand: bool = True,
    fillcolor=(0, 0, 0, 0),
) -> Image.Image:
    """Apply a discrete transform to an image."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    if tf == "original":
        return img
    if tf in ("rot90", "rotate90"):
        return img.rotate(90, resample=resample, expand=expand, fillcolor=fillcolor)
    if tf in ("rot180", "rotate180"):
        return img.rotate(180, resample=resample, expand=expand, fillcolor=fillcolor)
    if tf in ("rot270", "rotate270"):
        return img.rotate(270, resample=resample, expand=expand, fillcolor=fillcolor)
    if tf == "mirror":
        return ImageOps.mirror(img)
    if tf == "flip":
        return ImageOps.flip(img)
    if tf == "transpose":
        return img.transpose(Image.TRANSPOSE)
    if tf == "transverse":
        return img.transpose(Image.TRANSVERSE)
    raise ValueError(tf)


def rot(img: Image.Image, deg: int) -> Image.Image:
    """Rotate ``img`` by ``deg`` degrees using nearest-neighbor resampling."""
    return img.rotate(deg, resample=Image.NEAREST, expand=False)


def rot90(img: Image.Image) -> Image.Image:
    """Rotate ``img`` 90 degrees clockwise with nearest-neighbor fill."""
    return apply_transform(img, "rot90", resample=Image.NEAREST, expand=False)


def rot180(img: Image.Image) -> Image.Image:
    """Rotate ``img`` 180 degrees with nearest-neighbor fill."""
    return apply_transform(img, "rot180", resample=Image.NEAREST, expand=False)


def rot270(img: Image.Image) -> Image.Image:
    """Rotate ``img`` 270 degrees clockwise with nearest-neighbor fill."""
    return apply_transform(img, "rot270", resample=Image.NEAREST, expand=False)


def flip_h(img: Image.Image) -> Image.Image:
    """Mirror ``img`` across the vertical axis."""
    return ImageOps.mirror(img)


def flip_v(img: Image.Image) -> Image.Image:
    """Flip ``img`` across the horizontal axis."""
    return ImageOps.flip(img)


__all__ = [
    "TRANSFORMS",
    "apply_transform",
    "rot",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
]
