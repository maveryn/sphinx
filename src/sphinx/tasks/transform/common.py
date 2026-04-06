# sphinx/tasks/transform/common.py
from __future__ import annotations
import random
from typing import List, Tuple
from PIL import Image

from ...utils.drawing import graph_paper_bg, paste_rgba
from ...utils.transforms import apply_transform as apply_tf
from ...utils.rng import weighted_order  # re-export


def scale_patch(patch: Image.Image, tile_px: int, rng: random.Random) -> Image.Image:
    """Return a randomly scaled copy of ``patch``.

    The new width is a uniform fraction of ``tile_px`` in [0.45, 0.65]; height is
    scaled to preserve aspect ratio.
    """
    scale = rng.uniform(0.45, 0.65)
    target_w = max(12, int(tile_px * scale))
    w = max(1, patch.width)
    h = max(1, patch.height)
    target_h = max(12, int(h * (target_w / w)))
    return patch.resize((target_w, target_h), Image.LANCZOS)


def center_xy(tile_px: int, patch_w: int, patch_h: int) -> Tuple[int, int]:
    """Top-left position that centers a patch within a square tile."""
    return (tile_px // 2 - patch_w // 2, tile_px // 2 - patch_h // 2)


def bounds_ok(tile_px: int, x: int, y: int, w: int, h: int) -> bool:
    """True if the rectangle at ``(x,y,w,h)`` fits within the tile."""
    return x >= 0 and y >= 0 and (x + w) <= tile_px and (y + h) <= tile_px


def compose_tile_with_patch(tile_px: int, patch: Image.Image, offset: Tuple[int, int]) -> Image.Image:
    """Return a graph-paper tile with ``patch`` pasted at ``offset``."""
    tile = graph_paper_bg(tile_px)
    paste_rgba(tile, patch, offset)
    return tile


def apply_patch_transform(patch: Image.Image, tf_key: str) -> Image.Image:
    """Apply a discrete transform keyed by ``tf_key`` to ``patch``."""
    key_map = {
        "mirror_v": "mirror",
        "mirror_h": "flip",
        "diag_main": "transpose",
        "diag_anti": "transverse",
        "rot90": "rot90",
        "rot180": "rot180",
        "rot270": "rot270",
    }
    if tf_key == "translate":
        return patch
    return apply_tf(patch, key_map[tf_key])


def candidate_translations(tile_px: int, patch_w: int, patch_h: int,
                           centered_xy: Tuple[int, int]) -> List[Tuple[int, int]]:
    """List feasible step translations that keep a patch within bounds."""
    step = max(6, tile_px // 6)
    candidates = [
        ( step,  0), (-step,  0), (0,  step), (0, -step),
        ( step,  step), (-step,  step), ( step, -step), (-step, -step),
    ]
    cx, cy = centered_xy
    out: List[Tuple[int, int]] = []
    for dx, dy in candidates:
        if bounds_ok(tile_px, cx + dx, cy + dy, patch_w, patch_h):
            out.append((dx, dy))
    return out


def sample_translation(rng: random.Random, tile_px: int, patch_w: int, patch_h: int) -> Tuple[int, int]:
    """Sample a small translation that keeps the patch within bounds."""
    candidates = candidate_translations(
        tile_px, patch_w, patch_h, center_xy(tile_px, patch_w, patch_h)
    )
    if not candidates:
        return (0, 0)
    return rng.choice(candidates)


__all__ = [
    "scale_patch",
    "center_xy",
    "bounds_ok",
    "compose_tile_with_patch",
    "apply_patch_transform",
    "candidate_translations",
    "sample_translation",
    "weighted_order",
]
