# sphinx/tasks/symmetry/common.py
"""Common symmetry utilities and transforms."""

from ...utils.image_compare import (
    diff_frac,
    sig,
    strong_distinct,
    strong_same,
    pairwise_unique,
)  # noqa: F401
from ...utils.transforms import (
    apply_transform as apply_tf,
    flip_h,
    flip_v,
    rot,
    rot90,
    rot180,
    rot270,
)


# Mirror-only 2×2 rules (no rotation-only variants)
BASE_RULES_NO_ROT = {
    "mirror_v": ["original", "mirror", "original", "mirror"],
    "mirror_h": ["original", "original", "flip", "flip"],
    "mirror_hv": ["original", "mirror", "flip", "rot180"],
    "diag_main": ["original", "transpose", "transpose", "original"],
    "diag_anti": ["original", "transverse", "transverse", "original"],
}

ALL_TFS_ORDER = [
    "original",
    "mirror",
    "flip",
    "transpose",
    "transverse",
    "rot90",
    "rot180",
    "rot270",
]


__all__ = [
    "diff_frac",
    "sig",
    "strong_distinct",
    "strong_same",
    "rot",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "apply_tf",
    "pairwise_unique",
    "BASE_RULES_NO_ROT",
    "ALL_TFS_ORDER",
]

