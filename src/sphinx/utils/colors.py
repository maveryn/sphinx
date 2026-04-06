# sphinx/utils/colors.py
from __future__ import annotations
from sphinx.config import COLORS_NAMES


def color_label(hex_str: str) -> str:
    name = COLORS_NAMES.get(hex_str)
    if name is None:
        key = hex_str.lower()
        name = COLORS_NAMES.get(key) or COLORS_NAMES.get(hex_str.upper()) or "color"
    return f"{name} ({hex_str})"
