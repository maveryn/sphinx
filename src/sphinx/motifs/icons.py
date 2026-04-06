# sphinx/motifs/icons.py
import io
import random
import math
from pathlib import Path

import yaml
from PIL import Image

from ..base import Motif
from ..schema import MotifSpec
from ..config import SS_CELL, COLORS
from ..registry import register_motif
from .helpers import _down_on_background, _to_rgba


@register_motif
class IconsMotif(Motif):
    """Motif that renders icons from the ``icons`` asset collection.

    The motif reads the list of available icons and their train/test split from
    ``icons/available-icons.yaml``.  It samples one icon uniformly from the
    requested split and rasterises the corresponding SVG into the canonical
    motif tile size.
    """

    name = "icons"
    attr_ranges = {"count": (1, 1)}

    def __init__(self, split: str = "train"):
        super().__init__(split=split)
        root = Path(__file__).resolve().parents[1]
        with open(root / "icons" / "available-icons.yaml") as f:
            splits = yaml.safe_load(f)
        if split not in splits:
            raise ValueError(f"Unknown split '{split}'")
        self._icons = list(splits[split])
        self._svg_dir = root / "icons" / "svgs"

    # --- sampling ---
    def sample_spec(self, rng: random.Random) -> MotifSpec:
        seed = rng.randint(0, 2 ** 31 - 1)
        icon = rng.choice(self._icons)
        return MotifSpec(
            self.name,
            seed,
            rng.randrange(len(COLORS)),
            count=1,
            extra={"name": icon, "rotation": rng.uniform(0, 360.0)},
        )

    # --- normalization ---
    def clamp_spec(self, spec: MotifSpec) -> MotifSpec:
        ex = dict(spec.extra or {})
        icon = ex.get("name", self._icons[0])
        if icon not in self._icons:
            icon = self._icons[0]
        rot = float(ex.get("rotation", 0.0)) % 360.0
        ex = {"name": icon, "rotation": rot}
        return spec.clone(count=1, extra=ex)

    # --- rendering ---
    def render(self, spec: MotifSpec) -> Image.Image:
        import cairosvg

        s = self.clamp_spec(spec)

        # Deterministic RNG per-spec for scale jitter (as before)
        rng = random.Random(s.seed)
        base_scale = rng.uniform(0.5, 0.75)

        # Guarantee the rotated icon fits the SS_CELL canvas:
        # For a square of side L rotated by angle θ, the bounding side is L*(|cosθ| + |sinθ|).
        # So the max permitted scale is 1 / (|cosθ| + |sinθ|). We clamp with a tiny margin.
        rot_deg = float(s.extra.get("rotation", 0.0))
        theta = math.radians(rot_deg % 90.0)  # periodic every 90°
        max_scale_for_theta = 1.0 / (abs(math.cos(theta)) + abs(math.sin(theta)))
        scale = min(base_scale, 0.98 * max_scale_for_theta)

        size = int(SS_CELL * scale)

        svg_path = self._svg_dir / f"light-{s.extra['name']}.svg"
        png_bytes = cairosvg.svg2png(
            url=str(svg_path), output_width=size, output_height=size
        )
        icon = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        # Recolor the icon using the sampled color
        color = _to_rgba(COLORS[s.color_idx])
        colored = Image.new("RGBA", icon.size, color)
        colored.putalpha(icon.getchannel("A"))

        # Rotate with transparent fill; expand to preserve content
        rotated = colored.rotate(
            rot_deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
        )

        # Center the rotated icon on a transparent SS_CELL canvas
        canvas = Image.new("RGBA", (SS_CELL, SS_CELL), (255, 255, 255, 0))
        x = (SS_CELL - rotated.width) // 2
        y = (SS_CELL - rotated.height) // 2
        canvas.alpha_composite(rotated, (x, y))

        return _down_on_background(canvas)

