# sphinx/schema.py
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class MotifSpec:
    kind: str
    seed: int
    color_idx: int
    count: int = 1
    size: float = 1.0
    angle: float = 0.0
    thickness: int = 4
    holes: int = 0
    opening: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def clone(self, **updates) -> "MotifSpec":
        return replace(self, **updates)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TilingSpec:
    name: str
    seed: int
    width: int                 # number of fundamental cells in X
    height: int                # number of fundamental cells in Y

    # Rendering (kept implicit: we fit the geometry into SS_CELL with margins)
    margin_frac: float = 0.08  # fraction of SS_CELL used as outer margin

    # Coloring
    color_mode: str = "uniform"  # "uniform" | "non_uniform"
    # For uniform: {"scheme": "same" | "wythoffian" | "nonwythoffian",
    #               "variant": "parity" | "ring" (when nonwythoffian),
    #               "colors_idx": List[int]}
    uniform: Optional[Dict[str, Any]] = None
    # For non_uniform: {"colors_idx": List[int], "p": Optional[List[float]]}
    non_uniform: Optional[Dict[str, Any]] = None

    # Freeform extra knobs (per-tiling)
    extra: Optional[Dict[str, Any]] = None

    def clone(self, **kw) -> "TilingSpec":
        return replace(self, **kw)

@dataclass
class OptionItem:
    label: str
    image: Any  # PIL.Image.Image
    desc: str
    payload: Any = None

@dataclass
class InstanceImages:
    composite_path: str
    # (optional) additional images saved to disk in future

@dataclass
class InstanceMeta:
    id: str
    image_name: str
    task: str
    task_params: Dict[str, Any] = field(default_factory=dict)
    complexity: Optional[Dict[str, Any]] = None
    version: Dict[str, Any] = field(default_factory=lambda: {"schema": 2})
    seed: Optional[int] = None
    sample_seed: Optional[int] = None

@dataclass
class ComplexityReport:
    score: float
    level: int
    version: str
    parts: Dict[str, float]
    notes: Dict[str, Any] = field(default_factory=dict)
