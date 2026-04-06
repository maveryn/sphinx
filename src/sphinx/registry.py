# sphinx/registry.py
import random
import importlib
from typing import Dict, List, Type

from .config import DISABLED_MOTIFS, TASK_WEIGHTS
from .utils.rng import choice_weighted

# Global registries populated via decorators
MOTIF_REGISTRY: Dict[str, object] = {}
TASK_REGISTRY: Dict[str, object] = {}
TILING_REGISTRY: Dict[str, Type["Tiling"]] = {}


def register_motif(cls):
    """Class decorator to register a motif implementation."""
    MOTIF_REGISTRY[cls.name] = cls()
    return cls

def register_task(cls):
    """Class decorator to register a task implementation."""
    TASK_REGISTRY[cls.name] = cls()
    return cls


def register_tiling(cls):
    """Class decorator to register a tiling implementation."""
    TILING_REGISTRY[cls.name] = cls
    return cls


def build_motif_registry() -> Dict[str, object]:
    """Instantiate and return all enabled motifs."""
    importlib.import_module(".motifs", __package__)
    return {k: v for k, v in MOTIF_REGISTRY.items() if k not in DISABLED_MOTIFS}

def build_task_registry() -> Dict[str, object]:
    """Instantiate and return all task implementations."""
    importlib.import_module(".tasks", __package__)
    return TASK_REGISTRY


def build_tiling_registry() -> Dict[str, Type["Tiling"]]:
    """Return all registered tiling classes."""
    importlib.import_module(".tilings", __package__)
    return TILING_REGISTRY


def sample_task(rng: random.Random):
    tasks = build_task_registry()
    names = list(tasks.keys())
    weights = [TASK_WEIGHTS.get(n, 1.0) for n in names]
    pick = choice_weighted(rng, names, weights)
    return names.index(pick), tasks[pick]


def get_task_names() -> List[str]:
    return sorted(build_task_registry().keys())


def get_task(name: str):
    tasks = build_task_registry()
    if name not in tasks:
        raise KeyError(f"Unknown task '{name}'. Known: {sorted(tasks)}")
    return tasks[name]


def get_tiling_names() -> List[str]:
    return sorted(TILING_REGISTRY.keys())


def create_tiling(name: str):
    if name not in TILING_REGISTRY:
        raise KeyError(f"Tiling '{name}' not registered. Known: {get_tiling_names()}")
    return TILING_REGISTRY[name]()


def enabled_motif_names() -> List[str]:
    return [k for k in MOTIF_REGISTRY if k not in DISABLED_MOTIFS]
