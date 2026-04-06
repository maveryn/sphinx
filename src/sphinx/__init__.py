__all__ = [
    "generate_dataset",
    "generate_task_examples",
    "generate_smoke_examples",
    "get_task_names",
]
__version__ = "0.1.0"


def generate_dataset(*args, **kwargs):
    from .engine import generate_dataset as _generate_dataset

    return _generate_dataset(*args, **kwargs)


def generate_task_examples(*args, **kwargs):
    from .generate import generate_task_examples as _generate_task_examples

    return _generate_task_examples(*args, **kwargs)


def generate_smoke_examples(*args, **kwargs):
    from .smoke import generate_smoke_examples as _generate_smoke_examples

    return _generate_smoke_examples(*args, **kwargs)


def get_task_names():
    from .registry import get_task_names as _get_task_names

    return _get_task_names()
