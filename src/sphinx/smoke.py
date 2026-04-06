from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .generate import generate_task_examples


def generate_smoke_examples(
    out_dir: str | Path,
    task_names: Iterable[str] | None = None,
    samples_per_task: int = 25,
    base_seed: int = 42,
    workers: int | None = None,
) -> Path:
    return generate_task_examples(
        out_dir=out_dir,
        task_names=task_names,
        samples_per_task=samples_per_task,
        base_seed=base_seed,
        workers=workers,
        write_contact_sheets=True,
        write_metadata=True,
        fast_mode=True,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate smoke-test images for SPHINX tasks.")
    parser.add_argument("--out", type=str, default="smoke_outputs")
    parser.add_argument("--samples-per-task", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Worker processes per task. Defaults to CPU count.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task names. Defaults to all 25 SPHINX tasks.",
    )
    args = parser.parse_args()

    out_dir = generate_smoke_examples(
        out_dir=args.out,
        task_names=args.tasks,
        samples_per_task=args.samples_per_task,
        base_seed=args.seed,
        workers=args.workers or None,
    )
    print(f"Wrote smoke outputs to {out_dir}")


if __name__ == "__main__":
    main()
