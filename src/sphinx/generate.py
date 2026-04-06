from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image

from .registry import build_motif_registry, build_task_registry, get_task_names


_WORKER_TASKS = None
_WORKER_MOTIFS = None


def list_tasks() -> list[str]:
    return get_task_names()


def _stable_sample_seed(base_seed: int, task_name: str, sample_idx: int) -> int:
    task_offset = sum(ord(ch) for ch in task_name)
    return (base_seed * 1_000_003 + task_offset * 10_007 + sample_idx * 97) & ((1 << 63) - 1)


def _thumbnail(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = img.convert("RGB").copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - thumb.width) // 2
    y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def _write_contact_sheet(images: list[Path], out_path: Path, thumb_size: tuple[int, int] = (240, 240), columns: int = 5) -> None:
    if not images:
        return

    rows = (len(images) + columns - 1) // columns
    margin = 12
    canvas = Image.new(
        "RGB",
        (
            columns * thumb_size[0] + (columns + 1) * margin,
            rows * thumb_size[1] + (rows + 1) * margin,
        ),
        "white",
    )

    for idx, image_path in enumerate(images):
        with Image.open(image_path) as img:
            thumb = _thumbnail(img, thumb_size)
        row, col = divmod(idx, columns)
        x = margin + col * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + margin)
        canvas.paste(thumb, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _worker_context():
    global _WORKER_TASKS, _WORKER_MOTIFS
    if _WORKER_TASKS is None or _WORKER_MOTIFS is None:
        _WORKER_TASKS = build_task_registry()
        _WORKER_MOTIFS = build_motif_registry()
    return _WORKER_TASKS, _WORKER_MOTIFS


def _retry_policy(task_name: str, task, fast_mode: bool) -> tuple[int | None, int]:
    retries = getattr(task, "max_retries", None)
    if fast_mode and task_name == "sequence_arithmetic" and retries is not None:
        return 1, 64
    return retries, 1


def _generate_sample(task_name: str, sample_idx: int, base_seed: int, task_dir: str, fast_mode: bool) -> dict:
    tasks, motifs = _worker_context()
    task = tasks[task_name]
    retries, seed_attempts = _retry_policy(task_name, task, fast_mode)
    original_retries = getattr(task, "max_retries", None)

    last_error = None
    try:
        if retries is not None:
            task.max_retries = retries

        for attempt_idx in range(seed_attempts):
            sample_seed = _stable_sample_seed(base_seed + attempt_idx * 7_919, task_name, sample_idx)
            rng = random.Random(sample_seed)
            try:
                image, _cell_specs, meta = task.generate_instance(motifs, rng)
            except RuntimeError as exc:
                last_error = exc
                continue

            if image is None:
                last_error = RuntimeError(f"Task '{task_name}' returned no image for sample {sample_idx}")
                continue

            task_path = Path(task_dir)
            task_path.mkdir(parents=True, exist_ok=True)
            image_path = task_path / f"sample_{sample_idx:04d}.png"
            image.save(image_path)

            return {
                "task": task_name,
                "sample_idx": sample_idx,
                "seed": sample_seed,
                "image_name": image_path.name,
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
            }
    finally:
        if original_retries is not None:
            task.max_retries = original_retries

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Task '{task_name}' failed to generate sample {sample_idx}")


def generate_task_examples(
    out_dir: str | Path,
    task_names: Iterable[str] | None = None,
    samples_per_task: int = 1,
    base_seed: int = 42,
    workers: int | None = None,
    write_contact_sheets: bool = True,
    write_metadata: bool = True,
    fast_mode: bool = True,
    verbose: bool = True,
) -> Path:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    task_registry = build_task_registry()
    selected = list(task_names) if task_names is not None else sorted(task_registry.keys())

    summary = []
    for task_name in selected:
        if task_name not in task_registry:
            raise KeyError(f"Unknown task '{task_name}'. Known: {sorted(task_registry)}")

        task_dir = out_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[generate] Generating {samples_per_task} samples for {task_name}")

        saved_paths: list[Path] = []
        metadata_rows = []
        effective_workers = workers or min(samples_per_task, os.cpu_count() or 1)
        effective_workers = max(1, min(samples_per_task, effective_workers))

        if effective_workers == 1:
            for sample_idx in range(samples_per_task):
                row = _generate_sample(task_name, sample_idx, base_seed, str(task_dir), fast_mode)
                metadata_rows.append(row)
                saved_paths.append(task_dir / row["image_name"])
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = [
                    executor.submit(_generate_sample, task_name, sample_idx, base_seed, str(task_dir), fast_mode)
                    for sample_idx in range(samples_per_task)
                ]
                for future in as_completed(futures):
                    row = future.result()
                    metadata_rows.append(row)
                    saved_paths.append(task_dir / row["image_name"])

        metadata_rows.sort(key=lambda row: row["sample_idx"])
        saved_paths.sort()

        if write_metadata:
            with (task_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
                for row in metadata_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if write_contact_sheets:
            _write_contact_sheet(saved_paths, task_dir / "contact_sheet.png")

        summary.append({"task": task_name, "samples": samples_per_task, "dir": str(task_dir)})
        if verbose:
            print(f"[generate] Completed {task_name}")

    with (out_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-task SPHINX examples.")
    parser.add_argument("--out", type=str, default="generated/tasks")
    parser.add_argument("--samples-per-task", "--n", dest="samples_per_task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Worker processes per task. Defaults to CPU count.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of task names. Defaults to all tasks.")
    parser.add_argument("--list-tasks", action="store_true", help="Print available task names and exit.")
    parser.add_argument("--no-contact-sheets", action="store_true", help="Skip contact sheet generation.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip metadata.jsonl generation.")
    parser.add_argument("--full-retries", action="store_true", help="Disable the fast generation policy used for expensive tasks.")
    args = parser.parse_args()

    if args.list_tasks:
        for task_name in list_tasks():
            print(task_name)
        return

    out_dir = generate_task_examples(
        out_dir=args.out,
        task_names=args.tasks,
        samples_per_task=args.samples_per_task,
        base_seed=args.seed,
        workers=args.workers or None,
        write_contact_sheets=not args.no_contact_sheets,
        write_metadata=not args.no_metadata,
        fast_mode=not args.full_retries,
        verbose=True,
    )
    print(f"Wrote generated task examples to {out_dir}")


if __name__ == "__main__":
    main()
