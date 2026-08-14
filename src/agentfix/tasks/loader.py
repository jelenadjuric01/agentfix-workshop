from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_PROMPT = "The test suite for this project is failing. Find the bug and fix it."


@dataclass(frozen=True)
class Task:
    task_id: str
    root: Path
    template_dir: Path
    test_command: tuple[str, ...]
    expected_failures: tuple[str, ...]
    prompt: str


def load_task(task_dir: Path) -> Task:
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))

    command = tuple(meta.get("test_command", ["-m", "pytest", "-q"]))
    if command and command[0].startswith("-"):
        command = (sys.executable, *command)

    return Task(
        task_id=meta.get("task_id", task_dir.name),
        root=task_dir,
        template_dir=task_dir / "repo",
        test_command=command,
        expected_failures=tuple(meta.get("expected_failures", ())),
        prompt=meta.get("prompt", DEFAULT_PROMPT),
    )


@contextmanager
def workspace(task: Task) -> Iterator[Path]:
    """Copy the pristine template into a temp dir so every run starts identical."""
    temp_root = Path(tempfile.mkdtemp(prefix=f"agentfix_{task.task_id}_"))
    work_dir = temp_root / "repo"
    try:
        shutil.copytree(task.template_dir, work_dir)
        yield work_dir
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
