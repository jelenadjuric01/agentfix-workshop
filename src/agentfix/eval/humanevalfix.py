from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from agentfix.config import REPO_ROOT

VENDORED_SUBSET = REPO_ROOT / "tasks" / "humanevalfix" / "subset.json"


@dataclass(frozen=True)
class HumanEvalFixRow:
    task_id: str
    buggy_code: str
    tests: str
    entry_point: str


def load_vendored_rows(path: Path = VENDORED_SUBSET) -> list[HumanEvalFixRow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [HumanEvalFixRow(**row) for row in payload]


def write_task_dir(row: HumanEvalFixRow, dest: Path) -> Path:
    slug = row.task_id.replace("/", "-").lower()
    task_dir = Path(dest) / slug
    repo = task_dir / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    (repo / "candidate.py").write_text(row.buggy_code.rstrip() + "\n", encoding="utf-8")
    (repo / "test_candidate.py").write_text(row.tests.rstrip() + "\n", encoding="utf-8")

    task_dir.joinpath("task.json").write_text(
        json.dumps(
            {
                "task_id": slug,
                "test_command": ["-u", "test_candidate.py"],
                "expected_failures": [],
                "prompt": (
                    f"The function `{row.entry_point}` in candidate.py is buggy and its tests "
                    "fail. Fix it."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return task_dir


def load_hf_rows(sample: int | None = None, seed: int = 42) -> list[HumanEvalFixRow]:
    """Full dataset. Requires the [eval] extra: uv sync --extra eval"""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("The full benchmark needs: uv sync --extra eval") from error

    dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
    rows = [
        HumanEvalFixRow(
            task_id=item["task_id"],
            # buggy_solution is a function body only; declaration carries the def line
            # and imports, so buggy_code must be their concatenation to be valid Python.
            buggy_code=(item["declaration"] + item["buggy_solution"]).rstrip(),
            tests=f"from candidate import {item['entry_point']}\n\n{item['test'].rstrip()}",
            entry_point=item["entry_point"],
        )
        for item in dataset
    ]
    if sample is not None and sample < len(rows):
        random.seed(seed)
        rows = random.sample(rows, sample)
    return rows


def dump_rows(rows: list[HumanEvalFixRow], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8"
    )
