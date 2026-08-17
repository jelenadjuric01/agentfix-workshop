"""HumanEvalFix: a standard bug-fixing benchmark, adapted to this project's task format.

The workshop fixtures are hand-written and few, which makes them useful for teaching and
useless for measurement — three tasks cannot tell you much. HumanEvalFix supplies many
independent bugs with tests, so pass@1 over it means something.

Two ways in:

    load_vendored_rows()  — a small committed subset, no dependencies, always works
    load_hf_rows()        — the full dataset from HuggingFace, needs the [eval] extra

Either way, `write_task_dir` converts a row into the same task-directory layout the workshop
fixtures use, so the agent runs through identical machinery for both.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from agentfix.config import REPO_ROOT

VENDORED_SUBSET = REPO_ROOT / "tasks" / "humanevalfix" / "subset.json"


@dataclass(frozen=True)
class HumanEvalFixRow:
    """One benchmark item: broken code, the tests that catch it, and the function's name."""

    task_id: str
    buggy_code: str
    tests: str
    entry_point: str  # the function under test, named in the prompt so the model can find it


def load_vendored_rows(path: Path = VENDORED_SUBSET) -> list[HumanEvalFixRow]:
    """Load the committed subset. No network, no extras, no dataset library.

    `HumanEvalFixRow(**row)` unpacks each JSON object into keyword arguments, which also
    validates it: a missing or misspelled key raises TypeError here rather than surfacing as a
    confusing failure much later.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [HumanEvalFixRow(**row) for row in payload]


def write_task_dir(row: HumanEvalFixRow, dest: Path) -> Path:
    """Materialise one row as a task directory the loader can read.

    Produces exactly the layout described in the README's "Adding your own task":

        <dest>/<slug>/
        ├── task.json
        └── repo/
            ├── candidate.py        <- the buggy code
            └── test_candidate.py   <- the tests

    Called with a temp directory as `dest`, so these are created per eval run and discarded.
    """
    # "HumanEval/23" is not usable as a directory name.
    slug = row.task_id.replace("/", "-").lower()
    task_dir = Path(dest) / slug
    repo = task_dir / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    # `rstrip() + "\n"` normalises to exactly one trailing newline.
    (repo / "candidate.py").write_text(row.buggy_code.rstrip() + "\n", encoding="utf-8")
    (repo / "test_candidate.py").write_text(row.tests.rstrip() + "\n", encoding="utf-8")

    task_dir.joinpath("task.json").write_text(
        json.dumps(
            {
                "task_id": slug,
                # Runs the test file directly rather than through pytest: these tests are
                # plain asserts in a __main__ block, not pytest test functions. "-u" is
                # unbuffered output, so nothing is lost if the process is killed on timeout.
                # The loader prepends sys.executable because this starts with a flag.
                "test_command": ["-u", "test_candidate.py"],
                # Left empty: unlike the workshop fixtures, these are not pytest tests, so
                # there are no FAILED lines to name. The starts-red property comes from the
                # benchmark itself.
                "expected_failures": [],
                # More specific than the workshop prompt because there is only one file and
                # one function — naming the entry point saves the model a discovery step
                # without giving away what is wrong with it.
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
        # Imported here, not at module scope, so the whole eval package stays importable
        # without the optional dependency installed.
        from datasets import load_dataset
    except ImportError as error:
        # `raise ... from error` keeps the original traceback attached while replacing an
        # opaque ImportError with the command that fixes it.
        raise ImportError("The full benchmark needs: uv sync --extra eval") from error

    dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
    rows = [
        HumanEvalFixRow(
            task_id=item["task_id"],
            # buggy_solution is a function body only; declaration carries the def line
            # and imports, so buggy_code must be their concatenation to be valid Python.
            buggy_code=(item["declaration"] + item["buggy_solution"]).rstrip(),
            # The dataset's tests assume the function is already in scope, so an import of the
            # entry point is prepended to make the file runnable on its own.
            tests=f"from candidate import {item['entry_point']}\n\n{item['test'].rstrip()}",
            entry_point=item["entry_point"],
        )
        for item in dataset
    ]
    if sample is not None and sample < len(rows):
        # Seeded so a sampled run is reproducible: comparing two configurations is only
        # meaningful if both saw the same tasks.
        random.seed(seed)
        rows = random.sample(rows, sample)
    return rows


def dump_rows(rows: list[HumanEvalFixRow], path: Path) -> None:
    """Write rows back out as JSON — how the vendored subset was produced from the full set."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8"
    )
