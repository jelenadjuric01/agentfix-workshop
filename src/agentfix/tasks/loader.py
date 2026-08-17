"""What a "bug to fix" is, and how each run gets a clean copy of it.

A task on disk is a directory holding two things (see the README section "Adding your own
task"):

    tasks/workshop/01-shopcart/
    ├── task.json     <- metadata: what to run, what should fail, what to tell the model
    └── repo/         <- the buggy project; this is the agent's entire world

`load_task` reads the metadata. `workspace` copies `repo/` somewhere disposable so the
agent can write freely without ever touching the pristine original.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Fallback for a hand-written task.json that omits "prompt". Every task shipped in this
# repo supplies its own, so in practice this constant is never used.
#
# Note how little it says: not which test fails, not which file is wrong. That is
# deliberate — the agent has to discover the failure by running the tests. Handing it the
# failure up front would test a different, much easier skill. The success criterion
# ("not finished until the tests pass") lives in the system prompt in agent/loop.py.
DEFAULT_PROMPT = "The test suite for this project is failing. Find the bug and fix it."


@dataclass(frozen=True)
class Task:
    """One bug-fixing task, loaded from disk and never mutated afterwards."""

    task_id: str  # label used in output and in the temp directory name
    root: Path  # the task directory itself (where task.json lives)
    template_dir: Path  # root/repo — the pristine project, only ever read from
    test_command: tuple[str, ...]  # argv to run, from inside the workspace copy
    expected_failures: tuple[str, ...]  # test names that must be red before the agent starts
    prompt: str  # the first user message the model sees


def load_task(task_dir: Path) -> Task:
    """Read task.json into a Task, filling in defaults for anything it leaves out."""
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))

    # `meta.get(key, default)` throughout: every field is optional, so a minimal task.json
    # containing only {"task_id": "..."} still loads.
    command = tuple(meta.get("test_command", ["-m", "pytest", "-q"]))

    # A command starting with a flag ("-m", "-u") is meant for a Python interpreter but does
    # not name one. Prepending sys.executable pins it to *this* interpreter — the project's
    # virtualenv — rather than whatever "python" happens to be on PATH inside the sandbox.
    # (The Docker backend replaces this host path with the container's own python; see
    # sandbox/docker_backend.py.)
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
    """Copy the pristine template into a temp dir so every run starts identical.

    Used as:

        with workspace(task) as work_dir:
            ...          # work_dir is a throwaway copy the agent may write to
        # on leaving the block the copy is deleted, however the block ended

    `@contextmanager` turns a generator function into something usable with `with`, as a
    shorter alternative to writing a class with `__enter__` and `__exit__`. The `yield` is
    the boundary: everything before it is the setup (`__enter__`) and the value it yields
    is what `as work_dir` receives; everything in the `finally` is the teardown
    (`__exit__`). The generator must yield exactly once.

    The `try`/`finally` is load-bearing, not stylistic. If the body raises — a tool
    crashes, the model loops forever, you press Ctrl-C — the exception is thrown back into
    the generator at the `yield`, the `finally` still runs, and the temp directory is still
    deleted. Without it, every failed run would leak a copy of the project.

    Copying per run is what makes the agent safe to let loose: it rewrites whole files, and
    `tasks/` must be byte-identical afterwards so the next run starts from the same bug.
    """
    temp_root = Path(tempfile.mkdtemp(prefix=f"agentfix_{task.task_id}_"))
    work_dir = temp_root / "repo"
    try:
        shutil.copytree(task.template_dir, work_dir)
        yield work_dir
    finally:
        # ignore_errors=True: cleanup must never mask the real error from the block above.
        shutil.rmtree(temp_root, ignore_errors=True)
