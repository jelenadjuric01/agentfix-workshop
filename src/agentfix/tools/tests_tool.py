"""The run_tests tool — the agent's only oracle, and the only sandboxed tool.

This is the most consequential tool in the project. `is_done` in agent/loop.py consults
`last_result` and nothing else, so a run ends successfully only because the tests actually
passed — never because the model announced it was finished.

The module is named tests_tool.py rather than tests.py so that pytest does not try to
collect it as a test module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import ToolResult


class RunTestsTool:
    """Runs the task's test command through an execution backend and remembers the result."""

    name = "run_tests"
    # The model is told outright that this is the source of truth. It is also *asked*, never
    # told: nothing hands the agent the failing test output up front, so discovering the
    # failure is part of the task.
    description = "Run the project's test suite and return the result. This is the source of truth."
    # TODO(stage-1): the JSON Schema the model sees. run_tests needs no arguments.
    parameters: dict[str, Any] = {}

    def __init__(
        self,
        root: Path,
        command: tuple[str, ...],
        backend: ExecutionBackend,
        timeout_s: int = 10,
    ) -> None:
        self.root = root
        self.command = command
        # Injected rather than constructed here, so tests can pass a fake backend and the
        # subprocess/docker choice stays the caller's (runner.py calls get_backend()).
        self.backend = backend
        self.timeout_s = timeout_s

        # The one piece of mutable state in the tool layer, and the agent's whole verdict.
        # Annotated because the initial value is None while the attribute will later hold an
        # ExecResult: without the annotation a type checker infers the type as just None and
        # rejects the assignment in `run` below.
        self.last_result: ExecResult | None = None

    def invalidate(self) -> None:
        """The workspace changed, so the last test run is no longer evidence about it.

        Wired up in runner.py as `WriteFileTool(work_dir, on_write=run_tests.invalidate)`.
        Without it an agent could run the tests, see them pass, then write a file that breaks
        them — and `is_done` would still see the stale green result and report SOLVED.
        """
        self.last_result = None

    def run(self) -> ToolResult:
        # One contract note, because getting it wrong produces a very confusing agent:
        # failing tests are not a tool *error*. The tool worked; what it reports is that the
        # tests are red. That information is exactly what the agent needs in order to act.
        # TODO(stage-1): run the tests via self.backend, store self.last_result,
        # and return a ToolResult whose content tells the model what happened.
        raise NotImplementedError("stage 1: implement RunTestsTool.run")
