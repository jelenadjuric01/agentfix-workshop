"""The run_tests tool — the agent's only oracle, and the only sandboxed tool.

This is the most consequential tool in the project. `is_done` in agent/loop.py consults
`last_result` and nothing else, so a run ends successfully only because the tests actually
passed — never because the model announced it was finished.

The module is named tests_tool.py rather than tests.py so that pytest does not try to
collect it as a test module.
"""

from __future__ import annotations

from pathlib import Path

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import ToolResult


class RunTestsTool:
    """Runs the task's test command through an execution backend and remembers the result."""

    name = "run_tests"
    # The model is told outright that this is the source of truth. It is also *asked*, never
    # told: nothing hands the agent the failing test output up front, so discovering the
    # failure is part of the task.
    description = "Run the project's test suite and return the result. This is the source of truth."
    parameters = {"type": "object", "properties": {}}  # takes no arguments

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
        result = self.backend.run(self.root, self.command, timeout_s=self.timeout_s)
        self.last_result = result

        # Note `ToolResult(True, ...)` even when the tests fail: the *tool* worked. Failing
        # tests are the information the agent needs, not a tool error. The headline is
        # prepended because pytest output buries the verdict, and a 12B model reading 2,000
        # characters of traceback does better when the answer is in the first line.
        headline = "All tests passed." if result.passed else "Tests failed."
        return ToolResult(True, f"{headline}\n\n{result.output}".strip())
