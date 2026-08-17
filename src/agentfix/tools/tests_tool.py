from __future__ import annotations

from pathlib import Path
from typing import Any

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import ToolResult


class RunTestsTool:
    name = "run_tests"
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
        self.backend = backend
        self.timeout_s = timeout_s
        self.last_result: ExecResult | None = None

    def invalidate(self) -> None:
        """The workspace changed, so the last test run is no longer evidence about it."""
        self.last_result = None

    def run(self) -> ToolResult:
        # TODO(stage-1): run the tests via self.backend, store self.last_result,
        # and return a ToolResult whose content tells the model what happened.
        raise NotImplementedError("stage 1: implement RunTestsTool.run")
