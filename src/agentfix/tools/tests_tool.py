from __future__ import annotations

from pathlib import Path

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import ToolResult


class RunTestsTool:
    name = "run_tests"
    description = "Run the project's test suite and return the result. This is the source of truth."
    parameters = {"type": "object", "properties": {}}

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

    def run(self) -> ToolResult:
        result = self.backend.run(self.root, self.command, timeout_s=self.timeout_s)
        self.last_result = result

        headline = "All tests passed." if result.passed else "Tests failed."
        return ToolResult(True, f"{headline}\n\n{result.output}".strip())
