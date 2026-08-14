from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    output: str
    duration_s: float
    timed_out: bool = False


class ExecutionBackend(Protocol):
    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult: ...


def get_backend(name: str | None = None) -> ExecutionBackend:
    choice = (name or os.environ.get("AGENTFIX_SANDBOX") or "subprocess").lower()

    if choice == "subprocess":
        from agentfix.sandbox.subprocess_backend import SubprocessBackend

        return SubprocessBackend()

    if choice == "docker":
        from agentfix.sandbox.docker_backend import DockerBackend

        return DockerBackend()

    raise ValueError(f"Unknown AGENTFIX_SANDBOX={choice!r}; expected 'subprocess' or 'docker'")
