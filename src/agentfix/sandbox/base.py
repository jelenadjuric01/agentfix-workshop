"""How test commands get executed, and the switch between the two ways of doing it.

Only ONE tool in this project runs through a backend: `run_tests`. The file tools read and
write directly on the host. The reason is that a sandbox exists to contain *code
execution*, and writing bytes to a .py file executes nothing — those bytes stay inert
until pytest imports them. `run_tests` is the moment model-authored code actually runs, so
that is where the boundary goes.

A useful consequence: because the file tools write host-side, the container in
docker_backend.py can mount the workspace read-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one test run. `passed` is the agent's only source of truth."""

    passed: bool  # exit code was 0
    output: str  # stdout and stderr combined, already truncated
    duration_s: float
    timed_out: bool = False


class ExecutionBackend(Protocol):
    """One method: run this command in this directory. See llm/types.py on Protocol.

    Two implementations satisfy this without inheriting from it — `SubprocessBackend` and
    `DockerBackend`. Because they are interchangeable, the ~20 tests that assert on the
    Docker command line can run with no Docker daemon present.
    """

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult: ...


def get_backend(name: str | None = None) -> ExecutionBackend:
    """Pick a backend: the argument, else $AGENTFIX_SANDBOX, else subprocess.

    Note what the two backends actually promise, because "sandbox" oversells the default:

    subprocess (default) — hardened, not isolated. Same machine, same user, same
        filesystem. Adds a stripped environment, resource limits and a timeout. Test code
        can still open sockets and read anything you can read.
    docker (opt-in)      — real isolation: no network, read-only mount, dropped
        capabilities, memory and pid caps. Needs an image built first.

    The imports are inside the branches so that importing this module never requires
    Docker, and so an unused backend is never loaded.
    """
    choice = (name or os.environ.get("AGENTFIX_SANDBOX") or "subprocess").lower()

    if choice == "subprocess":
        from agentfix.sandbox.subprocess_backend import SubprocessBackend

        return SubprocessBackend()

    if choice == "docker":
        from agentfix.sandbox.docker_backend import DockerBackend

        return DockerBackend()

    # Fail loudly on a typo. Silently falling back to the weaker backend would mean someone
    # who asked for isolation ran without it and never found out.
    raise ValueError(f"Unknown AGENTFIX_SANDBOX={choice!r}; expected 'subprocess' or 'docker'")
