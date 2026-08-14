from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agentfix.sandbox.base import ExecResult
from agentfix.sandbox.subprocess_backend import TRUNCATION_MARKER

DEFAULT_IMAGE = "agentfix-sandbox:latest"


class DockerBackend:
    """Runs tests in a throwaway container with no network and hard resource caps."""

    def __init__(self, image: str = DEFAULT_IMAGE, max_output_chars: int = 2000) -> None:
        self.image = image
        self.max_output_chars = max_output_chars

    def build_argv(self, workspace: Path, command: tuple[str, ...]) -> list[str]:
        inner = ("python", *command[1:])

        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "--cpus",
            "1",
            "--user",
            "runner",
            "--volume",
            f"{workspace}:/work",
            "--workdir",
            "/work",
            self.image,
            *inner,
        ]

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult:
        argv = self.build_argv(workspace, command)
        start = time.time()

        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s + 10,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                False, f"TIMEOUT after {timeout_s}s", round(time.time() - start, 3), True
            )

        combined = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if len(combined) > self.max_output_chars:
            combined = combined[: self.max_output_chars] + TRUNCATION_MARKER

        return ExecResult(completed.returncode == 0, combined, round(time.time() - start, 3))
