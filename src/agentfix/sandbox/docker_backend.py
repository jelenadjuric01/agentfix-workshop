from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from agentfix.sandbox.base import ExecResult
from agentfix.sandbox.subprocess_backend import TRUNCATION_MARKER

DEFAULT_IMAGE = "agentfix-sandbox:latest"


class DockerBackend:
    """Runs tests in a throwaway container with no network and hard resource caps."""

    def __init__(self, image: str = DEFAULT_IMAGE, max_output_chars: int = 2000) -> None:
        self.image = image
        self.max_output_chars = max_output_chars

    def build_argv(
        self, workspace: Path, command: tuple[str, ...], name: str | None = None
    ) -> list[str]:
        # command[0] is the *host* interpreter path, which means nothing in the container.
        inner = ("python", *command[1:])
        if "pytest" in inner:
            # the mount is read-only, so pytest must not try to write .pytest_cache
            inner = (*inner, "-p", "no:cacheprovider")

        return [
            "docker",
            "run",
            "--rm",
            "--name",
            name or self._container_name(),
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
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--volume",
            # the filesystem tools write on the HOST, so the container never needs to
            f"{workspace}:/work:ro",
            "--workdir",
            "/work",
            self.image,
            *inner,
        ]

    @staticmethod
    def _container_name() -> str:
        return f"agentfix-{uuid.uuid4().hex[:12]}"

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult:
        name = self._container_name()
        argv = self.build_argv(workspace, command, name=name)
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
            # killing the docker CLI leaves the container running; kill it by name.
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            return ExecResult(
                False, f"TIMEOUT after {timeout_s}s", round(time.time() - start, 3), True
            )

        combined = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if len(combined) > self.max_output_chars:
            combined = combined[: self.max_output_chars] + TRUNCATION_MARKER

        return ExecResult(completed.returncode == 0, combined, round(time.time() - start, 3))
