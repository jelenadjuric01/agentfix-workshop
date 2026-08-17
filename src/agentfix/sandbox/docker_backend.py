"""The opt-in backend: real isolation, one throwaway container per test run.

Enabled with AGENTFIX_SANDBOX=docker, after building the image:

    docker build -t agentfix-sandbox -f Dockerfile.sandbox .

`build_argv` is a separate method from `run` so the command line can be asserted without a
Docker daemon present — most of the tests for this file check the flags rather than starting
containers, which is what keeps them runnable everywhere.
"""

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
        """The full `docker run` command line. Every flag here is asserted by a test."""
        # command[0] is the *host* interpreter path, which means nothing in the container.
        inner = ("python", *command[1:])
        if "pytest" in inner:
            # the mount is read-only, so pytest must not try to write .pytest_cache
            inner = (*inner, "-p", "no:cacheprovider")

        return [
            "docker",
            "run",
            "--rm",  # delete the container when it exits; no state survives a run
            "--name",
            name or self._container_name(),  # a known name, so a timeout can kill it
            "--network",
            "none",  # no network at all: the real difference from the subprocess backend
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "--cpus",
            "1",
            "--user",
            "runner",  # not root, even inside the container
            "--cap-drop",
            "ALL",  # drop every Linux capability
            "--security-opt",
            "no-new-privileges",  # and block regaining any via setuid binaries
            "--read-only",  # the container's own filesystem is immutable
            "--tmpfs",
            "/tmp",  # except /tmp, in memory, discarded with the container
            "--env",
            "HOME=/tmp",  # so anything writing a dotfile has somewhere to put it
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
        """A unique name per run, so concurrent runs cannot collide or kill each other."""
        return f"agentfix-{uuid.uuid4().hex[:12]}"

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult:
        # Generated here rather than inside build_argv so the same name is available below for
        # `docker kill`.
        name = self._container_name()
        argv = self.build_argv(workspace, command, name=name)
        start = time.time()

        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                # A grace margin over the caller's timeout: pulling and starting a container
                # costs time that is not the test's fault.
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

        # Note this is the *docker CLI's* exit code, which passes through the container's. A
        # missing image therefore surfaces as a failed test run with docker's error in the
        # output, rather than as an exception.
        return ExecResult(completed.returncode == 0, combined, round(time.time() - start, 3))
