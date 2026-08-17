"""The default backend: a hardened child process. Not a security boundary.

Be clear about what this does and does not give you. It runs the tests in a separate process
with a stripped environment, resource limits and a timeout, which stops runaway resource use
and accidental damage. It does NOT isolate the filesystem or the network: test code runs as
your user, on your machine, and can open sockets or read anything you can read.

That is an acceptable trade for a workshop where a local 12B model fixes arithmetic bugs. If
you need real isolation, use the Docker backend (AGENTFIX_SANDBOX=docker).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from agentfix.sandbox.base import ExecResult

TRUNCATION_MARKER = "\n[...truncated]"

# Caps chosen to stop a runaway test (an infinite loop allocating memory, a fork bomb)
# without interfering with a normal pytest run of a few small files.
MAX_ADDRESS_SPACE_BYTES = 2 * 1024**3
MAX_CPU_SECONDS = 30
MAX_FILE_SIZE_BYTES = 16 * 1024**2
MAX_PROCESSES = 64


def _apply_limits() -> None:
    """Constrain the child process. POSIX only; a no-op elsewhere.

    Runs in the child after fork but before the command is executed, so the limits apply to
    the test process and never to the agent itself. See `preexec_fn` below.
    """
    try:
        import resource
    except ImportError:  # Windows
        return

    # RLIMIT_AS is Linux-only: on macOS/Apple Silicon it aborts interpreter
    # startup entirely (CPython reserves a large virtual address space up
    # front), so it is only safe to apply on Linux, which is also where the
    # workshop's memory-constrained student environments actually run.
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES))

    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES))
    try:
        # Not permitted for every user on every platform, and a process-count cap is the
        # least important of these limits — so failing to set it must not abort the run.
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    except (ValueError, OSError):
        pass


class SubprocessBackend:
    """Runs tests in a child process with resource limits and a stripped environment."""

    def __init__(self, max_output_chars: int = 2000) -> None:
        self.max_output_chars = max_output_chars

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult:
        # A deliberately minimal environment, replacing the parent's rather than extending it,
        # so nothing leaks in: no API keys, no PYTHONPATH that could shadow the project's own
        # modules, no locale surprises. HOME points at the workspace so anything that writes a
        # dotfile writes it somewhere disposable.
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(workspace)}
        start = time.time()

        try:
            completed = subprocess.run(
                list(command),
                # Run from inside the workspace, so relative imports in the task's own tests
                # resolve against the copy rather than the repo.
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                # Test output can contain anything; a decode error must not crash the agent.
                errors="replace",
                timeout=timeout_s,
                # `preexec_fn` runs in the forked child before exec. Guarded on os.fork
                # because it does not exist on Windows.
                preexec_fn=_apply_limits if hasattr(os, "fork") else None,
                # check=False: a non-zero exit means "tests failed", which is a normal,
                # expected outcome here — not an exception.
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Reported as a normal result with timed_out=True rather than raised: a hanging
            # test is information the agent can act on.
            return ExecResult(
                passed=False,
                output=f"TIMEOUT after {timeout_s}s",
                duration_s=round(time.time() - start, 3),
                timed_out=True,
            )

        # stdout and stderr merged because pytest splits failure information across both, and
        # the model needs to read it as one narrative.
        combined = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return ExecResult(
            # The entire definition of success in this project: exit code 0.
            passed=completed.returncode == 0,
            output=self._truncate(combined),
            duration_s=round(time.time() - start, 3),
        )

    def _truncate(self, text: str) -> str:
        """Keep the head of the output. pytest puts the failure summary first."""
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + TRUNCATION_MARKER
