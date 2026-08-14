from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from agentfix.sandbox.base import ExecResult

TRUNCATION_MARKER = "\n[...truncated]"

# Currently unused: RLIMIT_AS with this value breaks subprocess startup on
# macOS. Kept for the controller's decision; see _apply_limits docstring.
MAX_ADDRESS_SPACE_BYTES = 2 * 1024**3
MAX_CPU_SECONDS = 30
MAX_FILE_SIZE_BYTES = 16 * 1024**2
MAX_PROCESSES = 64


def _apply_limits() -> None:
    """Constrain the child process. POSIX only; a no-op elsewhere.

    RLIMIT_AS is deliberately NOT set here: on macOS (verified empirically on
    Apple Silicon, Python 3.12) capping RLIMIT_AS to 2 GiB makes every child
    process fail at interpreter startup with
    `subprocess.SubprocessError: Exception occurred in preexec_fn.`, because
    CPython reserves a large virtual address space up front. That would make
    every sandboxed run report a false failure rather than enforce a memory
    limit. See task-3-report.md for the reproduction. Flagged for the
    controller to rule on (e.g. platform-conditional limit, or a different
    enforcement mechanism) rather than silently dropped.
    """
    try:
        import resource
    except ImportError:  # Windows
        return

    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    except (ValueError, OSError):
        pass


class SubprocessBackend:
    """Runs tests in a child process with resource limits and a stripped environment."""

    def __init__(self, max_output_chars: int = 2000) -> None:
        self.max_output_chars = max_output_chars

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult:
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(workspace)}
        start = time.time()

        try:
            completed = subprocess.run(
                list(command),
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_apply_limits if hasattr(os, "fork") else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                passed=False,
                output=f"TIMEOUT after {timeout_s}s",
                duration_s=round(time.time() - start, 3),
                timed_out=True,
            )

        combined = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return ExecResult(
            passed=completed.returncode == 0,
            output=self._truncate(combined),
            duration_s=round(time.time() - start, 3),
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + TRUNCATION_MARKER
