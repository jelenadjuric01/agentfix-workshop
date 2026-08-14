from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from agentfix.sandbox.docker_backend import DockerBackend


def _docker_unavailable_reason() -> str | None:
    if shutil.which("docker") is None:
        return "docker not installed"
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "docker daemon not available"
    if result.returncode != 0:
        return "docker daemon not available"
    return None


_SKIP_REASON = _docker_unavailable_reason()

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def test_passing_tests_report_passed(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"))
    assert result.passed is True


def test_network_is_unavailable_inside_the_container(tmp_path):
    (tmp_path / "test_net.py").write_text(
        "import socket\n"
        "def test_net():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('network was reachable')\n",
        encoding="utf-8",
    )
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"), timeout_s=30)
    assert result.passed is True


def test_command_is_passed_through_without_sys_executable(tmp_path):
    """The host's sys.executable path is meaningless inside the container."""
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    backend = DockerBackend()
    argv = backend.build_argv(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert sys.executable not in argv
    assert "python" in argv
