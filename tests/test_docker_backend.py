from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from agentfix.sandbox.docker_backend import DEFAULT_IMAGE, DockerBackend

# ---------------------------------------------------------------------------
# `build_argv` is a pure function. Everything that asserts on the isolation flags
# lives ABOVE the daemon skip marker, because a machine with no Docker daemon must
# still fail loudly if someone deletes `--network none`.
# ---------------------------------------------------------------------------

PYTEST_COMMAND = (sys.executable, "-m", "pytest", "-q")


def _argv(tmp_path, command=PYTEST_COMMAND) -> list[str]:
    return DockerBackend().build_argv(tmp_path, command, name="agentfix-test")


def test_command_is_passed_through_without_sys_executable(tmp_path):
    """The host's sys.executable path is meaningless inside the container."""
    argv = _argv(tmp_path)
    assert sys.executable not in argv
    assert "python" in argv


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--network", "none"),
        ("--memory", "512m"),
        ("--pids-limit", "128"),
        ("--cpus", "1"),
        ("--user", "runner"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--tmpfs", "/tmp"),
        ("--workdir", "/work"),
        ("--name", "agentfix-test"),
    ],
)
def test_every_isolation_flag_is_present_with_its_value(tmp_path, flag, value):
    argv = _argv(tmp_path)
    assert flag in argv, f"{flag} is missing — the sandbox is weaker than the docs claim"
    assert argv[argv.index(flag) + 1] == value


def test_the_container_filesystem_and_the_mount_are_both_read_only(tmp_path):
    argv = _argv(tmp_path)
    assert "--read-only" in argv
    assert f"{tmp_path}:/work:ro" in argv


def test_pytest_runs_without_its_cache_because_the_mount_is_read_only(tmp_path):
    argv = _argv(tmp_path)
    assert argv[-2:] == ["-p", "no:cacheprovider"]


def test_a_plain_script_command_is_not_given_pytest_flags(tmp_path):
    """HumanEvalFix tasks run `python -u test_candidate.py`, which has no -p option."""
    argv = _argv(tmp_path, (sys.executable, "-u", "test_candidate.py"))
    assert "no:cacheprovider" not in argv
    assert argv[-3:] == ["python", "-u", "test_candidate.py"]


def test_each_run_gets_a_distinct_container_name(tmp_path):
    backend = DockerBackend()
    first = backend.build_argv(tmp_path, PYTEST_COMMAND)
    second = backend.build_argv(tmp_path, PYTEST_COMMAND)
    assert first[first.index("--name") + 1] != second[second.index("--name") + 1]


# ---------------------------------------------------------------------------
# Below here a live daemon is required.
# ---------------------------------------------------------------------------


def _docker_unavailable_reason() -> str | None:
    if shutil.which("docker") is None:
        return "docker not installed"
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return "docker daemon not available"
    if result.returncode != 0:
        return "docker daemon not available"

    images = subprocess.run(
        ["docker", "images", "-q", DEFAULT_IMAGE], capture_output=True, text=True, check=False
    )
    if not images.stdout.strip():
        return (
            f"{DEFAULT_IMAGE} not built (docker build -t agentfix-sandbox -f Dockerfile.sandbox .)"
        )
    return None


_SKIP_REASON = _docker_unavailable_reason()

requires_docker = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


@requires_docker
def test_passing_tests_report_passed(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"), timeout_s=30)
    assert result.passed is True, result.output


@requires_docker
def test_failing_tests_report_failed(tmp_path):
    (tmp_path / "test_no.py").write_text("def test_no():\n    assert False\n", encoding="utf-8")
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"), timeout_s=30)
    assert result.passed is False
    assert "test_no" in result.output


@requires_docker
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
    assert result.passed is True, result.output


@requires_docker
def test_the_workspace_cannot_be_written_from_inside_the_container(tmp_path):
    (tmp_path / "test_ro.py").write_text(
        "import pytest\n"
        "def test_ro():\n"
        "    with pytest.raises(OSError):\n"
        "        open('escape.txt', 'w').write('x')\n",
        encoding="utf-8",
    )
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"), timeout_s=30)
    assert result.passed is True, result.output
    assert not (tmp_path / "escape.txt").exists()


@requires_docker
def test_a_hanging_test_times_out_and_leaves_no_container_behind(tmp_path):
    (tmp_path / "test_hang.py").write_text(
        "import time\ndef test_hang():\n    time.sleep(600)\n", encoding="utf-8"
    )
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"), timeout_s=1)

    assert result.timed_out is True
    listing = subprocess.run(
        ["docker", "ps", "--filter", "name=agentfix-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert listing.stdout.strip() == ""
