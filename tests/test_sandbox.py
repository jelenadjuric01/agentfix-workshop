import sys
from pathlib import Path

import pytest

from agentfix.sandbox import subprocess_backend as sb
from agentfix.sandbox.base import get_backend
from agentfix.sandbox.docker_backend import DockerBackend
from agentfix.sandbox.subprocess_backend import SubprocessBackend


def _write(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / name).write_text(body, encoding="utf-8")


def test_passing_tests_report_passed(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert 1 + 1 == 2\n")
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert result.passed is True
    assert result.timed_out is False


def test_failing_tests_report_failure_with_output(tmp_path):
    _write(tmp_path, "test_bad.py", "def test_bad():\n    assert 1 == 2\n")
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert result.passed is False
    assert "test_bad" in result.output


def test_infinite_loop_is_killed_by_timeout(tmp_path):
    _write(tmp_path, "test_hang.py", "def test_hang():\n    while True:\n        pass\n")
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"), timeout_s=3)
    assert result.timed_out is True
    assert result.passed is False
    assert "TIMEOUT" in result.output


def test_output_is_truncated_with_marker(tmp_path):
    _write(tmp_path, "test_loud.py", "def test_loud():\n    print('x' * 50000)\n    assert False\n")
    result = SubprocessBackend(max_output_chars=500).run(
        tmp_path, (sys.executable, "-m", "pytest", "-q")
    )
    assert len(result.output) < 800
    assert "[...truncated]" in result.output


def test_secrets_in_parent_env_are_not_visible_to_executed_code(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
    _write(
        tmp_path,
        "test_env.py",
        "import os\ndef test_env():\n    assert os.environ.get('MY_SECRET_TOKEN') is None\n",
    )
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert result.passed is True


def test_invalid_utf8_output_does_not_crash_the_backend(tmp_path):
    _write(
        tmp_path,
        "test_binary.py",
        "import sys\n"
        "def test_binary():\n"
        "    sys.stdout.buffer.write(b'\\xff\\xfe not valid utf-8')\n"
        "    sys.stdout.buffer.flush()\n"
        "    assert True\n",
    )
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q", "-s"))
    assert result.passed is True
    assert result.timed_out is False


def test_get_backend_defaults_to_subprocess(monkeypatch):
    monkeypatch.delenv("AGENTFIX_SANDBOX", raising=False)
    assert isinstance(get_backend(), SubprocessBackend)


def test_get_backend_returns_docker_backend(monkeypatch):
    monkeypatch.setenv("AGENTFIX_SANDBOX", "docker")
    assert isinstance(get_backend(), DockerBackend)


def test_get_backend_raises_on_unknown_value(monkeypatch):
    monkeypatch.setenv("AGENTFIX_SANDBOX", "bogus")
    with pytest.raises(ValueError):
        get_backend()


def test_apply_limits_sets_rlimit_as_only_on_linux(monkeypatch):
    class FakeResource:
        RLIMIT_AS = "AS"
        RLIMIT_CPU = "CPU"
        RLIMIT_FSIZE = "FSIZE"
        RLIMIT_NPROC = "NPROC"

        def __init__(self):
            self.calls = []

        def setrlimit(self, which, limits):
            self.calls.append(which)

    fake_resource = FakeResource()
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    monkeypatch.setattr(sb.sys, "platform", "linux")
    sb._apply_limits()
    assert "AS" in fake_resource.calls

    fake_resource.calls.clear()
    monkeypatch.setattr(sb.sys, "platform", "darwin")
    sb._apply_limits()
    assert "AS" not in fake_resource.calls
    assert {"CPU", "FSIZE", "NPROC"}.issubset(fake_resource.calls)
