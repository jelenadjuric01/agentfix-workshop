import sys
from pathlib import Path

from agentfix.sandbox.base import get_backend
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


def test_get_backend_defaults_to_subprocess(monkeypatch):
    monkeypatch.delenv("AGENTFIX_SANDBOX", raising=False)
    assert isinstance(get_backend(), SubprocessBackend)
