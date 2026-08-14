import sys

from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.tests_tool import RunTestsTool

PYTEST_CMD = (sys.executable, "-m", "pytest", "-q")


def _tool(root):
    return RunTestsTool(root=root, command=PYTEST_CMD, backend=SubprocessBackend())


def test_reports_failure_and_stores_last_result(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 1 == 2\n", encoding="utf-8")
    tool = _tool(tmp_path)

    result = tool.run()

    assert result.ok is True  # the tool ran successfully...
    assert "FAILED" in result.content or "failed" in result.content
    assert tool.last_result is not None
    assert tool.last_result.passed is False  # ...but the tests did not pass


def test_reports_success(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    tool = _tool(tmp_path)

    result = tool.run()

    assert "passed" in result.content.lower()
    assert tool.last_result.passed is True
