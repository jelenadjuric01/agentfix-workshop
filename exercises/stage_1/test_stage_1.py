"""Stage 1 — write the run_tests tool and its JSON schema."""

import sys
from pathlib import Path

from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.base import ToolRegistry
from agentfix.tools.tests_tool import RunTestsTool

PYTEST_CMD = (sys.executable, "-m", "pytest", "-q")


def test_tool_declares_a_valid_schema():
    tool = RunTestsTool(Path("."), PYTEST_CMD, SubprocessBackend())
    assert tool.name == "run_tests"
    assert tool.description, "the model chooses tools by their description — keep it"
    assert tool.parameters["type"] == "object"


def test_schema_is_exported_to_the_model():
    registry = ToolRegistry([RunTestsTool(Path("."), PYTEST_CMD, SubprocessBackend())])
    names = [schema["function"]["name"] for schema in registry.schemas()]
    assert "run_tests" in names


def test_running_failing_tests_reports_failure(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 1 == 2\n", encoding="utf-8")
    tool = RunTestsTool(tmp_path, PYTEST_CMD, SubprocessBackend())

    result = tool.run()

    assert result.ok is True, "the tool ran, so ok is True even when the tests fail"
    assert tool.last_result.passed is False, "the tests failed, so last_result.passed is False"


def test_running_passing_tests_reports_success(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    tool = RunTestsTool(tmp_path, PYTEST_CMD, SubprocessBackend())
    tool.run()
    assert tool.last_result.passed is True


def test_the_model_chooses_this_tool_when_told_tests_fail():
    """A schema the model cannot understand is a schema it will not call."""
    llm = FakeLLMClient([assistant_tool_call("run_tests", {}), assistant_text("done")])
    registry = ToolRegistry([RunTestsTool(Path("."), PYTEST_CMD, SubprocessBackend())])

    reply = llm.chat([{"role": "user", "content": "tests fail"}], tools=registry.schemas())

    assert reply.tool_calls[0].name == "run_tests"
