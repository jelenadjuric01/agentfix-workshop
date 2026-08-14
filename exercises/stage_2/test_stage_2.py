"""Stage 2 — dispatch tool calls in the loop and feed observations back."""

from pathlib import Path

from agentfix.agent.loop import run_agent
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool

FIXTURE = Path("tasks/workshop/01-shopcart")


def _build(work_dir, task):
    run_tests = RunTestsTool(work_dir, task.test_command, SubprocessBackend(), timeout_s=30)
    registry = ToolRegistry(
        [ListFilesTool(work_dir), ReadFileTool(work_dir), WriteFileTool(work_dir), run_tests]
    )
    return registry, run_tests


def test_tool_result_is_appended_as_a_tool_message():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="c1"), assistant_text("ok")]
        )
        run_agent(task, work_dir, llm, registry, run_tests, max_steps=2)

    observation = llm.calls[1][-1]
    assert observation["role"] == "tool", "the observation must go back as a tool message"
    assert "shopcart/cart.py" in observation["content"]


def test_tool_call_id_is_carried_back():
    """Omitting tool_call_id is the single most common mistake here."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="xyz789"), assistant_text("ok")]
        )
        run_agent(task, work_dir, llm, registry, run_tests, max_steps=2)

    assert llm.calls[1][-1].get("tool_call_id") == "xyz789"


def test_history_is_append_only():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="c1"), assistant_text("ok")]
        )
        run_agent(task, work_dir, llm, registry, run_tests, max_steps=2)

    first, second = llm.calls
    assert (
        second[: len(first)] == first
    ), "earlier messages must never change — it kills the KV cache"


def test_the_loop_keeps_going_after_a_tool_call():
    """A `break` after dispatching would stop at step 1. Dispatch, then loop back."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
                assistant_text("done"),
            ]
        )
        result = run_agent(task, work_dir, llm, registry, run_tests, max_steps=3)

    assert result.steps_used == 3
