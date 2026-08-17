"""Stage 3 — decide when the agent is actually done."""

import sys
from pathlib import Path

from agentfix.agent.loop import is_done, run_agent
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool

FIXTURE = Path("tasks/workshop/01-shopcart")
FIXED_CART = """from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices))
"""


def _build(work_dir, task):
    run_tests = RunTestsTool(work_dir, task.test_command, SubprocessBackend(), timeout_s=30)
    registry = ToolRegistry(
        [
            ListFilesTool(work_dir),
            ReadFileTool(work_dir),
            WriteFileTool(work_dir, on_write=run_tests.invalidate),
            run_tests,
        ]
    )
    return registry, run_tests


def test_not_done_before_the_tests_have_ever_run(tmp_path):
    run_tests = RunTestsTool(tmp_path, (sys.executable, "-m", "pytest", "-q"), SubprocessBackend())
    assert is_done(run_tests) is False


def test_not_done_when_the_model_only_claims_success():
    """If your is_done trusts the model's word, this test fails. That is the lesson."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_text("DONE. I have fixed the bug. All tests pass now."),
            ]
        )
        result = run_agent(task, llm, registry, run_tests, max_steps=2)

    assert result.solved is False


def test_done_only_once_the_tests_actually_pass():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c1"
                ),
                assistant_tool_call("run_tests", {}, call_id="c2"),
                assistant_text("fixed"),
            ]
        )
        result = run_agent(task, llm, registry, run_tests)

    assert result.solved is True
