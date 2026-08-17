import sys
from pathlib import Path

from agentfix.agent.loop import AgentResult, is_done, run_agent
from agentfix.agent.trace import Tracer
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
BROKEN_CART = FIXED_CART.replace("return sum(prices)", "return sum(prices) - 1")


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


def test_is_done_is_false_before_tests_run(tmp_path):
    run_tests = RunTestsTool(tmp_path, (sys.executable, "-m", "pytest", "-q"), SubprocessBackend())
    assert is_done(run_tests) is False


def test_agent_solves_the_task_when_it_writes_the_fix():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}, call_id="c2"),
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c3"
                ),
                assistant_tool_call("run_tests", {}, call_id="c4"),
                assistant_text("Fixed the stray subtraction."),
            ]
        )

        result = run_agent(task, llm, registry, run_tests)

    assert isinstance(result, AgentResult)
    assert result.solved is True
    assert result.steps_used == 5


def test_agent_is_not_solved_when_the_model_merely_claims_success():
    """The whole point of stage 3: assertions are not verification."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_text("DONE — I have fixed the bug."),
            ]
        )

        result = run_agent(task, llm, registry, run_tests, max_steps=2)

    assert result.solved is False


def test_a_pass_followed_by_a_breaking_write_is_not_solved():
    """`is_done` must not trust a green run that a later write invalidated."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c1"
                ),
                assistant_tool_call("run_tests", {}, call_id="c2"),
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": BROKEN_CART}, call_id="c3"
                ),
                assistant_text("all done"),
            ]
        )

        result = run_agent(task, llm, registry, run_tests, max_steps=4)

    assert result.solved is False


def test_the_agent_cannot_pass_by_rewriting_the_tests():
    """`run_tests` is the only oracle, so deleting the specification must not count as a fix."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call(
                    "write_file",
                    {"path": "tests/test_cart.py", "content": "def test_ok():\n    assert True\n"},
                    call_id="c1",
                ),
                assistant_tool_call("run_tests", {}, call_id="c2"),
                assistant_text("done"),
            ]
        )

        result = run_agent(task, llm, registry, run_tests, max_steps=3)

    assert result.solved is False
    refusal = llm.calls[1][-1]["content"]
    assert "tests are the specification" in refusal.lower()


def test_a_text_only_reply_does_not_end_the_run_while_the_tests_still_fail():
    """The stop condition is verification, not the model running out of things to say."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_text("The bug is the stray subtraction in cart.py."),
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c2"
                ),
                assistant_tool_call("run_tests", {}, call_id="c3"),
                assistant_text("fixed"),
            ]
        )

        result = run_agent(task, llm, registry, run_tests)

    assert result.solved is True
    assert result.steps_used == 5, "the diagnosis-only turn must not have ended the run"
    nudge = llm.calls[2][-1]
    assert nudge["role"] == "user"
    assert "have not passed" in nudge["content"]


def test_repeated_identical_calls_abandon_the_run():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("run_tests", {}, call_id=f"c{i}") for i in range(6)]
        )

        result = run_agent(task, llm, registry, run_tests)

    assert result.steps_used == 4, "one real call plus MAX_GUARD_HITS repeats, then abandon"
    assert result.solved is False


def test_the_guarded_call_is_recorded_in_the_trace():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
            ]
        )
        tracer = Tracer()

        run_agent(task, llm, registry, run_tests, max_steps=2, tracer=tracer)

    assert [event.kind for event in tracer.events] == ["llm", "tool", "llm", "tool"]
    assert "guarded" in tracer.events[-1].detail


def test_peak_prompt_tokens_is_the_largest_single_prompt():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("list_files", {}, call_id="c1", prompt_tokens=120),
                assistant_text("ok", prompt_tokens=90),
            ]
        )

        result = run_agent(task, llm, registry, run_tests, max_steps=2)

    assert result.peak_prompt_tokens == 120
    assert result.prompt_tokens == 210


def test_history_is_append_only_and_carries_tool_call_ids():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("run_tests", {}, call_id="c1"), assistant_text("giving up")]
        )
        run_agent(task, llm, registry, run_tests, max_steps=2)

    first_history, second_history = llm.calls
    assert second_history[: len(first_history)] == first_history
    assert second_history[-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "run_tests",
        "content": second_history[-1]["content"],
    }


def test_step_budget_is_a_hard_cap():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(2)]
        )

        result = run_agent(task, llm, registry, run_tests, max_steps=2)

    assert result.steps_used == 2
    assert result.solved is False


def test_repeated_identical_call_is_guarded_instead_of_re_executed():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}, call_id="c1"),
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}, call_id="c2"),
                assistant_text("stuck"),
            ]
        )
        tracer = Tracer()

        run_agent(task, llm, registry, run_tests, max_steps=3, tracer=tracer)

    third_history = llm.calls[2]
    assert "already called" in third_history[-1]["content"]


def test_trace_records_every_llm_and_tool_event():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="c1"), assistant_text("ok")]
        )
        tracer = Tracer()

        result = run_agent(task, llm, registry, run_tests, max_steps=2, tracer=tracer)

    kinds = [event.kind for event in tracer.events]
    assert kinds == ["llm", "tool", "llm"]
    assert result.trace == tuple(tracer.events)
