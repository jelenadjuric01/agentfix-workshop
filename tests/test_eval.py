import json

from agentfix.agent.loop import AgentResult
from agentfix.eval.humanevalfix import HumanEvalFixRow, load_vendored_rows, write_task_dir
from agentfix.eval.runner import EvalReport

FIXED_CART = """from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices))
"""


def _result(task_id: str, solved: bool) -> AgentResult:
    return AgentResult(task_id, solved, 3, 100, 50, 4.2, (), peak_prompt_tokens=80)


def test_pass_at_1_is_the_solved_fraction():
    report = EvalReport("workshop", (_result("a", True), _result("b", False)))
    assert report.pass_at_1 == 0.5


def test_pass_at_1_is_zero_for_an_empty_suite():
    assert EvalReport("workshop", ()).pass_at_1 == 0.0


def test_table_shows_steps_and_tokens():
    table = EvalReport("workshop", (_result("01-shopcart", True),)).format_table()
    assert "01-shopcart" in table
    assert "150" in table  # 100 prompt + 50 completion


def test_json_round_trips():
    payload = EvalReport("workshop", (_result("a", True),)).to_json()
    assert json.loads(json.dumps(payload))["pass_at_1"] == 1.0


def test_json_carries_the_peak_prompt_size():
    """Without this the only record of how close a run came to the context ceiling is lost."""
    payload = EvalReport("workshop", (_result("a", True), _result("b", False))).to_json()

    assert payload["peak_prompt_tokens"] == 80
    assert payload["results"][0]["peak_prompt_tokens"] == 80


def test_run_suite_evaluates_a_scripted_workshop_run_end_to_end(tmp_path, monkeypatch, capsys):
    """`agentfix eval` is a documented command; this is its offline coverage."""
    from agentfix.eval import runner as eval_runner
    from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call

    from agentfix.agent.loop import MAX_STEPS

    monkeypatch.setattr(eval_runner, "RESULTS_DIR", tmp_path)
    # 01-shopcart gets fixed; 02-invoice gets prose until the step budget runs out.
    llm = FakeLLMClient(
        [
            assistant_tool_call(
                "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c1"
            ),
            assistant_tool_call("run_tests", {}, call_id="c2"),
            assistant_text("fixed the stray subtraction"),
            *[assistant_text("still thinking") for _ in range(MAX_STEPS)],
        ]
    )

    exit_code = eval_runner.run_suite("workshop", limit=2, llm=llm)

    assert exit_code == 0
    payload = json.loads((tmp_path / "workshop.json").read_text(encoding="utf-8"))
    assert payload["pass_at_1"] == 0.5
    assert [r["task_id"] for r in payload["results"]] == ["01-shopcart", "02-invoice"]
    assert payload["results"][1]["steps_used"] == MAX_STEPS
    assert "pass@1" in capsys.readouterr().out


def test_vendored_subset_has_twenty_usable_rows():
    rows = load_vendored_rows()
    assert len(rows) == 20
    assert all(row.entry_point and row.buggy_code and row.tests for row in rows)


def _add_row(buggy_code: str) -> HumanEvalFixRow:
    return HumanEvalFixRow(
        task_id="HumanEval/0",
        buggy_code=buggy_code,
        tests=(
            "from candidate import add\n\n"
            "def check(add):\n"
            "    assert add(1, 2) == 3\n\n"
            "check(add)\n"
        ),
        entry_point="add",
    )


def test_write_task_dir_produces_a_loadable_red_task(tmp_path):
    from agentfix.sandbox.subprocess_backend import SubprocessBackend
    from agentfix.tasks.loader import load_task, workspace

    row = _add_row("def add(a, b):\n    return a - b\n")
    task_dir = write_task_dir(row, tmp_path)
    task = load_task(task_dir)

    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is False


def test_write_task_dir_recognises_a_correct_fix_as_a_pass(tmp_path):
    from agentfix.sandbox.subprocess_backend import SubprocessBackend
    from agentfix.tasks.loader import load_task, workspace

    row = _add_row("def add(a, b):\n    return a + b\n")
    task_dir = write_task_dir(row, tmp_path)
    task = load_task(task_dir)

    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is True
