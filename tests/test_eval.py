import json

from agentfix.agent.loop import AgentResult
from agentfix.eval.humanevalfix import HumanEvalFixRow, load_vendored_rows, write_task_dir
from agentfix.eval.runner import EvalReport


def _result(task_id: str, solved: bool) -> AgentResult:
    return AgentResult(task_id, solved, 3, 100, 50, 4.2, ())


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
