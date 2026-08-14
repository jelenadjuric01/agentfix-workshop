import sys
from pathlib import Path

import pytest

from agentfix.tasks.loader import load_task, workspace

FIXTURE = Path("tasks/workshop/01-shopcart")


def test_load_task_reads_metadata():
    task = load_task(FIXTURE)
    assert task.task_id == "01-shopcart"
    assert task.test_command[:3] == (sys.executable, "-m", "pytest")
    assert task.expected_failures


def test_workspace_yields_a_writable_copy_and_cleans_up():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        assert (work_dir / "shopcart" / "cart.py").is_file()
        (work_dir / "shopcart" / "cart.py").write_text("# clobbered\n", encoding="utf-8")
        recorded = work_dir
    assert not recorded.exists()


def test_template_is_never_mutated_by_a_workspace():
    task = load_task(FIXTURE)
    before = (task.template_dir / "shopcart" / "cart.py").read_text(encoding="utf-8")
    with workspace(task) as work_dir:
        (work_dir / "shopcart" / "cart.py").write_text("# clobbered\n", encoding="utf-8")
    assert (task.template_dir / "shopcart" / "cart.py").read_text(encoding="utf-8") == before


def test_fixture_starts_red():
    from agentfix.sandbox.subprocess_backend import SubprocessBackend

    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is False


@pytest.mark.parametrize(
    "task_dir",
    ["tasks/workshop/01-shopcart", "tasks/workshop/02-invoice", "tasks/workshop/03-parser"],
)
def test_every_workshop_fixture_starts_red(task_dir):
    from agentfix.sandbox.subprocess_backend import SubprocessBackend

    task = load_task(Path(task_dir))
    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is False


@pytest.mark.parametrize(
    "task_dir",
    ["tasks/workshop/01-shopcart", "tasks/workshop/02-invoice", "tasks/workshop/03-parser"],
)
def test_expected_failures_names_exactly_the_tests_that_fail(task_dir):
    """The field used to be dead data. Now it pins which tests are red, so fixture rot shows up."""
    from agentfix.sandbox.subprocess_backend import SubprocessBackend

    task = load_task(Path(task_dir))
    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)

    failed = sorted(
        line.split("::")[1].split()[0]
        for line in result.output.splitlines()
        if line.startswith("FAILED") and "::" in line
    )
    assert failed == sorted(task.expected_failures)


def test_02_invoice_bug_is_not_in_the_file_the_test_names():
    """The pedagogical contract of fixture 02 — asserted so it cannot silently regress."""
    task = load_task(Path("tasks/workshop/02-invoice"))
    failing_test_file = (task.template_dir / "tests" / "test_invoice.py").read_text(
        encoding="utf-8"
    )
    buggy_file = (task.template_dir / "billing" / "discounts.py").read_text(encoding="utf-8")
    assert "discounts" not in failing_test_file
    assert "quantity > BULK_THRESHOLD" in buggy_file
    assert ">=" not in buggy_file
