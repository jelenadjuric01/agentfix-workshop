import sys
from pathlib import Path

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
