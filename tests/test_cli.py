from agentfix.cli import main


def test_version_flag_prints_version(capsys):
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0.1.0" in captured.out


def test_unknown_command_returns_error():
    assert main(["nonsense"]) == 2


def test_solve_reports_a_legible_error_instead_of_a_traceback(capsys, monkeypatch):
    """R19: a student who runs the agent before finishing the exercises gets a sentence."""
    import agentfix.runner

    def boom(*args, **kwargs):
        raise RuntimeError("tool schema rejected")

    monkeypatch.setattr(agentfix.runner, "solve_task", boom)

    exit_code = main(["solve", "tasks/workshop/01-shopcart"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "RuntimeError: tool schema rejected" in captured.err
    assert "exercises/README.md" in captured.err
    assert "Traceback" not in captured.err
