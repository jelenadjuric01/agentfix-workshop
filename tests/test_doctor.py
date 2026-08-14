from agentfix.doctor import Check, report


def test_report_returns_zero_when_all_checks_pass(capsys):
    exit_code = report([Check("python", True, "3.12.9"), Check("ollama", True, "reachable")])
    assert exit_code == 0
    assert "READY" in capsys.readouterr().out


def test_report_returns_one_and_shows_remedy_when_a_check_fails(capsys):
    exit_code = report([Check("model", False, "not found — run: ollama pull ...")])
    captured = capsys.readouterr().out
    assert exit_code == 1
    assert "ollama pull" in captured
