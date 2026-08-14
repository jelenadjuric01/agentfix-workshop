from agentfix.cli import main


def test_version_flag_prints_version(capsys):
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0.1.0" in captured.out


def test_unknown_command_returns_error():
    assert main(["nonsense"]) == 2
