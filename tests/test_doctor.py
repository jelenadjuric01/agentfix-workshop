from agentfix import doctor
from agentfix.config import BASE_MODEL, LLMConfig
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


def test_a_stopped_server_is_reported_as_a_server_problem_not_a_missing_model(monkeypatch):
    """The old check ran `ollama list`, so a stopped server told students to re-pull 8 GB."""
    monkeypatch.setattr(doctor, "_get_json", lambda url, timeout_s=5.0: None)

    check = doctor._check_server(LLMConfig())

    assert check.ok is False
    assert "ollama serve" in check.detail
    assert "pull" not in check.detail


def test_a_pulled_but_underived_model_asks_for_ollama_create(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_get_json",
        lambda url, timeout_s=5.0: {"models": [{"name": f"{BASE_MODEL}:latest"}]},
    )

    check = doctor._check_model_present(LLMConfig())

    assert check.ok is False
    assert "ollama create" in check.detail


def test_a_derived_model_is_recognised(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_get_json",
        lambda url, timeout_s=5.0: {"models": [{"name": "agentfix-mellum2:latest"}]},
    )

    assert doctor._check_model_present(LLMConfig()).ok is True


def test_a_short_context_window_fails_with_the_remedy(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_get_json",
        lambda url, timeout_s=5.0: {
            "models": [{"name": "agentfix-mellum2:latest", "context_length": 4096}]
        },
    )

    check = doctor._check_context(LLMConfig())

    assert check.ok is False
    assert "4096" in check.detail
    assert "ollama create agentfix-mellum2 -f Modelfile" in check.detail


def test_the_documented_context_window_passes(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_get_json",
        lambda url, timeout_s=5.0: {
            "models": [{"name": "agentfix-mellum2:latest", "context_length": 16384}]
        },
    )

    assert doctor._check_context(LLMConfig()).ok is True


def test_ram_check_names_a_fallback_tier_when_the_machine_is_too_small(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_bytes", lambda: (8 * 1024**3, 4 * 1024**3))

    check = doctor._check_ram()

    assert check.ok is False
    assert "8.0 GB total" in check.detail
    assert "tier 3" in check.detail


def test_ram_check_passes_on_a_sixteen_gig_machine(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_bytes", lambda: (32 * 1024**3, 20 * 1024**3))
    assert doctor._check_ram().ok is True


def test_this_machine_reports_a_plausible_memory_reading():
    total, free = doctor._memory_bytes()
    assert total is None or total > 1024**3
    assert free is None or free >= 0
