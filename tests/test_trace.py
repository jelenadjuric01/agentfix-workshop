from agentfix.agent.trace import Tracer, TraceEvent


def test_records_events_and_serialises_them():
    tracer = Tracer()
    tracer.record(TraceEvent(1, "llm", "assistant", "asked for run_tests", 120, 0.4))

    assert len(tracer.events) == 1
    assert tracer.as_json()[0]["prompt_tokens"] == 120


def test_verbose_tracer_prints_each_event(capsys):
    Tracer(verbose=True).record(TraceEvent(2, "tool", "run_tests", "Tests failed.", 300, 1.2))
    assert "run_tests" in capsys.readouterr().out
