from __future__ import annotations

import subprocess
import sys

from agentfix.config import REPO_ROOT

# The `llm` marker is the only thing standing between "runs anywhere" and "needs a model
# on localhost", so the selection rules are worth testing rather than trusting. Each case
# runs pytest in a subprocess with --collect-only: no model is contacted, nothing executes.

LIVE_TEST = "tests/test_llm.py::test_live_model_answers_and_reports_usage"


def _collect(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_llm.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_live_test_is_excluded_by_default():
    """pyproject's addopts must keep a bare pytest run offline."""
    result = _collect()
    assert LIVE_TEST not in result.stdout
    assert "1 deselected" in result.stdout


def test_all_includes_the_live_test():
    result = _collect("--all")
    assert LIVE_TEST in result.stdout
    assert "deselected" not in result.stdout


def test_marker_alone_selects_only_the_live_test():
    result = _collect("-m", "llm")
    assert LIVE_TEST in result.stdout
    assert "deselected" in result.stdout


def test_all_with_an_explicit_marker_is_a_usage_error():
    """Two conflicting filters is a mistake worth reporting, not one to resolve silently."""
    result = _collect("--all", "-m", "llm")
    assert result.returncode == 4, result.stdout
    assert "--all cannot be combined with -m" in result.stderr + result.stdout
