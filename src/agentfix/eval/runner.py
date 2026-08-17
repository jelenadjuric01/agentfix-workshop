"""Measurement: run the agent over many tasks and report what happened.

The distinction worth understanding here is eval vs tests (ARCHITECTURE.md covers it at
length). The test suite asks "does the code do what I wrote it to do?" and answers
deterministically with a scripted fake model. Eval asks "is the agent any good?" and needs a
real model, so it is slow, costs tokens, and gives a different answer each run. Both are
necessary; neither substitutes for the other.

pass@1 is the headline: of N tasks, how many did the agent fix on its first and only attempt.
Steps, tokens and peak context are reported alongside it because an agent that solves a task
in 10 steps and 40k tokens is not the same product as one that does it in 4 and 8k.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentfix.agent.loop import MAX_STEPS, AgentResult
from agentfix.config import REPO_ROOT
from agentfix.eval.humanevalfix import load_vendored_rows, write_task_dir
from agentfix.llm.types import LLMClient
from agentfix.runner import solve_task

RESULTS_DIR = REPO_ROOT / "results"
WORKSHOP_TASKS_DIR = REPO_ROOT / "tasks" / "workshop"


@dataclass(frozen=True)
class EvalReport:
    """The results of one suite run, plus the derived numbers worth quoting."""

    suite: str
    results: tuple[AgentResult, ...]

    @property
    def pass_at_1(self) -> float:
        """Fraction of tasks solved. One attempt each — no retries, no best-of-n.

        `@property` makes this readable as `report.pass_at_1`, with no parentheses: computed
        on access, so it cannot drift out of sync with `results` the way a stored field could.
        """
        if not self.results:
            return 0.0  # an empty suite scores zero, rather than dividing by zero
        return sum(1 for result in self.results if result.solved) / len(self.results)

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "pass_at_1": self.pass_at_1,
            "peak_prompt_tokens": self.peak_prompt_tokens,
            "results": [
                # The trace is dropped: it is by far the largest field, and a full trace per
                # task would bury the numbers this file exists to report.
                {k: v for k, v in asdict(result).items() if k != "trace"}
                for result in self.results
            ],
        }

    @property
    def peak_prompt_tokens(self) -> int:
        """Largest single prompt across the whole suite — compare it to the context window."""
        # `default=0` because max() of an empty sequence raises.
        return max((result.peak_prompt_tokens for result in self.results), default=0)

    def format_table(self) -> str:
        """A fixed-width table for the terminal. `:<24` means left-align in 24 columns."""
        header = (
            f"{'task':<24} {'solved':<8} {'steps':<7} {'tokens':<9} {'peak ctx':<10} {'seconds':<8}"
        )
        rows = [
            f"{r.task_id:<24} {str(r.solved):<8} {r.steps_used:<7} "
            f"{r.prompt_tokens + r.completion_tokens:<9} {r.peak_prompt_tokens:<10} "
            f"{r.duration_s:<8}"
            for r in self.results
        ]
        summary = (
            f"\npass@1 = {self.pass_at_1:.2f}  ({len(self.results)} task(s))"
            f"  peak prompt = {self.peak_prompt_tokens} tok"
        )
        return "\n".join([header, "-" * len(header), *rows]) + summary


def evaluate(
    task_dirs: list[Path], llm: LLMClient | None = None, max_steps: int = MAX_STEPS
) -> EvalReport:
    """Run the agent over the given tasks, in order, one attempt each.

    Sequential on purpose: the local model serves one request at a time, so running tasks
    concurrently would not be faster, and it would make the timing numbers meaningless.
    """
    results = [solve_task(task_dir, llm=llm, max_steps=max_steps) for task_dir in task_dirs]
    return EvalReport(suite="custom", results=tuple(results))


def run_suite(suite: str, limit: int = 3, llm: LLMClient | None = None) -> int:
    """Run a named suite and write its report. Returns a process exit code.

    Note `limit` defaults to 3 and is applied *after* sorting, so a fourth workshop task is
    silently dropped unless you raise it — pass `--limit 4`.
    """
    if suite == "workshop":
        # Discovered by glob rather than hardcoded, so adding a task directory is enough.
        task_dirs = sorted(p.parent for p in WORKSHOP_TASKS_DIR.glob("*/task.json"))[:limit]
        report = EvalReport("workshop", evaluate(task_dirs, llm=llm).results)
        _publish(report)
        return 0

    # HumanEvalFix tasks do not exist on disk: they are generated from a vendored JSON subset
    # into a temp directory, used, and thrown away. Keeps a benchmark out of the repo while
    # letting it run through exactly the same task machinery as the workshop fixtures.
    with tempfile.TemporaryDirectory() as temp:
        rows = load_vendored_rows()[:limit]
        task_dirs = [write_task_dir(row, Path(temp)) for row in rows]
        report = EvalReport("humanevalfix", evaluate(task_dirs, llm=llm).results)

    # Published outside the `with` block: the temp tasks are gone by now, but the results are
    # values, not files, so nothing is lost.
    _publish(report)
    return 0


def _publish(report: EvalReport) -> None:
    """Print the table and save the JSON, so a run can be compared against a later one."""
    print(report.format_table())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # One file per suite, overwritten each run. results/precomputed/ holds the reference runs
    # quoted in the README; those are committed, these are not.
    target = RESULTS_DIR / f"{report.suite}.json"
    target.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
