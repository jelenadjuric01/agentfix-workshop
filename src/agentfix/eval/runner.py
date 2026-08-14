from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from agentfix.agent.loop import MAX_STEPS, AgentResult
from agentfix.eval.humanevalfix import load_vendored_rows, write_task_dir
from agentfix.llm.types import LLMClient
from agentfix.runner import solve_task

RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class EvalReport:
    suite: str
    results: tuple[AgentResult, ...]

    @property
    def pass_at_1(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.solved) / len(self.results)

    def to_json(self) -> dict:
        return {
            "suite": self.suite,
            "pass_at_1": self.pass_at_1,
            "results": [
                {k: v for k, v in asdict(result).items() if k != "trace"} for result in self.results
            ],
        }

    def format_table(self) -> str:
        header = f"{'task':<24} {'solved':<8} {'steps':<7} {'tokens':<9} {'seconds':<8}"
        rows = [
            f"{r.task_id:<24} {str(r.solved):<8} {r.steps_used:<7} "
            f"{r.prompt_tokens + r.completion_tokens:<9} {r.duration_s:<8}"
            for r in self.results
        ]
        summary = f"\npass@1 = {self.pass_at_1:.2f}  ({len(self.results)} task(s))"
        return "\n".join([header, "-" * len(header), *rows]) + summary


def evaluate(
    task_dirs: list[Path], llm: LLMClient | None = None, max_steps: int = MAX_STEPS
) -> EvalReport:
    results = [solve_task(task_dir, llm=llm, max_steps=max_steps) for task_dir in task_dirs]
    return EvalReport(suite="custom", results=tuple(results))


def run_suite(suite: str, limit: int = 3) -> int:
    if suite == "workshop":
        task_dirs = sorted(p.parent for p in Path("tasks/workshop").glob("*/task.json"))[:limit]
        report = EvalReport("workshop", evaluate(task_dirs).results)
        _publish(report)
        return 0

    with tempfile.TemporaryDirectory() as temp:
        rows = load_vendored_rows()[:limit]
        task_dirs = [write_task_dir(row, Path(temp)) for row in rows]
        report = EvalReport("humanevalfix", evaluate(task_dirs).results)

    _publish(report)
    return 0


def _publish(report: EvalReport) -> None:
    print(report.format_table())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / f"{report.suite}.json"
    target.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
