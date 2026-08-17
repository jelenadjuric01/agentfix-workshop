from __future__ import annotations

from pathlib import Path

from agentfix.agent.loop import MAX_STEPS, AgentResult, run_agent
from agentfix.agent.trace import Tracer
from agentfix.llm.types import LLMClient
from agentfix.sandbox.base import get_backend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool


def solve_task(
    task_dir: Path,
    llm: LLMClient | None = None,
    verbose: bool = False,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    if llm is None:
        from agentfix.llm.client import OllamaClient

        llm = OllamaClient()

    task = load_task(Path(task_dir))

    with workspace(task) as work_dir:
        run_tests = RunTestsTool(work_dir, task.test_command, get_backend(), timeout_s=30)
        registry = ToolRegistry(
            [
                ListFilesTool(work_dir),
                ReadFileTool(work_dir),
                # a write makes the last test result stale, so `is_done` must not trust it
                WriteFileTool(work_dir, on_write=run_tests.invalidate),
                run_tests,
            ]
        )
        return run_agent(
            task, llm, registry, run_tests, max_steps=max_steps, tracer=Tracer(verbose)
        )
