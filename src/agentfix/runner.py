"""Wiring: turns a task directory into a finished run.

Short, and worth reading closely — this is where the separate pieces are assembled, and
where two of the project's safety properties are actually established. Read it after
agent/loop.py.

    task dir -> load_task -> workspace copy -> tools bound to that copy -> run_agent
"""

from __future__ import annotations

from pathlib import Path

from agentfix.agent.loop import MAX_STEPS, AgentResult, run_agent
from agentfix.agent.trace import Tracer
from agentfix.llm.types import LLMClient
from agentfix.sandbox.base import get_backend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import (
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    _relative_files,
)
from agentfix.tools.tests_tool import RunTestsTool


def solve_task(
    task_dir: Path,
    llm: LLMClient | None = None,
    verbose: bool = False,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    """Run the agent on one task, from a fresh copy, and return what happened.

    `llm=None` means "make a real one". Tests pass a `FakeLLMClient` here instead, which is
    how the whole suite runs the real wiring with no model process anywhere.
    """
    if llm is None:
        # Imported inside the function, not at module scope, so that importing this module —
        # and therefore the test suite — never constructs an OpenAI client or reads the
        # environment. Nothing here requires a running Ollama until you actually ask for one.
        from agentfix.llm.client import OllamaClient

        llm = OllamaClient()

    task = load_task(Path(task_dir))

    # Read from the PRISTINE template, not the workspace copy, so the set cannot grow during a
    # run: an agent that managed to create a file could otherwise then write to it.
    writable = frozenset(_relative_files(task.template_dir))

    # Every run gets a disposable copy, and it is deleted when this block exits by any route.
    with workspace(task) as work_dir:
        # Built first because WriteFileTool needs its `invalidate` method below.
        run_tests = RunTestsTool(work_dir, task.test_command, get_backend(), timeout_s=30)

        # Each tool is constructed with `work_dir` bound to it. That is why `run_agent` needs
        # no workspace argument: by this point the workspace is baked into the tools, and the
        # loop is purely orchestration.
        registry = ToolRegistry(
            [
                ListFilesTool(work_dir),
                ReadFileTool(work_dir),
                # a write makes the last test result stale, so `is_done` must not trust it
                WriteFileTool(work_dir, on_write=run_tests.invalidate, allowed=writable),
                run_tests,
            ]
        )

        # `run_tests` is passed twice over: once inside the registry, for the model to call,
        # and once directly, so `is_done` can consult it without asking the model.
        return run_agent(
            task, llm, registry, run_tests, max_steps=max_steps, tracer=Tracer(verbose)
        )
