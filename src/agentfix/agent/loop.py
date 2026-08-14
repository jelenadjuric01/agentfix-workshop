from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agentfix.agent.trace import Tracer, TraceEvent
from agentfix.llm.types import LLMClient, LLMReply, ToolCall
from agentfix.tasks.loader import Task
from agentfix.tools.base import ToolRegistry
from agentfix.tools.tests_tool import RunTestsTool

MAX_STEPS = 6


@dataclass(frozen=True)
class AgentResult:
    task_id: str
    solved: bool
    steps_used: int
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    trace: tuple[TraceEvent, ...]


def system_prompt(registry: ToolRegistry) -> str:
    names = ", ".join(schema["function"]["name"] for schema in registry.schemas())
    return (
        "You are a Python bug-fixing agent working in a small project.\n"
        f"You have these tools: {names}.\n"
        "Work in this order: run the tests to see what fails, read the relevant file, "
        "then write the corrected file.\n"
        "When you call write_file you must supply the COMPLETE file contents, not a diff.\n"
        "Make the smallest change that fixes the failure. Do not rewrite unrelated code."
    )


def task_prompt(task: Task) -> str:
    return task.prompt


def is_done(run_tests: RunTestsTool) -> bool:
    """The agent is done when the tests actually pass — never because it says so."""
    return run_tests.last_result is not None and run_tests.last_result.passed


def _call_signature(call: ToolCall) -> tuple[str, str]:
    return call.name, repr(sorted(call.arguments.items()))


def run_agent(
    task: Task,
    work_dir: Path,
    llm: LLMClient,
    registry: ToolRegistry,
    run_tests: RunTestsTool,
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> AgentResult:
    tracer = tracer or Tracer()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(registry)},
        {"role": "user", "content": task_prompt(task)},
    ]

    prompt_tokens = 0
    completion_tokens = 0
    previous_signature: tuple[str, str] | None = None
    steps_used = 0
    started = time.time()

    for step in range(1, max_steps + 1):
        steps_used = step

        call_started = time.time()
        reply = llm.chat(messages, tools=registry.schemas())
        messages.append(reply.message)

        prompt_tokens += reply.prompt_tokens
        completion_tokens += reply.completion_tokens
        tracer.record(
            TraceEvent(
                step=step,
                kind="llm",
                name="assistant",
                detail=_describe(reply),
                prompt_tokens=reply.prompt_tokens,
                latency_s=round(time.time() - call_started, 2),
            )
        )

        if reply.tool_calls:
            for call in reply.tool_calls:
                signature = _call_signature(call)
                if signature == previous_signature:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": (
                                f"You already called {call.name} with these exact arguments and "
                                "got the result above. Try a different tool or different arguments."
                            ),
                        }
                    )
                    continue

                previous_signature = signature
                tool_started = time.time()
                outcome = registry.dispatch(call)
                messages.append(outcome.as_message())
                tracer.record(
                    TraceEvent(
                        step=step,
                        kind="tool",
                        name=call.name,
                        detail=outcome.result.content,
                        prompt_tokens=reply.prompt_tokens,
                        latency_s=round(time.time() - tool_started, 2),
                    )
                )
            continue

        if is_done(run_tests):
            break
        break

    return AgentResult(
        task_id=task.task_id,
        solved=is_done(run_tests),
        steps_used=steps_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_s=round(time.time() - started, 2),
        trace=tuple(tracer.events),
    )


def _describe(reply: LLMReply) -> str:
    if reply.tool_calls:
        return "calls " + ", ".join(call.name for call in reply.tool_calls)
    return (reply.message.get("content") or "").strip()
