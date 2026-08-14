# AgentFix Workshop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a teaching repository for a 90-minute workshop in which developers new to agents write three parts of a working coding agent — a tool and its JSON schema, the loop's tool dispatch, and a verification-based stop condition — running JetBrains Mellum2 locally and free.

**Architecture:** Hand-rolled agent, no framework. An `LLMClient` wraps an OpenAI-compatible endpoint (Ollama by default). A `ToolRegistry` turns `Tool` objects into API schemas and dispatches tool calls back into `role="tool"` messages. A ~15-line `run_agent` loop appends to history, dispatches tools, and stops when `run_tests` passes. Test execution goes through a swappable `ExecutionBackend` (hardened subprocess by default, Docker opt-in). A `FakeLLMClient` makes every exercise test offline, instant, and deterministic.

**Tech Stack:** Python 3.12, uv, `openai` SDK, pytest, Ollama serving `Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M`.

**Spec:** `docs/superpowers/specs/2026-08-14-agentfix-workshop-design.md`

## Global Constraints

- **Repo:** `~/Desktop/JetBrains/agentfix-workshop`. The predecessor repo `~/Desktop/JetBrains/Python Ai Agent` is **read-only** — copy files out of it, never modify it.
- **Package manager: uv only.** No `pip`, no `requirements.txt`. `uv.lock` is committed.
- **Python:** `>=3.12` (3.12.9 is installed).
- **Base dependencies:** `openai`, `pytest`. Nothing else. **No `torch`, `transformers`, `accelerate`, or `datasets` in the base set.**
- **Optional extras:** `[eval]` → `datasets>=2.20.0`; `[dev]` → `ruff`, `black`, `pytest-cov`.
- **Package name:** `agentfix`, `src/` layout. **CLI:** `agentfix` with subcommands `doctor`, `solve`, `eval`.
- **Model defaults, verbatim:** model id `hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M`, `base_url` `http://localhost:11434/v1`, `temperature` `0.6`, `top_p` `0.95`, `max_tokens` `1024`, `num_ctx` `16384`.
- **Env vars:** `MELLUM_BASE_URL`, `MELLUM_MODEL`, `AGENTFIX_SANDBOX` (`subprocess` | `docker`).
- **Message history is strictly append-only.** Never rewrite, reorder, or drop earlier messages — it invalidates Ollama's KV prefix cache and makes every turn pay full prefill (~480 tok/s measured).
- **Truncation limits:** `read_file` output 4000 chars; tool/test output 2000 chars. Enforced where the string is produced, with a literal `\n[...truncated]` marker.
- **`max_steps` default: 10.** Hard cap. (Raised from 6 after measurement — see spec.)
- **Sync only.** No `async`/`await` anywhere.
- **Style:** type annotations on all signatures, `@dataclass(frozen=True)` for data, snake_case, no comments except where logic is non-obvious. `ruff` and `black` clean.
- **Coverage:** ≥80% on `src/agentfix`.
- **Tests needing a live model** are marked `@pytest.mark.llm` and skipped by default.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project, deps, extras, `agentfix` script, pytest/ruff config |
| `src/agentfix/config.py` | `LLMConfig`, env loading |
| `src/agentfix/llm/types.py` | `ToolCall`, `LLMReply`, `LLMClient` Protocol |
| `src/agentfix/llm/client.py` | `OllamaClient` — OpenAI-compatible `chat()` |
| `src/agentfix/llm/fake.py` | `FakeLLMClient` + `assistant_text` / `assistant_tool_call` builders |
| `src/agentfix/sandbox/base.py` | `ExecResult`, `ExecutionBackend` Protocol, `get_backend()` |
| `src/agentfix/sandbox/subprocess_backend.py` | hardened subprocess execution (ported) |
| `src/agentfix/sandbox/docker_backend.py` | opt-in container execution |
| `src/agentfix/tools/base.py` | `ToolResult`, `ToolOutcome`, `Tool` Protocol, `ToolRegistry`, `truncate` |
| `src/agentfix/tools/fs.py` | `resolve_in_root`, `ListFilesTool`, `ReadFileTool`, `WriteFileTool` |
| `src/agentfix/tools/tests_tool.py` | `RunTestsTool` |
| `src/agentfix/tasks/loader.py` | `Task`, `load_task`, `workspace` context manager |
| `src/agentfix/agent/trace.py` | `TraceEvent`, `Tracer` |
| `src/agentfix/agent/loop.py` | `system_prompt`, `task_prompt`, `is_done`, `run_agent`, `AgentResult` |
| `src/agentfix/eval/runner.py` | `EvalReport`, `evaluate`, `format_table` |
| `src/agentfix/eval/humanevalfix.py` | `HumanEvalFixRow`, vendored-subset loader, task-dir generator |
| `src/agentfix/doctor.py` | preflight checks |
| `src/agentfix/cli.py` | argparse entry point |
| `tasks/workshop/01-shopcart/` | fixture: obvious bug |
| `tasks/workshop/02-invoice/` | fixture: bug not where the test points |
| `tasks/workshop/03-parser/` | fixture: two files involved |
| `tasks/humanevalfix/subset.json` | 20 vendored rows |
| `exercises/stage_{1,2,3}/` | README + tests for each stage |
| `notebooks/kaggle.ipynb` | tier-2 fallback |
| `README.md`, `WORKSHOP.md`, `ARCHITECTURE.md` | docs |

---

### Task 1: Project scaffold, uv, and CLI skeleton

**Files:**
- Create: `pyproject.toml`, `src/agentfix/__init__.py`, `src/agentfix/cli.py`, `tests/test_cli.py`, `.gitignore` (exists — extend)

**Interfaces:**
- Consumes: nothing
- Produces: `agentfix.__version__: str`; `agentfix.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Install uv**

`uv` is not currently installed. Homebrew is at `/opt/homebrew/bin/brew`.

```bash
brew install uv
uv --version   # expect: uv 0.x.y
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "agentfix"
version = "0.1.0"
description = "A teaching coding agent: tools, a loop, and verification"
requires-python = ">=3.12"
dependencies = ["openai>=1.40", "pytest>=8.3"]

[project.optional-dependencies]
eval = ["datasets>=2.20.0"]
dev = ["ruff>=0.6", "black>=24.8", "pytest-cov>=5.0"]

[project.scripts]
agentfix = "agentfix.cli:cli_entry"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentfix"]

[tool.pytest.ini_options]
testpaths = ["tests", "exercises"]
markers = ["llm: requires a running Ollama with the Mellum2 model (deselect with '-m \"not llm\"')"]
addopts = "-m 'not llm'"

[tool.ruff]
line-length = 100

[tool.black]
line-length = 100
```

- [ ] **Step 3: Write the failing test**

`tests/test_cli.py`:

```python
from agentfix.cli import main


def test_version_flag_prints_version(capsys):
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0.1.0" in captured.out


def test_unknown_command_returns_error():
    assert main(["nonsense"]) == 2
```

- [ ] **Step 4: Run test to verify it fails**

```bash
uv sync --extra dev
uv run pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.cli'`

- [ ] **Step 5: Implement the package skeleton**

`src/agentfix/__init__.py`:

```python
__version__ = "0.1.0"
```

`src/agentfix/cli.py`:

```python
from __future__ import annotations

import argparse
import sys

from agentfix import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentfix", description="A teaching coding agent")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("command", nargs="?", choices=["doctor", "solve", "eval"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, _rest = parser.parse_known_args(argv)

    if args.version:
        print(f"agentfix {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    print(f"'{args.command}' is not implemented yet")
    return 0


def cli_entry() -> None:
    sys.exit(main())
```

Note: `argparse` exits with code 2 on an invalid choice, which satisfies the second test.

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/agentfix tests/test_cli.py .gitignore
git commit -m "feat: uv project scaffold with agentfix CLI skeleton"
```

---

### Task 2: LLM client and fake client

**Files:**
- Create: `src/agentfix/config.py`, `src/agentfix/llm/__init__.py`, `src/agentfix/llm/types.py`, `src/agentfix/llm/client.py`, `src/agentfix/llm/fake.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LLMConfig(base_url, model, temperature, top_p, max_tokens, num_ctx)`, `LLMConfig.from_env() -> LLMConfig`
  - `ToolCall(id: str, name: str, arguments: dict)`
  - `LLMReply(message: dict, tool_calls: tuple[ToolCall, ...], prompt_tokens: int, completion_tokens: int)`
  - `LLMClient` Protocol with `chat(messages: list[dict], tools: list[dict] | None = None) -> LLMReply`
  - `OllamaClient(config: LLMConfig | None = None)`
  - `FakeLLMClient(replies: list[LLMReply])` with attribute `calls: list[list[dict]]`
  - `assistant_text(text: str, prompt_tokens: int = 10) -> LLMReply`
  - `assistant_tool_call(name: str, arguments: dict, call_id: str = "call_1", prompt_tokens: int = 10) -> LLMReply`

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:

```python
import pytest

from agentfix.config import LLMConfig
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.llm.types import LLMReply, ToolCall


def test_config_defaults_match_spec():
    config = LLMConfig()
    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.max_tokens == 1024
    assert config.num_ctx == 16384


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("MELLUM_BASE_URL", "http://gpu-box:8000/v1")
    monkeypatch.setenv("MELLUM_MODEL", "some/other-model")
    config = LLMConfig.from_env()
    assert config.base_url == "http://gpu-box:8000/v1"
    assert config.model == "some/other-model"


def test_assistant_text_builds_reply_with_no_tool_calls():
    reply = assistant_text("all done")
    assert reply.tool_calls == ()
    assert reply.message == {"role": "assistant", "content": "all done"}


def test_assistant_tool_call_builds_openai_shaped_message():
    reply = assistant_tool_call("run_tests", {}, call_id="abc")
    assert reply.tool_calls == (ToolCall(id="abc", name="run_tests", arguments={}),)
    assert reply.message["tool_calls"][0]["function"]["name"] == "run_tests"
    assert reply.message["tool_calls"][0]["id"] == "abc"


def test_fake_client_returns_scripted_replies_in_order_and_records_calls():
    client = FakeLLMClient([assistant_tool_call("run_tests", {}), assistant_text("fixed")])

    first = client.chat([{"role": "user", "content": "go"}])
    second = client.chat([{"role": "user", "content": "go"}, {"role": "tool", "content": "fail"}])

    assert first.tool_calls[0].name == "run_tests"
    assert second.message["content"] == "fixed"
    assert len(client.calls) == 2
    assert len(client.calls[1]) == 2


def test_fake_client_raises_when_script_is_exhausted():
    client = FakeLLMClient([assistant_text("only one")])
    client.chat([])
    with pytest.raises(AssertionError, match="exhausted"):
        client.chat([])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_llm.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.config'`

- [ ] **Step 3: Implement config and types**

`src/agentfix/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, replace

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024
    num_ctx: int = 16384

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return replace(
            cls(),
            base_url=os.environ.get("MELLUM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("MELLUM_MODEL", DEFAULT_MODEL),
        )
```

`src/agentfix/llm/__init__.py`: empty file.

`src/agentfix/llm/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMReply:
    message: dict
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply: ...
```

- [ ] **Step 4: Implement the fake client**

`src/agentfix/llm/fake.py`:

```python
from __future__ import annotations

import json

from agentfix.llm.types import LLMReply, ToolCall


def assistant_text(text: str, prompt_tokens: int = 10) -> LLMReply:
    return LLMReply(
        message={"role": "assistant", "content": text},
        tool_calls=(),
        prompt_tokens=prompt_tokens,
        completion_tokens=len(text.split()),
    )


def assistant_tool_call(
    name: str, arguments: dict, call_id: str = "call_1", prompt_tokens: int = 10
) -> LLMReply:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    return LLMReply(
        message=message,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
    )


class FakeLLMClient:
    """Scripted client so the agent loop is testable with no model running."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self._replies = list(replies)
        self._index = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        self.calls.append(list(messages))
        assert self._index < len(self._replies), (
            f"FakeLLMClient script exhausted after {self._index} call(s); "
            "the agent asked for more turns than the test scripted"
        )
        reply = self._replies[self._index]
        self._index += 1
        return reply
```

- [ ] **Step 5: Implement the real client**

`src/agentfix/llm/client.py`:

```python
from __future__ import annotations

import json

from openai import OpenAI

from agentfix.config import LLMConfig
from agentfix.llm.types import LLMReply, ToolCall


class OllamaClient:
    """OpenAI-compatible client. Works against Ollama, vLLM, or any /v1 endpoint."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = OpenAI(base_url=self.config.base_url, api_key="agentfix")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "extra_body": {"options": {"num_ctx": self.config.num_ctx}},
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        raw = choice.message

        calls: list[ToolCall] = []
        for call in raw.tool_calls or []:
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_parse_arguments(call.function.arguments),
                )
            )

        return LLMReply(
            message=raw.model_dump(exclude_none=True),
            tool_calls=tuple(calls),
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )


def _parse_arguments(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
```

`_parse_arguments` returning `{}` on malformed JSON is deliberate: a small model occasionally emits invalid argument JSON, and the tool layer's validation will turn that into an observation the model can recover from (Task 4) rather than crashing the run.

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm.py -v
```

Expected: 6 passed

- [ ] **Step 7: Add the live smoke test**

Append to `tests/test_llm.py`:

```python
@pytest.mark.llm
def test_live_model_answers_and_reports_usage():
    from agentfix.llm.client import OllamaClient

    reply = OllamaClient().chat([{"role": "user", "content": "Reply with exactly: ok"}])
    assert "ok" in (reply.message.get("content") or "").lower()
    assert reply.prompt_tokens > 0
```

Verify it is skipped by default and passes when selected:

```bash
uv run pytest tests/test_llm.py -v            # 6 passed, 1 deselected
uv run pytest tests/test_llm.py -m llm -v     # 1 passed (needs Ollama running)
```

- [ ] **Step 8: Commit**

```bash
git add src/agentfix/config.py src/agentfix/llm tests/test_llm.py
git commit -m "feat: OpenAI-compatible LLM client with scripted fake for offline tests"
```

---

### Task 3: Execution backends

**Files:**
- Create: `src/agentfix/sandbox/__init__.py`, `src/agentfix/sandbox/base.py`, `src/agentfix/sandbox/subprocess_backend.py`, `tests/test_sandbox.py`
- Source to port: `~/Desktop/JetBrains/Python Ai Agent/src/sandbox/runner.py` (read only)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ExecResult(passed: bool, output: str, duration_s: float, timed_out: bool)`
  - `ExecutionBackend` Protocol: `run(workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult`
  - `SubprocessBackend(max_output_chars: int = 2000)`
  - `get_backend(name: str | None = None) -> ExecutionBackend` — reads `AGENTFIX_SANDBOX`, default `subprocess`

- [ ] **Step 1: Write the failing test**

`tests/test_sandbox.py`:

```python
import sys
from pathlib import Path

from agentfix.sandbox.base import get_backend
from agentfix.sandbox.subprocess_backend import SubprocessBackend


def _write(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / name).write_text(body, encoding="utf-8")


def test_passing_tests_report_passed(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert 1 + 1 == 2\n")
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert result.passed is True
    assert result.timed_out is False


def test_failing_tests_report_failure_with_output(tmp_path):
    _write(tmp_path, "test_bad.py", "def test_bad():\n    assert 1 == 2\n")
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert result.passed is False
    assert "test_bad" in result.output


def test_infinite_loop_is_killed_by_timeout(tmp_path):
    _write(tmp_path, "test_hang.py", "def test_hang():\n    while True:\n        pass\n")
    result = SubprocessBackend().run(
        tmp_path, (sys.executable, "-m", "pytest", "-q"), timeout_s=3
    )
    assert result.timed_out is True
    assert result.passed is False
    assert "TIMEOUT" in result.output


def test_output_is_truncated_with_marker(tmp_path):
    _write(tmp_path, "test_loud.py", "def test_loud():\n    print('x' * 50000)\n    assert False\n")
    result = SubprocessBackend(max_output_chars=500).run(
        tmp_path, (sys.executable, "-m", "pytest", "-q")
    )
    assert len(result.output) < 800
    assert "[...truncated]" in result.output


def test_secrets_in_parent_env_are_not_visible_to_executed_code(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
    _write(
        tmp_path,
        "test_env.py",
        "import os\ndef test_env():\n    assert os.environ.get('MY_SECRET_TOKEN') is None\n",
    )
    result = SubprocessBackend().run(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert result.passed is True


def test_get_backend_defaults_to_subprocess(monkeypatch):
    monkeypatch.delenv("AGENTFIX_SANDBOX", raising=False)
    assert isinstance(get_backend(), SubprocessBackend)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sandbox.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.sandbox'`

- [ ] **Step 3: Implement `base.py`**

`src/agentfix/sandbox/__init__.py`: empty file.

`src/agentfix/sandbox/base.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    output: str
    duration_s: float
    timed_out: bool = False


class ExecutionBackend(Protocol):
    def run(
        self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10
    ) -> ExecResult: ...


def get_backend(name: str | None = None) -> ExecutionBackend:
    choice = (name or os.environ.get("AGENTFIX_SANDBOX") or "subprocess").lower()

    if choice == "subprocess":
        from agentfix.sandbox.subprocess_backend import SubprocessBackend

        return SubprocessBackend()

    if choice == "docker":
        from agentfix.sandbox.docker_backend import DockerBackend

        return DockerBackend()

    raise ValueError(f"Unknown AGENTFIX_SANDBOX={choice!r}; expected 'subprocess' or 'docker'")
```

- [ ] **Step 4: Implement `subprocess_backend.py`**

This ports the predecessor's `run_python` and adds rlimits, env stripping, and truncation.

```python
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agentfix.sandbox.base import ExecResult

TRUNCATION_MARKER = "\n[...truncated]"

MAX_ADDRESS_SPACE_BYTES = 2 * 1024**3
MAX_CPU_SECONDS = 30
MAX_FILE_SIZE_BYTES = 16 * 1024**2
MAX_PROCESSES = 64


def _apply_limits() -> None:
    """Constrain the child process. POSIX only; a no-op elsewhere."""
    try:
        import resource
    except ImportError:  # Windows
        return

    resource.setrlimit(resource.RLIMIT_AS, (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    except (ValueError, OSError):
        pass


class SubprocessBackend:
    """Runs tests in a child process with resource limits and a stripped environment."""

    def __init__(self, max_output_chars: int = 2000) -> None:
        self.max_output_chars = max_output_chars

    def run(
        self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10
    ) -> ExecResult:
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(workspace)}
        start = time.time()

        try:
            completed = subprocess.run(
                list(command),
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_apply_limits if hasattr(__import__("os"), "fork") else None,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                passed=False,
                output=f"TIMEOUT after {timeout_s}s",
                duration_s=round(time.time() - start, 3),
                timed_out=True,
            )

        combined = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return ExecResult(
            passed=completed.returncode == 0,
            output=self._truncate(combined),
            duration_s=round(time.time() - start, 3),
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + TRUNCATION_MARKER
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_sandbox.py -v
```

Expected: 6 passed. The timeout test takes ~3s.

- [ ] **Step 6: Commit**

```bash
git add src/agentfix/sandbox tests/test_sandbox.py
git commit -m "feat: hardened subprocess execution backend with rlimits and env stripping"
```

---

### Task 4: Tool layer

**Files:**
- Create: `src/agentfix/tools/__init__.py`, `src/agentfix/tools/base.py`, `tests/test_tools_base.py`

**Interfaces:**
- Consumes: `ToolCall` (Task 2)
- Produces:
  - `truncate(text: str, limit: int) -> str`
  - `ToolResult(ok: bool, content: str)`
  - `ToolOutcome(call_id: str, name: str, result: ToolResult)` with `as_message() -> dict`
  - `Tool` Protocol: attributes `name: str`, `description: str`, `parameters: dict`; method `run(**kwargs) -> ToolResult`
  - `ToolRegistry(tools: Sequence[Tool])` with `schemas() -> list[dict]`, `get(name: str) -> Tool`, `dispatch(call: ToolCall) -> ToolOutcome`

- [ ] **Step 1: Write the failing test**

`tests/test_tools_base.py`:

```python
from agentfix.llm.types import ToolCall
from agentfix.tools.base import ToolRegistry, ToolResult, truncate


class EchoTool:
    name = "echo"
    description = "Echo a message back."
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def run(self, message: str) -> ToolResult:
        return ToolResult(ok=True, content=f"echo: {message}")


def test_truncate_appends_marker_only_when_needed():
    assert truncate("short", 100) == "short"
    long_text = truncate("y" * 500, 100)
    assert long_text.startswith("y" * 100)
    assert "[...truncated]" in long_text


def test_schemas_are_openai_shaped():
    schemas = ToolRegistry([EchoTool()]).schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a message back.",
                "parameters": EchoTool.parameters,
            },
        }
    ]


def test_dispatch_returns_tool_message_with_call_id():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(ToolCall(id="call_7", name="echo", arguments={"message": "hi"}))

    assert outcome.result.ok is True
    assert outcome.as_message() == {
        "role": "tool",
        "tool_call_id": "call_7",
        "name": "echo",
        "content": "echo: hi",
    }


def test_unknown_tool_becomes_an_observation_not_an_exception():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(ToolCall(id="c1", name="nope", arguments={}))

    assert outcome.result.ok is False
    assert "nope" in outcome.result.content
    assert "echo" in outcome.result.content


def test_missing_required_argument_becomes_an_observation():
    registry = ToolRegistry([EchoTool()])
    outcome = registry.dispatch(ToolCall(id="c1", name="echo", arguments={}))

    assert outcome.result.ok is False
    assert "message" in outcome.result.content


def test_tool_exception_becomes_an_observation():
    class BoomTool:
        name = "boom"
        description = "Always explodes."
        parameters = {"type": "object", "properties": {}}

        def run(self) -> ToolResult:
            raise RuntimeError("kaboom")

    outcome = ToolRegistry([BoomTool()]).dispatch(ToolCall(id="c1", name="boom", arguments={}))
    assert outcome.result.ok is False
    assert "kaboom" in outcome.result.content
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_tools_base.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.tools'`

- [ ] **Step 3: Implement `tools/base.py`**

`src/agentfix/tools/__init__.py`: empty file.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from agentfix.llm.types import ToolCall

MAX_TOOL_OUTPUT_CHARS = 2000
MAX_FILE_READ_CHARS = 4000
TRUNCATION_MARKER = "\n[...truncated]"


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    name: str
    result: ToolResult

    def as_message(self) -> dict:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": self.result.content,
        }


class Tool(Protocol):
    name: str
    description: str
    parameters: dict

    def run(self, **kwargs) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def dispatch(self, call: ToolCall) -> ToolOutcome:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return ToolOutcome(
                call.id,
                call.name,
                ToolResult(False, f"No such tool: {call.name}. Available tools: {available}"),
            )

        missing = [
            key for key in tool.parameters.get("required", []) if key not in call.arguments
        ]
        if missing:
            return ToolOutcome(
                call.id,
                call.name,
                ToolResult(False, f"Missing required argument(s): {', '.join(missing)}"),
            )

        try:
            result = tool.run(**call.arguments)
        except TypeError as error:
            return ToolOutcome(call.id, call.name, ToolResult(False, f"Bad arguments: {error}"))
        except Exception as error:  # a tool crash must not kill the run
            return ToolOutcome(
                call.id, call.name, ToolResult(False, f"Tool raised {type(error).__name__}: {error}")
            )

        return ToolOutcome(call.id, call.name, result)
```

The broad `except Exception` is deliberate and is the layer's contract: a tool failure becomes an observation the model can act on. It is not silent — the exception type and message go straight into the content the model reads.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_tools_base.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentfix/tools tests/test_tools_base.py
git commit -m "feat: tool protocol, registry, and errors-as-observations dispatch"
```

---

### Task 5: Filesystem and test-running tools

**Files:**
- Create: `src/agentfix/tools/fs.py`, `src/agentfix/tools/tests_tool.py`, `tests/test_tools_fs.py`, `tests/test_tools_tests.py`

**Interfaces:**
- Consumes: `ToolResult`, `truncate`, `MAX_FILE_READ_CHARS` (Task 4); `ExecutionBackend`, `ExecResult` (Task 3)
- Produces:
  - `PathEscapeError(Exception)`
  - `resolve_in_root(root: Path, candidate: str) -> Path`
  - `ListFilesTool(root: Path)`, `ReadFileTool(root: Path)`, `WriteFileTool(root: Path)`
  - `RunTestsTool(root: Path, command: tuple[str, ...], backend: ExecutionBackend, timeout_s: int = 10)` with attribute `last_result: ExecResult | None`

- [ ] **Step 1: Write the failing test for fs tools**

`tests/test_tools_fs.py`:

```python
import pytest

from agentfix.tools.fs import (
    ListFilesTool,
    PathEscapeError,
    ReadFileTool,
    WriteFileTool,
    resolve_in_root,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text("def test_add():\n    pass\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    return tmp_path


def test_resolve_in_root_accepts_paths_inside(repo):
    assert resolve_in_root(repo, "src/calc.py") == repo / "src" / "calc.py"


@pytest.mark.parametrize("bad", ["../outside.py", "/etc/passwd", "src/../../escape.py"])
def test_resolve_in_root_rejects_escapes(repo, bad):
    with pytest.raises(PathEscapeError):
        resolve_in_root(repo, bad)


def test_list_files_lists_sources_and_hides_pycache(repo):
    result = ListFilesTool(repo).run()
    assert result.ok is True
    assert "src/calc.py" in result.content
    assert "tests/test_calc.py" in result.content
    assert "__pycache__" not in result.content


def test_read_file_returns_contents(repo):
    result = ReadFileTool(repo).run(path="src/calc.py")
    assert result.ok is True
    assert "return a - b" in result.content


def test_read_file_truncates_large_files(repo):
    (repo / "big.py").write_text("# pad\n" * 5000, encoding="utf-8")
    result = ReadFileTool(repo).run(path="big.py")
    assert "[...truncated]" in result.content


def test_read_missing_file_is_an_observation_listing_alternatives(repo):
    result = ReadFileTool(repo).run(path="src/typo.py")
    assert result.ok is False
    assert "src/calc.py" in result.content


def test_read_file_outside_root_is_refused(repo):
    result = ReadFileTool(repo).run(path="../../../etc/passwd")
    assert result.ok is False
    assert "outside" in result.content.lower()


def test_write_file_replaces_contents(repo):
    result = WriteFileTool(repo).run(path="src/calc.py", content="def add(a, b):\n    return a + b\n")
    assert result.ok is True
    assert (repo / "src" / "calc.py").read_text(encoding="utf-8").endswith("return a + b\n")


def test_write_file_rejects_syntax_errors_before_saving(repo):
    original = (repo / "src" / "calc.py").read_text(encoding="utf-8")
    result = WriteFileTool(repo).run(path="src/calc.py", content="def add(a, b)\n    return a + b\n")
    assert result.ok is False
    assert "syntax" in result.content.lower()
    assert (repo / "src" / "calc.py").read_text(encoding="utf-8") == original


def test_write_file_outside_root_is_refused(repo):
    result = WriteFileTool(repo).run(path="../evil.py", content="x = 1\n")
    assert result.ok is False
    assert "outside" in result.content.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_tools_fs.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.tools.fs'`

- [ ] **Step 3: Implement `tools/fs.py`**

```python
from __future__ import annotations

import ast
from pathlib import Path

from agentfix.tools.base import MAX_FILE_READ_CHARS, ToolResult, truncate

IGNORED_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the task root."""


def resolve_in_root(root: Path, candidate: str) -> Path:
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathEscapeError(f"{candidate} resolves outside the task root")
    return resolved


def _relative_files(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        found.append(str(path.relative_to(root)))
    return found


class ListFilesTool:
    name = "list_files"
    description = "List the Python files in the project, relative to the project root."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self) -> ToolResult:
        files = _relative_files(self.root)
        if not files:
            return ToolResult(True, "The project contains no Python files.")
        return ToolResult(True, truncate("\n".join(files)))


class ReadFileTool:
    name = "read_file"
    description = "Read the contents of one file in the project."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root"}
        },
        "required": ["path"],
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, path: str) -> ToolResult:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            return ToolResult(False, f"Refused: {path} is outside the project root.")

        if not target.is_file():
            available = ", ".join(_relative_files(self.root))
            return ToolResult(False, f"No such file: {path}. Files in this project: {available}")

        return ToolResult(True, truncate(target.read_text(encoding="utf-8"), MAX_FILE_READ_CHARS))


class WriteFileTool:
    name = "write_file"
    description = (
        "Replace the entire contents of one file. Provide the complete new file, not a diff."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root"},
            "content": {"type": "string", "description": "The complete new file contents"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, path: str, content: str) -> ToolResult:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            return ToolResult(False, f"Refused: {path} is outside the project root.")

        try:
            ast.parse(content)
        except SyntaxError as error:
            return ToolResult(
                False, f"Not written — the content has a syntax error on line {error.lineno}: {error.msg}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {len(content)} characters to {path}.")
```

The syntax pre-check in `WriteFileTool` is what stops a small model from bricking the repo and then spending its whole step budget on import errors. Rejecting the write and saying why is a much cheaper observation.

- [ ] **Step 4: Run fs tests to verify they pass**

```bash
uv run pytest tests/test_tools_fs.py -v
```

Expected: 13 passed

- [ ] **Step 5: Write the failing test for `RunTestsTool`**

`tests/test_tools_tests.py`:

```python
import sys

from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.tests_tool import RunTestsTool

PYTEST_CMD = (sys.executable, "-m", "pytest", "-q")


def _tool(root):
    return RunTestsTool(root=root, command=PYTEST_CMD, backend=SubprocessBackend())


def test_reports_failure_and_stores_last_result(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 1 == 2\n", encoding="utf-8")
    tool = _tool(tmp_path)

    result = tool.run()

    assert result.ok is True          # the tool ran successfully...
    assert "FAILED" in result.content or "failed" in result.content
    assert tool.last_result is not None
    assert tool.last_result.passed is False   # ...but the tests did not pass


def test_reports_success(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    tool = _tool(tmp_path)

    result = tool.run()

    assert "passed" in result.content.lower()
    assert tool.last_result.passed is True
```

The distinction asserted here matters: `ToolResult.ok` means *the tool worked*, while `last_result.passed` means *the tests passed*. Conflating them is how a stop condition ends up trusting the wrong signal.

- [ ] **Step 6: Run test to verify it fails**

```bash
uv run pytest tests/test_tools_tests.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.tools.tests_tool'`

- [ ] **Step 7: Implement `tools/tests_tool.py`**

```python
from __future__ import annotations

from pathlib import Path

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import ToolResult


class RunTestsTool:
    name = "run_tests"
    description = "Run the project's test suite and return the result. This is the source of truth."
    parameters = {"type": "object", "properties": {}}

    def __init__(
        self,
        root: Path,
        command: tuple[str, ...],
        backend: ExecutionBackend,
        timeout_s: int = 10,
    ) -> None:
        self.root = root
        self.command = command
        self.backend = backend
        self.timeout_s = timeout_s
        self.last_result: ExecResult | None = None

    def run(self) -> ToolResult:
        result = self.backend.run(self.root, self.command, timeout_s=self.timeout_s)
        self.last_result = result

        headline = "All tests passed." if result.passed else "Tests failed."
        return ToolResult(True, f"{headline}\n\n{result.output}".strip())
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
uv run pytest tests/test_tools_fs.py tests/test_tools_tests.py -v
```

Expected: 15 passed

- [ ] **Step 9: Commit**

```bash
git add src/agentfix/tools/fs.py src/agentfix/tools/tests_tool.py tests/test_tools_fs.py tests/test_tools_tests.py
git commit -m "feat: filesystem and run_tests tools with path confinement and syntax pre-check"
```

---

### Task 6: Task loader, workspace isolation, and the first fixture

**Files:**
- Create: `src/agentfix/tasks/__init__.py`, `src/agentfix/tasks/loader.py`, `tests/test_tasks.py`
- Create fixture: `tasks/workshop/01-shopcart/task.json`, `tasks/workshop/01-shopcart/repo/shopcart/{__init__.py,cart.py,pricing.py}`, `tasks/workshop/01-shopcart/repo/tests/test_cart.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Task(task_id: str, root: Path, template_dir: Path, test_command: tuple[str, ...], expected_failures: tuple[str, ...], prompt: str)`
  - `load_task(task_dir: Path) -> Task`
  - `workspace(task: Task) -> Iterator[Path]` — a `@contextmanager` yielding a temp copy of `template_dir`

- [ ] **Step 1: Write the failing test**

`tests/test_tasks.py`:

```python
import sys
from pathlib import Path

from agentfix.tasks.loader import load_task, workspace

FIXTURE = Path("tasks/workshop/01-shopcart")


def test_load_task_reads_metadata():
    task = load_task(FIXTURE)
    assert task.task_id == "01-shopcart"
    assert task.test_command[:3] == (sys.executable, "-m", "pytest")
    assert task.expected_failures


def test_workspace_yields_a_writable_copy_and_cleans_up():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        assert (work_dir / "shopcart" / "cart.py").is_file()
        (work_dir / "shopcart" / "cart.py").write_text("# clobbered\n", encoding="utf-8")
        recorded = work_dir
    assert not recorded.exists()


def test_template_is_never_mutated_by_a_workspace():
    task = load_task(FIXTURE)
    before = (task.template_dir / "shopcart" / "cart.py").read_text(encoding="utf-8")
    with workspace(task) as work_dir:
        (work_dir / "shopcart" / "cart.py").write_text("# clobbered\n", encoding="utf-8")
    assert (task.template_dir / "shopcart" / "cart.py").read_text(encoding="utf-8") == before


def test_fixture_starts_red():
    from agentfix.sandbox.subprocess_backend import SubprocessBackend

    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_tasks.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.tasks'`

- [ ] **Step 3: Create the `01-shopcart` fixture**

`tasks/workshop/01-shopcart/task.json`:

```json
{
  "task_id": "01-shopcart",
  "test_command": ["-m", "pytest", "-q"],
  "expected_failures": ["test_total_with_tax"],
  "prompt": "The test suite for this project is failing. Find the bug and fix it."
}
```

`tasks/workshop/01-shopcart/repo/shopcart/__init__.py`: empty file.

`tasks/workshop/01-shopcart/repo/shopcart/pricing.py`:

```python
TAX_RATE = 0.2


def with_tax(amount: float) -> float:
    return amount * (1 + TAX_RATE)
```

`tasks/workshop/01-shopcart/repo/shopcart/cart.py`:

```python
from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices)) - TAX_ROUNDING


TAX_ROUNDING = 0.01
```

`tasks/workshop/01-shopcart/repo/tests/test_cart.py`:

```python
from shopcart.cart import subtotal, total_with_tax


def test_subtotal():
    assert subtotal([1.0, 2.0, 3.0]) == 6.0


def test_total_with_tax():
    assert total_with_tax([10.0]) == 12.0
```

The bug is the stray `- TAX_ROUNDING`: obvious once read, and the failing test names the module it lives in. That is intentional for the first fixture.

- [ ] **Step 4: Implement `tasks/loader.py`**

`src/agentfix/tasks/__init__.py`: empty file.

```python
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_PROMPT = "The test suite for this project is failing. Find the bug and fix it."


@dataclass(frozen=True)
class Task:
    task_id: str
    root: Path
    template_dir: Path
    test_command: tuple[str, ...]
    expected_failures: tuple[str, ...]
    prompt: str


def load_task(task_dir: Path) -> Task:
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))

    command = tuple(meta.get("test_command", ["-m", "pytest", "-q"]))
    if command and command[0].startswith("-"):
        command = (sys.executable, *command)

    return Task(
        task_id=meta.get("task_id", task_dir.name),
        root=task_dir,
        template_dir=task_dir / "repo",
        test_command=command,
        expected_failures=tuple(meta.get("expected_failures", ())),
        prompt=meta.get("prompt", DEFAULT_PROMPT),
    )


@contextmanager
def workspace(task: Task) -> Iterator[Path]:
    """Copy the pristine template into a temp dir so every run starts identical."""
    temp_root = Path(tempfile.mkdtemp(prefix=f"agentfix_{task.task_id}_"))
    work_dir = temp_root / "repo"
    try:
        shutil.copytree(task.template_dir, work_dir)
        yield work_dir
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
```

Storing `test_command` with `sys.executable` prepended keeps the sandbox backend free of any knowledge about how tests are invoked.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_tasks.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/agentfix/tasks tasks/workshop/01-shopcart tests/test_tasks.py
git commit -m "feat: task loader with immutable fixtures and the 01-shopcart task"
```

---

### Task 7: Agent loop, trace, and stop condition

**Files:**
- Create: `src/agentfix/agent/__init__.py`, `src/agentfix/agent/trace.py`, `src/agentfix/agent/loop.py`, `tests/test_trace.py`, `tests/test_loop.py`

**Interfaces:**
- Consumes: `LLMClient`, `LLMReply` (Task 2); `ToolRegistry`, `ToolOutcome` (Task 4); `RunTestsTool` (Task 5); `Task` (Task 6)
- Produces:
  - `TraceEvent(step: int, kind: str, name: str, detail: str, prompt_tokens: int, latency_s: float)`
  - `Tracer(verbose: bool = False)` with `events: list[TraceEvent]`, `record(event) -> None`, `as_json() -> list[dict]`
  - `MAX_STEPS: int = 10`
  - `system_prompt(registry: ToolRegistry) -> str`, `task_prompt(task: Task) -> str`
  - `is_done(run_tests: RunTestsTool) -> bool`
  - `AgentResult(task_id, solved, steps_used, prompt_tokens, completion_tokens, duration_s, trace)`
  - `run_agent(task, work_dir, llm, registry, run_tests, max_steps=MAX_STEPS, tracer=None) -> AgentResult`

- [ ] **Step 1: Write the failing test for the tracer**

`tests/test_trace.py`:

```python
from agentfix.agent.trace import Tracer, TraceEvent


def test_records_events_and_serialises_them():
    tracer = Tracer()
    tracer.record(TraceEvent(1, "llm", "assistant", "asked for run_tests", 120, 0.4))

    assert len(tracer.events) == 1
    assert tracer.as_json()[0]["prompt_tokens"] == 120


def test_verbose_tracer_prints_each_event(capsys):
    Tracer(verbose=True).record(TraceEvent(2, "tool", "run_tests", "Tests failed.", 300, 1.2))
    assert "run_tests" in capsys.readouterr().out
```

- [ ] **Step 2: Run it and confirm it fails, then implement `trace.py`**

```bash
uv run pytest tests/test_trace.py -v      # FAIL: no module named 'agentfix.agent.trace'
```

`src/agentfix/agent/__init__.py`: empty file.

`src/agentfix/agent/trace.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TraceEvent:
    step: int
    kind: str
    name: str
    detail: str
    prompt_tokens: int
    latency_s: float


class Tracer:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.verbose:
            marker = "→" if event.kind == "llm" else "←"
            detail = event.detail.replace("\n", " ")[:100]
            print(
                f"  step {event.step} {marker} {event.kind}:{event.name}  "
                f"[ctx {event.prompt_tokens} tok, {event.latency_s:.1f}s]  {detail}"
            )

    def as_json(self) -> list[dict]:
        return [asdict(event) for event in self.events]
```

`prompt_tokens` is printed on every line on purpose: students watch the context grow turn by turn, which is what makes the prefill cost concrete.

```bash
uv run pytest tests/test_trace.py -v      # 2 passed
```

- [ ] **Step 3: Write the failing test for the loop**

`tests/test_loop.py`:

```python
import sys
from pathlib import Path

import pytest

from agentfix.agent.loop import AgentResult, is_done, run_agent
from agentfix.agent.trace import Tracer
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool

FIXTURE = Path("tasks/workshop/01-shopcart")
FIXED_CART = """from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices))
"""


def _build(work_dir, task):
    run_tests = RunTestsTool(work_dir, task.test_command, SubprocessBackend(), timeout_s=30)
    registry = ToolRegistry(
        [ListFilesTool(work_dir), ReadFileTool(work_dir), WriteFileTool(work_dir), run_tests]
    )
    return registry, run_tests


def test_is_done_is_false_before_tests_run(tmp_path):
    run_tests = RunTestsTool(tmp_path, (sys.executable, "-m", "pytest", "-q"), SubprocessBackend())
    assert is_done(run_tests) is False


def test_agent_solves_the_task_when_it_writes_the_fix():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}, call_id="c2"),
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c3"
                ),
                assistant_text("Fixed the stray subtraction."),
            ]
        )

        result = run_agent(task, work_dir, llm, registry, run_tests)

    assert isinstance(result, AgentResult)
    assert result.solved is True
    assert result.steps_used == 4


def test_agent_is_not_solved_when_the_model_merely_claims_success():
    """The whole point of stage 3: assertions are not verification."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_text("DONE — I have fixed the bug."),
            ]
        )

        result = run_agent(task, work_dir, llm, registry, run_tests)

    assert result.solved is False


def test_history_is_append_only_and_carries_tool_call_ids():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("run_tests", {}, call_id="c1"), assistant_text("giving up")]
        )
        run_agent(task, work_dir, llm, registry, run_tests)

    first_history, second_history = llm.calls
    assert second_history[: len(first_history)] == first_history
    assert second_history[-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "run_tests",
        "content": second_history[-1]["content"],
    }


def test_step_budget_is_a_hard_cap():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient([assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(2)])

        result = run_agent(task, work_dir, llm, registry, run_tests, max_steps=2)

    assert result.steps_used == 2
    assert result.solved is False


def test_repeated_identical_call_is_guarded_instead_of_re_executed():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}, call_id="c1"),
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}, call_id="c2"),
                assistant_text("stuck"),
            ]
        )
        tracer = Tracer()

        run_agent(task, work_dir, llm, registry, run_tests, tracer=tracer)

    third_history = llm.calls[2]
    assert "already called" in third_history[-1]["content"]


def test_trace_records_every_llm_and_tool_event():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient([assistant_tool_call("list_files", {}, call_id="c1"), assistant_text("ok")])
        tracer = Tracer()

        result = run_agent(task, work_dir, llm, registry, run_tests, tracer=tracer)

    kinds = [event.kind for event in tracer.events]
    assert kinds == ["llm", "tool", "llm"]
    assert result.trace == tuple(tracer.events)
```

- [ ] **Step 4: Run test to verify it fails**

```bash
uv run pytest tests/test_loop.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agentfix.agent.loop'`

- [ ] **Step 5: Implement `agent/loop.py`**

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agentfix.agent.trace import Tracer, TraceEvent
from agentfix.llm.types import LLMClient, ToolCall
from agentfix.tasks.loader import Task
from agentfix.tools.base import ToolRegistry
from agentfix.tools.tests_tool import RunTestsTool

MAX_STEPS = 10


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


def _describe(reply) -> str:
    if reply.tool_calls:
        return "calls " + ", ".join(call.name for call in reply.tool_calls)
    return (reply.message.get("content") or "").strip()
```

Note the two `break`s at the end: when the model stops calling tools the run is over either way, but `solved` is decided **only** by `is_done`. That separation is exactly what stage 3 teaches, and it is why `test_agent_is_not_solved_when_the_model_merely_claims_success` passes.

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_loop.py -v
```

Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add src/agentfix/agent tests/test_trace.py tests/test_loop.py
git commit -m "feat: agent loop with append-only history, loop guard, and verification-based stop"
```

---

### Task 8: CLI `solve` and `doctor`

**Files:**
- Modify: `src/agentfix/cli.py`
- Create: `src/agentfix/runner.py`, `src/agentfix/doctor.py`, `tests/test_runner.py`, `tests/test_doctor.py`

**Interfaces:**
- Consumes: everything from Tasks 2–7
- Produces:
  - `solve_task(task_dir: Path, llm: LLMClient | None = None, verbose: bool = False, max_steps: int = MAX_STEPS) -> AgentResult`
  - `Check(name: str, ok: bool, detail: str)`, `run_checks() -> list[Check]`, `report(checks: list[Check]) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:

```python
from pathlib import Path

from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.runner import solve_task

FIXTURE = Path("tasks/workshop/01-shopcart")
FIXED_CART = """from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices))
"""


def test_solve_task_wires_everything_together():
    llm = FakeLLMClient(
        [
            assistant_tool_call("run_tests", {}, call_id="c1"),
            assistant_tool_call(
                "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c2"
            ),
            assistant_text("done"),
        ]
    )

    result = solve_task(FIXTURE, llm=llm)

    assert result.solved is True
    assert result.task_id == "01-shopcart"
```

`tests/test_doctor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_runner.py tests/test_doctor.py -v
```

Expected: FAIL — no module named `agentfix.runner`

- [ ] **Step 3: Implement `runner.py`**

```python
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
                WriteFileTool(work_dir),
                run_tests,
            ]
        )
        return run_agent(
            task, work_dir, llm, registry, run_tests, max_steps=max_steps, tracer=Tracer(verbose)
        )
```

- [ ] **Step 4: Implement `doctor.py`**

```python
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from agentfix.config import LLMConfig

PULL_HINT = "run: ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _check_python() -> Check:
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 12)
    return Check("python", ok, f"{sys.version.split()[0]}" + ("" if ok else " — need >= 3.12"))


def _check_ollama_installed() -> Check:
    path = shutil.which("ollama")
    return Check("ollama installed", path is not None, path or "not found — install from ollama.com")


def _check_model_present(config: LLMConfig) -> Check:
    if shutil.which("ollama") is None:
        return Check("model present", False, f"cannot check — {PULL_HINT}")
    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    short = config.model.split("/")[-1].lower()
    ok = short in listing.stdout.lower()
    return Check("model present", ok, config.model if ok else f"missing — {PULL_HINT}")


def _check_generation(config: LLMConfig) -> Check:
    from agentfix.llm.client import OllamaClient

    try:
        started = time.time()
        reply = OllamaClient(config).chat([{"role": "user", "content": "Reply with: ok"}])
        elapsed = max(time.time() - started, 1e-6)
        rate = reply.completion_tokens / elapsed
        return Check("generation", True, f"{rate:.0f} tok/s")
    except Exception as error:
        return Check("generation", False, f"{type(error).__name__}: {error}")


def _check_sandbox() -> Check:
    import tempfile
    from pathlib import Path

    from agentfix.sandbox.base import get_backend

    with tempfile.TemporaryDirectory() as temp:
        Path(temp, "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        result = get_backend().run(Path(temp), (sys.executable, "-m", "pytest", "-q"))
    return Check("sandbox", result.passed, "executes tests" if result.passed else result.output)


def run_checks() -> list[Check]:
    config = LLMConfig.from_env()
    checks = [_check_python(), _check_ollama_installed(), _check_model_present(config)]
    if checks[-1].ok:
        checks.append(_check_generation(config))
    checks.append(_check_sandbox())
    return checks


def report(checks: list[Check]) -> int:
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")

    failed = [check for check in checks if not check.ok]
    if failed:
        print(f"\nNOT READY — {len(failed)} check(s) failed. Fix the FAIL lines above.")
        return 1

    rate = next((c.detail for c in checks if c.name == "generation"), "")
    print(f"\nREADY {rate}".rstrip())
    return 0
```

- [ ] **Step 5: Wire the CLI**

Replace `src/agentfix/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentfix import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentfix", description="A teaching coding agent")
    parser.add_argument("--version", action="store_true", help="print version and exit")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check that this machine is ready for the workshop")

    solve = sub.add_parser("solve", help="run the agent on one task")
    solve.add_argument("task_dir", type=Path)
    solve.add_argument("--verbose", action="store_true", help="print the agent's trace")
    solve.add_argument("--max-steps", type=int, default=MAX_STEPS)  # 10

    evaluate = sub.add_parser("eval", help="run the agent over a suite of tasks")
    evaluate.add_argument("--suite", default="workshop", choices=["workshop", "humanevalfix"])
    evaluate.add_argument("--limit", type=int, default=3)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"agentfix {__version__}")
        return 0

    if args.command == "doctor":
        from agentfix.doctor import report, run_checks

        return report(run_checks())

    if args.command == "solve":
        from agentfix.runner import solve_task

        result = solve_task(args.task_dir, verbose=args.verbose, max_steps=args.max_steps)
        status = "SOLVED" if result.solved else "NOT SOLVED"
        print(
            f"\n{status}  {result.task_id}  "
            f"steps={result.steps_used}  tokens={result.prompt_tokens + result.completion_tokens}  "
            f"{result.duration_s}s"
        )
        return 0 if result.solved else 1

    if args.command == "eval":
        from agentfix.eval.runner import run_suite

        return run_suite(args.suite, limit=args.limit)

    parser.print_help()
    return 0


def cli_entry() -> None:
    sys.exit(main())
```

`tests/test_cli.py` from Task 1 asserted exit code 2 for `main(["nonsense"])`; `add_subparsers` still produces an argparse error there, so that test continues to pass.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -v
```

Expected: all tests pass. The `eval` import will fail only if invoked — Task 10 creates it.

- [ ] **Step 7: Verify against the real model**

```bash
uv run agentfix doctor
uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

Expected: `doctor` prints `READY` with a tok/s figure; `solve` prints a trace and ends `SOLVED 01-shopcart`. If the agent does not solve it, capture the trace before changing anything — a prompt problem and a tool-contract problem look different in the trace.

- [ ] **Step 8: Commit**

```bash
git add src/agentfix/cli.py src/agentfix/runner.py src/agentfix/doctor.py tests/test_runner.py tests/test_doctor.py
git commit -m "feat: agentfix solve and doctor commands"
```

---

### Task 9: Fixtures 02-invoice and 03-parser

**Files:**
- Create: `tasks/workshop/02-invoice/{task.json,repo/…}`, `tasks/workshop/03-parser/{task.json,repo/…}`
- Modify: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `load_task`, `workspace` (Task 6)
- Produces: two more task directories usable by `solve_task` and the eval suite

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tasks.py`:

```python
import pytest


@pytest.mark.parametrize(
    "task_dir", ["tasks/workshop/01-shopcart", "tasks/workshop/02-invoice", "tasks/workshop/03-parser"]
)
def test_every_workshop_fixture_starts_red(task_dir):
    from agentfix.sandbox.subprocess_backend import SubprocessBackend

    task = load_task(Path(task_dir))
    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is False


def test_02_invoice_bug_is_not_in_the_file_the_test_names():
    """The pedagogical contract of fixture 02 — asserted so it cannot silently regress."""
    task = load_task(Path("tasks/workshop/02-invoice"))
    failing_test_file = (task.template_dir / "tests" / "test_invoice.py").read_text(encoding="utf-8")
    buggy_file = (task.template_dir / "billing" / "discounts.py").read_text(encoding="utf-8")
    assert "discounts" not in failing_test_file
    assert ">=" in buggy_file
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run pytest tests/test_tasks.py -v
```

Expected: FAIL — `tasks/workshop/02-invoice/task.json` does not exist

- [ ] **Step 3: Create `02-invoice` — the bug is not where the test points**

`tasks/workshop/02-invoice/task.json`:

```json
{
  "task_id": "02-invoice",
  "test_command": ["-m", "pytest", "-q"],
  "expected_failures": ["test_invoice_total_applies_bulk_discount"],
  "prompt": "The test suite for this project is failing. Find the bug and fix it."
}
```

`repo/billing/__init__.py`: empty file.

`repo/billing/discounts.py`:

```python
BULK_THRESHOLD = 10
BULK_RATE = 0.1


def bulk_discount(quantity: int, amount: float) -> float:
    if quantity > BULK_THRESHOLD:
        return amount * BULK_RATE
    return 0.0
```

`repo/billing/invoice.py`:

```python
from billing.discounts import bulk_discount


def line_total(unit_price: float, quantity: int) -> float:
    return unit_price * quantity


def invoice_total(unit_price: float, quantity: int) -> float:
    gross = line_total(unit_price, quantity)
    return gross - bulk_discount(quantity, gross)
```

`repo/tests/test_invoice.py`:

```python
from billing.invoice import invoice_total, line_total


def test_line_total():
    assert line_total(2.0, 5) == 10.0


def test_invoice_total_without_discount():
    assert invoice_total(2.0, 5) == 10.0


def test_invoice_total_applies_bulk_discount():
    assert invoice_total(2.0, 10) == 18.0
```

The bug is `>` instead of `>=` in `discounts.py`, but the only failing test lives in `test_invoice.py` and names `invoice_total`. The agent must call `list_files` and `read_file` to find it. This fixture is what justifies the tool layer.

- [ ] **Step 4: Create `03-parser` — two files involved**

`tasks/workshop/03-parser/task.json`:

```json
{
  "task_id": "03-parser",
  "test_command": ["-m", "pytest", "-q"],
  "expected_failures": ["test_parse_skips_comments", "test_parse_trims_whitespace"],
  "prompt": "The test suite for this project is failing. Find the bug and fix it."
}
```

`repo/config/__init__.py`: empty file.

`repo/config/tokens.py`:

```python
COMMENT_PREFIX = "//"


def is_comment(line: str) -> bool:
    return line.startswith(COMMENT_PREFIX)


def clean(line: str) -> str:
    return line
```

`repo/config/parser.py`:

```python
from config.tokens import clean, is_comment


def parse(text: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = clean(raw_line)
        if not line or is_comment(line):
            continue
        key, _, value = line.partition("=")
        settings[clean(key)] = clean(value)
    return settings
```

`repo/tests/test_parser.py`:

```python
from config.parser import parse


def test_parse_simple():
    assert parse("a=1\nb=2") == {"a": "1", "b": "2"}


def test_parse_skips_comments():
    assert parse("# ignored\na=1") == {"a": "1"}


def test_parse_trims_whitespace():
    assert parse("  a =  1  ") == {"a": "1"}
```

Two defects, both in `tokens.py`: `COMMENT_PREFIX` should be `"#"`, and `clean` should `strip()`. Fixing only one leaves a test red, so the agent must iterate — which is what makes the step budget and the loop visible.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_tasks.py -v
```

Expected: all pass, including the three parametrised red-fixture checks.

- [ ] **Step 6: Sanity-check with the real model**

```bash
uv run agentfix solve tasks/workshop/02-invoice --verbose
uv run agentfix solve tasks/workshop/03-parser --verbose
```

Record whether each is solved and in how many steps. If 03 exceeds the 6-step budget consistently, note it in `WORKSHOP.md` as the "agents have budgets" talking point rather than raising the cap.

- [ ] **Step 7: Commit**

```bash
git add tasks/workshop/02-invoice tasks/workshop/03-parser tests/test_tasks.py
git commit -m "feat: add 02-invoice and 03-parser task fixtures"
```

---

### Task 10: Eval runner and vendored HumanEvalFix subset

**Files:**
- Create: `src/agentfix/eval/__init__.py`, `src/agentfix/eval/runner.py`, `src/agentfix/eval/humanevalfix.py`, `scripts/vendor_humanevalfix.py`, `tasks/humanevalfix/subset.json`, `tests/test_eval.py`
- Copy from predecessor: `results/legacy/` (the old `*.json` and `comparative_results.csv`)

**Interfaces:**
- Consumes: `solve_task` (Task 8), `load_task` (Task 6)
- Produces:
  - `EvalReport(suite: str, results: tuple[AgentResult, ...])` with `pass_at_1: float`, `to_json() -> dict`, `format_table() -> str`
  - `evaluate(task_dirs: list[Path], llm=None, max_steps=MAX_STEPS) -> EvalReport`
  - `run_suite(suite: str, limit: int = 3) -> int`
  - `HumanEvalFixRow(task_id: str, buggy_code: str, tests: str, entry_point: str)`
  - `load_vendored_rows(path: Path = Path("tasks/humanevalfix/subset.json")) -> list[HumanEvalFixRow]`
  - `write_task_dir(row: HumanEvalFixRow, dest: Path) -> Path`
  - `load_hf_rows(sample: int | None = None, seed: int = 42) -> list[HumanEvalFixRow]` — requires the `[eval]` extra

Note: the predecessor's `humanevalfix.py` names its dataclass `Task`, which would collide with `agentfix.tasks.loader.Task`. It is renamed `HumanEvalFixRow` here.

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:

```python
import json
from pathlib import Path

from agentfix.agent.loop import AgentResult
from agentfix.eval.humanevalfix import HumanEvalFixRow, load_vendored_rows, write_task_dir
from agentfix.eval.runner import EvalReport


def _result(task_id: str, solved: bool) -> AgentResult:
    return AgentResult(task_id, solved, 3, 100, 50, 4.2, ())


def test_pass_at_1_is_the_solved_fraction():
    report = EvalReport("workshop", (_result("a", True), _result("b", False)))
    assert report.pass_at_1 == 0.5


def test_pass_at_1_is_zero_for_an_empty_suite():
    assert EvalReport("workshop", ()).pass_at_1 == 0.0


def test_table_shows_steps_and_tokens():
    table = EvalReport("workshop", (_result("01-shopcart", True),)).format_table()
    assert "01-shopcart" in table
    assert "150" in table       # 100 prompt + 50 completion


def test_json_round_trips():
    payload = EvalReport("workshop", (_result("a", True),)).to_json()
    assert json.loads(json.dumps(payload))["pass_at_1"] == 1.0


def test_vendored_subset_has_twenty_usable_rows():
    rows = load_vendored_rows()
    assert len(rows) == 20
    assert all(row.entry_point and row.buggy_code and row.tests for row in rows)


def test_write_task_dir_produces_a_loadable_red_task(tmp_path):
    from agentfix.sandbox.subprocess_backend import SubprocessBackend
    from agentfix.tasks.loader import load_task, workspace

    row = HumanEvalFixRow(
        task_id="HumanEval/0",
        buggy_code="def add(a, b):\n    return a - b\n",
        tests="from candidate import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        entry_point="add",
    )
    task_dir = write_task_dir(row, tmp_path)
    task = load_task(task_dir)

    with workspace(task) as work_dir:
        result = SubprocessBackend().run(work_dir, task.test_command, timeout_s=30)
    assert result.passed is False
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run pytest tests/test_eval.py -v
```

Expected: FAIL — no module named `agentfix.eval`

- [ ] **Step 3: Implement `eval/humanevalfix.py`**

`src/agentfix/eval/__init__.py`: empty file.

```python
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

VENDORED_SUBSET = Path("tasks/humanevalfix/subset.json")


@dataclass(frozen=True)
class HumanEvalFixRow:
    task_id: str
    buggy_code: str
    tests: str
    entry_point: str


def load_vendored_rows(path: Path = VENDORED_SUBSET) -> list[HumanEvalFixRow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [HumanEvalFixRow(**row) for row in payload]


def write_task_dir(row: HumanEvalFixRow, dest: Path) -> Path:
    slug = row.task_id.replace("/", "-").lower()
    task_dir = Path(dest) / slug
    repo = task_dir / "repo"
    (repo / "tests").mkdir(parents=True, exist_ok=True)

    (repo / "candidate.py").write_text(row.buggy_code.rstrip() + "\n", encoding="utf-8")
    (repo / "tests" / "test_candidate.py").write_text(row.tests.rstrip() + "\n", encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parent))\n",
        encoding="utf-8",
    )

    task_dir.joinpath("task.json").write_text(
        json.dumps(
            {
                "task_id": slug,
                "test_command": ["-m", "pytest", "-q"],
                "expected_failures": [],
                "prompt": (
                    f"The function `{row.entry_point}` in candidate.py is buggy and its tests "
                    "fail. Fix it."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return task_dir


def load_hf_rows(sample: int | None = None, seed: int = 42) -> list[HumanEvalFixRow]:
    """Full dataset. Requires the [eval] extra: uv sync --extra eval"""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("The full benchmark needs: uv sync --extra eval") from error

    dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
    rows = [
        HumanEvalFixRow(
            task_id=item["task_id"],
            buggy_code=item["buggy_solution"].rstrip(),
            tests=f"from candidate import {item['entry_point']}\n\n{item['test'].rstrip()}",
            entry_point=item["entry_point"],
        )
        for item in dataset
    ]
    if sample is not None and sample < len(rows):
        random.seed(seed)
        rows = random.sample(rows, sample)
    return rows


def dump_rows(rows: list[HumanEvalFixRow], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8"
    )
```

- [ ] **Step 4: Generate the vendored subset**

`scripts/vendor_humanevalfix.py`:

```python
"""Regenerate tasks/humanevalfix/subset.json. Needs: uv sync --extra eval"""

from pathlib import Path

from agentfix.eval.humanevalfix import VENDORED_SUBSET, dump_rows, load_hf_rows

if __name__ == "__main__":
    dump_rows(load_hf_rows(sample=20, seed=42), Path(VENDORED_SUBSET))
    print(f"wrote {VENDORED_SUBSET}")
```

Run it once and commit the output so the base install never needs `datasets`:

```bash
uv sync --extra eval
uv run python scripts/vendor_humanevalfix.py
uv sync                      # drop datasets again from the default env
```

- [ ] **Step 5: Implement `eval/runner.py`**

```python
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
                {k: v for k, v in asdict(result).items() if k != "trace"}
                for result in self.results
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
```

- [ ] **Step 6: Copy the legacy baseline**

```bash
mkdir -p results/legacy
cp "/Users/jelenaduric/Desktop/JetBrains/Python Ai Agent/results/"*.json results/legacy/ 2>/dev/null || true
cp "/Users/jelenaduric/Desktop/JetBrains/Python Ai Agent/results/comparative_results.csv" results/legacy/ 2>/dev/null || true
```

`.gitignore` already ignores `results/*.json` but not `results/legacy/`, so these are tracked while fresh run output is not.

- [ ] **Step 7: Run tests to verify they pass**

```bash
uv run pytest tests/test_eval.py -v
```

Expected: 6 passed

- [ ] **Step 8: Produce the pre-computed results the workshop discusses**

```bash
uv run agentfix eval --suite workshop --limit 3
uv run agentfix eval --suite humanevalfix --limit 20
```

Copy both JSON outputs to `results/precomputed/` and commit them — the eval segment is demo-only, so these are the numbers you show. Record the wall-clock time; put it in `WORKSHOP.md`.

- [ ] **Step 9: Commit**

```bash
git add src/agentfix/eval scripts tasks/humanevalfix results/legacy results/precomputed tests/test_eval.py
git commit -m "feat: eval runner with pass@1, vendored HumanEvalFix subset, and legacy baseline"
```

---

### Task 11: Exercises and teaching branches

**Files:**
- Create: `exercises/README.md`, `exercises/stage_1/{README.md,test_stage_1.py}`, `exercises/stage_2/{README.md,test_stage_2.py}`, `exercises/stage_3/{README.md,test_stage_3.py}`
- Modify (stub for students): `src/agentfix/tools/tests_tool.py`, `src/agentfix/agent/loop.py`

**Interfaces:**
- Consumes: everything
- Produces: `main` branch = student starting point; `solutions` branch + tags `stage-1-solution`, `stage-2-solution`, `stage-3-solution`

- [ ] **Step 1: Write the exercise tests (they pass now, against the finished code)**

`exercises/stage_1/test_stage_1.py`:

```python
"""Stage 1 — write the run_tests tool and its JSON schema."""

import sys
from pathlib import Path

from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.base import ToolRegistry
from agentfix.tools.tests_tool import RunTestsTool

PYTEST_CMD = (sys.executable, "-m", "pytest", "-q")


def test_tool_declares_a_valid_schema():
    tool = RunTestsTool(Path("."), PYTEST_CMD, SubprocessBackend())
    assert tool.name == "run_tests"
    assert tool.description, "the model chooses tools by their description — write one"
    assert tool.parameters["type"] == "object"


def test_schema_is_exported_to_the_model():
    registry = ToolRegistry([RunTestsTool(Path("."), PYTEST_CMD, SubprocessBackend())])
    names = [schema["function"]["name"] for schema in registry.schemas()]
    assert "run_tests" in names


def test_running_failing_tests_reports_failure(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 1 == 2\n", encoding="utf-8")
    tool = RunTestsTool(tmp_path, PYTEST_CMD, SubprocessBackend())

    result = tool.run()

    assert result.ok is True, "the tool ran, so ok is True even when the tests fail"
    assert tool.last_result.passed is False, "the tests failed, so last_result.passed is False"


def test_running_passing_tests_reports_success(tmp_path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    tool = RunTestsTool(tmp_path, PYTEST_CMD, SubprocessBackend())
    tool.run()
    assert tool.last_result.passed is True


def test_the_model_chooses_this_tool_when_told_tests_fail():
    """A schema the model can't understand is a schema it won't call."""
    llm = FakeLLMClient([assistant_tool_call("run_tests", {}), assistant_text("done")])
    registry = ToolRegistry([RunTestsTool(Path("."), PYTEST_CMD, SubprocessBackend())])

    reply = llm.chat([{"role": "user", "content": "tests fail"}], tools=registry.schemas())

    assert reply.tool_calls[0].name == "run_tests"
```

`exercises/stage_2/test_stage_2.py`:

```python
"""Stage 2 — dispatch tool calls in the loop and feed observations back."""

from pathlib import Path

from agentfix.agent.loop import run_agent
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool

FIXTURE = Path("tasks/workshop/01-shopcart")


def _build(work_dir, task):
    run_tests = RunTestsTool(work_dir, task.test_command, SubprocessBackend(), timeout_s=30)
    registry = ToolRegistry(
        [ListFilesTool(work_dir), ReadFileTool(work_dir), WriteFileTool(work_dir), run_tests]
    )
    return registry, run_tests


def test_tool_result_is_appended_as_a_tool_message():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="c1"), assistant_text("ok")]
        )
        run_agent(task, work_dir, llm, registry, run_tests)

    observation = llm.calls[1][-1]
    assert observation["role"] == "tool", "the observation must go back as a tool message"
    assert "shopcart/cart.py" in observation["content"]


def test_tool_call_id_is_carried_back():
    """Omitting tool_call_id is the single most common mistake here."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="xyz789"), assistant_text("ok")]
        )
        run_agent(task, work_dir, llm, registry, run_tests)

    assert llm.calls[1][-1].get("tool_call_id") == "xyz789"


def test_history_is_append_only():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [assistant_tool_call("list_files", {}, call_id="c1"), assistant_text("ok")]
        )
        run_agent(task, work_dir, llm, registry, run_tests)

    first, second = llm.calls
    assert second[: len(first)] == first, "earlier messages must never change — it kills the KV cache"


def test_the_loop_keeps_going_after_a_tool_call():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
                assistant_text("done"),
            ]
        )
        result = run_agent(task, work_dir, llm, registry, run_tests)

    assert result.steps_used == 3
```

`exercises/stage_3/test_stage_3.py`:

```python
"""Stage 3 — decide when the agent is actually done."""

import sys
from pathlib import Path

from agentfix.agent.loop import is_done, run_agent
from agentfix.llm.fake import FakeLLMClient, assistant_text, assistant_tool_call
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.base import ToolRegistry
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool

FIXTURE = Path("tasks/workshop/01-shopcart")
FIXED_CART = """from shopcart.pricing import with_tax


def subtotal(prices: list[float]) -> float:
    return sum(prices)


def total_with_tax(prices: list[float]) -> float:
    return with_tax(subtotal(prices))
"""


def _build(work_dir, task):
    run_tests = RunTestsTool(work_dir, task.test_command, SubprocessBackend(), timeout_s=30)
    registry = ToolRegistry(
        [ListFilesTool(work_dir), ReadFileTool(work_dir), WriteFileTool(work_dir), run_tests]
    )
    return registry, run_tests


def test_not_done_before_the_tests_have_ever_run(tmp_path):
    run_tests = RunTestsTool(tmp_path, (sys.executable, "-m", "pytest", "-q"), SubprocessBackend())
    assert is_done(run_tests) is False


def test_not_done_when_the_model_only_claims_success():
    """If your is_done trusts the model's word, this test fails. That is the lesson."""
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call("run_tests", {}, call_id="c1"),
                assistant_text("DONE. I have fixed the bug. All tests pass now."),
            ]
        )
        result = run_agent(task, work_dir, llm, registry, run_tests)

    assert result.solved is False


def test_done_only_once_the_tests_actually_pass():
    task = load_task(FIXTURE)
    with workspace(task) as work_dir:
        registry, run_tests = _build(work_dir, task)
        llm = FakeLLMClient(
            [
                assistant_tool_call(
                    "write_file", {"path": "shopcart/cart.py", "content": FIXED_CART}, call_id="c1"
                ),
                assistant_tool_call("run_tests", {}, call_id="c2"),
                assistant_text("fixed"),
            ]
        )
        result = run_agent(task, work_dir, llm, registry, run_tests)

    assert result.solved is True
```

- [ ] **Step 2: Verify all exercise tests pass against the finished code**

```bash
uv run pytest exercises -v
```

Expected: 12 passed. If any fail, the implementation is wrong, not the test — fix the implementation.

- [ ] **Step 3: Write the exercise READMEs**

`exercises/README.md`:

```markdown
# Exercises

Three stages. Each one edits real source files in `src/agentfix/` and has tests that
run **without a model** — you can finish every stage offline.

| Stage | You write | File | Test |
|---|---|---|---|
| 1 | the `run_tests` tool + its JSON schema | `src/agentfix/tools/tests_tool.py` | `uv run pytest exercises/stage_1` |
| 2 | the loop's tool dispatch | `src/agentfix/agent/loop.py` | `uv run pytest exercises/stage_2` |
| 3 | the stop condition | `src/agentfix/agent/loop.py` | `uv run pytest exercises/stage_3` |

Stuck? Jump ahead without falling behind the room:

    git checkout stage-1-solution     # or stage-2-solution, stage-3-solution
```

`exercises/stage_1/README.md`:

```markdown
# Stage 1 — Give the agent a tool

Open `src/agentfix/tools/tests_tool.py`. Two `TODO(stage-1)` markers.

1. **`parameters`** — the JSON Schema the model sees. `run_tests` takes no arguments,
   so this is an object with no properties. The `description` matters more than you
   think: it is the only thing telling the model when to reach for this tool.
2. **`run()`** — call `self.backend.run(...)`, store the result on `self.last_result`,
   and return a `ToolResult`.

Watch the distinction: `ToolResult.ok` means *the tool worked*.
`last_result.passed` means *the tests passed*. They are not the same, and stage 3
depends on the difference.

    uv run pytest exercises/stage_1 -v
```

`exercises/stage_2/README.md`:

```markdown
# Stage 2 — Close the loop

Open `src/agentfix/agent/loop.py`, find `TODO(stage-2)` inside `run_agent`.

For each tool call the model made: dispatch it through the registry and append the
result to `messages`.

Two rules the tests enforce:

- The observation goes back as a **`role="tool"`** message carrying the matching
  **`tool_call_id`**. Drop the id and the model cannot match the answer to its question.
- **Only ever append.** Never rewrite or reorder earlier messages — the server reuses
  its KV cache for an unchanged prefix, and mutating history makes every turn re-read
  the whole conversation from scratch.

    uv run pytest exercises/stage_2 -v
```

`exercises/stage_3/README.md`:

```markdown
# Stage 3 — When is it done?

Open `src/agentfix/agent/loop.py`, find `TODO(stage-3)` in `is_done`.

The tempting answers are both wrong:

- "the model stopped calling tools" — it may have given up, or hallucinated success
- "the model said DONE" — models say DONE about code that does not work

The agent is done when **the tests pass**. Verification by execution, not by assertion.
That is the difference between a demo and something you would let near real code.

    uv run pytest exercises/stage_3 -v
```

- [ ] **Step 4: Commit the finished implementation, then build the teaching branches**

```bash
git add exercises
git commit -m "test: add stage 1-3 exercise tests and instructions"
git branch solutions
git tag stage-3-solution
```

- [ ] **Step 5: Stub the three pieces on `main`**

In `src/agentfix/tools/tests_tool.py`, replace `parameters` and `run`:

```python
    # TODO(stage-1): the JSON Schema the model sees. run_tests needs no arguments.
    parameters: dict = {}

    def run(self) -> ToolResult:
        # TODO(stage-1): run the tests via self.backend, store self.last_result,
        # and return a ToolResult whose content tells the model what happened.
        raise NotImplementedError("stage 1: implement RunTestsTool.run")
```

In `src/agentfix/agent/loop.py`, replace the dispatch block and `is_done`:

```python
def is_done(run_tests: RunTestsTool) -> bool:
    # TODO(stage-3): the agent is done when the tests actually pass.
    # Not when the model stops calling tools. Not when it says "DONE".
    raise NotImplementedError("stage 3: implement is_done")
```

```python
        if reply.tool_calls:
            for call in reply.tool_calls:
                # TODO(stage-2): dispatch the call through the registry and append the
                # resulting tool message to `messages`. Keep the tool_call_id.
                raise NotImplementedError("stage 2: dispatch the tool call")
            continue
```

Keep the loop-guard code in place on `main` — it is not part of any exercise, and
students should see it working.

- [ ] **Step 6: Verify the student starting point behaves as intended**

```bash
uv run pytest exercises/stage_1 -v     # FAIL — NotImplementedError (expected)
uv run pytest tests -v -k "not loop and not runner and not tools_tests"
```

The second command confirms that the parts students are *not* writing still pass, so a
fresh clone is not a sea of red.

- [ ] **Step 7: Commit the student starting point and tag the intermediate solutions**

```bash
git add src/agentfix/tools/tests_tool.py src/agentfix/agent/loop.py
git commit -m "chore: stub stages 1-3 as the student starting point"

git checkout solutions
git checkout main -- .                                   # start from the stubbed state
git checkout stage-3-solution -- src/agentfix/tools/tests_tool.py
git commit -am "solution: stage 1 — the run_tests tool"
git tag stage-1-solution

git checkout stage-3-solution -- src/agentfix/agent/loop.py
git commit -am "solution: stages 2 and 3 — dispatch and stop condition"
git tag stage-2-solution

git checkout main
```

`main` ends at the stubbed starting point; `solutions` and the three tags hold the
answers. Verify:

```bash
git log --oneline --all --decorate | head -10
uv run pytest exercises/stage_1 -v          # FAIL on main
git checkout stage-1-solution -q && uv run pytest exercises/stage_1 -v   # PASS
git checkout main -q
```

---

### Task 12: Docker backend

**Files:**
- Create: `src/agentfix/sandbox/docker_backend.py`, `Dockerfile.sandbox`, `tests/test_docker_backend.py`

**Interfaces:**
- Consumes: `ExecResult` (Task 3)
- Produces: `DockerBackend(image: str = "agentfix-sandbox:latest", max_output_chars: int = 2000)` satisfying `ExecutionBackend`

- [ ] **Step 1: Write the test**

`tests/test_docker_backend.py`:

```python
import shutil
import sys

import pytest

from agentfix.sandbox.docker_backend import DockerBackend

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")


def test_passing_tests_report_passed(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"))
    assert result.passed is True


def test_network_is_unavailable_inside_the_container(tmp_path):
    (tmp_path / "test_net.py").write_text(
        "import socket\n"
        "def test_net():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('network was reachable')\n",
        encoding="utf-8",
    )
    result = DockerBackend().run(tmp_path, ("python", "-m", "pytest", "-q"), timeout_s=30)
    assert result.passed is True


def test_command_is_passed_through_without_sys_executable(tmp_path):
    """The host's sys.executable path is meaningless inside the container."""
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    backend = DockerBackend()
    argv = backend.build_argv(tmp_path, (sys.executable, "-m", "pytest", "-q"))
    assert sys.executable not in argv
    assert "python" in argv
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run pytest tests/test_docker_backend.py -v
```

Expected: FAIL — no module named `agentfix.sandbox.docker_backend` (or skipped if Docker is absent, which is also acceptable on your machine — the code still gets written)

- [ ] **Step 3: Write `Dockerfile.sandbox`**

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.2 && useradd --create-home runner
USER runner
WORKDIR /work
```

Build it:

```bash
docker build -f Dockerfile.sandbox -t agentfix-sandbox:latest .
```

- [ ] **Step 4: Implement `docker_backend.py`**

```python
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agentfix.sandbox.base import ExecResult
from agentfix.sandbox.subprocess_backend import TRUNCATION_MARKER

DEFAULT_IMAGE = "agentfix-sandbox:latest"


class DockerBackend:
    """Runs tests in a throwaway container with no network and hard resource caps."""

    def __init__(self, image: str = DEFAULT_IMAGE, max_output_chars: int = 2000) -> None:
        self.image = image
        self.max_output_chars = max_output_chars

    def build_argv(self, workspace: Path, command: tuple[str, ...]) -> list[str]:
        inner = ["python" if part.endswith("python") or "python" in Path(part).name else part
                 for part in command]
        if inner and inner[0] != "python":
            inner[0] = "python"

        return [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--pids-limit", "128",
            "--cpus", "1",
            "--volume", f"{workspace}:/work",
            "--workdir", "/work",
            self.image,
            *inner,
        ]

    def run(
        self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10
    ) -> ExecResult:
        argv = self.build_argv(workspace, command)
        start = time.time()

        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s + 10)
        except subprocess.TimeoutExpired:
            return ExecResult(False, f"TIMEOUT after {timeout_s}s", round(time.time() - start, 3), True)

        combined = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if len(combined) > self.max_output_chars:
            combined = combined[: self.max_output_chars] + TRUNCATION_MARKER

        return ExecResult(completed.returncode == 0, combined, round(time.time() - start, 3))
```

The container mount is read-write because `write_file` edits happen on the host before
`run_tests` is called — the workspace is already a disposable temp copy.

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_docker_backend.py -v
AGENTFIX_SANDBOX=docker uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

Expected: tests pass (or skip cleanly without Docker); the `solve` run behaves identically to the subprocess backend, just slower.

- [ ] **Step 6: Commit**

```bash
git add src/agentfix/sandbox/docker_backend.py Dockerfile.sandbox tests/test_docker_backend.py
git commit -m "feat: opt-in Docker execution backend with no network and resource caps"
```

---

### Task 13: Kaggle notebook and documentation

**Files:**
- Create: `notebooks/kaggle.ipynb`, `README.md`, `WORKSHOP.md`, `ARCHITECTURE.md`

**Interfaces:**
- Consumes: everything
- Produces: no code interfaces

- [ ] **Step 1: Build the Kaggle notebook**

Create `notebooks/kaggle.ipynb` with these cells, in order. **Everything runs inside the container — there is no tunnel.**

Cell 1 (markdown): explain that this is the tier-2 fallback for laptops that cannot run an 8 GB model, that Internet must be enabled in notebook settings (requires a phone-verified Kaggle account), and that a GPU accelerator should be selected.

Cell 2 (code) — install Ollama and start it:

```python
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama --version
```

Cell 3 (code) — pull the model (~8 GB; mention the Kaggle Dataset trick in a markdown cell above it):

```python
!ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
```

Cell 4 (code) — clone and install:

```python
!git clone https://github.com/YOURNAME/agentfix-workshop.git
%cd agentfix-workshop
!curl -LsSf https://astral.sh/uv/install.sh | sh
!~/.local/bin/uv sync
```

Cell 5 (code) — verify:

```python
!~/.local/bin/uv run agentfix doctor
```

Cell 6 (code) — the workshop commands:

```python
!~/.local/bin/uv run pytest exercises/stage_1 -v
!~/.local/bin/uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

`MELLUM_BASE_URL` stays at its default `http://localhost:11434/v1` — inside this container that *is* local.

- [ ] **Step 2: Verify the notebook on Kaggle**

Upload and run it end to end. This path is unverified in the spec and is a listed risk. If Ollama will not run under Kaggle, fall back to tier 3 (`MELLUM_MODEL=qwen2.5-coder:1.5b`) and say so in `README.md` rather than leaving a broken notebook in the repo.

- [ ] **Step 3: Write `README.md`**

Must contain, in this order: what this is (one paragraph); the three tiers table with RAM requirements; tier-1 setup as five copy-pasteable commands ending in `uv run agentfix doctor`; a note that the 8 GB pull must happen **before** the workshop; the tier-2 pointer to `notebooks/kaggle.ipynb`; tier-3 instructions (`MELLUM_MODEL=qwen2.5-coder:1.5b`); the command reference (`doctor`, `solve`, `eval`, `pytest exercises/stage_N`); a pointer to `exercises/README.md`; and the measured performance table from the spec so expectations are calibrated.

- [ ] **Step 4: Write `WORKSHOP.md`**

The instructor runsheet. Must contain: the minute-by-minute table from the spec; for each segment, what to say and which command to run; the documented cut order (eval → Docker demo → fold concepts into the demo) with the minutes each recovers; the "stage 3 is protected" note; the checkpoint-tag rescue command; the actual wall-clock timings you recorded in Task 10 Step 8; and a pre-workshop checklist (email sent, doctor outputs collected, USB sticks prepared, endpoint decision made).

- [ ] **Step 5: Write `ARCHITECTURE.md`**

The annotated loop. Must contain: the `run_agent` body with a note on each line; the module table from this plan's File Structure section; why history is append-only, with the measured ~480 tok/s prefill number; why `write_file` instead of `apply_diff`; why errors are observations; the two security boundaries (tool-layer path confinement vs sandbox execution); and a "what we deliberately left out" section (planning, reflection, parallel tools, async) pointing at the Thinking variant as the next step.

- [ ] **Step 6: Final verification**

```bash
uv run pytest --cov=src/agentfix --cov-report=term-missing
uv run ruff check .
uv run black --check .
uv run agentfix doctor
```

Expected: coverage ≥80% on `src/agentfix`, ruff and black clean. Note that `main` is the stubbed branch, so run the coverage check on `solutions`:

```bash
git checkout solutions && uv run pytest --cov=src/agentfix --cov-report=term-missing && git checkout main
```

- [ ] **Step 7: Commit**

```bash
git add notebooks README.md WORKSHOP.md ARCHITECTURE.md
git commit -m "docs: README, instructor runsheet, architecture notes, and Kaggle fallback notebook"
```

---

## Self-Review

**Spec coverage:** every spec section maps to a task — inference layer → Task 2; three tiers → Tasks 2, 13; tool layer → Tasks 4, 5; sandbox both backends → Tasks 3, 12; agent loop, stop condition, loop guard, trace, discovery-not-preloading → Task 7; tasks and eval, vendored subset, `[eval]` extra → Tasks 6, 9, 10; exercises, timeline, checkpoint tags, offline exercise tests → Task 11; packaging, doctor, docs → Tasks 1, 8, 13; inherited files → Tasks 3 (runner), 10 (humanevalfix, legacy results).

**Type consistency:** `ToolCall`/`LLMReply` (Task 2) are consumed unchanged in Tasks 4 and 7. `ToolResult`/`ToolOutcome.as_message()` (Task 4) are used in Tasks 5, 7, 11. `ExecResult`/`ExecutionBackend.run()` (Task 3) is implemented identically by `SubprocessBackend` and `DockerBackend` (Task 12). `RunTestsTool.last_result` (Task 5) is what `is_done` reads (Task 7). `Task`/`workspace` (Task 6) are used in Tasks 7, 8, 10, 11. The predecessor's colliding `Task` dataclass is renamed `HumanEvalFixRow` (Task 10).

**Known deviation from the spec:** the spec's file table lists `src/agentfix/sandbox/subprocess.py`; this plan uses `subprocess_backend.py` to avoid shadowing the stdlib `subprocess` module, and `tests_tool.py` rather than `tests.py` for the same reason with pytest collection.
