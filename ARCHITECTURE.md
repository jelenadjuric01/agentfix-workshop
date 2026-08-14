# Architecture

No framework. Everything below is real code in this repo, not a simplification of it — the whole
point of the workshop is that a coding agent fits in something you can read end to end.

## The loop, annotated

`src/agentfix/agent/loop.py`, `run_agent`:

```python
def run_agent(task, work_dir, llm, registry, run_tests, max_steps=MAX_STEPS, tracer=None):
    tracer = tracer or Tracer()
    messages = [
        {"role": "system", "content": system_prompt(registry)},
        {"role": "user", "content": task_prompt(task)},
    ]
    # ^ the only two messages that exist before the model has done anything.

    previous_signature = None
    for step in range(1, max_steps + 1):
        # ^ bounded by construction: an agent with no cap is an unbounded wait and bill.
        # Raised to 10 from an original 6 after measurement — this tool granularity needs
        # run_tests + list_files + one read_file per implicated file + write_file + a
        # verifying run_tests, which is 8 steps for a three-file read.

        reply = llm.chat(messages, tools=registry.schemas())
        messages.append(reply.message)
        # ^ append-only. Never rewritten, reordered, or dropped. See "Why append-only" below.

        if reply.tool_calls:
            for call in reply.tool_calls:
                signature = _call_signature(call)
                if signature == previous_signature:
                    # ^ loop guard: same tool + same args twice in a row gets an observation
                    # instead of a re-execution. See "Loop guard limitation" below — it does
                    # not stop here, it `continue`s to the next call.
                    messages.append({...})
                    continue
                previous_signature = signature
                messages.append(registry.dispatch(call).as_message())
                # ^ the only two methods that bridge model output to real effects:
                # `schemas()` above, `dispatch()` here.
            continue
            # ^ tool calls never end the run on their own — always loop back for another turn.

        if is_done(run_tests):
            # ^ verification by execution, not by the model's say-so. The highest-value idea
            # in the whole design. `run_tests` is asked, not told: nothing hands the agent the
            # failing test output up front (see "Discovery over pre-loading" below).
            break
        break
        # ^ NOTE: in the shipped stub this second `break` exists because `is_done` always
        # returns False before Stage 3 is implemented, so the loop must still terminate on a
        # text-only reply rather than raising. Stage 3's fix targets `is_done`, not this line.

    return AgentResult(...)
```

## Module map

| File | Responsibility |
|---|---|
| `src/agentfix/config.py` | `LLMConfig`, env loading (`MELLUM_BASE_URL`, `MELLUM_MODEL`) |
| `src/agentfix/llm/types.py` | `ToolCall`, `LLMReply`, `LLMClient` Protocol |
| `src/agentfix/llm/client.py` | `OllamaClient` — OpenAI-compatible `chat()` |
| `src/agentfix/llm/fake.py` | `FakeLLMClient` + scripted-reply builders — offline, instant, deterministic |
| `src/agentfix/sandbox/base.py` | `ExecResult`, `ExecutionBackend` Protocol, `get_backend()` |
| `src/agentfix/sandbox/subprocess_backend.py` | hardened subprocess execution (default) |
| `src/agentfix/sandbox/docker_backend.py` | opt-in container execution |
| `src/agentfix/tools/base.py` | `ToolResult`, `ToolOutcome`, `Tool` Protocol, `ToolRegistry`, `truncate` |
| `src/agentfix/tools/fs.py` | `resolve_in_root`, `ListFilesTool`, `ReadFileTool`, `WriteFileTool` |
| `src/agentfix/tools/tests_tool.py` | `RunTestsTool` |
| `src/agentfix/tasks/loader.py` | `Task`, `load_task`, `workspace()` — the copy-to-tempdir context manager |
| `src/agentfix/agent/trace.py` | `TraceEvent`, `Tracer` |
| `src/agentfix/agent/loop.py` | `system_prompt`, `task_prompt`, `is_done`, `run_agent`, `AgentResult`, `MAX_STEPS` |
| `src/agentfix/runner.py` | `solve_task` — wires a `Task` into a fresh workspace, registry, and `run_agent` call |
| `src/agentfix/eval/runner.py` | `EvalReport`, `evaluate`, `run_suite` — pass@1 + steps + tokens + wall time |
| `src/agentfix/eval/humanevalfix.py` | `HumanEvalFixRow`, vendored-subset loader, task-dir generator |
| `src/agentfix/doctor.py` | preflight checks |
| `src/agentfix/cli.py` | argparse entry point: `doctor`, `solve`, `eval` |

Deviation from the original design doc: the spec names `sandbox/subprocess.py` and `tools/tests.py`;
the code uses `subprocess_backend.py` and `tests_tool.py` instead, to avoid shadowing the stdlib
`subprocess` module and colliding with pytest's `test_*` collection convention, respectively.

## Why history is strictly append-only

llama.cpp/Ollama reuse the KV cache for a shared message-history prefix. Measured prefill on this
machine is **~480 tok/s** — a 3,438-token prompt took ~7s before the first output token. Rewriting
or reordering any earlier message invalidates that cached prefix, so every later turn pays full
prefill again instead of only paying for what's new. In a multi-turn agent loop where the same
growing history is re-sent every turn, that difference compounds: append-only keeps each turn's
cost proportional to what changed, not to the whole conversation so far.

## Why `write_file` instead of `apply_diff`

Production agents typically edit via diffs. At 12B, small models reliably emit invalid unified
diffs — drifting line numbers, mismatched context — and burn turns failing to apply a patch rather
than fixing anything. A full-file rewrite of a small file is far more reliable at this model size.
The general lesson, not specific to this repo: tool ergonomics have to match model capability, and
a contract the model can't reliably satisfy is a leading cause of agents that "don't work" even
though the loop and the model are both fine in isolation.

## Why errors are observations, not exceptions

A tool never raises into the loop (`ToolRegistry.dispatch` catches broadly and turns failures into
a `ToolResult(ok=False, content=...)`). A missing file becomes `"No such file: src/typo.py.
Available: ..."`, fed back to the model as a normal tool result — and the model self-corrects on
the next turn. The natural instinct is to wrap the loop in `try/except` and treat a tool failure as
fatal; the fix that actually improves reliability is feeding failures back to the model as
information instead of killing the run over them.

## Two security boundaries — do not conflate them

1. **Tool-layer path confinement.** Every filesystem tool resolves the requested path against the
   task root and refuses anything that escapes it (`resolve_in_root` in `tools/fs.py`). This guards
   *access* — which files the model-driven tool calls are even allowed to name.
2. **Sandbox execution.** `run_tests` executes model-written code through an `ExecutionBackend`.
   This guards *execution* — what that code can do once it runs: `SubprocessBackend` applies
   `resource.setrlimit` (address space, CPU time, file size, process count), a stripped `env` with
   no inherited secrets, and output truncation at the source; `DockerBackend` adds `--network none`,
   memory/pid/cpu caps, and a non-root user, at the cost of requiring Docker.

These are separate layers on purpose: path confinement stops the model from *naming* a file outside
the project; the sandbox stops code *inside* the project from doing damage once it runs. Neither
substitutes for the other.

### `RLIMIT_AS` is Linux-only

`resource.setrlimit(resource.RLIMIT_AS, ...)` — the address-space cap — aborts CPython startup on
macOS/Apple Silicon; verified empirically on this machine (`_apply_limits` is called via
`preexec_fn`, and the child process fails to start at all when `RLIMIT_AS` is set on this
platform). On macOS, memory is effectively uncapped by `SubprocessBackend`; the CPU-time,
file-size, and process-count limits still apply, plus the wall-clock `timeout_s` on the
`subprocess.run` call itself. Real memory capping on macOS comes only from `DockerBackend`. This
matters less than it sounds: the Kaggle tier (tier 2) runs on Linux, so the memory cap holds
exactly where the weakest laptops in the room are running — the machines with no cap are, by
construction, the ones with 16 GB+ of their own RAM to begin with.

### Docker isolation is unverified in this repo

The Docker daemon was not running during development, so `tests/test_docker_backend.py`'s three
tests skip rather than pass. `DockerBackend` and its tests exist and are written to the same
`ExecutionBackend` protocol as `SubprocessBackend`, but "the code is written" is not the same claim
as "this has been run against a container." Run it against a live daemon before the sandbox-safety
demo depends on it working.

## HumanEvalFix runs as a plain script, not pytest

The workshop tasks (`tasks/workshop/*`) run via `pytest`. HumanEvalFix tasks do not: each vendored
row's `tests` field defines a `check(fn)` function and calls it at module level, so there is no
`test_*` function for pytest to collect — pytest would report "0 collected" and silently pass.
Separately, the dataset's `buggy_solution` is a function *body* only; it is not valid Python until
the matching `declaration` (imports + `def` line) is prepended, which `load_hf_rows` in
`eval/humanevalfix.py` does at vendor time. Both defects independently made pass@1 a structural
0.0 before they were fixed — not a model failure, a harness bug. This is exactly why `Task` carries
its own `test_command`: `tasks/workshop/*/task.json` says `["-m", "pytest", "-q"]`, while
`write_task_dir` generates `["-u", "test_candidate.py"]` for HumanEvalFix rows. Same `Task`
abstraction, same agent, same tools, two different ways of finding out whether the fix worked.

## Ollama context: what's confirmed and what isn't

`LLMConfig.num_ctx = 16384` is sent on every request via `extra_body={"options": {"num_ctx":
...}}` against the OpenAI-compatible `/v1` endpoint. Observed on this machine: Ollama loaded the
model with `-c 4096` and `--context-shift`, which are its own defaults, not clearly the value this
client requested. **Whether `/v1` actually honours the `num_ctx` option has not been confirmed.**
Runs so far total roughly 9k tokens *across* an entire run's turns rather than in one prompt, so
nothing has broken in observed practice — but do not read that as proof the setting works; it may
simply mean no run has yet needed enough context for a 4096-token ceiling to matter. Setting
`OLLAMA_CONTEXT_LENGTH=16384` before starting the Ollama server is the belt-and-braces fix
documented in `README.md`, and is the thing to check first if a long run behaves as though it lost
track of early tool calls.

## Loop-guard limitation

The guard compares only the *immediately previous* call's signature (tool name + sorted argument
items). It blocks one exact repeat, but it does not detect or break a longer cycle: a model that
calls `run_tests` four times in a row still trips the guard on calls 2, 3, and 4 individually, gets
an "already called" observation each time, and can keep doing this until the step budget runs out
without ever trying something different. See `WORKSHOP.md`'s failure-mode traces for a measured
example of exactly this happening. The guard is honestly a workaround for small-model capability
("don't waste a whole turn re-reading the same file"), not an architectural cycle-detector — say so
if a student asks why it didn't save that run.

## What was deliberately left out

No planning phase, no reflection/self-critique, no parallel tool calls, no multi-agent
orchestration, no retry-at-different-temperature. These are all real techniques in production
agents; they are absent here because the workshop's budget is 90 minutes and three concepts (a
tool, dispatch, a stop condition), and each of these would add a decision point without adding to
that lesson. The natural next step, mentioned but not built, is the Mellum2 **Thinking** variant —
same code, one environment variable, visible `<think>` blocks — which is the smallest possible step
from this repo toward a model that plans before it acts.
