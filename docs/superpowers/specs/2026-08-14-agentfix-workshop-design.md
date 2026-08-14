# AgentFix Workshop — Design

**Date:** 2026-08-14
**Status:** Approved design, pending implementation plan
**Repo:** `~/Desktop/JetBrains/agentfix-workshop` (fresh git history)
**Predecessor:** `~/Desktop/JetBrains/Python Ai Agent` — left untouched; three files copied forward

## Goal

A teaching repository for a **90-minute workshop** that shows developers new to agents how a
coding agent actually works. Students build three pieces of a working agent themselves: a tool
and its schema, the loop's tool dispatch, and the stop condition. Everything runs locally and
free on their own laptops.

The predecessor project fixed single functions with a monolithic `policy.py` that mixed model
config, prompting, generation, and output parsing, and had no tool abstraction. This rebuild
separates those concerns so each one can be taught.

## Non-goals

- Beating a benchmark. Pass rate is a teaching artifact, not the objective.
- A production agent framework. No planning phase, no reflection, no parallel tool calls,
  no multi-agent orchestration.
- Multi-language support. Python only.

## Constraints

| Constraint | Value |
|---|---|
| Duration | 90 minutes, developers new to agents |
| Cost to students | $0 |
| Hardware floor | 16 GB RAM for the default tier; fallbacks below that |
| Package manager | uv exclusively |
| Model | JetBrains Mellum2-12B-A2.5B-Instruct (Apache 2.0) |

## Verified measurements

Measured 2026-08-14 on an Apple M4, 24 GB unified memory, Ollama, `Q4_K_M` GGUF (8.07 GB):

| Metric | Result |
|---|---|
| Generation throughput | **51 tok/s** (700 tokens / 13.7 s) |
| Cold model load | ~3.5 s, one-time |
| Prefill throughput | **~480 tok/s** (3,438-token prompt ≈ 7 s before first token) |
| Tool calling via Ollama `/v1` | Works — correct name and args, `finish_reason: tool_calls` |
| Bug fix sanity check | Correct, no prose, no code fences |

Two design consequences follow from the prefill number and are load-bearing throughout:

1. **Message history is strictly append-only.** llama.cpp/Ollama reuse the KV cache for a shared
   prefix. Rewriting or reordering earlier messages invalidates it and makes every subsequent
   turn pay full prefill.
2. **Tool observations are truncated where they are produced.** One untruncated traceback taxes
   every later turn in the run.

Expect roughly 3–4× slower on an older Intel laptop. This is why tier 2 exists.

## Architecture

Hand-rolled, no agent framework. A framework would hide the loop, the dispatch, and the message
history — precisely the three things students are here to write.

```
src/agentfix/
  llm/client.py          # OpenAI-compatible client; LLMConfig from env
  llm/fake.py            # scripted client — makes every exercise test offline and instant
  tools/base.py          # Tool protocol, ToolResult, ToolRegistry
  tools/fs.py            # list_files, read_file, write_file
  tools/tests.py         # run_tests
  sandbox/base.py        # ExecutionBackend protocol
  sandbox/subprocess.py  # hardened; default; copied from predecessor
  sandbox/docker.py      # opt-in via AGENTFIX_SANDBOX=docker
  agent/loop.py          # the ~15-line loop — STAGES 2 and 3
  agent/trace.py         # structured trace, --verbose tree
  tasks/loader.py        # Task = template dir + test cmd + expected failures
  eval/runner.py         # pass@1 + steps + tokens + wall time
  eval/humanevalfix.py   # copied from predecessor; behind [eval] extra
  doctor.py              # preflight
  cli.py                 # agentfix doctor | solve | eval
tasks/workshop/          # 3 hand-built buggy repos (01-shopcart, 02-invoice, 03-parser)
tasks/humanevalfix/      # vendored 20-task JSON subset
exercises/stage_{1,2,3}/ # TODOs + failing tests
notebooks/kaggle.ipynb   # tier 2
```

### Inference layer

```python
@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
    temperature: float = 0.6     # JetBrains' recommended value
    top_p: float = 0.95
    max_tokens: int = 1024
    num_ctx: int = 16384         # explicit — Ollama's default truncates tool history
```

Built from `MELLUM_BASE_URL` / `MELLUM_MODEL` with these as defaults. `LLMClient` exposes one
method, `chat(messages, tools) -> Response`, returning assistant text or tool calls.

Decisions:

- **`transformers` and `torch` are gone.** Roughly 150 lines of the predecessor's `policy.py`
  disappear, along with dtype/device handling, model caching, and beam search. Base install
  becomes `openai` + `pytest`, so `uv sync` finishes in seconds — which matters for 20 people on
  venue wifi.
- **Sync, not async.** The loop is strictly sequential; `async` would put event-loop questions
  between students and the lesson. Where async belongs (parallel eval runs) is called out verbally.
- **`num_ctx` explicit and taught.** With four tools and several turns, the default context would
  silently drop early messages and the agent would loop re-reading the same file.

### Three inference tiers

| Tier | Who | Endpoint |
|---|---|---|
| 1 (default) | 16 GB+ laptop | local Ollama, `localhost:11434/v1` |
| 2 | weak laptop | Kaggle notebook, `localhost:11434/v1` inside the container — **no tunnel** |
| 3 | last resort | `qwen2.5-coder:1.5b` (~1 GB) locally, or pair up |

Tier 2 runs the entire repo inside the notebook rather than exposing Kaggle as a remote server.
Tunnelling into a classroom is the fragile option, and it fails for exactly the person whose
setup is already weakest. Kaggle supplies a T4, so the weakest laptop may end up fastest.

Kaggle notes: Internet must be enabled (requires a phone-verified account — belongs in the
pre-work email); free GPU quota ~30 h/week; upload the GGUF once as a private Kaggle Dataset and
mount it to avoid re-pulling 8 GB per session.

### Tool layer

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict            # hand-written JSON Schema — STAGE 1
    def run(self, **kwargs) -> ToolResult: ...

@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
```

`ToolRegistry` does two things: `schemas()` builds the API `tools=[...]` payload, and
`dispatch(tool_call)` resolves the name, validates args, runs the tool, and returns a
`role="tool"` message carrying the `tool_call_id`. Those two methods are the whole bridge between
model output and real effects.

| Tool | Notes |
|---|---|
| `list_files` | forces a first decision with no obvious answer |
| `read_file(path)` | truncated ~4 KB with an explicit `[...truncated]` marker |
| `write_file(path, content)` | full-file replacement |
| `run_tests()` | the sandbox; the only source of ground truth |

Decisions:

- **`write_file`, not `apply_diff`.** Production agents edit via diffs, but small models emit
  invalid unified diffs — drifting line numbers, mismatched context — and burn their turn budget
  failing to apply patches. Full-file rewrite of a small file is far more reliable at 12B. The
  lesson: tool ergonomics must match model capability, and a contract the model cannot satisfy is
  a leading cause of agents that "don't work."
- **Errors are observations, not exceptions.** A tool never raises into the loop. Missing file →
  `ToolResult(ok=False, content="No such file: src/typo.py. Available: ...")`, and the model
  self-corrects. Students' instinct is to wrap the loop in `try/except`; the fix is feeding
  failures back to the model.
- **Path confinement lives in the tool layer.** Every fs tool resolves against the task root and
  rejects escapes. Tools guard *access*; the sandbox guards *execution*. Two distinct boundaries.
- **Truncation is a tool responsibility**, enforced where the string is produced.

### Sandbox

`ExecutionBackend` protocol, selected by `AGENTFIX_SANDBOX=subprocess|docker`.

**`SubprocessBackend` (default).** The predecessor's runner — temp dir, `python -I -B`, timeout —
hardened with:

- `resource.setrlimit` for address space, CPU time, file size, and process count (POSIX only;
  Windows students get timeout-only, documented honestly)
- a stripped `env` dict, so executed code cannot see tokens or credentials
- output truncation at the source

**`DockerBackend` (opt-in).** `--network none`, read-only mounts, memory and pid limits, non-root,
tmpfs workdir. Instructor demos it; students do not install Docker.

Rationale for not defaulting to Docker: the Kaggle tier cannot run nested containers, so a
subprocess backend must exist regardless; and adding Docker Desktop to pre-work roughly doubles
the setup-failure surface on a 90-minute clock. Per-call container overhead (~0.3–1 s) is not the
objection — setup burden is.

This split also yields a 5-minute segment most agent tutorials skip: your agent executes
model-written code, here is what that means, here is what production agents do about it.

### Agent loop

```python
def run_agent(task, llm, tools, max_steps=10, tracer=None) -> AgentResult:
    messages = [system_prompt(tools), task_prompt(task)]

    for step in range(max_steps):
        reply = llm.chat(messages, tools=tools.schemas())
        messages.append(reply.message)              # append-only, always

        if reply.tool_calls:                         # ── STAGE 2
            for call in reply.tool_calls:
                messages.append(tools.dispatch(call).as_message())
            continue

        if is_done(task, tools):                     # ── STAGE 3
            break

    return AgentResult(...)
```

Decisions:

- **Stop condition is verification by execution.** The agent is done when `run_tests` passes — not
  when the model stops calling tools, and not when it says "DONE". This is the highest-value idea
  in the workshop and the difference between a demo and something trustworthy.
- **Bounded by construction.** `max_steps=10` is a hard cap. An unbounded agent is an unbounded
  wait and an unbounded bill. (Raised from 6 after measurement: this tool granularity needs
  `run_tests` + `list_files` + one `read_file` per implicated file + `write_file` + a verifying
  `run_tests` — 8 steps for a three-file read — so 6 made the demo task unsolvable by
  construction. The teaching point, a bounded budget, is unchanged.)
- **Loop guard.** Identical tool + identical args twice in a row → inject an observation saying so
  instead of re-executing. Small models get stuck re-reading the same file. Framed honestly as a
  workaround for model capability, not architecture.
- **Discovery over pre-loading.** The agent is *not* handed the failing test output; it must call
  `run_tests` to find out. Costs one turn (~5–10 s at measured speed), makes tool use necessary
  rather than decorative, and "the agent chose to look before editing" is a far better first
  impression.
- **The trace is a first-class output.** Each step records tool, args, truncated result,
  `prompt_tokens`, and latency. `--verbose` prints a readable tree; JSON goes to `results/`.
  Watching `prompt_tokens` climb per turn makes the context-budget lesson concrete.

Deliberately omitted: planning/reflection phases, parallel tool execution, self-critique,
retry-at-different-temperature. The Mellum2 **Thinking** variant is mentioned as the natural next
step — same code, one env var, visible `<think>` blocks.

### Tasks and evaluation

A `Task` is a directory template + test command + expected-failing tests. Each run copies the
pristine template into a temp workspace, so fixtures are never mutated and runs are reproducible.

This unifies both suites: **a HumanEvalFix task is just a one-file repo** (`candidate.py` +
`test_candidate.py`), so the same agent, tools, and loop handle both.

```
tasks/
  workshop/
    01-shopcart/     # obvious bug; failing test names the module
    02-invoice/      # bug is NOT in the file the failing test points at
    03-parser/       # two defects, co-located in tokens.py (see R16)
  humanevalfix/      # vendored 20-task JSON subset
```

Task 02 is what earns the tool layer. If every bug sits where the test points, `list_files` and
`read_file` are decoration.

Eval reports **pass@1** plus per-task steps, tokens, and wall time. "Passed in 6 steps and 40k
tokens" is a different result from "passed in 2" — agent cost is part of the lesson.

Timing: ~30–60 s per task at measured speed, so a 20-task eval is 10–20 minutes and **cannot** be
a live activity. Students run 3–5 tasks (~3 min, genuinely theirs); the repo ships pre-computed
full results for discussion.

`datasets` sits behind the `[eval]` extra — it drags in pyarrow and would make `uv sync` a coffee
break. The vendored JSON subset covers workshop use with no extra dependency.

## Exercises and timeline

| Time | Segment |
|---|---|
| 0:00–0:08 | Setup triage; what an agent is (loop, tools, verification) |
| 0:08–0:16 | Live demo: finished agent fixes task 01 with `--verbose` |
| 0:16–0:24 | The tool-calling contract: schema → `tool_call` → observation → append |
| 0:24–0:36 | **Stage 1** — write `run_tests` and its schema |
| 0:36–0:50 | **Stage 2** — loop dispatch; run the agent on task 01 |
| 0:50–1:08 | **Stage 3** — stop condition; run on task 02 where the naive version fails |
| 1:08–1:16 | Eval discussion (pre-computed results; demo-only) |
| 1:16–1:30 | Sandbox safety + Docker demo; Thinking variant; next steps |

Two structural safeguards, because the schedule has little slack:

- **Exercise tests never need the model.** Every stage test runs against `llm/fake.py` — instant,
  deterministic, offline. A student with broken Ollama still completes all three stages and passes
  every test. Real inference is the reward, never a prerequisite. This de-risks the session more
  than anything else in the design.
- **Checkpoint tags** `stage-1-solution`, `stage-2-solution`, `stage-3-solution`. Anyone stuck
  checks out the next tag and rejoins the group.

Each stage is 5–15 lines to write:

- **Stage 1** — `parameters` schema + `run()`. Test asserts the schema is valid and that the model
  selects this tool when told a test is failing.
- **Stage 2** — the dispatch block. Test asserts a scripted tool call produces the right message
  sequence; the classic omission of `tool_call_id` has its own assertion and error message.
- **Stage 3** — `is_done()`. The scripted model *claims* success while tests still fail, so the
  naive implementation goes red. That is where the lesson lands.

**Cut order when the room runs slow:** eval segment (−8 min), then the Docker demo (−4), then fold
the tool-calling concepts into the live demo. Stage 3 is protected at all costs.

## Packaging

`pyproject.toml` with a committed `uv.lock` so every machine resolves identically.

- base: `openai`, `pytest`
- `[eval]`: `datasets`
- `[dev]`: `ruff`, `black`, `pytest-cov`

One console script, three subcommands:

```
uv sync
uv run agentfix doctor
uv run agentfix solve tasks/workshop/01-shopcart --verbose
uv run agentfix eval --suite workshop
uv run pytest exercises/stage_1
```

`agentfix doctor` is the pre-work artifact: checks Python version, RAM (total, with free
reported), Ollama installation, server reachability, derived-model presence, the loaded context
window, one warmed-up timed generation, and one sandbox execution — then prints
`READY — 51 tok/s` or a specific remedy per failure with the exact command to run. Students send
the output days ahead, converting setup chaos into a triage list.

Docs, three files: `README.md` (setup, three tiers), `WORKSHOP.md` (minute-by-minute runsheet and
what to say at each beat), `ARCHITECTURE.md` (the annotated loop). Plus `notebooks/kaggle.ipynb`.

## Testing

- Unit: tool layer, registry, truncation, path confinement, loop against the fake client — all
  offline and fast.
- Integration: real Ollama, marked `@pytest.mark.llm`, skipped by default so CI and student
  machines stay green without a model.
- Target ≥80% coverage on new code — realistic because the fake client makes the loop fully
  testable.

## Inherited from the predecessor

Copied forward; the predecessor repo is not modified.

| File | Fate |
|---|---|
| `src/sandbox/runner.py` | copied, hardened, wrapped as `SubprocessBackend` |
| `src/eval/humanevalfix.py` | copied; becomes a task-suite generator behind `[eval]` |
| `results/*.json`, `comparative_results.csv` | copied to `results/legacy/` as the Qwen-1.5B baseline |
| `src/agent/policy.py` | not carried over — replaced by `llm/client.py` + `agent/loop.py` |
| `src/eval/run_benchmark.py`, `run_one.py` | not carried over — replaced by `cli.py` + `eval/runner.py` |
| `requirements.txt` | not carried over — replaced by `pyproject.toml` + `uv.lock` |

The legacy results are worth keeping *because* they are from a 1.5B model: 1.5B vs Mellum2 on the
same harness is a free extra slide.

## Build order

Sequenced so the repo is runnable end-to-end early:

1. uv scaffold + package skeleton + `doctor`
2. `LLMClient` + `FakeLLMClient`
3. Tool layer + hardened `SubprocessBackend`
4. Agent loop + trace
5. Task fixtures (workshop 01–03)
6. Eval runner + vendored HumanEvalFix subset
7. Exercises + checkpoint tags + solutions
8. `DockerBackend`, Kaggle notebook, docs

## Open risks

| Risk | Mitigation |
|---|---|
| 8 GB model download × N students on venue wifi | pre-work email a week ahead; USB sticks; `doctor` output collected in advance |
| 8 GB-RAM laptops cannot run the default tier | tier 2 (Kaggle) and tier 3 (1 GB model) both documented and tested |
| Old laptop is 3–4× slower; live runs drag | exercise tests need no model; pre-recorded traces available |
| 90 minutes has near-zero slack | documented cut order; checkpoint tags; eval is demo-only |
| Kaggle + Ollama not yet verified with this model | verify before the workshop; tier 3 is the fallback if it fails |
| Ollama context truncation silently breaks the agent | `num_ctx` set explicitly in config, and taught |
