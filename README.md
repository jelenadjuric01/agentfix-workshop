# agentfix

A teaching repository for a workshop that shows developers new to agents how a coding
agent actually works, by having them build one. You write three pieces of a real agent yourself —
a tool and its JSON schema, the loop's tool dispatch, and a verification-based stop condition —
then watch it fix real bugs, locally, for $0, using [JetBrains
Mellum2](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF) served by Ollama. There
is no framework: the loop itself is about 15 lines and the rest of `run_agent` is tracing and
token accounting. See `ARCHITECTURE.md` for the annotated version.

Every exercise test runs against a scripted fake model, so the workshop does not depend on your
Ollama setup working — real inference is the reward, not a prerequisite.

When something breaks, go to [`TROUBLESHOOT.md`](TROUBLESHOOT.md) rather than reading this file end
to end — every problem that has actually come up, one collapsible hint each. When you are done and
want the machine back as it was, [`CLEANUP.md`](CLEANUP.md) removes the models, Ollama, and
everything else this workshop installed.

## Which setup option should you use?

| Option | Who | RAM | Endpoint |
|---|---|---|---|
| 1 (default) | 16 GB+ laptop | 16 GB+ | local Ollama, `http://localhost:11434/v1` |
| 2 | weaker laptop, can't run the 8 GB Mellum2 model | much lower | `qwen3:1.7b` locally (~1.4 GB) |
| 3 | can't run either model locally | any | local exercises + Google Colab notebook — `notebooks/agentfix.ipynb` (**tested**) |

Options 1 and 2 run on macOS, Linux, WSL2, and native Windows — the per-OS install commands are in
the local setup sections below, and the handful of differences that actually matter (sandbox limits,
the RAM check) are in [Platform notes](#platform-notes). Option 3 needs only a browser.

Windows users: prefer WSL2.

Option 3 does not change the lesson. Those learners work through the whole workshop locally like
everyone else — the exercise tests need no model — and use the notebook for the one step that
needs real inference. They skip `agentfix doctor` locally, where it is expected to fail.

## Option 1: Mellum2 local setup

Do this **before the workshop** — the model pull is 8 GB and venue wifi will not survive 20 people
doing it at once.

Only step 1 differs by operating system. Steps 2 and 3 are byte-for-byte identical everywhere,
which is the whole reason this repo derives the model with a `Modelfile` instead of setting a server
environment variable (see "The context window" below).

### Step 1 — install `uv` and Ollama

<details open>
<summary><b>macOS</b> (verified — all measurements in this README come from here)</summary>

```bash
brew install uv ollama
ollama serve &                  # or: open -a Ollama   (the app starts the same server)
```

Homebrew's `ollama` and the Ollama.app are the same server on `localhost:11434` — use either, but
not both at once. Without Homebrew: install `uv` with
`curl -LsSf https://astral.sh/uv/install.sh | sh` and Ollama from
[ollama.com/download](https://ollama.com/download).
</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The install script registers a systemd service, so the server is already listening on
`localhost:11434`; `systemctl status ollama` tells you. If you installed the tarball by hand
instead, run `ollama serve` in its own terminal. A GPU is not required — CPU inference works, it is
just slower than the numbers below.
</details>

<details>
<summary><b>Windows — WSL2 (recommended)</b></summary>

Run the workshop inside WSL2 and treat it as Linux. This is the Windows path to prefer: the
sandbox that executes the agent's test runs is POSIX-shaped (see "Platform notes" below), so WSL2
avoids the one part of this repo that is untested on native Windows.

In PowerShell, once:

```powershell
wsl --install -d Ubuntu
```

Then, inside the Ubuntu shell, follow the **Linux** instructions above and do everything else —
`git clone`, `uv`, `ollama`, the exercises — inside WSL2. Keep the clone on the Linux filesystem
(`~/agentfix-workshop`, not `/mnt/c/...`); pytest across the `/mnt/c` bridge is slow enough to be
annoying.

WSL2 gets a fraction of your total RAM by default (50%, capped at 8 GB on older builds), and that
fraction — not your machine's spec sheet — is what has to hold an 8 GB model. If `free -g` inside
WSL2 shows less than 16 GB total, raise it in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

then `wsl --shutdown` in PowerShell and reopen the shell.
</details>

<details>
<summary><b>Windows — native PowerShell (works for the exercises; sandbox untested)</b></summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --id Ollama.Ollama      # or the installer from ollama.com/download
```

The installer runs Ollama as a background app, so the server is already on `localhost:11434` (look
for the tray icon). Then use the same commands as everywhere else — `uv run ...` is identical in
PowerShell, and forward slashes in task paths are fine.

Two caveats, both detailed under "Platform notes": `agentfix doctor` cannot read RAM on Windows and
skips that check rather than failing it, and the subprocess sandbox has not been run on native
Windows. If `uv run agentfix doctor` reports a `sandbox` failure, switch to WSL2 rather than
debugging it during the workshop.
</details>

### Step 2 — pull and derive the model

Identical on macOS, Linux, WSL2, and PowerShell:

```bash
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
ollama create agentfix-mellum2 -f Modelfile      # required — see "The context window" below
```

### Step 3 — install the project and check it

```bash
uv sync --extra dev
uv run agentfix doctor
```

`agentfix doctor` checks your Python version and free RAM, that Ollama is installed, that its
server answers, that the derived model exists, that the **loaded context window is 16384**, does
one warmed-up timed generation, and runs one sandboxed test execution. It prints `[PASS]`/`[FAIL]`
per check and a final `READY <rate> tok/s`, or a remedy command for each failure. Send its output
to the instructor days ahead — that turns setup problems into a triage list instead of a
90-minute-clock emergency. Expected on a healthy 24 GB M4:

```
[PASS] python: 3.12.9
[PASS] ram: 24.0 GB total, 10.2 GB free
[PASS] ollama installed: /usr/local/bin/ollama
[PASS] ollama server: reachable at http://localhost:11434
[PASS] model present: agentfix-mellum2
[PASS] generation: 51 tok/s (372 tokens in 7.3s)
[PASS] context window: 16384 tokens
[PASS] sandbox: executes tests
```

Two lines read differently off macOS. On Linux the `ollama installed` path is usually
`/usr/local/bin/ollama` or `/usr/bin/ollama`; on native Windows it is the `ollama.exe` under
`%LOCALAPPDATA%\Programs\Ollama`. And the `ram` check only knows how to read `/proc/meminfo`
(Linux) and `sysctl`/`vm_stat` (macOS), so on native Windows it prints
`[PASS] ram: could not read memory on this platform — check by hand` — a deliberate pass, not a
measurement. Check it yourself there: you want 16 GB total and ~9 GB free before loading an 8 GB
model. Inside WSL2 the check works, and reports WSL2's allocation rather than the machine's.

### The context window — do not skip the `ollama create`

The agent re-sends its whole accumulated tool history every turn, and the longest measured
HumanEvalFix prompts reach ~3,100 tokens. Ollama's default is 4,096 with `--context-shift`, which
drops the *earliest* messages first — starting with the system prompt that tells the agent it is
not finished until the tests pass. Measured on ollama 0.32.4, reading the `CONTEXT` column of
`ollama ps`:

| how the context was requested | resulting `CONTEXT` |
|---|---|
| `/v1` + `extra_body={"options": {"num_ctx": 16384}}` | **4096** — silently ignored |
| `export OLLAMA_CONTEXT_LENGTH=16384` in your own shell, then a request | **4096** — wrong process |
| `ollama create agentfix-mellum2 -f Modelfile` (`PARAMETER num_ctx 16384`) | **16384** ✅ |

So the `Modelfile` route is the one this repo uses: one command, the same on every platform, and
it survives whichever endpoint the client talks to. The client still sends `num_ctx` in
`extra_body` because vLLM and Ollama's native `/api/chat` both honour it — but on Ollama's `/v1`
it does nothing, so do not rely on it.

`OLLAMA_CONTEXT_LENGTH` also works, but only if you get it into the **server process's**
environment, which is a different command on every platform — that is exactly why it is the
fallback and not the instruction:

| where the server runs | how to set it |
|---|---|
| any platform, server in your own terminal | `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` |
| macOS, Ollama.app | `launchctl setenv OLLAMA_CONTEXT_LENGTH 16384`, then restart the app |
| Linux, systemd service | `systemctl edit ollama` → `[Service]` / `Environment="OLLAMA_CONTEXT_LENGTH=16384"`, then `systemctl daemon-reload && systemctl restart ollama` |
| Windows, background app | `setx OLLAMA_CONTEXT_LENGTH 16384`, then quit Ollama from the tray and reopen it |

The `ollama create` above makes all four rows unnecessary. Do that instead.

## Option 2: Qwen3 local — the 1.4 GB fallback

macOS, Linux, WSL2:

```bash
ollama pull qwen3:1.7b
printf 'FROM qwen3:1.7b\nPARAMETER num_ctx 16384\n' > /tmp/Modelfile.agentfix-qwen3
ollama create agentfix-qwen3 -f /tmp/Modelfile.agentfix-qwen3
MELLUM_MODEL=agentfix-qwen3 uv run agentfix doctor
MELLUM_MODEL=agentfix-qwen3 uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

Native Windows PowerShell — same four steps, but there is no `/tmp` and no `VAR=value command`
prefix, so the Modelfile goes in the repo and the variable is set for the session:

```powershell
ollama pull qwen3:1.7b
Set-Content Modelfile.agentfix-qwen3 @('FROM qwen3:1.7b', 'PARAMETER num_ctx 16384')
ollama create agentfix-qwen3 -f Modelfile.agentfix-qwen3
$env:MELLUM_MODEL = 'agentfix-qwen3'
uv run agentfix doctor
uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

`$env:MELLUM_MODEL` stays set for the rest of that PowerShell session — use
`Remove-Item Env:\MELLUM_MODEL` to go back to Mellum2. (In `cmd.exe` it is
`set MELLUM_MODEL=agentfix-qwen3`.)

The `ollama create` is for the same reason as Option 1: without it the context is 4,096 and long
runs lose their own history. Skipping it makes `doctor`'s `context window` check fail.

If you swap in a different small model here, check that it emits real tool calls. Many small
models — coder-tuned ones especially — print the call as ordinary text instead of making one, and
then `tool_calls` comes back empty, no tool ever runs, and the loop you built never gets to do
anything. `qwen3:1.7b` emits real tool calls.

It is a much smaller model than Mellum2 and may not do as well — expect it to need more steps, or
to fail a task Mellum2 solves. It also reasons before it acts, so turns take longer. Read the
`--verbose` trace rather than the verdict: what matters here is that the loop behaves correctly.
Good enough to see that; not the demo model.

## Option 3: local exercises + Google Colab for the model (tested)

For machines that cannot comfortably hold either model. Only one thing moves to the browser: the
real model. Everything else stays where it is.

**Do the workshop locally, unchanged.** Clone the repo, install with `uv sync`, build the same
three parts — the `run_tests` tool and its JSON schema, the loop's tool dispatch, the
verification-based stop condition — and run the exercise tests:

```bash
uv run pytest exercises/stage_1        # or stage_2, stage_3
uv run pytest                          # every test that needs no model
```

Those run against a scripted fake model. No Ollama, no model download, no GPU. They pass on any
machine, including a 3.4 GB Chromebook.

**Skip `agentfix doctor` on this option.** It will fail, and that is expected — it looks for
Ollama, a derived model, and a server on `localhost:11434`, none of which exist on this laptop by
design. Nothing is broken and there is nothing to fix.

**Run the real model in the browser.** Open `notebooks/agentfix.ipynb` in Google Colab when you
reach the point of running the agent for real. The notebook does only that step: it installs
Ollama inside the Colab runtime, derives `agentfix-qwen3` with `num_ctx 16384`, puts the finished
agent in place, runs `agentfix doctor` there, and then `agentfix solve` on the workshop tasks.
That `doctor` run — in Colab, not on the laptop — is the one that has to pass.

The notebook checks out the `stage-3-solution` versions of the two exercise files rather than
uploading local edits, so what runs against the real model is guaranteed to work. The learner's
own implementation is verified where they wrote it, by the exercise tests above.

This path has been tested end to end.

## Command reference

```bash
uv run agentfix doctor                                       # preflight check
uv run agentfix solve tasks/workshop/01-shopcart --verbose    # run the agent on one task
uv run agentfix eval --suite workshop --limit 3               # run the agent over a suite
uv run agentfix eval --suite humanevalfix --limit 3
uv run pytest exercises/stage_1                                # or stage_2, stage_3
uv run pytest                                                  # every test that needs no model
uv run pytest --all                                            # add the ones that need Ollama
uv run pytest -m llm                                           # only the ones that need Ollama
```

`solve` also accepts `--max-steps N` (default 10). `eval --suite` accepts `workshop` or
`humanevalfix`.

### Tests and the `llm` marker

One test in this repo talks to a real model; everything else runs offline against a scripted fake
(`src/agentfix/llm/fake.py`). That test carries `@pytest.mark.llm`, and `pyproject.toml` sets
`addopts = "-m 'not llm'"`, so a bare `uv run pytest` is always offline-safe — a broken Ollama
cannot fail your suite. `--all` opts back in, `-m llm` runs that test alone, and combining `--all`
with an explicit `-m` is a usage error rather than a silent override.

Every command above is identical on macOS, Linux, WSL2, and Windows PowerShell — forward slashes in
task paths work in PowerShell too. Only environment variables differ: `MELLUM_MODEL=x uv run ...`
inline on POSIX shells, `$env:MELLUM_MODEL = 'x'` on its own line first in PowerShell. The two
variables the CLI reads are `MELLUM_MODEL` (default `agentfix-mellum2`) and `MELLUM_BASE_URL`
(default `http://localhost:11434/v1`), plus `AGENTFIX_SANDBOX` (`subprocess` or `docker`).

Working through the exercises yourself? Start at `exercises/README.md` — it lays out the three
stages, which file you edit for each, and the `git checkout stage-N-solution` escape hatch if you
fall behind.

### Every pytest option you are likely to want

Two settings in `pyproject.toml` change what a bare `uv run pytest` does, so it is worth knowing
they are there:

```toml
testpaths = ["tests", "exercises"]   # what gets collected when you name no path
addopts   = "-m 'not llm'"           # silently prepended to EVERY invocation
```

`addopts` means `uv run pytest` is really `uv run pytest -m 'not llm'`, even though nothing on
your command line says so. And `testpaths` **differs by branch**: on `main` (the exercise branch)
it is `["exercises"]` only, because the full `tests/` suite would fail against the unimplemented
stubs; on `solutions` it is both directories. So the same command collects 15 tests on one branch
and 150 on the other.

**Project-specific — these exist only because this repo added them:**

| Command | What it does |
|---|---|
| `uv run pytest` | everything that needs no model (149 collected, 1 deselected on `solutions`) |
| `uv run pytest --all` | adds the one test that needs a running Ollama |
| `uv run pytest -m llm` | *only* that test — useful for checking your model setup alone |
| `uv run pytest --all -m llm` | deliberate usage error: `--all cannot be combined with -m 'llm'` |

`--all` is defined in `conftest.py` at the repo root, which pytest auto-loads. It works by
clearing the marker filter that `addopts` injected. The alternative incantation is
`-m "llm or not llm"`, which is accurate, unmemorable, and looks like a typo.

**Choosing what runs:**

| Command | What it does |
|---|---|
| `uv run pytest tests/test_loop.py` | one file |
| `uv run pytest tests/test_loop.py::test_step_budget_is_a_hard_cap` | one test, by node id |
| `uv run pytest exercises/stage_1` | one directory — how you work through the exercises |
| `uv run pytest -k "guard or dispatch"` | any test whose *name* matches the expression |
| `uv run pytest --lf` | only the tests that failed last run ("last failed") |
| `uv run pytest --ff` | all of them, but failures first ("failed first") |
| `uv run pytest -x` | stop at the first failure |
| `uv run pytest --maxfail=3` | stop after three |
| `uv run pytest --co -q` | collect and list test ids without running anything |

Naming a path **overrides `testpaths`**, which is how you run `tests/` on `main` even though the
default there is `exercises/` only.

**Controlling the output:**

| Command | What it does |
|---|---|
| `uv run pytest -q` | quiet: one dot per test |
| `uv run pytest -v` | verbose: one line per test, with its name |
| `uv run pytest -s` | do not capture stdout — needed to see `print()` from inside a test |
| `uv run pytest -rA` | a summary line for every test, not just failures |
| `uv run pytest --tb=short` | shorter tracebacks (`long`, `short`, `line`, `no`) |
| `uv run pytest --durations=10` | the ten slowest tests — the Docker ones dominate here |

**Coverage** (`pytest-cov` is in the `dev` extra):

```bash
uv run pytest --cov=src/agentfix --cov-report=term-missing   # per-file, with unhit line numbers
uv run pytest --cov=src/agentfix --cov-report=html           # writes htmlcov/index.html
```

**Debugging a failure:**

| Command | What it does |
|---|---|
| `uv run pytest --pdb` | drop into the debugger at the point of failure |
| `uv run pytest -l` | show local variable values in tracebacks |
| `uv run pytest --setup-show` | show fixture setup and teardown around each test |

Two of these are worth combining while working: `uv run pytest --lf -x -vv` reruns just what broke,
stops at the first one, and shows you the full assertion diff.

## Running things in Docker

Everything above runs the tests directly on your machine. The Docker path is different and narrower:
it swaps out **one thing** — how `run_tests` executes the task's test suite — via the
`AGENTFIX_SANDBOX` environment variable. The agent itself, the model client, and the file tools
still run on the host either way. See `src/agentfix/sandbox/` for why the boundary sits there.

### Build the image first

```bash
docker build -t agentfix-sandbox -f Dockerfile.sandbox .
```

The trailing `.` is the build context, not punctuation — leaving it off is the usual mistake. The
image is about 235 MB and pins pytest to the version in `uv.lock`, so both backends verify a fix
identically.

Check it exists:

```bash
docker images agentfix-sandbox
```

### Run the agent with the container sandbox

```bash
# POSIX shells (macOS, Linux, WSL2) — inline, applies to this one command
AGENTFIX_SANDBOX=docker uv run agentfix doctor
AGENTFIX_SANDBOX=docker uv run agentfix solve tasks/workshop/01-shopcart --verbose
AGENTFIX_SANDBOX=docker uv run agentfix eval --suite workshop --limit 3

# ...or export it once for the whole shell session
export AGENTFIX_SANDBOX=docker
uv run agentfix solve tasks/workshop/02-invoice --verbose
unset AGENTFIX_SANDBOX          # back to the subprocess backend
```

```powershell
# Windows PowerShell — its own line, before the command
$env:AGENTFIX_SANDBOX = 'docker'
uv run agentfix solve tasks/workshop/01-shopcart --verbose
Remove-Item Env:\AGENTFIX_SANDBOX
```

`agentfix doctor` is the quickest confirmation: its `sandbox` check runs a trivial passing test
through whichever backend is configured, so `[PASS] sandbox: executes tests` with
`AGENTFIX_SANDBOX=docker` set means the image works. A typo fails loudly rather than quietly
falling back:

```
$ AGENTFIX_SANDBOX=podman uv run agentfix doctor
ValueError: Unknown AGENTFIX_SANDBOX='podman'; expected 'subprocess' or 'docker'
```

### Run the Docker backend's own tests

```bash
uv run pytest tests/test_docker_backend.py           # 20 passed with the image built
uv run pytest tests/test_docker_backend.py -v        # see which are flag checks vs live runs
uv run pytest tests/test_sandbox_image.py            # pytest-version pin; needs no daemon
uv run pytest tests/test_docker_backend.py --durations=5
```

Fifteen of those assert on the generated `docker run` command line and need no daemon at all;
five actually start containers and skip until the image is built. With it built, all 20 pass —
including no-network, unwritable-workspace, and timeout-leaves-nothing-behind.

### What the container actually gives you

The default subprocess backend is *hardened* — stripped environment, resource limits, a timeout —
but it is not isolated: test code runs as your user and can reach your filesystem and the network.
The container is the real boundary:

| Flag | Effect |
|---|---|
| `--network none` | no network at all |
| `--read-only` + `:ro` mount | nothing inside can be written; the file tools write on the host |
| `--user runner` | not root, even inside |
| `--cap-drop ALL`, `--security-opt no-new-privileges` | no Linux capabilities, none regainable |
| `--memory 512m`, `--pids-limit 128`, `--cpus 1` | hard resource caps |
| `--rm` | container deleted on exit; nothing survives a run |

Inspect the exact command line without running anything:

```bash
uv run python -c "
from pathlib import Path
from agentfix.sandbox.docker_backend import DockerBackend
print(' '.join(DockerBackend().build_argv(Path('/tmp/demo'), ('python', '-m', 'pytest', '-q'))))
"
```

Containers are started with `--rm`, so nothing to clean up afterwards. If a run is interrupted
mid-container, `docker ps -a --filter name=agentfix-` will show any stragglers.

## Adding your own task

The three workshop fixtures are deliberately tiny. Once the agent solves them, the most useful
thing you can do is point it at a bug of your own — that is where you find out what a 12B model can
and cannot do.

A task is a directory with exactly two things in it: a `task.json` and a `repo/` folder holding a
self-contained project.

```
tasks/workshop/01-shopcart/
├── task.json                  # metadata: what to run, what should fail
└── repo/                      # the buggy project, copied fresh into a tempdir per run
    ├── shopcart/
    │   ├── __init__.py
    │   ├── cart.py            # ← the bug lives here
    │   └── pricing.py
    └── tests/
        └── test_cart.py       # ← the specification; the agent may not edit this
```

`repo/` is the agent's whole world. Nothing outside it is visible, and nothing inside it is
imported by the workshop itself — so it needs its own `__init__.py` files and its own tests, and it
must run with no dependencies beyond pytest and the standard library. The agent never touches your
copy: `workspace()` copies `repo/` into a temp dir per run and deletes it afterwards.

`task.json` has four fields, all optional except in practice you want all four:

```json
{
  "task_id": "04-mytask",
  "test_command": ["-m", "pytest", "-q"],
  "expected_failures": ["test_the_one_that_is_red"],
  "prompt": "The test suite for this project is failing. Find the bug and fix it."
}
```

- **`task_id`** — label used in output and the temp-dir name. Defaults to the directory name.
- **`test_command`** — run from inside `repo/`. A leading `-` flag gets `sys.executable` prepended,
  so `["-m", "pytest", "-q"]` becomes `python -m pytest -q`.
- **`expected_failures`** — the test *function* names that must be red before the agent starts.
  Bare names, not `path::name`.
- **`prompt`** — the first user message. Keep it free of hints; see below.

Three rules that make a task work:

1. **It must start red.** `uv run pytest` inside `repo/` should fail before the agent runs. A task
   that starts green is solved before it begins.
2. **The bug goes in the source, never in the tests.** `write_file` refuses any path under `tests/`
   or named `test_*.py`, because `run_tests` is the agent's only oracle — a bug in the test suite is
   unfixable by construction.
3. **Keep files small.** `write_file` takes complete file contents, not a diff, and the model has a
   4k-token context in the derived Modelfile. A 300-line file the agent has to rewrite in full is a
   task about context limits, not about debugging. `read_file` truncates at 4,000 characters.

Then run it:

```bash
uv run agentfix solve tasks/workshop/04-mytask --verbose
```

`--verbose` prints the trace — every model turn and every tool call — which is the point of trying
your own task. A `NOT SOLVED` with a readable trace teaches more than a `SOLVED`.

Two things worth knowing:

- **The eval suite globs, but `--limit` defaults to 3.** `agentfix eval --suite workshop` picks up
  any directory under `tasks/workshop/` containing a `task.json`, sorted by name, then truncates to
  `--limit`. A fourth task sorts last and is silently cut off, so run
  `uv run agentfix eval --suite workshop --limit 4` to include it. Adding a task also changes the
  pass@1 denominator, so it won't be comparable to the measured numbers below.
- **`tests/test_tasks.py` won't check your task.** Its fixture-validation tests hardcode the
  original three paths, so your task is neither verified nor broken by them. To have
  `expected_failures` and the starts-red rule enforced automatically, add your path to the
  `@pytest.mark.parametrize` lists at `tests/test_tasks.py:44` and `:57`.

## Measured performance

Measured on an Apple M4, 24 GB, against a local Ollama running
`hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M`. Expect roughly 3-4x slower on an older
Intel laptop.

| Metric | Result |
|---|---|
| Generation throughput | 51 tok/s (372 tokens in 7.3s, as `agentfix doctor` reports it) |
| Prefill throughput | ~480 tok/s (a 3,438-token prompt took ~7s before the first output token) |
| Cold model load | ~3.5s, one-time |
| GGUF size on disk | 8.07 GB |
| Loaded context window | 16,384 tokens (`ollama ps`, via the derived model) |
| Workshop suite (`01`–`03`), pass@1 | 1.00 (3/3), 44.5s wall clock, peak prompt 1,456 tok |
| HumanEvalFix (20 vendored tasks), pass@1 | 0.60 (12/20), median 7 steps, max 10, 185,235 tokens, 8m09s wall clock, peak prompt 2,998 tok |

pass@1 on HumanEvalFix was **0.50 before** the loop's stop condition was made real. The old loop
ended a run on any text-only reply, so four failures stopped at 3–5 steps of 10 with the budget
unused; now a text-only reply while the tests are red gets a nudge and another step, and every one
of the 8 remaining failures uses all 10 steps. The context-window fix landed in the same
measurement, so the two effects are not separated — note that the largest single prompt was 2,998
tokens against the old 3,072-token usable window, i.e. the longest runs really were at the edge.
Nothing was tuned to move the number.

The wall-clock figure is exactly why that eval segment is demo-only in the workshop — it does
not fit in a 90-minute session as a live activity. `results/precomputed/` ships both runs so
students can discuss the numbers without waiting for them.

For contrast, the predecessor project's baseline: `Qwen/Qwen2.5-Coder-1.5B-Instruct` on a GPU, over
all 164 HumanEvalFix Python tasks, **pass@1 = 0.305** (50/164) with greedy decoding. That run was
**single-shot** — one patch per task, graded by running the tests, no loop and no tools, which is the
difference that matters here and not the parameter count. It also came with a 13-config decoding
sweep (temperature, top-p, beams, repetition penalty) spanning 0.262–0.317; with one run per config
and n=164, the binomial standard error is ±0.036, so that whole spread is noise. Only 31 of the 164
tasks changed verdict across all 13 configs. Decoding parameters bought almost nothing; adding a
loop and a test-execution oracle roughly doubled the fix rate.

Not an apples-to-apples comparison — different task count, different hardware, different harness —
but the mechanism is the point. The raw reports used to ship in `results/legacy/`; they were removed
from the repo, and `git log --diff-filter=D -- results/legacy` finds the commit that dropped them if
you want them back.

## Platform notes

What actually differs between operating systems, beyond the install commands:

| | macOS | Linux | WSL2 | native Windows |
|---|---|---|---|---|
| verified here | yes (M4, 24 GB) | no | no | no |
| `doctor` RAM check | `sysctl` + `vm_stat` | `/proc/meminfo` | `/proc/meminfo`, reports WSL2's slice | skipped with a PASS — check by hand |
| subprocess sandbox limits | CPU + file size + process count | CPU + file size + process count + 2 GB address space | same as Linux | none applied |
| Docker sandbox | Docker Desktop | Docker Engine | Docker Desktop w/ WSL2 backend, or Engine inside WSL2 | Docker Desktop |

Only the macOS column has been run end to end. Linux, WSL2, and native Windows are documented from
the code and the vendors' own install instructions, not from a run in this environment — if you hit
something the README gets wrong, tell the instructor so this table can be corrected.

The sandbox is the one place where the difference is structural rather than cosmetic.
`src/agentfix/sandbox/subprocess_backend.py` runs each test execution in a child process with a
stripped environment (`PATH=/usr/bin:/bin`) and POSIX `resource` limits applied through
`preexec_fn`. Both of those are POSIX-only: on native Windows the `resource` import fails and
`preexec_fn` is skipped, so the child gets the stripped environment but **no CPU, file-size, or
process caps** — a runaway test cannot be cut off by anything except the 10-second wall-clock
timeout. `RLIMIT_AS` (the 2 GB address-space cap) is deliberately Linux-only as well, because
applying it on Apple Silicon aborts interpreter startup. This is the reason WSL2 is the recommended
Windows path, and it is also why the `sandbox` line in `doctor` is worth reading rather than
skimming: it is the only signal that test execution works at all on your machine.

If you want the tighter isolation on any platform, build the image once and switch backends — the
commands are the same everywhere:

```bash
docker build -t agentfix-sandbox -f Dockerfile.sandbox .
AGENTFIX_SANDBOX=docker uv run agentfix doctor     # PowerShell: $env:AGENTFIX_SANDBOX = 'docker'
```

Verified on macOS: with the image built, `AGENTFIX_SANDBOX=docker uv run agentfix doctor` reports
`[PASS] sandbox: executes tests` and the container test file is 20 passed, no skips.

## Known limitations

- **The Docker sandbox needs one build step, and then it is verified.** Every isolation flag
  (`--network none`, the memory/pid/cpu caps, `--read-only`, `--cap-drop ALL`, …) is asserted by
  tests that run without a daemon. Five more actually start containers and check behaviour; they
  skip until you build the image:
  `docker build -t agentfix-sandbox -f Dockerfile.sandbox .` (note the trailing `.` — it is the
  build context). With the image built, `uv run pytest tests/test_docker_backend.py` is **20
  passed, no skips** — measured, including no-network, unwritable-workspace, and
  timeout-leaves-nothing-behind. The image's pytest is pinned to the version in `uv.lock` so both
  backends verify fixes identically (confirmed in-container: pytest 9.1.1);
  `tests/test_sandbox_image.py` enforces that pin without needing a daemon.
- **The Google Colab notebook is the supported browser path.** `notebooks/agentfix.ipynb` has been tested end to end. It covers only the real-model run: those learners do the exercises locally against the fake model and skip the local `doctor` check, which is expected to fail for them.
- **Known failure class: the agent's only oracle is `run_tests`.** `write_file` refuses paths under
  `tests/` or named `test_*.py`, and a successful write invalidates the last test result, so
  neither rewriting the specification nor a stale green run can produce a false `SOLVED`. Both were
  reproducible before those guards; see `ARCHITECTURE.md` for what remains open (nothing stops the
  agent from special-casing the exact inputs the tests use).

## License

MIT — see [LICENSE](LICENSE).
