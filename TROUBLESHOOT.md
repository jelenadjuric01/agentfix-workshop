# Troubleshooting

Everything that has actually gone wrong for someone running this workshop, and what fixed it.
Nothing here is a bug in your exercise work — these are environment problems, and each one has a
short answer.

Click a heading to open it.

## Environment and tooling

<details>
<summary><b><code>ModuleNotFoundError: No module named 'openai'</code> (or <code>datasets</code>, or anything else)</b></summary>

Something ran outside the project environment. `uv run` creates and uses `.venv` for you, so the
usual cause is a bare `python`/`pytest` invocation instead of `uv run …`.

```bash
uv sync --extra dev
uv run agentfix doctor
```

`datasets` specifically is an optional extra — it is only needed for
`uv run agentfix eval --suite humanevalfix`, and it comes from `uv sync --extra eval`.

If you are running commands without `uv`, activate the environment first
(`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) so `python` means the
project's interpreter.
</details>

<details>
<summary><b>Your IDE underlines <code>agentfix</code> imports in red, but the tests pass</b></summary>

The IDE is looking at a different interpreter than `uv run` is. The package lives under `src/`, so
it resolves only when the IDE is pointed at the project's `.venv`, where the package is installed
in editable mode.

In PyCharm: **Settings** → **Project: agentfix-workshop** → **Python Interpreter** → select the
existing `.venv` in the repo root. In VS Code: **Python: Select Interpreter** → the same `.venv`.

If the imports still look wrong afterwards, mark `src` as the sources root (PyCharm: right-click
`src` → **Mark Directory as** → **Sources Root**). Neither change affects how the code runs — only
what the editor understands.
</details>

<details>
<summary><b>Tests pass in the terminal but fail in the IDE, or the reverse</b></summary>

Two different interpreters again. The terminal uses `uv run` (or whatever is on your `PATH`); the
IDE uses the one in its settings. Point both at the repo's `.venv`.

Worth knowing about the suite itself: `pyproject.toml` sets `addopts = "-m 'not llm'"`, so a bare
`uv run pytest` never touches Ollama. If your IDE's run configuration overrides that, it will try
to run the model-backed test and fail for reasons that have nothing to do with your code.
</details>

<details>
<summary><b>Windows: you installed Ollama or set <code>MELLUM_MODEL</code>, and nothing sees it</b></summary>

Close your IDE and your terminals, and open them again.

Windows hands a process its environment variables when the process starts and never updates them.
The Ollama installer's `PATH` entry, `setx`, and the **Environment Variables** dialog all apply to
*future* processes — anything already running keeps the environment it was launched with. That
includes an IDE and every terminal inside it.

Check in a **new** terminal:

```powershell
echo $env:MELLUM_MODEL
ollama --version
```

Note that in this repo `MELLUM_MODEL` is normally set per session
(`$env:MELLUM_MODEL = 'agentfix-qwen3'`) or per command
(`MELLUM_MODEL=agentfix-qwen3 uv run agentfix …` on POSIX), so it is gone the moment you open a new
shell. That is expected — set it again, or use `setx` if you want it to persist.
</details>

## Ollama and the model

<details>
<summary><b><code>doctor</code> says <code>context window: 4096</code> instead of <code>16384</code></b></summary>

The `ollama create` step was skipped, so you are talking to the base model rather than the derived
one:

```bash
ollama create agentfix-mellum2 -f Modelfile
```

or, on Option 2, the `ollama create agentfix-qwen3` command from the README.

Do not skip it. At 4,096 tokens Ollama drops the *earliest* messages once the conversation grows
past the limit — and the earliest message is the system prompt telling the agent it is not done
until the tests pass. The symptom is an agent that appears to forget the task halfway through,
which looks like a weak model rather than a misconfigured one.

`ollama ps` after one request shows the same thing: the CONTEXT column must read 16384.
</details>

<details>
<summary><b>Connection refused on <code>localhost:11434</code></b></summary>

Ollama is installed but the server is not running.

```bash
brew services start ollama       # macOS, Homebrew
open -a Ollama                   # macOS, the app
sudo systemctl start ollama      # Linux
ollama serve                     # anywhere, including WSL2 without systemd
```

On macOS the Homebrew service and the tray app are the same server on the same port — run one, not
both.
</details>

<details>
<summary><b>The wrong model is being used</b></summary>

`MELLUM_MODEL` decides, and it defaults to `agentfix-mellum2`. Confirm what exists and what is set:

```bash
ollama list
echo $MELLUM_MODEL               # PowerShell: echo $env:MELLUM_MODEL
```

To go back to the default, clear the override: `unset MELLUM_MODEL`, or
`Remove-Item Env:\MELLUM_MODEL` in PowerShell.
</details>

<details>
<summary><b>The model is painfully slow, or the machine runs out of memory</b></summary>

Mellum2 is an 8 GB model and wants 16 GB of RAM to be comfortable. Switch to Option 2
(`qwen3:1.7b`, ~1.4 GB) — the README has the commands. The loop behaves the same, but the model is
much smaller and may not do as well, so expect more steps or a task it cannot fix. It also reasons
before it acts, so turns take longer.

Under 8 GB, use Option 3: do the exercises locally against the fake model and run
`notebooks/agentfix.ipynb` in Colab for the real-model run.
</details>

<details>
<summary><b>On Option 3, <code>agentfix doctor</code> fails locally</b></summary>

Expected, and nothing is broken. On that option there is no Ollama, no model and no server on this
machine by design — they live in the Colab runtime. Skip `doctor` here;
`notebooks/agentfix.ipynb` runs the same check inside Colab, and that is the one that has to pass.

Every exercise still runs locally, because the exercise tests use a scripted fake model.
</details>

## Running the agent

<details>
<summary><b><code>agentfix solve</code> on <code>main</code> exits with an error about the tool schema</b></summary>

That is the intended starting point, not a broken install. `main` ships the exercises stubbed, so
the `run_tests` schema is empty and the model provider rejects the request — the CLI catches it,
prints a readable message, and exits 1.

Implement the stages, or see the finished agent with `git checkout solutions` (or the
`stage-1-solution` / `stage-2-solution` / `stage-3-solution` tags).
</details>

<details>
<summary><b>The agent prints <code>NOT SOLVED</code></b></summary>

Not necessarily your bug. Real models do not fix every task, and the Option 2 model
(`qwen3:1.7b`) is much smaller than Mellum2 and may not do as well.

Read the `--verbose` trace before assuming the code is wrong. You want the shape of a working
loop: the model calls `run_tests`, looks around with `list_files` / `read_file`, writes a file, and
runs the tests again. If that shape is there and the run simply ran out of steps, the
implementation is doing its job.

Two shapes that *do* point at your code: the agent never dispatches a tool at all (Stage 2), or the
run ends while the tests are still red (Stage 3). For the first one, check the next entry before
you go looking at your dispatch — a model that cannot emit tool calls produces the same trace.

At the `stage-2-solution` checkpoint `NOT SOLVED` is guaranteed — the stop condition does not exist
yet, so the run always burns its full 10-step budget. That is the point of the checkpoint, not a
failure.
</details>

<details>
<summary><b>Every trace line is the model printing JSON, and no tool ever runs</b></summary>

A trace where every step is an `llm:` line printing a JSON object, with no `tool:` line between
them, means the model is *describing* a tool call as text instead of making one. The client reads
native `tool_calls` off the reply and nothing else, so an empty `tool_calls` field means no tool is
dispatched, `run_tests` never runs, the stop condition never sees a passing suite, and the run
spends its whole budget. The loop guard cannot help either: it compares one tool call against the
previous one, and there are no tool calls to compare.

**This is the model, not your code.** Not every model does OpenAI-style function calling, and
small coder-tuned ones frequently do not. It only comes up if you pointed `MELLUM_MODEL` at a
model of your own choosing — the two the README names both emit real tool calls.

Check which model actually served the run before anything else:

```bash
ollama ps
echo $MELLUM_MODEL
```
</details>

<details>
<summary><b>Docker sandbox tests skip, or <code>doctor</code> reports a <code>sandbox</code> failure</b></summary>

The Docker backend needs a running daemon **and** a built image. A skip means "unproven at
runtime", not "broken" — the default subprocess backend still works, so the workshop is unaffected.

```bash
docker info
docker build -t agentfix-sandbox -f Dockerfile.sandbox .     # the trailing . is the build context
uv run pytest tests/test_docker_backend.py -v                # 20 passed, no skips
```

Leaving off the trailing `.` is the usual mistake. On native Windows the subprocess backend applies
no resource limits at all (the POSIX `resource` module does not exist there) — if you need real
isolation on Windows, use Docker or WSL2.
</details>

## Nothing here matches

Run `uv run agentfix doctor` and read it top to bottom. It prints `[PASS]`/`[FAIL]` per check with
a remedy command for each failure, which is faster than guessing.
