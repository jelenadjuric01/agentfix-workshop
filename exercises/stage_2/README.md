# Stage 2 — Close the loop

Open `src/agentfix/agent/loop.py`, find `TODO(stage-2)` inside `run_agent`.

For each tool call the model made: dispatch it through the registry and append the
result to `messages`.

Four rules the tests enforce:

- The observation goes back as a **`role="tool"`** message carrying the matching
  **`tool_call_id`**. Drop the id and the model cannot match the answer to its question.
- **Only ever append.** Never rewrite or reorder earlier messages — the server reuses
  its KV cache for an unchanged prefix, and mutating history makes every turn re-read
  the whole conversation from scratch.
- Go through **`registry.dispatch(call)`**. Looking the tool up yourself with `registry.get`
  and calling `.run(**call.arguments)` looks equivalent and is not: `dispatch` is where a
  hallucinated tool name, malformed JSON, a missing argument, and a crashing tool all become
  observations instead of exceptions that end the run.
- **Record a `TraceEvent`** for the call, like the guard branch above already does. The trace
  is what `--verbose` prints, and it is how you will debug your own agent ten minutes from now.

    uv run pytest exercises/stage_2 -v
