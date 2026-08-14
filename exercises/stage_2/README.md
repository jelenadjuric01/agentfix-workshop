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
