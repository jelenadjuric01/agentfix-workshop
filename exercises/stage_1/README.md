# Stage 1 — Give the agent a tool

Open `src/agentfix/tools/tests_tool.py`. Two `TODO(stage-1)` markers.

1. **`parameters`** — the JSON Schema the model sees. `run_tests` takes no arguments, so this
   is an object with an *empty* `properties` — empty, not absent. Read the `description` above
   it while you are there: it is already written, and it is the only thing telling the model
   when to reach for this tool. Ask yourself whether you would pick this tool from that
   sentence alone — a schema is an API for a reader who cannot ask follow-up questions.
2. **`run()`** — call `self.backend.run(...)`, store the result on `self.last_result`,
   and return a `ToolResult`.

Watch the distinction: `ToolResult.ok` means *the tool worked*.
`last_result.passed` means *the tests passed*. They are not the same, and stage 3
depends on the difference.

And watch where each one is *read*. `last_result` is for `is_done` in stage 3; the model never
sees it. The model sees only `ToolResult.content`, so that string has to carry both the verdict
and the pytest output. Return an empty one and `is_done` still works perfectly while the agent
has no idea what failed.

    uv run pytest exercises/stage_1 -v
