"""The three file tools: list_files, read_file, write_file.

Each is a plain class matching the `Tool` protocol in tools/base.py — class attributes for
the schema the model sees, and a `run` method with its own named parameters that
`ToolRegistry.dispatch` fills in from the model's JSON.

These run in-process on the host, with no subprocess and no container (see sandbox/base.py
for why). Their safety therefore comes from validating *before* acting, and the four checks
that do it are `resolve_in_root`, `is_test_path`, `ast.parse`, and `truncate`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from agentfix.tools.base import MAX_FILE_READ_CHARS, ToolResult, truncate

# Noise the model should never spend context on.
IGNORED_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

PROTECTED_HINT = (
    "Refused: {path} is part of the test suite. The tests are the specification — "
    "fix the source instead."
)


NEW_FILE_HINT = (
    "Refused: {path} is not one of this project's existing files. Fix a file that is "
    "already there rather than creating a new one."
)


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the task root."""


def resolve_in_root(root: Path, candidate: str) -> Path:
    """Resolve `candidate` under `root`, refusing anything that escapes it.

    This is the containment boundary for every file operation. Because the agent process
    itself is not sandboxed, this function is the only thing standing between a
    model-supplied path and your real filesystem — hence a raised exception rather than a
    warning.

    `.resolve()` is what makes it work: it collapses "..", follows symlinks and produces an
    absolute path, so "../../../../etc/passwd" and a symlink pointing outside the workspace
    are both caught. Comparing the raw strings would not catch either.
    """
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    # Allowed if it IS the root, or if the root is one of its parents.
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathEscapeError(f"{candidate} resolves outside the task root")
    return resolved


def is_test_path(root: Path, target: Path) -> bool:
    """`run_tests` is the only oracle, so letting the agent rewrite the tests is a bypass.

    Without this, an agent that cannot fix the bug can still turn the suite green by
    deleting the failing assertion — and `is_done` would report SOLVED. That was
    reproducible before this check existed.
    """
    try:
        relative = target.relative_to(root.resolve())
    except ValueError:
        # Not under the root at all. resolve_in_root already refused it; say "not a test
        # file" rather than crashing.
        return False
    # Both comparisons are case-INSENSITIVE, and that is not pedantry. macOS ships a
    # case-insensitive filesystem, so "Tests/TEST_CART.PY" addresses the very same inode as
    # "tests/test_cart.py" — while a case-sensitive check calls it an ordinary source file and
    # waves it through. Reproduced: the agent overwrote the protected suite with a trivially
    # passing test and the run reported SOLVED with the bug untouched.
    if any(part.lower() == "tests" for part in relative.parts):
        return True
    return relative.name.lower().startswith("test_")


def _relative_files(root: Path) -> list[str]:
    """Every .py file under `root`, sorted, as paths relative to it.

    Relative and sorted on purpose: absolute paths would leak the temp directory name into
    the model's context and change on every run, which wastes tokens and makes traces
    impossible to compare.
    """
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        found.append(str(path.relative_to(root)))
    return found


class ListFilesTool:
    """Orientation: what files exist. Takes no arguments."""

    # These three class attributes are what `ToolRegistry.schemas()` turns into the model's
    # tool description. Empty "properties" with no "required" list means "no arguments".
    name = "list_files"
    description = "List the Python files in the project, relative to the project root."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, root: Path) -> None:
        # Per-instance state, unlike the schema above: each run has a different workspace.
        self.root = root

    def run(self) -> ToolResult:
        files = _relative_files(self.root)
        if not files:
            # An empty project is not an error, so report it as a normal observation.
            return ToolResult(True, "The project contains no Python files.")
        return ToolResult(True, truncate("\n".join(files)))


class ReadFileTool:
    """Read one file. The model is told to read only files the failure implicates."""

    name = "read_file"
    description = "Read the contents of one file in the project."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root"}
        },
        # "required" is not decoration: dispatch reads this list to reject a call that omits
        # `path` before `run` is ever entered.
        "required": ["path"],
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, path: str) -> ToolResult:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            # Refusals are observations, not exceptions — the model can try another path.
            return ToolResult(False, f"Refused: {path} is outside the project root.")

        if not target.is_file():
            # Listing what does exist turns a typo into a one-turn recovery.
            available = ", ".join(_relative_files(self.root))
            return ToolResult(False, f"No such file: {path}. Files in this project: {available}")

        return ToolResult(True, truncate(target.read_text(encoding="utf-8"), MAX_FILE_READ_CHARS))


class WriteFileTool:
    """Replace one file wholesale. Deliberately not a diff tool.

    Production agents usually edit via diffs. At 12B, models reliably emit invalid unified
    diffs — drifting line numbers, mismatched context — and burn their step budget failing
    to apply a patch instead of fixing anything. A full-file rewrite is far more reliable at
    this model size. The general lesson: a tool contract the model cannot satisfy looks
    exactly like a broken agent.
    """

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

    def __init__(
        self,
        root: Path,
        on_write: Callable[[], None] | None = None,
        allowed: frozenset[str] | None = None,
    ) -> None:
        self.root = root
        # The relative paths this tool may write, taken from the task's pristine template.
        #
        # An allow-list rather than more refusal rules, because "does this path look dangerous"
        # is an unwinnable game. Two reproduced escapes made the case: writing `pytest.py` at
        # the workspace root shadows the module `run_tests` executes (a red suite then exits 0,
        # so `is_done` reports SOLVED with the bug untouched), and a `.pth` file under a
        # workspace-relative site-packages path runs arbitrary code at interpreter startup.
        # Neither goes anywhere near a name `is_test_path` inspects.
        #
        # The agent's job is to repair a file that already exists, so it never needs to create
        # one. Saying exactly that closes every "write a NEW file that changes what the tests
        # do" route at once. `None` disables the check, which only tests use.
        self.allowed = allowed
        # A callback invoked after a successful write. runner.py passes
        # `run_tests.invalidate`, so writing a file discards the previous test result: that
        # result described the code as it was, and stale green tests would let `is_done`
        # report SOLVED for a file that was just changed.
        self.on_write = on_write

    def run(self, path: str, content: str) -> ToolResult:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            return ToolResult(False, f"Refused: {path} is outside the project root.")

        # Guard the oracle: the tests are the specification, so they are not writable.
        if is_test_path(self.root, target):
            return ToolResult(False, PROTECTED_HINT.format(path=path))

        # Then the allow-list. Compared on the resolved path's own relative form, so a
        # case-variant alias of a real file ("Tests/TEST_CART.PY") does not match the template
        # entry it aliases, and is refused even on a case-insensitive filesystem.
        if self.allowed is not None:
            relative = str(target.relative_to(self.root.resolve()))
            if relative not in self.allowed:
                return ToolResult(False, NEW_FILE_HINT.format(path=path))

        # Parse before writing. `ast.parse` compiles the text without executing any of it,
        # so it is a syntax check with no side effects. Rejecting a broken file up front
        # gives a precise error ("line 12: unexpected indent") instead of leaving the
        # project unimportable and making the next `run_tests` fail for a confusing reason.
        try:
            ast.parse(content)
        except SyntaxError as error:
            return ToolResult(
                False,
                f"Not written — the content has a syntax error on line {error.lineno}: {error.msg}",
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if self.on_write is not None:
            self.on_write()
        # Report the size rather than echoing the content: the model just sent it, and
        # repeating it back would double its cost in context for no new information.
        return ToolResult(True, f"Wrote {len(content)} characters to {path}.")
