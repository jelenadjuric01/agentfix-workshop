from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from agentfix.tools.base import MAX_FILE_READ_CHARS, ToolResult, truncate

IGNORED_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}
PROTECTED_HINT = (
    "Refused: {path} is part of the test suite. The tests are the specification — "
    "fix the source instead."
)


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the task root."""


def resolve_in_root(root: Path, candidate: str) -> Path:
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathEscapeError(f"{candidate} resolves outside the task root")
    return resolved


def is_test_path(root: Path, target: Path) -> bool:
    """`run_tests` is the only oracle, so letting the agent rewrite the tests is a bypass."""
    try:
        relative = target.relative_to(root.resolve())
    except ValueError:
        return False
    return "tests" in relative.parts or relative.name.startswith("test_")


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

    def __init__(self, root: Path, on_write: Callable[[], None] | None = None) -> None:
        self.root = root
        self.on_write = on_write

    def run(self, path: str, content: str) -> ToolResult:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            return ToolResult(False, f"Refused: {path} is outside the project root.")

        if is_test_path(self.root, target):
            return ToolResult(False, PROTECTED_HINT.format(path=path))

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
        return ToolResult(True, f"Wrote {len(content)} characters to {path}.")
