import sys

import pytest

from agentfix.tools.fs import (
    ListFilesTool,
    PathEscapeError,
    ReadFileTool,
    WriteFileTool,
    is_test_path,
    resolve_in_root,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "def test_add():\n    pass\n", encoding="utf-8"
    )
    # A .py file, not a .pyc: `rglob("*.py")` never matches a .pyc, so a .pyc fixture would
    # pass whether or not IGNORED_DIRS is applied at all. The directories that actually matter
    # are the ones full of real .py files — a stray .venv inside a task repo above all.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("junk = 1\n", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "vendored.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_resolve_in_root_accepts_paths_inside(repo):
    assert resolve_in_root(repo, "src/calc.py") == repo / "src" / "calc.py"


@pytest.mark.parametrize("bad", ["../outside.py", "/etc/passwd", "src/../../escape.py"])
def test_resolve_in_root_rejects_escapes(repo, bad):
    with pytest.raises(PathEscapeError):
        resolve_in_root(repo, bad)


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating a symlink needs elevated rights on Windows"
)
def test_resolve_in_root_rejects_a_symlink_that_points_outside(repo, tmp_path_factory):
    """The reason `resolve_in_root` calls `.resolve()` rather than comparing strings.

    Every case above is textual — `..` and an absolute path — and a prefix check on the raw
    string would catch them. A symlink is the case that needs the filesystem: the path is
    entirely inside the root right up until it is followed.
    """
    outside = tmp_path_factory.mktemp("outside") / "secret.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    (repo / "innocent.py").symlink_to(outside)

    with pytest.raises(PathEscapeError):
        resolve_in_root(repo, "innocent.py")


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating a symlink needs elevated rights on Windows"
)
def test_the_file_tools_refuse_a_symlink_that_points_outside(repo, tmp_path_factory):
    """And the refusal reaches the model as an observation, not as a leaked file."""
    outside = tmp_path_factory.mktemp("outside") / "secret.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    (repo / "innocent.py").symlink_to(outside)

    read = ReadFileTool(repo).run(path="innocent.py")
    assert read.ok is False
    assert "SECRET" not in read.content

    written = WriteFileTool(repo).run(path="innocent.py", content="x = 2\n")
    assert written.ok is False
    assert outside.read_text(encoding="utf-8") == "SECRET = 1\n", "the target must be untouched"


def test_list_files_lists_sources_and_hides_ignored_directories(repo):
    result = ListFilesTool(repo).run()
    assert result.ok is True
    assert "src/calc.py" in result.content
    assert "tests/test_calc.py" in result.content
    # Every byte listed here is a byte the model re-reads on every later turn.
    assert "__pycache__" not in result.content
    assert ".venv" not in result.content


def test_is_test_path_says_no_for_a_path_outside_the_root(repo, tmp_path_factory):
    """`resolve_in_root` has already refused these, so the answer is "not a test file"."""
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "tests" / "test_other.py"
    assert is_test_path(repo, elsewhere) is False


def test_list_files_reports_an_empty_project_as_an_observation(tmp_path):
    result = ListFilesTool(tmp_path).run()
    assert result.ok is True
    assert "no Python files" in result.content


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
    result = WriteFileTool(repo).run(
        path="src/calc.py", content="def add(a, b):\n    return a + b\n"
    )
    assert result.ok is True
    assert (repo / "src" / "calc.py").read_text(encoding="utf-8").endswith("return a + b\n")


def test_write_file_rejects_syntax_errors_before_saving(repo):
    original = (repo / "src" / "calc.py").read_text(encoding="utf-8")
    result = WriteFileTool(repo).run(
        path="src/calc.py", content="def add(a, b)\n    return a + b\n"
    )
    assert result.ok is False
    assert "syntax" in result.content.lower()
    assert (repo / "src" / "calc.py").read_text(encoding="utf-8") == original


def test_write_file_outside_root_is_refused(repo):
    result = WriteFileTool(repo).run(path="../evil.py", content="x = 1\n")
    assert result.ok is False
    assert "outside" in result.content.lower()


@pytest.mark.parametrize("protected", ["tests/test_calc.py", "test_calc.py", "tests/conftest.py"])
def test_write_file_refuses_to_touch_the_test_suite(repo, protected):
    result = WriteFileTool(repo).run(path=protected, content="def test_ok():\n    assert True\n")
    assert result.ok is False
    assert "tests are the specification" in result.content.lower()
    assert (repo / "tests" / "test_calc.py").read_text(encoding="utf-8") == (
        "def test_add():\n    pass\n"
    )


def test_a_successful_write_notifies_its_observer(repo):
    notified = []
    result = WriteFileTool(repo, on_write=lambda: notified.append(True)).run(
        path="src/calc.py", content="def add(a, b):\n    return a + b\n"
    )
    assert result.ok is True
    assert notified == [True]


def test_a_refused_write_does_not_notify(repo):
    notified = []
    tool = WriteFileTool(repo, on_write=lambda: notified.append(True))
    tool.run(path="src/calc.py", content="def add(a, b)\n")
    tool.run(path="tests/test_calc.py", content="x = 1\n")
    assert notified == []
