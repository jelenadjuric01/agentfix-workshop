import pytest

from agentfix.tools.fs import (
    ListFilesTool,
    PathEscapeError,
    ReadFileTool,
    WriteFileTool,
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
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    return tmp_path


def test_resolve_in_root_accepts_paths_inside(repo):
    assert resolve_in_root(repo, "src/calc.py") == repo / "src" / "calc.py"


@pytest.mark.parametrize("bad", ["../outside.py", "/etc/passwd", "src/../../escape.py"])
def test_resolve_in_root_rejects_escapes(repo, bad):
    with pytest.raises(PathEscapeError):
        resolve_in_root(repo, bad)


def test_list_files_lists_sources_and_hides_pycache(repo):
    result = ListFilesTool(repo).run()
    assert result.ok is True
    assert "src/calc.py" in result.content
    assert "tests/test_calc.py" in result.content
    assert "__pycache__" not in result.content


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
