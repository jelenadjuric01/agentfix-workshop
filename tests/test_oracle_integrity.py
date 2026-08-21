"""The agent must not be able to make the tests pass without fixing the bug.

`run_tests` is the only oracle, so any write that changes what the suite *does* — rather than
what the code does — is a privilege escalation. Every case here was a reproduced escape before
the checks that stop it existed, so these are regression tests, not hypotheticals.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agentfix.tools.fs import WriteFileTool, _relative_files, is_test_path


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cart.py").write_text("original\n", encoding="utf-8")
    (tmp_path / "cart.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tool(project):
    return WriteFileTool(project, allowed=frozenset(_relative_files(project)))


@pytest.mark.parametrize(
    "path", ["tests/test_cart.py", "Tests/TEST_CART.PY", "TESTS/Test_Cart.py"]
)
def test_is_test_path_is_case_insensitive(project, path):
    """macOS is case-insensitive: "Tests/TEST_CART.PY" is the same inode as the real suite."""
    assert is_test_path(project, project / path)


def test_a_case_variant_write_cannot_reach_the_test_file(project, tool):
    result = tool.run("Tests/TEST_CART.PY", "def test_ok():\n    assert True\n")
    assert not result.ok
    assert (project / "tests" / "test_cart.py").read_text() == "original\n"


@pytest.mark.parametrize(
    "path", ["pytest.py", "conftest.py", ".local/lib/python3.12/site-packages/evil.pth"]
)
def test_a_new_file_that_could_change_the_oracle_is_refused(project, tool, path):
    assert not tool.run(path, "import sys\n").ok
    assert not (project / path).exists()


def test_shadowing_the_test_runner_really_would_have_forged_a_pass(tmp_path):
    """Proves the allow-list is load-bearing rather than guarding nothing."""
    (tmp_path / "test_x.py").write_text("def test_red():\n    assert 1 == 2\n", encoding="utf-8")
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    assert subprocess.run(argv, cwd=tmp_path, capture_output=True).returncode != 0

    (tmp_path / "pytest.py").write_text("import sys\n\nsys.exit(0)\n", encoding="utf-8")
    forged = subprocess.run(argv, cwd=tmp_path, capture_output=True)
    assert forged.returncode == 0, "shadowing forges a pass — hence the allow-list"


def test_an_existing_file_is_still_writable(project, tool):
    """The fix must not stop the agent doing its actual job."""
    assert tool.run("cart.py", "x = 2\n").ok
    assert (project / "cart.py").read_text() == "x = 2\n"
