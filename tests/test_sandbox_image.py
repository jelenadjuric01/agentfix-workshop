from __future__ import annotations

import re
import tomllib

import pytest

from agentfix.config import REPO_ROOT

# ---------------------------------------------------------------------------
# The agent's only oracle is `run_tests`, so the pytest that runs it must be the
# same one in both execution backends. The subprocess backend inherits the host's
# resolved environment (uv.lock); the Docker backend gets whatever the image was
# built with. Nothing at runtime compares the two, and the container tests skip
# until the image is built — so a drifted pin would silently change the oracle.
# These tests are the comparison. They need no daemon and no image.
# ---------------------------------------------------------------------------

DOCKERFILE = REPO_ROOT / "Dockerfile.sandbox"
LOCKFILE = REPO_ROOT / "uv.lock"

ARG_PATTERN = re.compile(r"^ARG\s+PYTEST_VERSION=(?P<version>\S+)\s*$", re.MULTILINE)


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _image_pytest_version() -> str:
    match = ARG_PATTERN.search(_dockerfile_text())
    assert match is not None, "Dockerfile.sandbox must declare `ARG PYTEST_VERSION=<version>`"
    return match.group("version")


def _locked_pytest_version() -> str:
    packages = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))["package"]
    versions = [package["version"] for package in packages if package["name"] == "pytest"]
    assert len(versions) == 1, f"expected exactly one locked pytest, got {versions}"
    return versions[0]


def test_image_pytest_matches_the_lockfile():
    """Bump `ARG PYTEST_VERSION` whenever `uv lock` moves pytest, or the two backends diverge."""
    assert _image_pytest_version() == _locked_pytest_version(), (
        f"Dockerfile.sandbox pins pytest {_image_pytest_version()} but uv.lock resolves "
        f"{_locked_pytest_version()} — the Docker backend would verify fixes with a different "
        "pytest than the subprocess backend. Update the ARG default."
    )


def test_the_pin_is_exact():
    """A range would let two builds of the same commit disagree with each other."""
    assert re.fullmatch(
        r"\d+\.\d+(\.\d+)?", _image_pytest_version()
    ), f"PYTEST_VERSION={_image_pytest_version()!r} is not an exact version"


def test_the_arg_is_actually_used_by_the_install():
    """An ARG nobody interpolates is a comment that tests would happily keep green."""
    assert "pytest==${PYTEST_VERSION}" in _dockerfile_text()


@pytest.mark.parametrize("locked", ["9.1.1", "8.3.2"])
def test_the_comparison_can_fail(locked, monkeypatch):
    """Guard the guard: a typo'd regex or lookup must not make this suite vacuously pass."""
    monkeypatch.setattr(f"{__name__}._locked_pytest_version", lambda: locked)
    if locked == _image_pytest_version():
        test_image_pytest_matches_the_lockfile()
    else:
        with pytest.raises(AssertionError):
            test_image_pytest_matches_the_lockfile()
