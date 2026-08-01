"""Do the skills actually reach the wheel?

Every other test in this suite reads the checkout, so all of them pass whether
or not the skills ship. If they stop shipping, imports still work, this suite is
still green, and `find_skills` quietly returns only whatever is in the user's own
directory. The runtime deliberately cannot catch that -- it is on the agent's
path and must degrade rather than raise (`_skills._warn_empty_once` leaves a
breadcrumb, nothing more) -- so it is caught here, by building the thing that
ships and looking inside it.

**Two independent mechanisms** each put the files in the wheel, and either alone
is sufficient: `include-package-data = true` fed by setuptools_scm's git file
finder (every tracked file under the package), and the explicit
`[tool.setuptools.package-data]` glob. Breaking one is harmless -- verified, both
ways round -- which is why this asserts the *outcome* rather than either
mechanism. The glob is therefore redundant today and worth keeping anyway: it is
what survives if include-package-data is ever turned off.

`build/lib` is cleared first, and that is not housekeeping. setuptools stages
into it and does not prune, so a tree left by an earlier build re-supplies files
the current config would drop -- with both mechanisms broken and a stale
`build/lib` present, this test passes. CI is clean so it would have looked fine
there; a developer checkout, which has a stale tree after any local build, is
exactly where a false green would land.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from .conftest import SKILLS_DIR

# biopb-mcp/ -- the package root, four levels up from _tests/skills/.
PACKAGE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = PACKAGE_ROOT.parent


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> zipfile.ZipFile:
    """Build biopb-mcp and hand back the wheel, read-only."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    if not (REPO_ROOT / ".git").exists():
        # setuptools_scm derives the version from git describe; a source export
        # cannot build, and that is not a packaging regression.
        pytest.skip("not a git checkout")

    # See the module docstring: a stale staging tree silently re-supplies files
    # the current config would drop. Both are gitignored build artifacts that any
    # build regenerates, so removing them costs nothing but correctness.
    shutil.rmtree(PACKAGE_ROOT / "build", ignore_errors=True)
    for egg in (PACKAGE_ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)

    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        ["uv", "build", "--package", "biopb-mcp", "--wheel", "-o", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stderr}"
    built = list(out.glob("*.whl"))
    assert len(built) == 1, f"expected one wheel, got {built}"
    return zipfile.ZipFile(built[0])


def _shipped_md(wheel: zipfile.ZipFile) -> set[str]:
    prefix = "biopb_mcp/mcp/_skills_data/"
    return {
        n[len(prefix) :]
        for n in wheel.namelist()
        if n.startswith(prefix) and n.endswith(".md")
    }


def test_every_skill_in_the_tree_is_in_the_wheel(wheel):
    on_disk = {p.name for p in SKILLS_DIR.glob("*.md")}
    assert on_disk, f"no skills at {SKILLS_DIR}"
    assert _shipped_md(wheel) == on_disk


def test_a_shipped_skill_is_not_empty(wheel):
    # A glob can match the names while something upstream truncates the content.
    for name in sorted(_shipped_md(wheel)):
        raw = wheel.read(f"biopb_mcp/mcp/_skills_data/{name}")
        assert raw.strip(), f"{name} ships empty"


def test_the_test_suite_does_not_ship(wheel):
    # The other half of the same pair of globs (#666). Asserted here because a
    # change to one is usually a change to the other.
    assert not [n for n in wheel.namelist() if "_tests" in n.split("/")]
