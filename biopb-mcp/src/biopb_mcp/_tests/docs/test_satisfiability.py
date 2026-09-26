"""A doc may not declare a package that damages the environment to install.

The rule, plainly: every package a shipped doc declares must install into
a biopb environment **without moving anything already there**. A doc that
fails this is not a doc with a caveat — it is a bug, and it should be rejected
in review rather than shipped with a workaround.

The failure mode is not a conflict a user would notice. It is a resolver that
succeeds by moving something else:

    uv pip install --dry-run basicpy
      + basicpy==2.0.0
      - numpy==2.3.5      + numpy==1.26.4
      - pandas==3.0.3     + pandas==2.3.3
      - scipy==1.18.0     + scipy==1.12.0

Nothing errors. And the damage is worse than a version change, because the
kernel is *live*: numpy 2 is already imported and dask, pyarrow and napari
extensions are compiled against it, so the next C-extension import hits an ABI
mismatch — and after the `restart_kernel` that suggests, the whole session comes
back up on the numpy the tensor stack was deliberately moved off.

Why this cannot be fixed in the doc body. The `kernel` doc offers the agent
three answers to a missing package: the user installs it, the agent installs
it after consent, or the doc's degraded path. The second is the harmful one,
and a warning in one doc's prose does not make it safe — the agent is following
a general guide, and the next doc would have to repeat the warning. Nor does
"install it in its own environment" help: the agent's only execution surface is
the kernel's interpreter, so a package in some other venv is not importable.
A package that needs its own environment belongs behind the algorithm plane, as
an `ops` server that is called rather than imported.

So this gate is unconditional. There is no allowlist and no xfail: those would
be a place to record that a known-bad doc ships anyway, which is the outcome
the gate exists to prevent.

**One question, not two.** Whether the package can be installed on this platform
*at all* is a different failure and lives in `test_availability.py` (#680): a
missing wheel is loud, arrives before anything runs, and the agent can route
around it. Nothing here bends that way -- a downgrade is silent, arrives after
the fact, and no fallback in a body saves the user from it.

Marked `satisfiability` and deselected by default — each token costs a real
resolver run. CI runs the marker as its own step, on every matrix cell.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from .conftest import DOCS_DIR, declared_packages, offered_files, write_doc_file

pytestmark = pytest.mark.satisfiability

# Skipped: these are the workspace itself. `packages:` is for third-party APIs a
# body quotes, so a workspace name there is a mistake rather than a requirement
# to resolve -- resolving it would reach for the *published* biopb-mcp and report
# the workspace being "downgraded" to it.
_WORKSPACE = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}


def _pkg_requirements(directory: Path = DOCS_DIR) -> list[str]:
    """Every third-party `packages:` requirement in *directory*."""
    out = []
    for path in offered_files(directory):
        for spec in declared_packages(path):
            # A PEP 508 requirement already: `name`, `name>=X`, `name~=X`. Parse
            # it rather than splitting on operators, so a new bound spelling
            # cannot silently make a workspace name look third-party.
            if Requirement(spec).name.lower() not in _WORKSPACE:
                out.append(spec)
    return sorted(set(out))


def _plan(requirement: str) -> tuple[dict, dict]:
    """(removals, additions) that installing *requirement* here would make.

    ``uv pip install --dry-run`` prints the plan as ``- name==version`` /
    ``+ name==version`` lines against the current environment. Nothing is
    downloaded or installed.
    """
    proc = subprocess.run(
        ["uv", "pip", "install", "--dry-run", requirement],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"{requirement!r} cannot be installed alongside this environment:\n"
        f"{proc.stderr.strip()}"
    )
    removals, additions = {}, {}
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if line[:2] not in ("- ", "+ ") or "==" not in line:
            continue
        name, _, version = line[2:].partition("==")
        (removals if line[0] == "-" else additions)[name.strip()] = version.strip()
    return removals, additions


REQUIREMENTS = _pkg_requirements()


@pytest.fixture(scope="session", autouse=True)
def _needs_uv():
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")


def test_the_extractor_finds_declared_packages(docs_dir):
    """The gate below is parametrized over what this returns, so prove the
    extractor works on a tree that declares one and it cannot go vacuously
    green."""
    write_doc_file(
        docs_dir,
        "needs-things",
        "packages: [biopb-mcp>=0.13.0, some-package>=2.0]\n",
    )
    assert _pkg_requirements(docs_dir) == ["some-package>=2.0"]


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_a_declared_package_installs_without_moving_anything(requirement):
    removals, additions = _plan(requirement)

    from packaging.version import InvalidVersion, Version

    moved = []
    for name, old in removals.items():
        new = additions.get(name)
        if new is None:
            moved.append(f"{name} {old} -> removed")
            continue
        try:
            if Version(new) < Version(old):
                moved.append(f"{name} {old} -> {new}")
        except InvalidVersion:  # pragma: no cover - non-PEP440 local versions
            continue

    assert not moved, (
        f"a doc declares a requirement on {requirement}, and installing it "
        f"would move packages this environment already has:\n  "
        + "\n  ".join(moved)
        + "\n\n"
        "The install succeeds, so the agent and the user both get this silently, "
        "under a live kernel that has already imported the old versions.\n"
        "This is not fixable with a warning in the doc body -- the kernel doc "
        "offers the agent an install-it-for-you path, and a package in a "
        "separate environment is not importable from the kernel at all.\n"
        "Either the doc drops the dependency (use its degraded path as the "
        "only path), or the package moves behind the algorithm plane as an "
        "`ops` server that is called rather than imported."
    )
