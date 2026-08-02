"""A skill may not declare a package that damages the environment to install.

The rule, plainly: every `pkg:` token a shipped skill declares must install into
a biopb environment **without moving anything already there**. A skill that
fails this is not a skill with a caveat — it is a bug, and it should be rejected
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

Why this cannot be fixed in the skill body. `guide://kernel` offers the agent
three answers to a missing `pkg:` token: the user installs it, the agent installs
it after consent, or the skill's degraded path. The second is the harmful one,
and a warning in one skill's prose does not make it safe — the agent is following
a general guide, and the next skill would have to repeat the warning. Nor does
"install it in its own environment" help: the agent's only execution surface is
the kernel's interpreter, so a package in some other venv is not importable.
A package that needs its own environment belongs behind the algorithm plane, as
an `ops:<kind>` server that is called rather than imported.

So this gate is unconditional. There is no allowlist and no xfail: those would
be a place to record that a known-bad skill ships anyway, which is the outcome
the gate exists to prevent.

**One question, not two.** Whether the package can be installed on this platform
*at all* is a different failure with a different verdict, and it lives in
`test_availability.py` (#680): a missing wheel is loud, arrives before anything
runs, and a `suggests:` package may legitimately lack one on some cell. Nothing
here bends that way -- a downgrade is silent, arrives after the fact, and no
degraded path in a body saves the user from it, so `requires:` and `suggests:`
are judged identically below.

Marked `satisfiability` and deselected by default — each token costs a real
resolver run. CI runs the marker as its own step, on every matrix cell.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from ._validate import validate
from .conftest import SKILLS_DIR, write_skill

pytestmark = pytest.mark.satisfiability

# Skipped: these are the workspace itself, and a `pkg:biopb-mcp>=X` floor is a
# statement about this repo's own release history, not about whether a third
# party can be installed. Resolving it here would reach for the *published*
# biopb-mcp and report the workspace being "downgraded" to it. What such a floor
# should mean now that skills and runtime ship together is an open question in
# biopb-mcp/docs/skill-testing.md §1.
_WORKSPACE = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}


def _pkg_requirements(directory: Path = SKILLS_DIR) -> list[str]:
    """Every third-party `pkg:` token in *directory*, as pip requirement strings."""
    entries, _ = validate(directory)
    out = []
    for e in entries:
        # Both keys. Optionality is about whether the *skill* still works
        # without the package; it says nothing about what installing it does to
        # the environment, and the agent installs a suggested package on
        # exactly the same path.
        for token in [*e.requires, *e.suggests]:
            if not token.startswith("pkg:"):
                continue
            spec = token.split(":", 1)[1]
            # A PEP 508 requirement already: `name`, `name>=X`, `name~=X`. Parse
            # it rather than splitting on operators, so a new bound spelling
            # cannot silently make a workspace token look third-party.
            if Requirement(spec).name.lower() in _WORKSPACE:
                continue
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


def test_the_extractor_finds_pkg_tokens(skills_dir):
    """No shipped skill declares a third-party package today, which would leave
    the gate below parametrized over nothing. That is the correct state, not a
    reason to stop checking the machinery -- so prove the extractor works on a
    tree that does declare one, and the gate cannot go vacuously green.

    The `suggests:` token is here for the same reason it is in the gate: an
    optional package is installed by the same agent into the same live kernel,
    so skipping it would leave the damage question unasked for exactly the
    packages users are most likely to add mid-session."""
    write_skill(
        skills_dir,
        "needs-things",
        frontmatter=(
            "description: A sentence.\ntitle: T\nversion: 1.0.0\n"
            "requires: [viewer, pkg:biopb-mcp>=0.13.0, pkg:some-package>=2.0]\n"
            "suggests: [pkg:optional-package~=1.2]\n"
        ),
    )
    assert _pkg_requirements(skills_dir) == [
        "optional-package~=1.2",
        "some-package>=2.0",
    ]


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_a_skills_package_installs_without_moving_anything(requirement):
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
        f"a skill declares pkg:{requirement}, and installing it would move "
        f"packages this environment already has:\n  " + "\n  ".join(moved) + "\n\n"
        "The install succeeds, so the agent and the user both get this silently, "
        "under a live kernel that has already imported the old versions.\n"
        "This is not fixable with a warning in the skill body -- guide://kernel "
        "offers the agent an install-it-for-you path, and a package in a "
        "separate environment is not importable from the kernel at all.\n"
        "Either the skill drops the dependency (use its degraded path as the "
        "only path), or the package moves behind the algorithm plane as an "
        "ops:<kind> server that is called rather than imported."
    )
