"""Can a skill's declared packages actually be installed alongside biopb?

The half of the contract layer that runs in CI. `test_contracts.py` asks whether
an API still looks the way a body says; this asks the prior question — whether a
user following the skill can get that package at all, in *this* environment.

The failure mode is not a hard conflict. It is a resolver that succeeds by
moving something else:

    uv pip install --dry-run basicpy
      + basicpy==2.0.0
      - numpy==2.3.5      + numpy==1.26.4
      - pandas==3.0.3     + pandas==2.3.3
      - scipy==1.18.0     + scipy==1.12.0

Nothing errors. The user now runs biopb on a numpy two years older than the one
the tensor stack was migrated to, and the first thing to break will look
unrelated to the skill that caused it.

This resolves against **the installed environment**, not against PyPI metadata
for the last release — which is the reason it has to live in this repo. Asking
PyPI answers for whatever biopb-mcp was published as; asking the venv answers
for the workspace under test. (Resolving `biopb-mcp[mcp]` from PyPI yields
`napari==0.8.0` while the source pins `napari[all]==0.7.0` exactly.)

Marked `satisfiability` and deselected by default: each token costs a real
resolver run over a torch-sized dependency graph. CI runs the marker as its own
step.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from ._validate import validate
from .conftest import SKILLS_DIR

pytestmark = pytest.mark.satisfiability

# Skipped: these are the workspace itself, and a `pkg:biopb-mcp>=X` floor is a
# statement about this repo's own release history, not about whether a third
# party can be installed. Resolving it here would reach for the *published*
# biopb-mcp and report the workspace being "downgraded" to it. What such a floor
# should mean now that skills and runtime ship together is an open question in
# docs/skill-testing.md §11.
_WORKSPACE = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}


def _pkg_requirements() -> list[str]:
    """Every `pkg:` token across the shipped skills, as pip requirement strings."""
    entries, _ = validate(SKILLS_DIR)
    out = []
    for e in entries:
        for token in e.requires:
            if not token.startswith("pkg:"):
                continue
            spec = token.split(":", 1)[1]
            name = spec.split(">=")[0].split("==")[0].strip()
            if name.lower() in _WORKSPACE:
                continue
            out.append(spec)
    return sorted(set(out))


def _plan(requirement: str) -> tuple[dict, dict]:
    """(removals, additions) that installing *requirement* here would make.

    ``uv pip install --dry-run`` prints the plan as ``- name==version`` /
    ``+ name==version`` lines against the current environment.
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


# Known-bad, recorded rather than tolerated. `strict=True`, so fixing one turns
# this into a failure that says "delete the entry" -- an xfail that has silently
# started passing is how a gate rots.
#
# Neither is fixable from this repo: both want an older pin than the biopb stack
# carries, so the resolutions are a skill-authoring decision (bound the token,
# or say in the body that the package wants its own environment).
_KNOWN_DOWNGRADES = {
    "basicpy": (
        "basicpy 2.0.0 pins scipy<1.13, which drags numpy back with it "
        "(2.3.5 -> 1.26.4, plus pandas and scipy). It does not pin numpy "
        "directly. The tensor stack moved to numpy 2.x in the bioio migration, "
        "so this install quietly reverts it -- under a live kernel that already "
        "imported numpy 2. There is no older basicpy to fall back to: 1.1.0 is "
        "jax-backed and takes no scipy pin, but its source uses pydantic v1 "
        "@root_validator/class Config while declaring pydantic>=1.9.1, so a "
        "resolver leaves pydantic 2.x in place and `import basicpy` fails at "
        "class definition. Take the skill's degraded path instead "
        "(FLATFIELD_METHOD = 'smoothed-median')."
    ),
    "m2stitch": (
        "m2stitch takes pandas 3.0.3 -> 2.3.3. Milder than basicpy -- pandas is "
        "a direct biopb-mcp dependency rather than the array layer everything "
        "else is compiled against -- but still an unrequested change to the "
        "running environment. Degraded path: PLACEMENT = 'nominal-grid'."
    ),
}


def _params():
    for requirement in _pkg_requirements():
        reason = _KNOWN_DOWNGRADES.get(requirement)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(requirement, marks=marks, id=requirement)


REQUIREMENTS = _pkg_requirements()


@pytest.fixture(scope="session", autouse=True)
def _needs_uv():
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")


def test_there_is_something_to_check():
    # A silent empty parametrize would make this file look green while checking
    # nothing -- the shape of failure this whole layer exists to prevent.
    assert REQUIREMENTS, "no pkg: tokens found in the shipped skills"


def test_every_known_downgrade_is_still_declared():
    """The xfail list must not outlive the tokens it describes -- an entry for a
    package no skill declares any more is a note nobody will read."""
    stale = set(_KNOWN_DOWNGRADES) - set(REQUIREMENTS)
    assert not stale, f"_KNOWN_DOWNGRADES mentions unshipped packages: {sorted(stale)}"


@pytest.mark.parametrize("requirement", list(_params()))
def test_installing_a_skills_package_does_not_downgrade_the_environment(requirement):
    removals, additions = _plan(requirement)

    from packaging.version import InvalidVersion, Version

    downgrades = []
    for name, old in removals.items():
        new = additions.get(name)
        if new is None:
            downgrades.append(f"{name} {old} -> removed")
            continue
        try:
            if Version(new) < Version(old):
                downgrades.append(f"{name} {old} -> {new}")
        except InvalidVersion:  # pragma: no cover - non-PEP440 local versions
            continue

    assert not downgrades, (
        f"installing {requirement!r} would move packages this environment "
        f"already has:\n  " + "\n  ".join(downgrades) + "\n"
        "The install succeeds, so a user following the skill gets this silently. "
        "Either bound the skill's requires: token to a version that co-resolves, "
        "or say in the body that the package needs its own environment."
    )
