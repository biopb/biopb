"""Is a declared package installable *here*, on every platform biopb supports?

The other half of the split #680 asked for. `test_satisfiability.py` next door
asks whether installing a package would **damage** the environment; this asks
whether it can be installed at all, and the two deserve different answers:

    damage        a live kernel, after the fact   silent -- an ABI mismatch on
                                                  the next C-extension import
    unavailable   install time, before anything   loud, and the user sees it

Damage is fatal everywhere and cannot be worked around from a skill body, so it
is a gate. Unavailability is not that: `checklist:` informs the agent and gates
nothing, so a missing wheel is a gap the agent names to the user and works
around -- with the body's fallback where there is one, and an improvised one
otherwise. This layer therefore **reports** the platform holes rather than
rejecting them, which is the point of running the whole grid: an author who can
see "3 of 9, all the 3.12 cells" can decide whether the skill needs a fallback
spelled out, and nobody has to find out from a user.

One floor is still a gate. A package that installs on **no** supported cell is
not a platform gap, it is a declaration nobody can ever satisfy -- the agent
would improvise its way around the token every single time, and the catalog
would be advertising a path that does not exist.

**Why `uv pip compile` and not another `--dry-run`.** `--dry-run` answers for
the interpreter and platform it runs on, so the question needs one CI cell per
answer -- and the matrix is sparse (3.10 and 3.11 on Linux only), so 4 of 9
combinations are unscreened. Worse, each cell runs pytest independently, so
"failed on 1 of 5" is not knowable anywhere, which is what made all-or-nothing
the only rule the old shape could express. `compile` resolves for an interpreter
and platform that are not present, so all nine run from one job, in one process,
where a per-cell verdict can actually be rendered. About a second a cell.

**`--only-binary`, scoped to the declared package.** Refusing an sdist is the
whole detection mechanism: `uv pip install --dry-run psfmodels` succeeds on 3.12
by resolving to an sdist and never attempting the build, so the gate reported
success for an install that needs a C++ toolchain -- MSVC Build Tools on Windows
-- that a stock machine does not have. The strictness is deliberate: it refuses
pure-Python sdists too, and that is the right proxy, since a pure-Python project
essentially always ships a `py3-none-any` wheel. It is scoped to the declared
package rather than global because the workspace itself is a local source tree
and must still be built to co-resolve; a transitive dependency that ships only
an sdist is therefore not screened here.

Marked `availability` and deselected by default -- it is a network resolve per
cell. CI runs it once, in a job of its own, because one job is the point.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from ._validate import validate
from .conftest import SKILLS_DIR, write_skill

# What a user installs, resolved from this checkout rather than from PyPI: the
# published biopb-mcp yields napari 0.8.0 where the source pins napari[all]
# 0.7.0, so PyPI would answer for the last release instead of this branch.
WORKSPACE = Path(__file__).resolve().parents[5] / "biopb-mcp"

# What `install.sh` accepts (MIN_MINOR/MAX_MINOR), crossed with the three
# platforms the catalog ships to. Nine cells, not the CI matrix's five: there is
# no reason to leave macOS-3.10 unscreened when the answer costs a second.
PYTHONS = ("3.10", "3.11", "3.12")
PLATFORMS = ("linux", "macos", "windows")

_WORKSPACE_DISTS = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}


def _declared(directory: Path = SKILLS_DIR) -> list[tuple[str, str]]:
    """`(skill_id, requirement)` for every third-party `pkg:` token."""
    entries, _ = validate(directory)
    out = []
    for e in entries:
        for token in e.checklist:
            if not token.startswith("pkg:"):
                continue
            spec = token.split(":", 1)[1]
            if Requirement(spec).name.lower() in _WORKSPACE_DISTS:
                continue
            out.append((e.id, spec))
    return sorted(set(out))


def _resolves(requirement: str, python: str, platform: str) -> str | None:
    """None if *requirement* co-resolves with the workspace on that cell.

    Otherwise the resolver's own explanation, which already distinguishes "no
    usable wheels" from a version conflict -- both are reasons a user cannot
    have this package here, and the message is what makes the difference
    readable in CI.
    """
    name = Requirement(requirement).name
    with tempfile.TemporaryDirectory() as tmp:
        reqs = Path(tmp) / "requirements.in"
        reqs.write_text(f"{WORKSPACE}[mcp]\n{requirement}\n")
        proc = subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--only-binary",
                name,
                "--python-version",
                python,
                "--python-platform",
                platform,
                "--quiet",
                str(reqs),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    return None if proc.returncode == 0 else proc.stderr.strip()


DECLARED = _declared()


def test_the_extractor_finds_third_party_tokens(skills_dir):
    """The grid below is parametrized over what this returns, so it must not go
    quietly empty."""
    write_skill(
        skills_dir,
        "needs-things",
        frontmatter=(
            "description: A sentence.\ntitle: T\nversion: 1.0.0\n"
            "checklist: [viewer, pkg:biopb-mcp>=0.13.0, pkg:some-package>=2.0]\n"
        ),
    )
    assert _declared(skills_dir) == [("needs-things", "some-package>=2.0")]


def verdict(skill_id, requirement, unavailable) -> str | None:
    """The rejection message for a package missing on *unavailable* cells, or None.

    Split out from the grid so the rule is testable without a resolver: what a
    coverage hole earns is the whole substance of #680, and it should not be
    checkable only by shipping a bad skill.
    """
    if len(unavailable) < len(PYTHONS) * len(PLATFORMS):
        return None  # a hole is reported by the caller; the agent routes around it
    return (
        f"{skill_id} declares pkg:{requirement}, which installs on none of the "
        f"{len(PYTHONS) * len(PLATFORMS)} supported interpreter/platform cells.\n\n"
        "That is not a platform gap the agent can work around -- it is a token no "
        "session will ever satisfy, so every run improvises past it while the "
        "catalog advertises a path nobody has. Drop the dependency, or move it "
        "behind the algorithm plane as an ops:<kind> server."
    )


class TestVerdict:
    """The rule, exercised without touching the network."""

    ALL = {(p, plat): "no wheels" for p in PYTHONS for plat in PLATFORMS}
    SOME = {("3.12", plat): "no wheels" for plat in PLATFORMS}

    def test_available_everywhere_passes(self):
        assert verdict("s", "pkg>=1", {}) is None

    def test_a_platform_hole_is_reported_not_rejected(self):
        """psfmodels has no cp312 wheel, and 3.12 is what macOS and Windows users
        get by default -- but `checklist:` informs rather than gates, so the skill
        still ships and those sessions route around the token."""
        assert verdict("s", "psfmodels", self.SOME) is None

    def test_but_installing_nowhere_is_a_dead_declaration(self):
        msg = verdict("s", "nowhere", self.ALL)
        assert msg and "installs on none of the 9" in msg


@pytest.mark.availability
@pytest.mark.parametrize(
    ("skill_id", "requirement"),
    DECLARED,
    ids=[f"{s}:{r}" for s, r in DECLARED],
)
def test_a_declared_package_can_be_installed_where_we_ship(skill_id, requirement):
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    unavailable = {
        (python, platform): reason
        for python in PYTHONS
        for platform in PLATFORMS
        if (reason := _resolves(requirement, python, platform)) is not None
    }
    if unavailable:
        # The report this layer exists for. Printed even when it passes: an
        # author who can see which cells are missing can decide whether the body
        # needs a fallback spelled out, and the resolver's own message is what
        # tells a missing wheel from a version conflict.
        cells = ", ".join(f"py{p} {plat}" for p, plat in sorted(unavailable))
        print(f"\n{skill_id} · {requirement}: unavailable on {len(unavailable)}/9")
        print(f"  {cells}\n{next(iter(unavailable.values()))}")

    if message := verdict(skill_id, requirement, unavailable):
        pytest.fail(message)
