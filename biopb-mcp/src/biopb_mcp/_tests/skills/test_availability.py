"""Is a declared package installable *here*, on every platform biopb supports?

The other half of the split #680 asked for. `test_satisfiability.py` next door
asks whether installing a package would **damage** the environment; this asks
whether it can be installed at all, and the two deserve different verdicts:

    damage        a live kernel, after the fact   silent -- an ABI mismatch on
                                                  the next C-extension import
    unavailable   install time, before anything   loud, and the user sees it

Damage is fatal everywhere and cannot be worked around from a skill body.
Unavailability is not a damage failure at all: it is an ordinary gap, and the
skill format already answers it -- step 1 resolves the requirement and a missing
`suggests:` package routes to the degraded path the body names. So a package
with no wheel on some cell rejects a **required** token and is merely recorded
for a **suggested** one.

Recorded, not ignored. A suggested package that resolves *nowhere* still fails:
the degraded path would be the only path anyone ever ran, and the declaration
would be advertising a preferred path that does not exist.

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


def _declared(directory: Path = SKILLS_DIR) -> list[tuple[str, str, bool]]:
    """`(skill_id, requirement, optional)` for every third-party `pkg:` token."""
    entries, _ = validate(directory)
    out = []
    for e in entries:
        for token, optional in [(t, False) for t in e.requires] + [
            (t, True) for t in e.suggests
        ]:
            if not token.startswith("pkg:"):
                continue
            spec = token.split(":", 1)[1]
            if Requirement(spec).name.lower() in _WORKSPACE_DISTS:
                continue
            out.append((e.id, spec, optional))
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


def test_the_extractor_separates_required_from_suggested(skills_dir):
    """The grid below is parametrized over what this returns, so it must not go
    quietly empty or quietly collapse the two keys into one verdict."""
    write_skill(
        skills_dir,
        "needs-things",
        frontmatter=(
            "description: A sentence.\ntitle: T\nversion: 1.0.0\n"
            "requires: [viewer, pkg:biopb-mcp>=0.13.0, pkg:hard-dep>=2.0]\n"
            "suggests: [pkg:soft-dep~=1.2]\n"
        ),
    )
    assert _declared(skills_dir) == [
        ("needs-things", "hard-dep>=2.0", False),
        ("needs-things", "soft-dep~=1.2", True),
    ]


def verdict(skill_id, requirement, optional, unavailable) -> str | None:
    """The rejection message for a package missing on *unavailable* cells, or None.

    Split out from the grid so the rule itself is testable without a resolver:
    which verdict a hole in the coverage earns is the whole substance of #680,
    and it should not be checkable only by shipping a bad skill.
    """
    if not unavailable:
        return None
    cells = ", ".join(f"py{p} {plat}" for p, plat in sorted(unavailable))

    if not optional:
        return (
            f"{skill_id} requires pkg:{requirement}, which cannot be installed "
            f"on {len(unavailable)} of {len(PYTHONS) * len(PLATFORMS)} supported "
            f"cells: {cells}\n\n"
            "A required package has to be there for the skill to run at all, so "
            "this ships a workflow those users cannot execute -- and find_skills "
            "still retrieves it for them.\n"
            "Either the skill grows a real degraded path and the token moves to "
            "suggests:, or it drops the dependency."
        )

    if len(unavailable) == len(PYTHONS) * len(PLATFORMS):
        return (
            f"{skill_id} suggests pkg:{requirement}, which installs nowhere.\n\n"
            "Optional is not the same as imaginary: nobody would ever run the "
            "preferred path, so the body proves a fallback while advertising "
            "something else. Drop the token and let the degraded path be the "
            "only path."
        )
    return None  # an optional package with a platform hole is what suggests: is for


class TestVerdict:
    """The table #680 argued for, exercised without touching the network."""

    ALL = {(p, plat): "no wheels" for p in PYTHONS for plat in PLATFORMS}
    SOME = {("3.12", plat): "no wheels" for plat in PLATFORMS}

    def test_available_everywhere_passes_either_way(self):
        assert verdict("s", "pkg>=1", False, {}) is None
        assert verdict("s", "pkg>=1", True, {}) is None

    def test_a_required_package_may_not_have_a_hole(self):
        msg = verdict("s", "psfmodels", False, self.SOME)
        assert msg and "3 of 9" in msg and "py3.12 windows" in msg
        # And it names the way out, since "reject" was the old answer to this.
        assert "moves to\nsuggests:" in msg or "suggests:" in msg

    def test_an_optional_package_may(self):
        """The relaxation itself: psfmodels has no cp312 wheel, and 3.12 is what
        macOS and Windows users get by default -- yet the skill still works for
        them through its degraded path."""
        assert verdict("s", "psfmodels", True, self.SOME) is None

    def test_but_not_a_hole_everywhere(self):
        msg = verdict("s", "nowhere", True, self.ALL)
        assert msg and "installs nowhere" in msg


@pytest.mark.availability
@pytest.mark.parametrize(
    ("skill_id", "requirement", "optional"),
    DECLARED,
    ids=[f"{s}:{r}" for s, r, _ in DECLARED],
)
def test_a_declared_package_is_available_where_it_has_to_be(
    skill_id, requirement, optional
):
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    unavailable = {
        (python, platform): reason
        for python in PYTHONS
        for platform in PLATFORMS
        if (reason := _resolves(requirement, python, platform)) is not None
    }
    if unavailable:
        # Printed whichever way the verdict goes: a skill shipping with a hole in
        # its platform coverage is a thing the author should have to see, and the
        # resolver's own message is what tells a missing wheel from a conflict.
        cells = ", ".join(f"py{p} {plat}" for p, plat in sorted(unavailable))
        print(f"\n{skill_id} · {requirement}: unavailable on {len(unavailable)}/9")
        print(f"  {cells}\n{next(iter(unavailable.values()))}")

    if message := verdict(skill_id, requirement, optional, unavailable):
        pytest.fail(message)
