"""A skill body is an un-versioned assertion about someone else's API.

`docs/skill-testing.md` §3. The satisfiability gate next door answers "may this
package be installed at all"; this answers the question after it — "does the
surface the body quotes still look like that". Both have to be true before a
`pkg:` token is honest.

The layer exists because of what has actually broken. `m2stitch.stitch_images`
defaults `row_col_transpose=True` and swaps rows and cols for you, so a body that
passed them positionally produced exactly the diagonal staircase its own failure
table described. Nothing about that is findable by testing the agent; it is a
default in a library nobody in this repo owns.

It went unmanned once already: the original module was written entirely for
`flatfield-and-stitch-tiles` and was deleted with it in #667, leaving §3 with the
note "returns with the first skill whose package passes §3a". That is
`drift-correction`. To stop the same thing happening silently a second time,
`test_every_declared_package_is_covered_here` fails when a shipped skill declares
a third-party package this module says nothing about -- the same shape as the
phrasing-table check in test_retrieval.py.

**When these run, and why not more often.** `.github/workflows/skill-contracts.yaml`,
on pull requests that touch a skill file or this module -- not on every PR and
not on a schedule.

The trigger follows the risk. A skill declares a *bounded* package range
(`~=`, so a floor plus an upper bound at the next minor), which means the API
these assertions pin cannot move underneath a shipped skill: what a user
resolves is inside the range the assertions were proved against. Upstream
releasing a new minor is therefore not an event this layer needs to hear about,
which rules out a cron. What remains is our own editing -- a body rewritten to
pass `tmats` positionally, a new skill written against an API nobody ran -- and
that is change-triggered, so it belongs on the PR that does it.

A package going stale inside its own range is caught elsewhere, by the
satisfiability gate: it resolves every declared token on every matrix cell, so a
range that stops installing fails there rather than here.

Each package is installed into its **own ephemeral env** by that workflow, never
into the shared test env. Two skills' packages are then never resolved together,
so a future pair that cannot co-exist costs neither skill its contract test --
and no unrelated PR pays for a dependency it does not use.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as dist_version

import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from scipy import ndimage

from ._validate import validate
from .conftest import SKILLS_DIR

# The workspace's own distributions. A `pkg:biopb-mcp>=X` floor is a statement
# about this repo's release history, not about a third party's API, and there is
# no external surface to assert.
_WORKSPACE = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}

# Which skill's claims each package below is asserted on behalf of. Adding a
# package-tier skill without adding an entry fails the coverage test.
COVERED: dict[str, set[str]] = {
    "drift-correction": {"pystackreg"},
}


def _third_party_requirements(entry) -> dict[str, SpecifierSet]:
    """`{name: specifier}` for the `pkg:` tokens that name a third party.

    The token is already a PEP 508 requirement (`name`, `name>=X`, `name~=X`),
    so it is parsed rather than split on operators.
    """
    out: dict[str, SpecifierSet] = {}
    for token in entry.requires:
        if not token.startswith("pkg:"):
            continue
        req = Requirement(token.split(":", 1)[1])
        if req.name.lower() not in _WORKSPACE:
            out[req.name] = req.specifier
    return out


@pytest.fixture(scope="module")
def entries():
    ents, _ = validate(SKILLS_DIR)
    return ents


def test_every_declared_package_is_covered_here(entries):
    """The guard against this layer going unmanned again. A skill that declares
    a third-party package with nothing asserting its surface is a skill nobody
    checked against the library it drives."""
    missing = []
    for e in entries:
        declared = set(_third_party_requirements(e))
        if uncovered := declared - COVERED.get(e.id, set()):
            missing.append(f"{e.id}: {sorted(uncovered)}")
    assert not missing, (
        "shipped skills declare packages with no contract test:\n"
        + "\n".join(missing)
        + "\n\nAdd assertions and list the package in COVERED."
    )


def test_covered_is_not_stale(entries):
    """The other direction: a COVERED entry naming a skill or a package that no
    longer ships is a table to update, not a failure to debug."""
    live = {e.id: set(_third_party_requirements(e)) for e in entries}
    stale = [
        f"{skill_id}: {sorted(pkgs - live.get(skill_id, set()))}"
        for skill_id, pkgs in COVERED.items()
        if pkgs - live.get(skill_id, set())
    ]
    assert not stale, "COVERED names what no skill declares:\n" + "\n".join(stale)


def test_the_installed_version_is_inside_every_declared_range(entries):
    """Ties the assertions below to the frontmatter, in both directions.

    A floor alone would let this module prove the body against a version no user
    will ever get. The upper bound is what makes the proof transferable: the
    assertions hold for the whole declared range, and the range is what the agent
    resolves. So the version under test has to be *inside* it, not merely above
    the floor.
    """
    for e in entries:
        for name, spec in _third_party_requirements(e).items():
            if not spec:
                continue
            try:
                installed = dist_version(name)
            except PackageNotFoundError:
                # Each env carries one declared package (skill_contracts.py), so
                # the others are legitimately absent here rather than missing.
                continue
            assert spec.contains(installed), (
                f"{e.id} declares {name}{spec}, installed is {installed} -- "
                "these assertions would prove nothing about what a user resolves"
            )


# --- pystackreg, for drift-correction -------------------------------------
#
# Guarded per-package, not at module scope: skill_contracts.py runs this module
# once per declared package, so in another package's env pystackreg is
# legitimately absent and only its own assertions should skip. A module-level
# importorskip would take the whole file down with it, including the checks
# above that need no package at all.


@pytest.fixture(scope="module")
def StackReg():
    return pytest.importorskip("pystackreg").StackReg


@pytest.fixture(scope="module")
def drifting_movie():
    """A (T, Y, X) movie with a known per-frame shift -- the shape the skill's
    `MOVIE` parameter describes."""
    rng = np.random.default_rng(0)
    img = 100 + 3000 * ndimage.gaussian_filter(
        (rng.random((128, 128)) < 0.006).astype(np.float32), 3
    )
    truth = np.array([(1.3 * i, -0.8 * i) for i in range(12)])
    movie = np.array(
        [ndimage.shift(img, t, order=3, mode="nearest") for t in truth],
        np.float32,
    )
    return movie, truth


def test_the_modes_the_body_names_exist(StackReg):
    """`sr = StackReg(getattr(StackReg, MODE))` with MODE in {TRANSLATION,
    RIGID_BODY} -- step 3, and the `MODE` row of the parameter table."""
    assert hasattr(StackReg, "TRANSLATION")
    assert hasattr(StackReg, "RIGID_BODY")


def test_register_stack_still_defaults_to_previous(StackReg):
    """Step 3 tells the agent to pass `reference="previous"` and explains why.
    It is also the library default, which means an agent that omits the argument
    is still safe *today*. If upstream ever flips it to "first", that stops being
    true silently -- and the body reads as advice rather than a requirement, so
    nothing else would catch it.
    """
    import inspect

    sig = inspect.signature(StackReg.register_stack)
    assert sig.parameters["reference"].default == "previous"


def test_the_translation_lives_where_step_4_reads_it(StackReg, drifting_movie):
    """The load-bearing indexing claim: `dy = m[1, 2]`, `dx = m[0, 2]`.

    Step 4 stops the workflow on these numbers, so a transposed read would not
    raise -- it would report a plausible trajectory for the wrong axis and pass
    its own sanity check.
    """
    movie, truth = drifting_movie
    tmats = StackReg(StackReg.TRANSLATION).register_stack(movie, reference="previous")

    assert tmats.shape == (len(movie), 3, 3)
    dy = np.array([m[1, 2] for m in tmats])
    dx = np.array([m[0, 2] for m in tmats])
    assert np.abs(dy - truth[:, 0]).max() < 0.05
    assert np.abs(dx - truth[:, 1]).max() < 0.05


def test_transform_stack_takes_tmats_and_actually_stabilises(StackReg, drifting_movie):
    """Step 5's `sr.transform_stack(MOVIE, tmats=tmats)` -- the keyword name, and
    that reusing a foreign `tmats` array is a supported call rather than an
    accident of the signature."""
    movie, _ = drifting_movie
    sr = StackReg(StackReg.TRANSLATION)
    tmats = sr.register_stack(movie, reference="previous")
    corrected = sr.transform_stack(movie, tmats=tmats)

    assert corrected.shape == movie.shape
    inner = (slice(20, -20), slice(20, -20))
    before = np.abs(movie[-1][inner] - movie[0][inner]).mean()
    after = np.abs(corrected[-1][inner] - corrected[0][inner]).mean()
    assert after < before / 10


def test_transform_stack_returns_float64(StackReg, drifting_movie):
    """Step 5 warns that interpolation resamples intensities. It also promotes
    the dtype, which doubles the footprint of a float32 movie -- a surprise worth
    catching here if it ever changes."""
    movie, _ = drifting_movie
    sr = StackReg(StackReg.TRANSLATION)
    tmats = sr.register_stack(movie, reference="previous")
    assert movie.dtype == np.float32
    assert sr.transform_stack(movie, tmats=tmats).dtype == np.float64


# --- skimage, for the degraded path ---------------------------------------


def test_phase_cross_correlation_still_defaults_to_phase_normalization():
    """The first row of the skill's failure table exists because this default
    whitens frequency bins holding only numerical noise, so a smooth
    low-contrast movie recovers ~0 drift and reports success. The body's fix is
    `normalization=None`. Both halves are asserted here: if skimage ever changed
    the default the row would be wrong, and if the parameter went away the fix
    would be unrunnable.
    """
    import inspect

    from skimage.registration import phase_cross_correlation

    params = inspect.signature(phase_cross_correlation).parameters
    assert "normalization" in params
    assert params["normalization"].default == "phase"
