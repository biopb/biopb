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

Whether these gate is a separate decision, taken in #673: putting `pystackreg`
in the `testing` group is what makes them run in CI, and reverses §10's plan of
workstation-only. Until then the module skips wherever the package is absent,
which includes CI -- the assertions are here and correct, but they are not yet
watching anything.
"""

from __future__ import annotations

from importlib.metadata import version as dist_version

import numpy as np
import pytest
from packaging.version import Version
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


def _third_party_requirements(entry) -> dict[str, str | None]:
    """`{name: floor or None}` for the `pkg:` tokens that name a third party."""
    out: dict[str, str | None] = {}
    for token in entry.requires:
        if not token.startswith("pkg:"):
            continue
        spec = token.split(":", 1)[1]
        name, _, floor = spec.partition(">=")
        if name not in _WORKSPACE:
            out[name] = floor or None
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


def test_the_installed_version_satisfies_every_declared_floor(entries):
    """Ties the assertions below to the frontmatter. Without this the module
    asserts against whatever happened to resolve, while the declaration promises
    something else."""
    for e in entries:
        for name, floor in _third_party_requirements(e).items():
            if floor is None:
                continue
            installed = dist_version(name)
            assert Version(installed) >= Version(floor), (
                f"{e.id} declares {name}>={floor}, installed is {installed}"
            )


# --- pystackreg, for drift-correction -------------------------------------

pystackreg = pytest.importorskip("pystackreg")


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


def test_the_modes_the_body_names_exist():
    """`sr = StackReg(getattr(StackReg, MODE))` with MODE in {TRANSLATION,
    RIGID_BODY} -- step 3, and the `MODE` row of the parameter table."""
    from pystackreg import StackReg

    assert hasattr(StackReg, "TRANSLATION")
    assert hasattr(StackReg, "RIGID_BODY")


def test_register_stack_still_defaults_to_previous():
    """Step 3 tells the agent to pass `reference="previous"` and explains why.
    It is also the library default, which means an agent that omits the argument
    is still safe *today*. If upstream ever flips it to "first", that stops being
    true silently -- and the body reads as advice rather than a requirement, so
    nothing else would catch it.
    """
    import inspect

    from pystackreg import StackReg

    sig = inspect.signature(StackReg.register_stack)
    assert sig.parameters["reference"].default == "previous"


def test_the_translation_lives_where_step_4_reads_it(drifting_movie):
    """The load-bearing indexing claim: `dy = m[1, 2]`, `dx = m[0, 2]`.

    Step 4 stops the workflow on these numbers, so a transposed read would not
    raise -- it would report a plausible trajectory for the wrong axis and pass
    its own sanity check.
    """
    from pystackreg import StackReg

    movie, truth = drifting_movie
    tmats = StackReg(StackReg.TRANSLATION).register_stack(movie, reference="previous")

    assert tmats.shape == (len(movie), 3, 3)
    dy = np.array([m[1, 2] for m in tmats])
    dx = np.array([m[0, 2] for m in tmats])
    assert np.abs(dy - truth[:, 0]).max() < 0.05
    assert np.abs(dx - truth[:, 1]).max() < 0.05


def test_transform_stack_takes_tmats_and_actually_stabilises(drifting_movie):
    """Step 5's `sr.transform_stack(MOVIE, tmats=tmats)` -- the keyword name, and
    that reusing a foreign `tmats` array is a supported call rather than an
    accident of the signature."""
    from pystackreg import StackReg

    movie, _ = drifting_movie
    sr = StackReg(StackReg.TRANSLATION)
    tmats = sr.register_stack(movie, reference="previous")
    corrected = sr.transform_stack(movie, tmats=tmats)

    assert corrected.shape == movie.shape
    inner = (slice(20, -20), slice(20, -20))
    before = np.abs(movie[-1][inner] - movie[0][inner]).mean()
    after = np.abs(corrected[-1][inner] - corrected[0][inner]).mean()
    assert after < before / 10


def test_transform_stack_returns_float64(drifting_movie):
    """Step 5 warns that interpolation resamples intensities. It also promotes
    the dtype, which doubles the footprint of a float32 movie -- a surprise worth
    catching here if it ever changes."""
    from pystackreg import StackReg

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
