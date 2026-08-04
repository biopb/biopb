"""A skill body is an un-versioned assertion about someone else's API.

`biopb-mcp/docs/skill-testing.md` §4. The satisfiability gate next door answers "may this
package be installed at all"; this answers the two questions after it — "does it
import once installed" (#670: `stardist` resolves clean and imports nothing,
because its TensorFlow dependency hides under an extra) and "does the surface the
body quotes still look like that". All three have to be true before a `pkg:`
token is honest.

The layer exists because of what has actually broken. `m2stitch.stitch_images`
defaults `row_col_transpose=True` and swaps rows and cols for you, so a body that
passed them positionally produced exactly the diagonal staircase its own failure
table described. Nothing about that is findable by testing the agent; it is a
default in a library nobody in this repo owns.

It went unmanned once already: the original module was written entirely for
`flatfield-and-stitch-tiles` and was deleted with it in #667, leaving §4 with the
note "returns with the first skill whose package passes §4a". That is
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

import importlib
from importlib.metadata import PackageNotFoundError, version as dist_version

import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
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
    "track-objects": {"laptrack"},
}


def _third_party_requirements(entry) -> dict[str, SpecifierSet]:
    """`{name: specifier}` for the `pkg:` tokens that name a third party.

    The token is already a PEP 508 requirement (`name`, `name>=X`, `name~=X`),
    so it is parsed rather than split on operators.
    """
    out: dict[str, SpecifierSet] = {}
    for token in entry.checklist:
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


def test_every_installed_declared_package_actually_imports(entries):
    """Installing is not working, and the satisfiability gate cannot tell them apart.

    `uv pip install --dry-run stardist` resolves clean and moves nothing, so §3a
    passes it -- and `import stardist` still fails, because `csbdeep` declares
    TensorFlow only under a `[tf1]` extra. The gate asks "can this be installed
    without damage"; nothing asked "will this import". This does.

    Unlike the rest of the file this is not a claim about anyone's API, so it
    needs no hand-written assertion per package: it runs over whatever the
    catalog declares. Conditioned on the distribution being *installed* rather
    than on which env this is -- skill_contracts.py gives each package its own,
    so elsewhere the others are legitimately absent, and unimportable-when-
    present is fatal on every platform either way.
    """
    from importlib.metadata import packages_distributions

    modules = packages_distributions()
    for e in entries:
        for name in _third_party_requirements(e):
            try:
                dist_version(name)
            except PackageNotFoundError:
                continue  # not this package's env
            canonical = canonicalize_name(name)
            top_level = sorted(
                mod
                for mod, dists in modules.items()
                if any(canonicalize_name(d) == canonical for d in dists)
            )
            # A namespace package can leave the mapping empty; the import name
            # then differs from the distribution name only by the separator.
            for module in top_level or [name.replace("-", "_")]:
                try:
                    importlib.import_module(module)
                # Not just ImportError: a package can raise anything at import.
                except Exception as exc:
                    pytest.fail(
                        f"{e.id} declares {name}, which is installed "
                        f"({dist_version(name)}) and does not import:\n"
                        f"  import {module} -> {type(exc).__name__}: {exc}\n"
                        "A declared package the agent cannot import is a skill that "
                        "dead-ends at step 1 on every platform."
                    )


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


# --- laptrack, for track-objects ------------------------------------------
#
# Every assertion here is a sentence in the body. Three of them are about
# defaults rather than about arity, because that is where this library's
# surface is sharp: a wrong default here does not raise, it returns tracks.
#
# This skill ships only because biopb-mcp excludes scipy 1.15 for its own
# reasons (see its pyproject, and scipy#22501). Had it not, §4a would have
# rejected the token on 3.10 -- where 1.15.3 is the newest scipy -- since
# laptrack pins against exactly that series and the resolver's only answer is
# to move a live kernel's scipy backwards.


@pytest.fixture(scope="module")
def LapTrack():
    return pytest.importorskip("laptrack").LapTrack


@pytest.fixture(scope="module")
def two_frames():
    """Two frames, four objects, all moving 4 px in +x -- far enough apart that
    the correct assignment is unambiguous and any cutoff question is about the
    cutoff rather than about the linking."""
    import pandas as pd

    y = np.array([10.0, 60.0, 110.0, 160.0])
    return pd.DataFrame(
        {
            "frame": [0] * 4 + [1] * 4,
            "label": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": np.concatenate([y, y]),
            "x": np.concatenate([np.full(4, 20.0), np.full(4, 24.0)]),
        }
    )


def test_the_metric_is_squared_and_the_cutoffs_are_squared_with_it(LapTrack):
    """The claim step 4 is built around, and the first failure row.

    `cutoff` is compared against a *sqeuclidean* distance, so a body that
    passes a distance silently gets its square root. Both halves are pinned:
    if the default metric ever became `euclidean` the instruction to square
    would itself be the bug.
    """
    fields = LapTrack.model_fields
    for name in ("metric", "gap_closing_metric", "splitting_metric"):
        assert fields[name].default == "sqeuclidean", name
    assert fields["cutoff"].default == 15**2
    assert fields["gap_closing_cutoff"].default == 15**2


def test_squaring_is_what_the_cutoff_means(LapTrack, two_frames):
    """The same claim, executed rather than read off a default: at a true step
    of 4 px, `cutoff=5` (which is 5 px^2 = 2.2 px) links nothing and
    `cutoff=5**2` links everything.

    Gap closing is switched off to isolate round 1 -- with it on, its own
    (much larger) default cutoff re-links across a frame difference of 1 and
    hides the mistake. Which is itself worth knowing: the two rounds are not
    independent, and an undersized `cutoff` is partly papered over by
    `gap_closing_cutoff` rather than being obvious.
    """
    kw = {"coordinate_cols": ["y", "x"], "frame_col": "frame"}
    loose, _, _ = LapTrack(cutoff=5**2, gap_closing_cutoff=False).predict_dataframe(
        two_frames, **kw
    )
    tight, _, _ = LapTrack(cutoff=5, gap_closing_cutoff=False).predict_dataframe(
        two_frames, **kw
    )
    assert loose["track_id"].nunique() == 4
    assert tight["track_id"].nunique() == 8


def test_splitting_and_merging_are_off_by_default(LapTrack):
    """Step 4 turns `splitting_cutoff` on for dividing cells and says why. If
    upstream ever defaulted it to a number, the body would be describing an
    opt-in that had become an opt-out -- and the lineage counts in its failure
    table would be wrong in the other direction."""
    fields = LapTrack.model_fields
    assert fields["splitting_cutoff"].default is False
    assert fields["merging_cutoff"].default is False


def test_the_frame_count_is_a_difference_not_a_number_of_missed_frames(
    LapTrack, two_frames
):
    """The third failure row. A single missed frame puts the two surviving
    detections two frames apart, so `gap_closing_max_frame_count=1` closes
    nothing -- which is the opposite of how the name reads."""
    import pandas as pd

    # object 1 present at frames 0 and 2, absent at 1
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "label": [1, 2, 1],
            "y": [10.0, 200.0, 10.0],
            "x": [20.0, 200.0, 28.0],
        }
    )
    kw = {"coordinate_cols": ["y", "x"], "frame_col": "frame"}
    lt = LapTrack(cutoff=15**2, gap_closing_cutoff=30**2)
    one, _, _ = lt.model_copy(
        update={"gap_closing_max_frame_count": 1}
    ).predict_dataframe(df, **kw)
    two, _, _ = lt.model_copy(
        update={"gap_closing_max_frame_count": 2}
    ).predict_dataframe(df, **kw)
    assert one["track_id"].nunique() == 3
    assert two["track_id"].nunique() == 2


def test_predict_dataframe_returns_the_columns_step_5_merges_on(LapTrack, two_frames):
    """`track_df, split_df, merge_df = ...`, and the two id columns step 5
    reads. The `label` column is the load-bearing part: the join in step 5 only
    works because the caller's own columns come back alongside the ids."""
    result = LapTrack(cutoff=15**2).predict_dataframe(
        two_frames, coordinate_cols=["y", "x"], frame_col="frame"
    )
    assert len(result) == 3
    assert {"frame", "label", "y", "x", "track_id", "tree_id"} <= set(result[0].columns)


def test_a_division_appears_in_split_df_and_keeps_one_tree_id(LapTrack):
    """What `splitting_cutoff` buys, and the `track_id` vs `tree_id` sentence
    in step 5: after a division there are three `track_id`s and one `tree_id`,
    and `split_df` names the parent.

    An empty `split_df` carries no columns at all, so the names in the *Next
    steps* section are only assertable on a frame that has a row in it.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "frame": [0, 1, 1, 2, 2],
            "label": [1, 1, 2, 1, 2],
            "y": [10.0, 6.0, 14.0, 5.0, 15.0],
            "x": [20.0, 24.0, 24.0, 28.0, 28.0],
        }
    )
    track_df, split_df, _ = LapTrack(
        cutoff=10**2, splitting_cutoff=20**2
    ).predict_dataframe(df, coordinate_cols=["y", "x"], frame_col="frame")

    assert {"parent_track_id", "child_track_id"} <= set(split_df.columns)
    assert len(split_df) == 2
    assert track_df["track_id"].nunique() == 3
    assert track_df["tree_id"].nunique() == 1


def test_the_returned_rows_are_frame_sorted_and_carry_no_way_back(LapTrack, two_frames):
    """Why step 5 merges instead of assigning.

    On an input that is not already frame-sorted, the result comes back sorted
    by frame under a fresh 0..N-1 index -- so `det["track_id"] =
    track_df["track_id"].values` pairs each detection with a different one's
    id, without raising. Measured on the case fixture, that recovers 0.2% of
    the true links.

    Pinned as *observed behaviour*, not as a promise: if a future release
    returned the caller's order, or a MultiIndex that maps back (which is what
    the docstring describes), this fails and step 5 can be simplified. Either
    way the merge stays correct.
    """
    shuffled = two_frames.iloc[[7, 0, 5, 2, 6, 1, 4, 3]].reset_index(drop=True)
    track_df, _, _ = LapTrack(cutoff=15**2).predict_dataframe(
        shuffled, coordinate_cols=["y", "x"], frame_col="frame"
    )
    assert track_df.index.names == [None]
    assert list(track_df.index) == list(range(len(shuffled)))
    assert (np.diff(track_df["frame"].to_numpy()) >= 0).all()
    assert not (track_df["label"].to_numpy() == shuffled["label"].to_numpy()).all()


def test_the_distribution_version_is_the_one_the_checklist_can_be_resolved_on():
    """Step 1 tells the agent to read `importlib.metadata.version` and not
    `laptrack.__version__`, because the module attribute is stale: 0.17.1 ships
    `__version__ = "0.17.0"`. An agent resolving `pkg:laptrack~=0.17.1` off the
    attribute reports a correct install as unmet.

    Asserted as the *inequality*, so this stops being a special case the day
    upstream fixes it -- at which point step 1's sentence should go.
    """
    laptrack = pytest.importorskip("laptrack")
    assert dist_version("laptrack") != getattr(laptrack, "__version__", None)


# --- skimage, for the degraded path ---------------------------------------


def test_phase_cross_correlation_still_defaults_to_phase_normalization():
    """A failure row in two bodies exists because this default whitens
    frequency bins holding only numerical noise: `drift-correction` recovers ~0
    drift on a smooth movie and reports success, and `stitch-tiles` measured the
    share of tile pairs it registers correctly falling from 92% to 68%. Both fix
    it with `normalization=None`. Both halves are asserted here: if skimage ever
    changed the default the rows would be wrong, and if the parameter went away
    the fix would be unrunnable.
    """
    import inspect

    from skimage.registration import phase_cross_correlation

    params = inspect.signature(phase_cross_correlation).parameters
    assert "normalization" in params
    assert params["normalization"].default == "phase"


def test_phase_cross_correlation_still_returns_three_values():
    """`stitch-tiles` unpacks `shift, _, _ = phase_cross_correlation(...)`.

    The arity is not obvious from the docs — older releases returned the error
    and phase difference only on request — and a body that unpacks the wrong
    number of values fails at the first pair, on every run, for every user.
    """
    import numpy as np
    from skimage.registration import phase_cross_correlation

    reference = np.random.default_rng(0).random((64, 64))
    result = phase_cross_correlation(
        reference, np.roll(reference, 3, axis=1), normalization=None
    )
    assert isinstance(result, tuple) and len(result) == 3
    shift = np.asarray(result[0], float)
    assert shift.shape == (2,)
    # And it means what the body's `pair_offset` assumes: the offset that maps
    # the moving image onto the reference, sign included.
    assert tuple(shift) == (0.0, -3.0)
