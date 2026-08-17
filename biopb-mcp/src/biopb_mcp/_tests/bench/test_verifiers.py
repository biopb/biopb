"""Checks that belong to one case, not to every case.

`test_cases.py` asserts what has to be true of any case — the persona holds
back, the truth is not in the data, an empty attempt scores nothing. That is
the floor, and it is not enough for the part of a case that can be wrong
*quietly*.

**The verifier is that part.** A fixture that fails to build raises and a bad
prompt shows up in the transcript, but a verifier that scores a wrong answer as
a pass produces a clean green report meaning nothing — and it will keep
producing one for as long as anybody trusts it. So a case whose scoring has a
specific way of being fooled writes the test for that here, and the shape to
write is always the same four: a perfect run, a run that did nothing, a missing
deliverable, and whatever *looking* right without *being* right is for this
particular task.

A case with no entry here is not exempt; it is a case whose verifier nobody has
found a specific way to fool yet.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..agentbench._fixture import Attempt, Fixture
from .cases import (
    align_channels_from_landmarks as landmarks,
    kymograph_velocity,
    multiview_reconstruction as multiview,
)
from .test_cases import built_fixture

# --- drift-correction: the movie must not paint the answer on its own edges --

#: Widest run of identical rows or columns tolerated at a frame border. A
#: synthetic field is sparse, so some edge really is flat background: measured
#: over six seeds, every frame of every channel stays under 6 px. Rendering
#: frame-sized and shifting in place instead of cropping a padded canvas reached
#: 25 px, and the width tracked the offset.
MAX_FLAT_BORDER_PX = 10


def _flat_border_px(frame, tol=1e-3) -> int:
    """Rows at the top edge of `frame` that are copies of their neighbour."""
    varies = np.abs(np.diff(frame, axis=0)).mean(axis=1) > tol
    return int(np.argmax(varies)) if varies.any() else frame.shape[0]


def test_the_drifted_movie_invents_no_pixels():
    """The same leak as `test_the_fixture_keeps_its_truth_out_of_the_data`, by
    the other route: not a truth *key* left in `data`, but the truth painted
    into the pixels.

    A stage that moves reveals sample that was outside the field of view; it
    does not create pixels. Shift a frame-sized image and the interpolator has
    to invent the vacated border, and the width of what it invents *is* the
    shift — the withheld trajectory, readable off the edges with no registration
    at all, and a band of flat correlated structure sitting inside the very data
    the run registers on.
    """
    from .cases import drift_correction

    movie = np.asarray(built_fixture(drift_correction.CASE).data["movie"])
    worst = {"px": 0, "frame": -1, "channel": -1, "edge": -1}
    for t, frame in enumerate(movie):
        for c, plane in enumerate(frame):
            # All four edges: flip to bring each one to the top in turn.
            for edge, view in enumerate((plane, plane[::-1], plane.T, plane.T[::-1])):
                width = _flat_border_px(view)
                if width > worst["px"]:
                    worst = {"px": width, "frame": t, "channel": c, "edge": edge}
    assert worst["px"] <= MAX_FLAT_BORDER_PX, (
        f"drift-correction: {worst['px']} px of flat border at frame "
        f"{worst['frame']}, channel {worst['channel']}, edge {worst['edge']} — "
        "the field of view is showing pixels no acquisition produced"
    )


# --- align-channels-from-landmarks: the honesty metric ----------------------


def _fixture_with(truth: np.ndarray) -> Fixture:
    return Fixture(
        provenance="test",
        data={},
        truth={"probe_truth": truth},
        tolerance={
            "median_error_px": landmarks.ERROR_LIMIT_PX,
            "quality_honesty": landmarks.HONESTY_LIMIT,
        },
    )


def _truth(n: int = landmarks.N_PROBES) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 960, size=(n, 2))


def _attempt(mapped, quality) -> Attempt:
    arrays = {}
    if mapped is not None:
        arrays["probe_mapped"] = np.asarray(mapped, float)
    if quality is not None:
        arrays["quality_px"] = np.asarray(float(quality))
    return Attempt(subject="test", arrays=arrays)


def test_a_perfect_run_passes():
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth, 0.5))
    assert outcome.passed
    assert {m.name: m.value for m in outcome.metrics}["median_error_px"] == 0.0


def test_a_run_that_did_nothing_fails():
    """Identity: `probe_mapped` left equal to the input. On this fixture the
    real displacement is ~52 px, so doing nothing is not a near miss."""
    truth = _truth()
    identity = truth + 52.0
    outcome = landmarks._verify(_fixture_with(truth), _attempt(identity, 1.0))
    assert not outcome.passed
    by_name = {m.name: m for m in outcome.metrics}
    assert by_name["median_error_px"].value == pytest.approx(52 * np.sqrt(2), rel=0.01)


def test_quoting_the_fitting_residual_fails_honesty():
    """The failure the whole metric exists for.

    A spline interpolates its control points exactly, so the residual there is
    ~0 whatever the warp does in between. A run 20 px wrong that reports
    0.002 px has not made a small reporting slip — it has produced a number
    that cannot be told from a good one.
    """
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth + 20.0, 0.002))
    by_name = {m.name: m for m in outcome.metrics}
    assert by_name["quality_honesty"].scored
    assert not by_name["quality_honesty"].passed
    assert by_name["quality_honesty"].value > 10


def test_hedging_wildly_also_fails_honesty():
    """Overstating is symmetric. A run that reports 500 px on a 1 px result has
    not reported its accuracy either, and must not score as if it had."""
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth + 1.0, 500.0))
    by_name = {m.name: m for m in outcome.metrics}
    assert not by_name["quality_honesty"].passed


def test_an_honest_estimate_passes():
    truth = _truth()
    # ~2.8 px actual, claimed 3 -- the shape of a run that cross-validated.
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth + 2.0, 3.0))
    assert outcome.passed


def test_a_missing_result_is_unscorable_not_a_pass():
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(None, 3.0))
    by_name = {m.name: m for m in outcome.metrics}
    assert not outcome.passed
    assert not by_name["median_error_px"].scored
    assert by_name["deliverables_unusable"].value == 1.0


def test_a_wrong_shape_is_unscorable_not_a_pass():
    """Bound, but to the wrong thing. Must not pass on the strength of the
    other deliverable."""
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth[:10], 3.0))
    by_name = {m.name: m for m in outcome.metrics}
    assert not outcome.passed
    assert not by_name["median_error_px"].scored
    assert by_name["deliverables_unusable"].value == 1.0


def test_non_finite_output_is_unscorable():
    """A thin-plate spline extrapolating past its support is a real way to
    produce inf, and it must not crash the scorer."""
    truth = _truth()
    mapped = truth.copy()
    mapped[0] = np.inf
    outcome = landmarks._verify(_fixture_with(truth), _attempt(mapped, 3.0))
    by_name = {m.name: m for m in outcome.metrics}
    assert not outcome.passed
    assert not by_name["median_error_px"].scored


def test_a_missing_quality_still_scores_the_error():
    """The two metrics are independent: a run that mapped well but reported no
    estimate has done most of the task, and the report should say so."""
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth, None))
    by_name = {m.name: m for m in outcome.metrics}
    assert by_name["median_error_px"].scored
    assert not by_name["quality_honesty"].scored
    assert not outcome.passed  # deliverables_unusable = 1 fails


def test_the_landmark_persona_does_not_describe_the_transform():
    """`test_a_task_persona_holds_no_deliverable` catches the deliverable by
    name. This catches it by description: a persona that says what the warp
    *is* has handed over the answer without using the word for it."""
    facts = " ".join(landmarks.CASE.persona.facts.values()).casefold()
    for phrase in ("the transform is", "the warp is", "the shift is"):
        assert phrase not in facts, f"the persona describes the answer: {phrase!r}"


# --- multiview-reconstruction: every view, in microns -----------------------
#
# The registration lives here and not in the case module for the same reason as
# the kymograph estimator: how a run reaches an answer is not the case's
# business. It exists because the docstring's route table is a claim.


def _kabsch(src, dst):
    """Rigid (R, t) with ``dst ~ src @ R.T + t``, by SVD."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ flip @ u.T
    return r, cd - r @ cs


#: Coarse to fine. The nominal transform is a few degrees and a few microns
#: out, so the first pass has to pair points much further apart than the beads
#: are spaced; a schedule that starts at the final tolerance never starts.
CUTOFF_SCHEDULE_UM = (10.0, 6.0, 4.0, 2.5, 1.5, 1.0, 0.8, 0.8, 0.8)


def _icp(moving, fixed, r, t):
    tree = cKDTree(fixed)
    for cutoff in CUTOFF_SCHEDULE_UM:
        distance, nearest = tree.query(moving @ r.T + t)
        keep = distance < cutoff
        if keep.sum() < 8:
            break
        r, t = _kabsch(moving[keep], fixed[nearest[keep]])
    return r, t


def _register(fixture, scale, centroid=True, refine=True):
    """Every view straight to view 0 -- no link graph and no composition, so
    #690's spanning-tree-versus-least-squares question cannot arise here."""
    clouds = [
        np.asarray(fixture.data[f"view{k}"], float) * scale
        for k in range(multiview.N_VIEWS)
    ]
    nominal = multiview._transforms(exact=False)
    maps = [(np.eye(3), np.zeros(3))]
    for k in range(1, multiview.N_VIEWS):
        r_k, t_k = nominal[k]
        r, t = r_k.T, -r_k.T @ t_k  # view k -> the reference, as the stage has it
        if centroid:
            t = t + clouds[0].mean(0) - (clouds[k] @ r.T + t).mean(0)
        if refine:
            r, t = _icp(clouds[k], clouds[0], r, t)
        maps.append((r, t))
    return maps


def _principal_axes(fixture, scale):
    """Correspondence-free: match centroids and principal axes, nothing else."""
    clouds = [
        np.asarray(fixture.data[f"view{k}"], float) * scale
        for k in range(multiview.N_VIEWS)
    ]
    centre = clouds[0].mean(0)
    _, _, axes = np.linalg.svd(clouds[0] - centre, full_matrices=False)
    maps = [(np.eye(3), np.zeros(3))]
    for cloud in clouds[1:]:
        here = cloud.mean(0)
        _, _, mine = np.linalg.svd(cloud - here, full_matrices=False)
        flipped = axes.copy()
        r = axes.T @ mine
        if np.linalg.det(r) < 0:
            flipped[2] *= -1
            r = flipped.T @ mine
        maps.append((r, centre - r @ here))
    return maps


def _probe_um(fixture, maps, scale):
    probes = np.asarray(fixture.data["probes"], float)
    out = np.empty_like(probes)
    for i, k in enumerate((1, 2, 3)):
        rows = slice(
            i * multiview.N_PROBES_PER_VIEW, (i + 1) * multiview.N_PROBES_PER_VIEW
        )
        out[rows] = (probes[rows] * scale) @ maps[k][0].T + maps[k][1]
    return out


def _confirmed(fixture, maps, scale, gap_um=1.1):
    """Clusters carrying detections from two or more views.

    The only count that is well defined here: a spurious detection appears in
    one list, and so does a real bead only one view was shallow enough to see,
    and nothing in the data tells those two apart.
    """
    points, source = [], []
    for k in range(multiview.N_VIEWS):
        cloud = np.asarray(fixture.data[f"view{k}"], float) * scale
        points.append(cloud @ maps[k][0].T + maps[k][1])
        source.append(np.full(len(cloud), k))
    points, source = np.vstack(points), np.concatenate(source)

    tree = cKDTree(points)
    taken = np.zeros(len(points), bool)
    count = 0
    for i in np.argsort(points[:, 0]):
        if taken[i]:
            continue
        group = [j for j in tree.query_ball_point(points[i], gap_um) if not taken[j]]
        taken[group] = True
        count += len(set(source[group])) >= 2
    return count


def _score(fixture, maps, scale, count=None):
    return multiview._verify(
        fixture,
        Attempt(
            subject="route",
            arrays={
                "probe_um": _probe_um(fixture, maps, scale),
                "n_beads_confirmed": (
                    _confirmed(fixture, maps, scale) if count is None else count
                ),
            },
            notes="",
        ),
    )


def _named(outcome):
    return {m.name: m for m in outcome.metrics}


@pytest.fixture(scope="module")
def bead_field():
    return built_fixture(multiview.CASE)


@pytest.fixture(scope="module")
def microns():
    return multiview.VOXEL_UM


def test_multiview_the_reference_route_registers_every_view(bead_field, microns):
    """Winnability, recomputed rather than quoted: convert to microns, start
    from the stage angle, put the centroids together, refine."""
    outcome = _score(bead_field, _register(bead_field, microns), microns)
    assert outcome.passed
    by_name = _named(outcome)
    assert by_name["worst_view_median_um"].value < 0.1
    assert by_name["bead_count_error"].value == 0.0


def test_multiview_registering_in_voxel_indices_fails(bead_field, microns):
    """The failure the modality has and a 2D tile grid cannot: z is 3.1x
    coarser than xy, and the views differ by rotations about the axis that
    trades z for x, so a rotation fitted to indices is fitted to a different
    geometry."""
    ones = np.ones(3)
    outcome = _score(bead_field, _register(bead_field, ones), ones)
    assert not outcome.passed
    assert _named(outcome)["worst_view_median_um"].value > 100.0


def test_multiview_the_nominal_stage_transform_is_not_the_answer(bead_field, microns):
    """The stage records what it was told to do, not where the sample settled."""
    outcome = _score(
        bead_field,
        _register(bead_field, microns, centroid=False, refine=False),
        microns,
    )
    assert not outcome.passed
    assert _named(outcome)["worst_view_median_um"].value > 4.0


def test_multiview_icp_needs_the_centroid_start(bead_field, microns):
    """The finding this fixture exists to keep: the nominal transform is a good
    enough start for three views and not for the fourth, whose stage shift is
    6.1 um. Outside its basin ICP does not diverge, it converges 8.7 degrees off
    -- and the run has no way to tell from its own output."""
    outcome = _score(
        bead_field, _register(bead_field, microns, centroid=False), microns
    )
    assert not outcome.passed
    assert _named(outcome)["worst_view_median_um"].value > 1.0


def test_multiview_principal_axes_without_correspondence_fail(bead_field, microns):
    """The back door: register the clouds by their moments and never match a
    bead to a bead. It fails because the subsets differ -- each view's centroid
    and inertia are computed over a different sample of the specimen."""
    outcome = _score(bead_field, _principal_axes(bead_field, microns), microns)
    assert not outcome.passed
    assert _named(outcome)["worst_view_median_um"].value > 10.0


def test_multiview_perfect_transforms_do_not_excuse_never_merging(bead_field, microns):
    """Why there are two metrics. A run that registers exactly and keeps every
    detection has done the hard half and not the task: it scores 0.03 um and
    reports 3.1x too many beads."""
    maps = _register(bead_field, microns)
    total = sum(len(bead_field.data[f"view{k}"]) for k in range(multiview.N_VIEWS))
    outcome = _score(bead_field, maps, microns, count=total)
    by_name = _named(outcome)
    assert by_name["worst_view_median_um"].passed
    assert not by_name["bead_count_error"].passed
    assert not outcome.passed


def test_multiview_view_zeros_list_alone_fails_the_count(bead_field, microns):
    """The other half of the same point, and the closer of the two: a run that
    reports what one view saw is out by 23%, not by a factor."""
    outcome = _score(
        bead_field,
        _register(bead_field, microns),
        microns,
        count=len(bead_field.data["view0"]),
    )
    assert not _named(outcome)["bead_count_error"].passed


def test_multiview_the_worst_view_is_scored_not_the_pooled_median(bead_field, microns):
    """The metric's own failure mode, asserted because the first version of
    this verifier had it. Two views perfect and one 5 um out leaves the pooled
    median at zero -- the failure sits in the top third and never reaches it."""
    truth = np.asarray(bead_field.truth["probe_um"], float)
    mapped = truth.copy()
    mapped[2 * multiview.N_PROBES_PER_VIEW :] += 5.0

    pooled = np.median(np.linalg.norm(mapped - truth, axis=1))
    assert pooled == 0.0, "the premise of this test is gone"

    outcome = multiview._verify(
        bead_field,
        Attempt(
            subject="two of three",
            arrays={
                "probe_um": mapped,
                "n_beads_confirmed": bead_field.truth["n_confirmed"],
            },
            notes="",
        ),
    )
    assert not outcome.passed
    assert _named(outcome)["worst_view_median_um"].value > 4.0


def test_multiview_the_four_views_are_not_the_same_subset(bead_field):
    """If every view saw every bead, correspondence would be sorting and the
    count would be free. Detection falls off with depth so that it is not."""
    sizes = [len(bead_field.data[f"view{k}"]) for k in range(multiview.N_VIEWS)]
    assert len(set(sizes)) > 1, sizes
    assert bead_field.truth["n_confirmed"] < sum(sizes) / 2


def test_multiview_an_empty_attempt_scores_nothing(bead_field):
    outcome = multiview._verify(
        bead_field, Attempt(subject="nothing", arrays={}, notes="")
    )
    assert not any(m.scored for m in outcome.metrics)
    assert not outcome.passed


def test_multiview_half_an_answer_is_not_half_a_pass(bead_field, microns):
    """The count reported and the transforms not. The scored metric is green,
    so without `deliverables_unusable` the run would be."""
    outcome = multiview._verify(
        bead_field,
        Attempt(
            subject="half",
            arrays={"n_beads_confirmed": bead_field.truth["n_confirmed"]},
            notes="",
        ),
    )
    by_name = _named(outcome)
    assert by_name["bead_count_error"].passed
    assert not by_name["worst_view_median_um"].scored
    assert by_name["deliverables_unusable"].value == 1.0
    assert not outcome.passed


# --- lumos-spectral-unmixing: accuracy must be over the field, not the keep --


def _lumos_truth():
    from .cases import lumos_spectral_unmixing as lumos

    return np.asarray(built_fixture(lumos.CASE).truth["dye_map"])


def _lumos_score(labels):
    from .cases import lumos_spectral_unmixing as lumos

    fixture = built_fixture(lumos.CASE)
    arrays = {} if labels is None else {"dye_labels": np.asarray(labels)}
    outcome = lumos.verify(fixture, Attempt(subject="test", arrays=arrays))
    return {m.name: m for m in outcome.metrics}, outcome


def test_lumos_a_perfect_run_passes():
    truth = _lumos_truth()
    by_name, outcome = _lumos_score(truth)
    assert outcome.passed
    assert by_name["dye_error"].value == 0.0
    assert by_name["background_error"].value == 0.0


def test_lumos_a_relabelled_run_passes():
    """Which dye gets which number is a cluster id, not an answer. A run that
    permuted 1-4 has done the task and must not be marked down for it."""
    truth = _lumos_truth()
    permuted = np.select(
        [truth == 1, truth == 2, truth == 3, truth == 4],
        [3, 4, 1, 2],
        default=0,
    )
    by_name, outcome = _lumos_score(permuted)
    assert outcome.passed
    assert by_name["dye_error"].value == 0.0


def test_lumos_a_run_that_did_nothing_fails():
    """Everything called background: the degenerate answer, and the one an
    over-cut converges to."""
    truth = _lumos_truth()
    by_name, outcome = _lumos_score(np.zeros_like(truth))
    assert not outcome.passed
    assert by_name["dye_error"].value > 0.5
    assert by_name["background_error"].value == pytest.approx(0.6, abs=0.01)


def test_lumos_perfect_clustering_over_a_bad_background_cut_fails():
    """**The specific way of looking right for this case.**

    The run's partition of the pixels it kept is *exactly* the truth -- every
    check it can compute on retained pixels comes back clean, which is what the
    prescreen's M2 arm did while discarding a seventh of the image. Scoring
    accuracy over the retained pixels would call this perfect. Scoring it over
    the field is what makes the discarded third visible, and it is the whole
    reason `background_error` is a metric rather than a footnote.
    """
    truth = _lumos_truth()
    rng = np.random.default_rng(0)
    over_cut = truth.copy()
    # Vacate a third of the labelled pixels -- perfect where it kept, silent
    # about where it did not.
    labelled = np.flatnonzero(truth.ravel() != 0)
    drop = rng.choice(labelled, size=int(0.33 * labelled.size), replace=False)
    flat = over_cut.ravel()
    flat[drop] = 0
    by_name, outcome = _lumos_score(over_cut)

    kept = over_cut != 0
    assert np.array_equal(over_cut[kept], truth[kept]), (
        "the setup is not perfect-on-keep"
    )
    assert not outcome.passed
    assert not by_name["background_error"].passed
    assert not by_name["dye_error"].passed


def test_lumos_a_missing_result_is_unscorable_not_a_pass():
    by_name, outcome = _lumos_score(None)
    assert not outcome.passed
    assert not by_name["dye_error"].scored
    assert "left no" in by_name["dye_error"].unavailable


def test_lumos_a_foreign_labelling_convention_is_unscorable():
    """Five clusters, or background numbered last, is a different task's answer.
    Scoring it anyway would silently score something the case did not ask."""
    truth = _lumos_truth()
    by_name, outcome = _lumos_score(np.where(truth == 0, 5, truth))
    assert not outcome.passed
    assert not by_name["dye_error"].scored


def test_lumos_intensity_alone_cannot_pass():
    """§11's back door, asserted rather than remembered.

    The within-dye brightness range is a decade and identical across dyes, so
    total intensity carries no dye identity. Binning it -- with the bin edges
    fitted against the answer key, which no run could do -- must land far
    outside the tolerance, or the fixture is scoring brightness.
    """
    from .cases import lumos_spectral_unmixing as lumos

    fixture = built_fixture(lumos.CASE)
    truth = np.asarray(fixture.truth["dye_map"])
    total = np.asarray(fixture.data["lambda_stack"], float).sum(axis=0)

    keep = truth != 0  # oracle foreground: strictly better than any run's cut
    edges = np.quantile(total[keep], np.linspace(0, 1, lumos.N_DYES + 1)[1:-1])
    labels = np.zeros(truth.shape, int)
    labels[keep] = np.digitize(total[keep], edges) + 1

    by_name, _ = _lumos_score(labels)
    assert by_name["dye_error"].value > 3 * lumos.TOLERANCE["dye_error"], (
        "an intensity-only route scores near the tolerance, so this fixture "
        "is measuring brightness rather than spectrum (§11 back door)"
    )


def test_lumos_the_operators_estimate_lands_inside_the_pass_band():
    """The safeguard has to point at the answer, and it is easy to word so it
    does not.

    The persona's coverage estimate is what turns this case from a draw off the
    unordered auto-threshold menu into something a run can check against. But
    the pass band is asymmetric -- under-cutting background costs accuracy fast,
    because background pixels forced into a dye cluster are wrong twice -- so an
    estimate that reads high fails a run that believed it. A first draft said
    "call it two thirds", which is 0.333 background against a band that starts
    at 0.38, and a run following the operator exactly would have scored 0.791.

    So: take the range a reader would take from the persona's own words, and
    require both ends to pass. This is the test that fails when someone rounds
    the operator's number for readability.
    """
    from .cases import lumos_spectral_unmixing as lumos

    fixture = built_fixture(lumos.CASE)
    cube = np.moveaxis(np.asarray(fixture.data["lambda_stack"], float), 0, -1)
    total = cube.sum(axis=-1)

    # "a bit more than half ... not much more" -> 0.55 to 0.62 covered.
    for coverage in (0.55, 0.62):
        keep = total > np.quantile(total, 1.0 - coverage)
        vectors = cube[keep]
        vectors = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9
        )
        from sklearn.cluster import KMeans

        labels = np.zeros(total.shape, int)
        labels[keep] = KMeans(4, n_init=10, random_state=1).fit_predict(vectors) + 1
        _, outcome = _lumos_score(labels)
        assert outcome.passed, (
            f"a run that believed the operator ({coverage:.0%} covered) fails: "
            f"{[str(m) for m in outcome.metrics]}"
        )


def test_lumos_the_operators_patch_count_catches_a_shattered_mask():
    """The other half of the safeguard, and the one that answers 'would a bad
    cut look obviously wrong'.

    It does: a failing cut breaks the field into 119-1165 components against the
    truth's 10. The operator says "ten or so big irregular areas", so a run that
    asked has the number to compare against -- and this asserts the comparison
    actually discriminates, rather than both masks being in the same ballpark.
    """
    from scipy import ndimage as ndi
    from skimage import filters

    from .cases import lumos_spectral_unmixing as lumos

    fixture = built_fixture(lumos.CASE)
    truth = np.asarray(fixture.truth["dye_map"])
    total = np.asarray(fixture.data["lambda_stack"], float).sum(axis=0)

    _, true_components = ndi.label(truth != 0)
    _, bad_components = ndi.label(total > filters.threshold_otsu(total))
    assert true_components <= 20, (
        f"the truth is {true_components} components, which is not 'ten or so' "
        "-- the persona's description has drifted from the fixture"
    )
    assert bad_components > 5 * true_components, (
        f"an over-cut gives {bad_components} components against the truth's "
        f"{true_components}, which is not a difference anyone would notice"
    )


# --- reconstruction-fidelity-qc: the ranking must survive its own shortcuts --


def _squirrel():
    from .cases import reconstruction_fidelity_qc as squirrel

    return squirrel, built_fixture(squirrel.CASE)


def _squirrel_parts():
    """``(module, fixture, reconstructions, widefield, true order)``."""
    squirrel, fixture = _squirrel()
    widefield = np.asarray(fixture.data[squirrel.WIDEFIELD], float)
    recons = {
        name: np.asarray(array, float)
        for name, array in fixture.data.items()
        if name != squirrel.WIDEFIELD
    }
    order = [str(x) for x in np.asarray(fixture.truth["fidelity_order"]).tolist()]
    return squirrel, fixture, recons, widefield, order


def _squirrel_score(ranking):
    squirrel, fixture = _squirrel()
    arrays = {} if ranking is None else {"fidelity_ranking": np.asarray(ranking)}
    outcome = squirrel.verify(fixture, Attempt(subject="test", arrays=arrays))
    return {m.name: m.value for m in outcome.metrics}, outcome


def _rank_by(scores):
    """Best first. Alphabetical is the tie-break, so a statistic that cannot
    separate two layers falls back to presentation order — which this case has
    already measured as worth nothing (Spearman 0.036)."""
    return sorted(sorted(scores), key=lambda name: -scores[name])


def test_squirrel_a_perfect_run_passes():
    _, _, _, _, order = _squirrel_parts()
    by_name, outcome = _squirrel_score(order)
    assert outcome.passed
    assert by_name["ranking_error"] == 0.0


def test_squirrel_the_mark_is_not_a_knife_edge():
    """The prescreen's own threshold caveat: with seven items one adjacent swap
    moves Spearman by 0.036, so a mark sitting one swap from a clean run is
    measuring the draw rather than the work.

    Each adjacent transposition costs ``sum d^2 = 2`` and the limit is at 8, so
    every one of the three disjoint adjacent swaps that fit in seven items
    passes. What must not pass is a wholesale reordering: moving one layer
    three places costs 12.
    """
    squirrel, _, _, _, order = _squirrel_parts()
    for pairs in range(4):
        ranking = list(order)
        for i in range(pairs):
            lo = 2 * i
            ranking[lo], ranking[lo + 1] = ranking[lo + 1], ranking[lo]
        by_name, outcome = _squirrel_score(ranking)
        assert outcome.passed, (
            f"{pairs} disjoint adjacent swap(s) scores "
            f"{by_name['ranking_error']:.4f} against a limit of "
            f"{squirrel.TOLERANCE['ranking_error']}, and a case this coarse "
            "must leave room for a near-miss"
        )
    displaced = order[1:4] + [order[0]] + order[4:]
    by_name, outcome = _squirrel_score(displaced)
    assert not outcome.passed, (
        "moving the most faithful reconstruction three places down scores "
        f"{by_name['ranking_error']:.4f} and passes"
    )
    _, outcome = _squirrel_score(list(reversed(order)))
    assert not outcome.passed, "the reversed ranking is a pass"


def test_squirrel_matching_resolution_passes_and_skipping_it_does_not():
    """§11 triviality, computed rather than asserted from a table.

    Correlating each reconstruction against the widefield is the whole idea and
    every arm in the prescreen found it unaided. What separated the tiers was
    blurring to the camera's resolution first. So the un-blurred one-liner must
    not reach the mark, or this case measures nothing the prescreen did not
    already give away.
    """
    squirrel, _, recons, widefield, _ = _squirrel_parts()
    from scipy import ndimage as ndi

    def rebin(a):
        return a.reshape(squirrel.CAM, squirrel.BIN, squirrel.CAM, squirrel.BIN).mean(
            (1, 3)
        )

    blurred = _rank_by(
        {
            name: squirrel._pearson(
                rebin(ndi.gaussian_filter(a, squirrel.PSF_SIGMA_FINE)), widefield
            )
            for name, a in recons.items()
        }
    )
    raw = _rank_by(
        {name: squirrel._pearson(rebin(a), widefield) for name, a in recons.items()}
    )

    _, blurred_outcome = _squirrel_score(blurred)
    raw_by_name, raw_outcome = _squirrel_score(raw)
    assert blurred_outcome.passed, "the reference route does not pass its own case"
    assert not raw_outcome.passed, (
        "correlating without matching resolution scores "
        f"{raw_by_name['ranking_error']:.4f}, inside the limit — the one step "
        "this case exists to measure is not load-bearing"
    )


def test_squirrel_the_operators_psf_lands_well_inside_the_pass_band():
    """The safeguard, and the reason handing the PSF over concedes nothing.

    The prescreen found a *fitted* resolution-scaling function and the *known
    optical PSF* rank all seven identically, so this case must not be a
    PSF-guessing game. Both numbers the persona offers — the measured bead
    width, and an Abbe estimate from the NA and wavelength it also states —
    have to pass, and so does a wide band either side of them.
    """
    squirrel, _, recons, widefield, _ = _squirrel_parts()
    from scipy import ndimage as ndi

    def score(fwhm_nm):
        sigma = fwhm_nm / 2.3548 / squirrel.FINE_NM
        ranking = _rank_by(
            {
                name: squirrel._pearson(
                    ndi.gaussian_filter(a, sigma)
                    .reshape(squirrel.CAM, squirrel.BIN, squirrel.CAM, squirrel.BIN)
                    .mean((1, 3)),
                    widefield,
                )
                for name, a in recons.items()
            }
        )
        return _squirrel_score(ranking)

    # what the persona says beads measure
    assert score(squirrel.PSF_FWHM_NM)[1].passed
    # 0.51 * lambda / NA from the objective the persona names
    assert score(0.51 * 600.0 / 1.35)[1].passed
    for multiple in (0.6, 0.85, 1.25, 2.0, 3.0):
        by_name, outcome = score(squirrel.PSF_FWHM_NM * multiple)
        assert outcome.passed, (
            f"a PSF {multiple}x the true one scores "
            f"{by_name['ranking_error']:.4f} and fails, which makes this a "
            "test of PSF estimation rather than of matching resolution at all"
        )


def test_squirrel_ranking_by_apparent_sharpness_fails():
    """The prescreen's first keepsake: sharpness is *anti*-correlated with
    fidelity, so "it looks more resolved" inverts the answer."""
    squirrel, _, recons, _, _ = _squirrel_parts()
    ranking = _rank_by(
        {
            name: float(
                np.mean(np.gradient(a)[0] ** 2 + np.gradient(a)[1] ** 2)
                / (a.var() + 1e-12)
            )
            for name, a in recons.items()
        }
    )
    by_name, outcome = _squirrel_score(ranking)
    assert not outcome.passed
    assert by_name["ranking_error"] > 1.0, (
        "ranking by sharpness scores "
        f"{by_name['ranking_error']:.4f}; the prescreen measured this route as "
        "worse than chance and the fixture should reproduce that"
    )


def test_squirrel_demoting_the_signed_reconstruction_is_fatal_on_its_own():
    """The prescreen's second keepsake, and this case's specific way of being
    wrong while looking careful.

    Odd-order SOFI cumulants are legitimately signed. Two of six arms demoted
    one reconstruction as "corrupted" for having 7.5% negative pixels, and that
    single error was the whole of one arm's failure. Here it must be fatal on
    its own — everything else ranked perfectly.
    """
    squirrel, fixture, recons, _, order = _squirrel_parts()
    signed = squirrel.LAYER_NAMES["sofi3_signed"]
    assert order.index(signed) == 1, "the signed reconstruction is no longer rank 2"

    array = recons[signed]
    negative = float((array < 0).mean())
    assert 0.05 < negative < 0.12, (
        f"the signed reconstruction is {negative:.1%} negative; the trap is "
        "that this reads as corruption to a run that does not ask"
    )
    for name, other in recons.items():
        if name != signed:
            assert other.min() >= 0.0, f"{name} is also signed"

    ranking = [n for n in order if n != signed] + [signed]
    by_name, outcome = _squirrel_score(ranking)
    assert not outcome.passed, (
        "throwing away the signed reconstruction and ranking everything else "
        "perfectly is a pass, so the case cannot see the error the prescreen "
        "says decided a whole arm"
    )
    assert by_name["ranking_error"] > 0.5


def test_squirrel_no_per_image_statistic_reproduces_the_key():
    """§11's back door, at oracle strength: the statistic AND its sign chosen
    against the answer key, which no run could do.

    The first build of this fixture failed here. A residual background is
    low-frequency power, and it had been made monotone in fidelity, so the
    high-frequency power fraction — one number per image, no comparison with
    anything — reproduced the key at |rho| 0.964 on two of six seeds.
    """
    squirrel, _, recons, _, _ = _squirrel_parts()

    def statistics(a):
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
        half = squirrel.FINE // 2
        ky, kx = np.mgrid[-half:half, -half:half]
        radius = np.hypot(ky, kx)
        positive = np.clip(a - a.min(), 0, None)
        positive = positive / (positive.sum() + 1e-12)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "skew": float(((a - a.mean()) ** 3).mean() / (a.std() ** 3 + 1e-12)),
            "kurtosis": float(((a - a.mean()) ** 4).mean() / (a.std() ** 4 + 1e-12)),
            "negative_fraction": float((a < 0).mean()),
            "negative_depth": float(a.min() / a.max()),
            "max_over_std": float(a.max() / (a.std() + 1e-12)),
            "gradient_energy": float(np.mean(np.gradient(a)[0] ** 2)),
            "high_frequency_fraction": float(
                spectrum[radius > squirrel.FINE * 0.18].sum() / (spectrum.sum() + 1e-12)
            ),
            "low_frequency_fraction": float(
                spectrum[radius < squirrel.FINE * 0.02].sum() / (spectrum.sum() + 1e-12)
            ),
            "entropy": float(
                -(positive[positive > 0] * np.log(positive[positive > 0])).sum()
            ),
        }

    menu = {name: statistics(a) for name, a in recons.items()}
    worst = ("", 0.0)
    for statistic in next(iter(menu.values())):
        values = {name: menu[name][statistic] for name in recons}
        for sign in (1.0, -1.0):
            by_name, outcome = _squirrel_score(
                _rank_by({k: sign * v for k, v in values.items()})
            )
            assert not outcome.passed, (
                f"ranking by {statistic} alone (sign {sign:+.0f}) scores "
                f"{by_name['ranking_error']:.4f} and passes — one number per "
                "image, computed without looking at the widefield at all"
            )
            if 1.0 - by_name["ranking_error"] > worst[1]:
                worst = (statistic, 1.0 - by_name["ranking_error"])
    assert worst[1] < 0.6, (
        f"the best per-image statistic ({worst[0]}) reaches Spearman "
        f"{worst[1]:.3f}; §11 wants a route that ignores the premise near chance"
    )


def test_squirrel_the_presentation_order_is_worth_nothing():
    """The letters were permuted against the fidelity order on purpose."""
    _, _, recons, _, _ = _squirrel_parts()
    by_name, outcome = _squirrel_score(sorted(recons))
    assert not outcome.passed
    assert by_name["ranking_error"] > 0.5, (
        "listing the layers in the order they are presented scores "
        f"{by_name['ranking_error']:.4f}, so the anonymised names carry the answer"
    )


def test_squirrel_a_missing_result_is_unscorable_not_a_pass():
    by_name, outcome = _squirrel_score(None)
    assert not outcome.passed
    assert by_name["ranking_error"] is None


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda o: o[:-1], id="one-short"),
        pytest.param(lambda o: o + [o[0]], id="one-too-many"),
        pytest.param(lambda o: [o[0]] + o[2:] + [o[0]], id="repeats-a-name"),
        pytest.param(lambda o: o[:-1] + ["method_H"], id="names-a-missing-layer"),
        pytest.param(lambda o: [[n] for n in o], id="not-a-flat-list"),
    ],
)
def test_squirrel_a_ranking_that_is_not_a_ranking_is_unusable(damage):
    """A ranking of six, or one that repeats a layer, has not answered the
    question — and scoring it anyway would score a different task."""
    _, _, _, _, order = _squirrel_parts()
    by_name, outcome = _squirrel_score(damage(list(order)))
    assert not outcome.passed
    assert by_name["ranking_error"] is None


def _squirrel_rebin(squirrel, a):
    return a.reshape(squirrel.CAM, squirrel.BIN, squirrel.CAM, squirrel.BIN).mean(
        (1, 3)
    )


def test_squirrel_the_case_does_not_require_squirrel():
    """The scope note, pinned.

    The prescreen's conclusion was that the whole candidate is "match
    resolution before correlating" -- not a resolution-scaling function, not an
    intensity fit, not even a Gaussian. If any of these cheaper routes failed,
    the case would quietly be scoring something the prescreen said it was not.
    """
    squirrel, _, recons, widefield, _ = _squirrel_parts()
    from scipy import ndimage as ndi

    blurred = {
        name: ndi.gaussian_filter(a, squirrel.PSF_SIGMA_FINE)
        for name, a in recons.items()
    }

    def upsampled(w):
        return np.repeat(np.repeat(w, squirrel.BIN, 0), squirrel.BIN, 1)

    def by_rse(a):
        binned = _squirrel_rebin(squirrel, a)
        design = np.stack([binned.ravel(), np.ones(binned.size)], 1)
        coefficients, *_ = np.linalg.lstsq(design, widefield.ravel(), rcond=None)
        residual = design @ coefficients - widefield.ravel()
        return -float(np.sqrt(np.mean(residual**2)))

    routes = {
        "blurred but not rebinned": {
            name: squirrel._pearson(a, upsampled(widefield))
            for name, a in blurred.items()
        },
        "a box blur of roughly the right width": {
            name: squirrel._pearson(
                _squirrel_rebin(
                    squirrel, ndi.uniform_filter(a, int(2.5 * squirrel.PSF_SIGMA_FINE))
                ),
                widefield,
            )
            for name, a in recons.items()
        },
        "scored by RSE instead of RSP": {
            name: by_rse(a) for name, a in blurred.items()
        },
    }
    for label, scores in routes.items():
        by_name, outcome = _squirrel_score(_rank_by(scores))
        assert outcome.passed, (
            f"{label} scores {by_name['ranking_error']:.4f} and fails, so this "
            "case is measuring more than matching resolution"
        )


def test_squirrel_the_remaining_shortcuts_fail():
    """The rest of the docstring's table, so no row of it is a number somebody
    typed once: comparing against an upsampled widefield without blurring, and
    ranking by agreement with the other six -- which never looks at the data."""
    squirrel, _, recons, widefield, _ = _squirrel_parts()

    upsampled = np.repeat(np.repeat(widefield, squirrel.BIN, 0), squirrel.BIN, 1)
    shortcuts = {
        "upsample the widefield, no blur": {
            name: squirrel._pearson(a, upsampled) for name, a in recons.items()
        },
    }
    standardised = [(a - a.mean()) / (a.std() + 1e-12) for a in recons.values()]
    consensus = np.mean(standardised, axis=0)
    shortcuts["consensus of the seven"] = {
        name: squirrel._pearson(a, consensus) for name, a in recons.items()
    }
    for label, scores in shortcuts.items():
        by_name, outcome = _squirrel_score(_rank_by(scores))
        assert not outcome.passed, (
            f"{label} scores {by_name['ranking_error']:.4f} and passes, and it "
            "is not the work this case is for"
        )


# --- landmark-registration: the warp must actually need a spline -------------


def _landmark_parts():
    """``(module, fixture, clicked source, clicked target, probes, truth)``."""
    from .cases import landmark_registration as synthetic

    fixture = built_fixture(synthetic.CASE)
    return (
        synthetic,
        fixture,
        np.asarray(fixture.data["moving_pts"], float),
        np.asarray(fixture.data["fixed_pts"], float),
        np.asarray(fixture.data["probe_pts"], float),
        np.asarray(fixture.truth["probe_truth"], float),
    )


def _landmark_score(mapped, quality):
    synthetic, fixture, *_ = _landmark_parts()
    arrays = {}
    if mapped is not None:
        arrays["probe_mapped"] = np.asarray(mapped, float)
    if quality is not None:
        arrays["quality_px"] = np.asarray(float(quality))
    outcome = synthetic.CASE.score(fixture, Attempt(subject="test", arrays=arrays))
    return {m.name: m for m in outcome.metrics}, outcome


def _fit_affine(source, target, points):
    design = np.hstack([source, np.ones((len(source), 1))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.hstack([points, np.ones((len(points), 1))]) @ coefficients


def _fit_spline(source, target, points):
    from scipy.interpolate import RBFInterpolator

    return RBFInterpolator(source, target, kernel="thin_plate_spline")(points)


def test_landmarks_synthetic_the_reference_route_passes():
    """§4 winnability: a thin-plate spline through the shipped clicks must
    clear the limit, or the case asks for something it does not supply."""
    _, _, source, target, probes, truth = _landmark_parts()
    mapped = _fit_spline(source, target, probes)
    actual = float(np.median(np.linalg.norm(mapped - truth, axis=1)))
    by_name, outcome = _landmark_score(mapped, actual)
    assert outcome.passed, f"the reference route scores {actual:.2f} px and fails"
    assert actual < 5.0


def test_landmarks_synthetic_a_global_affine_fails():
    """The tier gap the prescreen actually found, and the reason this fixture
    exists at all.

    Both Haiku arms fitted an affine at every landmark budget and paid 5.8x on
    the set that supported a spline. The record also says the *first* synthetic
    fixture could not show this -- affine and spline landed within ~1 px -- so
    this assertion is the one that says the rebuild worked.
    """
    _, _, source, target, probes, truth = _landmark_parts()
    affine = _fit_affine(source, target, probes)
    spline = _fit_spline(source, target, probes)
    affine_px = float(np.median(np.linalg.norm(affine - truth, axis=1)))
    spline_px = float(np.median(np.linalg.norm(spline - truth, axis=1)))

    by_name, outcome = _landmark_score(affine, affine_px)
    assert not outcome.passed, (
        f"a global affine scores {affine_px:.2f} px and passes, so this fixture "
        "has the same defect the prescreen recorded in its synthetic first try"
    )
    assert affine_px / spline_px > 3.0, (
        f"affine {affine_px:.2f} px against spline {spline_px:.2f} px is a "
        f"ratio of {affine_px / spline_px:.2f}; the deformation is too nearly "
        "affine for the choice of model to be what is measured"
    )


def test_landmarks_synthetic_a_second_order_polynomial_is_not_enough():
    """More than affine and still global. It should land between the two and
    still fail -- documented because it is the obvious middle route."""
    _, _, source, target, probes, truth = _landmark_parts()

    def features(points):
        row, col = points[:, 0], points[:, 1]
        return np.stack(
            [np.ones_like(row), row, col, row * row, row * col, col * col], 1
        )

    coefficients, *_ = np.linalg.lstsq(features(source), target, rcond=None)
    mapped = features(probes) @ coefficients
    actual = float(np.median(np.linalg.norm(mapped - truth, axis=1)))
    _, outcome = _landmark_score(mapped, actual)
    assert not outcome.passed, f"a quadratic scores {actual:.2f} px and passes"


def test_landmarks_synthetic_doing_nothing_fails():
    _, _, _, _, probes, truth = _landmark_parts()
    displacement = float(np.median(np.linalg.norm(probes - truth, axis=1)))
    by_name, outcome = _landmark_score(probes, displacement)
    assert not outcome.passed
    assert by_name["median_error_px"].value == pytest.approx(displacement, rel=1e-6)


def test_landmarks_synthetic_quoting_the_fitting_residual_fails_honesty():
    """The trap the honesty metric exists for, on this fixture's own numbers.

    A spline interpolates its control points exactly, so its residual at the
    clicks is ~0 whatever the warp does in between. A run reporting that number
    is not making a small reporting slip.
    """
    _, _, source, target, probes, truth = _landmark_parts()
    residual = float(
        np.median(np.linalg.norm(_fit_spline(source, target, source) - target, axis=1))
    )
    assert residual < 0.01, (
        f"the spline's fitting residual is {residual:.4f} px, so the trap this "
        "metric exists for is not present in this fixture"
    )
    affine = _fit_affine(source, target, probes)
    by_name, outcome = _landmark_score(affine, residual)
    assert not outcome.passed
    assert by_name["quality_honesty"].scored
    assert not by_name["quality_honesty"].passed


def test_landmarks_synthetic_the_channels_are_not_one_image_twice():
    """§11 back door: if `fixed` were `moving` pushed through the truth map,
    intensity registration would be the answer and the clicked points would be
    decoration. Both halves are checked -- how much the channels share, and
    whether the cheapest intensity route gets anywhere."""
    from skimage.registration import phase_cross_correlation

    _, fixture, _, _, probes, truth = _landmark_parts()
    moving = np.asarray(fixture.data["moving"], float)
    fixed = np.asarray(fixture.data["fixed"], float)

    a = moving.ravel() - moving.mean()
    b = fixed.ravel() - fixed.mean()
    correlation = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(correlation) < 0.30, (
        f"the channels correlate {correlation:.3f} -- close enough to one image "
        "twice that intensity registration becomes the answer"
    )

    shift, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=4)
    by_name, outcome = _landmark_score(probes + shift, 5.0)
    assert not outcome.passed, (
        "phase correlation alone scores "
        f"{by_name['median_error_px'].value:.2f} px and passes"
    )


def test_landmarks_an_empty_attempt_scores_nothing_on_both_fixtures():
    """The shared contract this verifier used to violate.

    `deliverables_unusable` carried a limit of 0.0 -- a silent always-fail --
    and reported 2-of-2 on an empty attempt, which the case contract forbids.
    It was never caught because `align-channels-from-landmarks` is `OnDisk` and
    skips everywhere; its procedural sibling does not.
    """
    from .cases import landmark_registration as synthetic

    for case, fixture in (
        (synthetic.CASE, built_fixture(synthetic.CASE)),
        (landmarks.CASE, _fixture_with(_truth())),
    ):
        outcome = case.score(fixture, Attempt(subject="left-nothing"))
        assert not outcome.passed
        assert outcome.metrics
        assert all(not m.scored for m in outcome.metrics), (
            f"{case.label}: an empty attempt produced a scored metric"
        )
        for metric in outcome.metrics:
            assert metric.limit > 0, f"{case.label}: {metric.name} has no usable limit"


# --- kymograph-velocity: the distance axis, and what hides a wrong one -------
#
# The estimator lives here rather than in the case module because it is not the
# case's business how a run gets its answer -- only what it reports. It is here
# at all because the docstring's route table is a claim, and a claim about what
# a fixture separates has to be recomputed against the fixture that ships.


def _kymograph(movie, path, linewidth=5):
    """(T, S) intensities along *path*, averaged perpendicular to it."""
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(linewidth) - (linewidth - 1) / 2
    out = np.zeros((len(movie), len(path)))
    for i, frame in enumerate(movie):
        frame = np.asarray(frame, float)
        out[i] = sum(
            ndi.map_coordinates(frame, (path + off * normal).T, order=1, mode="nearest")
            for off in offsets
        ) / len(offsets)
    return out


def _equal_count(vertices, per_segment):
    """The trap: `per_segment` samples between every pair of clicks, whatever
    the distance between them."""
    out = [
        a + (b - a) * np.linspace(0, 1, per_segment, endpoint=False)[:, None]
        for a, b in zip(vertices[:-1], vertices[1:], strict=True)
    ]
    return np.vstack(out + [vertices[-1][None, :]])


def _speeds(kymo, step_um, vmax=9.0):
    """One speed each way, by shearing the kymograph and summing over time: a
    trace of slope v lines up with itself only at v, so the sum's variance
    peaks there. Returns (forward, backward) in um/s, both positive."""
    velocities = np.linspace(-vmax, vmax, 721)
    columns = np.arange(kymo.shape[1])
    power = np.array(
        [
            np.var(
                sum(
                    np.interp(columns + v * t, columns, row, left=0.0, right=0.0)
                    for t, row in enumerate(kymo)
                )
            )
            for v in velocities
        ]
    )
    out = []
    for side in (+1, -1):
        keep = np.where(side * velocities > 0.4)[0]
        best = keep[int(np.argmax(power[keep]))]
        out.append(abs(velocities[best]) * step_um / kymograph_velocity.DT_S)
    return tuple(out)


def _reference_path(fixture, per_segment=None):
    roi = np.asarray(fixture.data["roi"], float)
    if per_segment is not None:
        path = _equal_count(roi, per_segment)
        # What the run believes one sample is worth when it never asked: the
        # whole traced length divided by however many samples it took.
        return path, kymograph_velocity._arclength(roi)[-1] / (len(path) - 1)
    length = kymograph_velocity._arclength(roi)[-1]
    return kymograph_velocity._resample(roi, int(round(length))), 1.0


def _measure(fixture, per_segment=None, linewidth=5, destationary=True):
    movie = np.asarray(fixture.data["transport"])
    path, step_px = _reference_path(fixture, per_segment)
    kymo = _kymograph(movie, path, linewidth)
    if destationary:
        kymo = kymo - np.median(kymo, axis=0, keepdims=True)
    return _speeds(kymo, step_px * kymograph_velocity.PIXEL_UM)


def _errors(fixture, **kwargs):
    forward, backward = _measure(fixture, **kwargs)
    return (
        abs(forward - fixture.truth["forward_um_per_s"])
        / fixture.truth["forward_um_per_s"],
        abs(backward - fixture.truth["backward_um_per_s"])
        / fixture.truth["backward_um_per_s"],
    )


@pytest.fixture(scope="module")
def transport():
    return built_fixture(kymograph_velocity.CASE)


def test_kymograph_the_reference_route_reaches_both_speeds(transport):
    """Winnability, recomputed rather than quoted. Resample the traced ROI by
    arc length, take the stationary component out, find one slope each way."""
    forward, backward = _errors(transport)
    assert forward < 0.05, forward
    assert backward < 0.05, backward
    # And comfortably, not marginally -- a limit the reference only just clears
    # is measuring the estimator rather than the run.
    assert max(forward, backward) < kymograph_velocity.SPEED_LIMIT / 3


def test_kymograph_equal_counts_per_segment_fail(transport):
    """The trap the dropped candidate recorded on its way out. Same movie, same
    ROI, same estimator -- only the meaning of one sample changes, and nothing
    in the kymograph says so."""
    for per_segment in (60, 100):
        forward, backward = _errors(transport, per_segment=per_segment)
        assert max(forward, backward) > kymograph_velocity.SPEED_LIMIT, (
            f"{per_segment} samples per segment scored "
            f"{forward:.1%}/{backward:.1%}, inside the limit"
        )


def test_kymograph_the_roi_earns_that_trap_rather_than_being_drawn_for_it(
    transport,
):
    """`<L><1/L>` is the factor an equal-count route puts on a per-segment
    average, and the prescreen measured 1.21x on its own polyline. This ROI
    reaches the same figure from a rule about how far a chord may bow off the
    filament -- so the trap is a property of tracing a curve, not of a vertex
    list chosen to make the point."""
    roi = np.asarray(transport.data["roi"], float)
    segments = np.diff(kymograph_velocity._arclength(roi))
    assert 1.15 < segments.mean() * np.mean(1 / segments) < 1.30
    assert segments.max() / segments.min() > 4.0, "the segments are too even"


def test_kymograph_the_roi_stays_on_the_filament(transport):
    """A ROI that wandered off the ridge would fail every route for a reason
    that is not the one being measured."""
    roi = np.asarray(transport.data["roi"], float)
    movie = np.asarray(transport.data["transport"], float)
    bright = movie.mean(axis=0)
    dense = kymograph_velocity._resample(roi, 3000)
    along = ndi.map_coordinates(bright, dense.T, order=1, mode="nearest")
    assert along.min() > np.percentile(bright, 97), (
        "the traced path leaves the filament somewhere along it"
    )


def test_kymograph_leaving_the_stationary_component_in_fails(transport):
    """The filament is labelled along its whole length and some cargo is
    parked, so the brightest thing in the kymograph is a horizontal stripe. A
    run that looks for slopes without taking it out is answering about the
    stripe."""
    forward, backward = _errors(transport, destationary=False)
    assert max(forward, backward) > kymograph_velocity.SPEED_LIMIT


def test_kymograph_perpendicular_averaging_is_not_what_this_separates(
    transport,
):
    """Recorded because the docstring says so and a claim of no effect is as
    falsifiable as any other. Sixty frames of coherent integration have already
    bought what averaging five pixels across the filament would."""
    wide = _errors(transport)
    narrow = _errors(transport, linewidth=1)
    assert max(narrow) < 0.05, narrow
    assert abs(max(wide) - max(narrow)) < 0.01


def test_kymograph_a_canonical_guess_fails(transport):
    """The triviality screen. Fast anterograde transport is quoted at about
    1 um/s and retrograde at about half that; a run that reports the numbers it
    remembers instead of the ones in front of it must not pass, which is why
    the fixture's speeds are 2.35 and 0.95."""
    outcome = kymograph_velocity._verify(
        transport,
        Attempt(
            subject="guess",
            arrays={
                "speed_forward_um_per_s": 1.0,
                "speed_backward_um_per_s": 0.5,
            },
            notes="",
        ),
    )
    assert not outcome.passed
    failed = [m.name for m in outcome.metrics if m.scored and not m.passed]
    assert failed == ["forward_speed_error", "backward_speed_error"]


def test_kymograph_one_speed_for_both_directions_fails(transport):
    """A run that finds the dominant population and reports it twice has
    measured one thing and answered two questions with it."""
    fast = transport.truth["forward_um_per_s"]
    outcome = kymograph_velocity._verify(
        transport,
        Attempt(
            subject="blend",
            arrays={
                "speed_forward_um_per_s": fast,
                "speed_backward_um_per_s": fast,
            },
            notes="",
        ),
    )
    assert not outcome.passed
    by_name = {m.name: m for m in outcome.metrics}
    assert by_name["forward_speed_error"].passed
    assert not by_name["backward_speed_error"].passed


def test_kymograph_half_an_answer_is_not_half_a_pass(transport):
    """One direction reported perfectly and the other not at all. The scored
    metric is green, so without `deliverables_unusable` the run would be."""
    outcome = kymograph_velocity._verify(
        transport,
        Attempt(
            subject="half",
            arrays={"speed_forward_um_per_s": transport.truth["forward_um_per_s"]},
            notes="",
        ),
    )
    by_name = {m.name: m for m in outcome.metrics}
    assert by_name["forward_speed_error"].passed
    assert not by_name["backward_speed_error"].scored
    assert by_name["deliverables_unusable"].value == 1.0
    assert not outcome.passed


def test_kymograph_an_empty_attempt_scores_nothing(transport):
    """The shared contract, asserted here too because this verifier reports a
    count and a count of nothing is 0, which reads as a pass."""
    outcome = kymograph_velocity._verify(
        transport, Attempt(subject="nothing", arrays={}, notes="")
    )
    assert not any(m.scored for m in outcome.metrics)
    assert not outcome.passed


def test_kymograph_a_negative_speed_is_not_a_speed(transport):
    """The task asks for magnitudes and says so. A signed answer is not a near
    miss to be scored generously -- it is a different quantity, and reading it
    as a speed would score a run that got the direction backwards as perfect."""
    truth = transport.truth
    outcome = kymograph_velocity._verify(
        transport,
        Attempt(
            subject="signed",
            arrays={
                "speed_forward_um_per_s": truth["forward_um_per_s"],
                "speed_backward_um_per_s": -truth["backward_um_per_s"],
            },
            notes="",
        ),
    )
    by_name = {m.name: m for m in outcome.metrics}
    assert not by_name["backward_speed_error"].scored
    assert not outcome.passed
