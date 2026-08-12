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
from scipy.spatial import cKDTree

from ..agentbench._fixture import Attempt, Fixture
from .cases import (
    align_channels_from_landmarks as landmarks,
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
