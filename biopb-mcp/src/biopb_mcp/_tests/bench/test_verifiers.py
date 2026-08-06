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

from ..agentbench._fixture import Attempt, Fixture
from .cases import align_channels_from_landmarks as landmarks
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
