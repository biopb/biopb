"""Every case's task, persona and verifier, with no model and no session.

This is the half that runs with the ordinary suite. A broken verifier, a task
that forgets to name where results go, or a persona that hands over the answer
should surface as a normal red test — not be discovered by someone mid-run with
a paid session open.

The verifier tests are the load-bearing ones. A scorer is the only part of a
case that can be wrong *quietly*: a fixture that fails to build raises, a task
prompt that reads badly shows up in the transcript, but a verifier that scores
a wrong answer as a pass produces a clean green report that means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..agentbench._fixture import Attempt, Fixture
from ._runner import (
    HARNESS_ERROR,
    OK,
    WRONG_ANSWER,
    Sample,
    TaskCase,
    samples_wanted,
)
from .cases import CASES, align_channels_from_landmarks as landmarks


@pytest.fixture(params=CASES, ids=lambda c: c.case_id)
def case(request) -> TaskCase:
    return request.param


def test_at_least_one_case_ships():
    assert CASES, "cases/ is empty, so this suite asserts nothing about anything"


def test_the_task_names_every_result_it_collects(case: TaskCase):
    """A name the task never mentions cannot be bound by an agent that never
    saw it, and would score as `no-result` for a reason that is the harness's
    fault rather than the run's."""
    for kernel_name in case.collect.values():
        assert kernel_name in case.task, (
            f"{case.case_id} collects `{kernel_name}` but never names it in the "
            "task text"
        )


def test_the_task_names_every_layer_it_is_given(case: TaskCase):
    for layer in case.layers:
        assert layer.name in case.task, (
            f"{case.case_id} puts a layer `{layer.name}` on the viewer that the "
            "task text never mentions"
        )


def test_a_tensor_case_says_where_its_ids_arrive(case: TaskCase):
    """An id is minted at run time, so it cannot be written into a prompt in
    advance. A case presenting on the plane has to name the handle."""
    from ._runner import TENSOR_HANDLE

    if any(layer.lazy for layer in case.layers):
        assert TENSOR_HANDLE in case.task or all(
            layer.name in case.task for layer in case.layers if layer.lazy
        ), (
            f"{case.case_id} presents on the plane but the task names neither "
            f"{TENSOR_HANDLE} nor the layers it creates"
        )


def test_the_persona_volunteers_nothing(case: TaskCase):
    assert "volunteer" in case.persona.background.lower() or "not" in (
        case.persona.background.lower()
    ), f"{case.persona.name} is not told to hold back what it was not asked"


def test_the_persona_holds_no_number_the_verifier_checks(case: TaskCase):
    """A persona that can be asked for the answer measures nothing.

    This suite's personas exist for realism, not elicitation: the task is
    self-sufficient, so asking must neither rescue nor penalise a run.
    """
    lowered = " ".join(case.persona.facts).lower()
    for banned in ("probe_mapped", "quality_px", "the transform is", "the warp is"):
        assert banned not in lowered, (
            f"{case.persona.name} knows `{banned}`, which is what the run is "
            "supposed to work out"
        )


# --- the landmark verifier -------------------------------------------------


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


# --- sample classification -------------------------------------------------


def test_a_provider_truncation_is_a_harness_error_not_a_capability():
    """The distinction the status vocabulary exists for: a completion cut off
    by `max_tokens` looks exactly like a model deciding to stop."""
    from ..agentbench._conversation import AGENT_TRUNCATED

    sample = Sample(index=1, outcome=None, stopped=AGENT_TRUNCATED)
    assert sample.status == HARNESS_ERROR


def test_a_passing_outcome_reports_ok():
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth, 0.5))
    assert Sample(index=1, outcome=outcome, stopped="finished").status == OK


def test_a_failing_outcome_reports_wrong_answer():
    truth = _truth()
    outcome = landmarks._verify(_fixture_with(truth), _attempt(truth + 40.0, 40.0))
    assert Sample(index=1, outcome=outcome, stopped="finished").status == WRONG_ANSWER


def test_samples_default_to_one(monkeypatch):
    from ._runner import SAMPLES_ENV

    monkeypatch.delenv(SAMPLES_ENV, raising=False)
    assert samples_wanted() == 1


@pytest.mark.parametrize("raw,want", [("3", 3), ("0", 1), ("-2", 1), ("nonsense", 1)])
def test_samples_are_read_from_the_environment(monkeypatch, raw, want):
    from ._runner import SAMPLES_ENV

    monkeypatch.setenv(SAMPLES_ENV, raw)
    assert samples_wanted() == want
