"""Does following `drift-correction` actually stabilise a drifting movie?

Every expected-to-fail row in `EXPECTED` below is a claim the skill file makes
in prose, turned into a measurement:

- step 3's "``reference="previous"`` is load-bearing, not a tuning choice",
- the failure table's first row, "recovered drift is ~0 while the movie visibly
  moves ... pass ``normalization=None``",
- step 1's "translation-only and less precise" about the degraded path — which
  this fixture set makes concrete, and stronger than the body currently says.

The expected-to-*pass* rows are the calibration. A verifier that only ever sees
correct runs cannot be told apart from one that always returns green, which is
the failure that left the contract layer unmanned for a release. Both halves are
needed, and neither is interesting alone.
"""

from __future__ import annotations

import pytest

from . import _drift
from ._drift import SKILL, save_artifacts, verify
from ._outcome import FIXTURE_DIR_ENV, providers_for

pytestmark = pytest.mark.outcome

#: subject -> (runner, the package it needs)
SUBJECTS = {
    "as-the-skill-says": (_drift.as_the_skill_says, "pystackreg"),
    "degraded-path": (_drift.the_degraded_path, None),
    "against-the-first-frame": (_drift.against_the_first_frame, "pystackreg"),
    "default-normalization": (_drift.with_default_normalization, None),
}

#: (case, subject) -> (should it pass, which claim this row is)
EXPECTED = {
    ("blobs-slow", "as-the-skill-says"): (True, "steps 3 and 5, verbatim"),
    ("blobs-slow", "degraded-path"): (
        True,
        "step 1: the fallback is a real one on a field with structure",
    ),
    ("blobs-slow", "against-the-first-frame"): (
        False,
        "step 3: the displacement outgrows the capture range and the fit "
        "returns a confident, wrong transform",
    ),
    ("blobs-slow", "default-normalization"): (
        False,
        "failure table row 1: phase whitening buries the peak, ~0 drift is "
        "recovered, and nothing errors",
    ),
    ("blobs-fast", "as-the-skill-says"): (True, "steps 3 and 5 at 4 px/frame"),
    ("blobs-fast", "degraded-path"): (
        True,
        "the fallback registers to frame 0 and is unbothered by the rate",
    ),
    ("blobs-fast", "against-the-first-frame"): (
        False,
        "step 3, at the rate where the body reports 4 of 4 runs losing lock",
    ),
    ("blobs-fast", "default-normalization"): (
        False,
        "failure table row 1, at 4 px/frame",
    ),
    ("smooth-low-contrast", "as-the-skill-says"): (
        True,
        "a pyramid optimiser is untroubled by low contrast",
    ),
    ("smooth-low-contrast", "degraded-path"): (
        False,
        "step 1 calls the fallback 'less precise'. Here that means ~5 px of "
        "error on a 39 px drift -- smooth error, so step 4's largest-step "
        "check does not catch it. See this module's docstring",
    ),
    ("smooth-low-contrast", "against-the-first-frame"): (
        True,
        "the counter-example: low-frequency content has a wide capture range, "
        "so reference='first' is fine here. The rule earns its place on the "
        "cases above, not on every image",
    ),
    ("smooth-low-contrast", "default-normalization"): (
        False,
        "failure table row 1, on exactly the smooth low-contrast field it names",
    ),
}

# `tier="outcome"` is not decoration: `_drift_channels` registers its own case
# under this same skill, and without the filter this module would collect it --
# with four subjects that assume a single-channel movie -- as soon as pytest
# happened to import the two modules together.
OUTCOME_TIER = providers_for(SKILL, tier="outcome")
SYNTHETIC = [p for p in OUTCOME_TIER if p.kind == "synthetic"]
BY_CASE = {p.case_id: p for p in OUTCOME_TIER}

#: On a machine with no curated tree this is one skip carrying instructions,
#: rather than nothing at all. The tier is worth advertising -- it is where real
#: data goes, and an invisible seam is one nobody uses.
CURATED = [p for p in OUTCOME_TIER if p.kind == "curated"] or [
    pytest.param(
        None,
        marks=pytest.mark.skip(
            reason=f"no curated fixtures here; point {FIXTURE_DIR_ENV} at a tree"
        ),
    )
]


def _case_of(provider) -> str:
    return getattr(provider, "case_id", "none")


def _fixture_for(case_id: str):
    provider = BY_CASE[case_id]
    usable, why = provider.available()
    if not usable:
        pytest.skip(f"{provider.case_id}: {why}")
    return provider.build()


# --- the fixtures themselves ----------------------------------------------


@pytest.mark.parametrize("provider", SYNTHETIC, ids=_case_of)
def test_the_fixture_keeps_its_truth_out_of_the_data(provider):
    """The invariant the whole layer rests on. A truth key that also appears in
    `data` is visible to the run, and a fixture that leaks its answer reports a
    perfect score for a procedure that never worked."""
    f = provider.build()
    assert f.truth, f"{f.label} carries no truth, so nothing can be scored"
    assert not set(f.truth) & set(f.data), (
        f"{f.label} exposes {sorted(set(f.truth) & set(f.data))} to the run"
    )


@pytest.mark.parametrize("provider", SYNTHETIC, ids=_case_of)
def test_the_fixture_is_reproducible(provider):
    """Procedural, from a seed -- so nothing binary lands in git and two runs
    of the suite compare like for like. Without this, a tolerance tuned today
    means nothing tomorrow."""
    import numpy as np

    a, b = provider.build(), provider.build()
    assert np.array_equal(a.data["movie"], b.data["movie"])
    assert np.array_equal(a.truth["offsets"], b.truth["offsets"])


@pytest.mark.parametrize("provider", SYNTHETIC, ids=_case_of)
def test_the_movie_really_drifts(provider):
    """A fixture that forgot to move is a fixture every subject passes."""
    import numpy as np

    f = provider.build()
    excursion = float(np.hypot(*np.asarray(f.truth["offsets"])[-1]))
    assert excursion > 10.0, f"{f.label} drifts only {excursion:.1f} px"


def test_the_expectation_table_covers_every_case():
    """The anti-rot guard, the same shape as `COVERED` in the contract layer. A
    case or a subject added without an expectation is one nobody decided the
    right answer for."""
    want = {(p.case_id, s) for p in SYNTHETIC for s in SUBJECTS}
    assert not (want - set(EXPECTED)), (
        f"no expectation for {sorted(want - set(EXPECTED))}"
    )
    assert not (set(EXPECTED) - want), (
        f"EXPECTED names what is not registered: {sorted(set(EXPECTED) - want)}"
    )


# --- the layer's actual question ------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "subject"),
    sorted(EXPECTED),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_the_verifier_tells_a_good_run_from_a_bad_one(case_id, subject, record):
    should_pass, claim = EXPECTED[(case_id, subject)]
    runner, needs = SUBJECTS[subject]
    if needs:
        pytest.importorskip(needs)

    fixture = _fixture_for(case_id)
    outcome = record(verify, save_artifacts, fixture, runner(fixture))

    assert outcome.scored, "nothing was measured, so nothing was tested"
    assert outcome.passed is should_pass, (
        f"{claim}\n\n"
        f"expected this run to {'pass' if should_pass else 'fail'}, and it did "
        f"not:\n{outcome.summary()}"
    )


@pytest.mark.parametrize("provider", CURATED, ids=_case_of)
def test_a_curated_movie_is_corrected_by_the_procedure(provider, record):
    """Real data, when this machine has any — the substitution the synthetic
    cases leave the door open for.

    Only the two correct subjects run. Whether a real movie is one that
    `reference="first"` fails on is a property of that acquisition, not
    something the layer may assume, so there is no negative control here: the
    calibration lives on the synthetic cases, where the answer is known.
    """
    pytest.importorskip("pystackreg")
    usable, why = provider.available()
    if not usable:
        pytest.skip(why)

    fixture = provider.build()
    outcome = record(verify, save_artifacts, fixture, _drift.as_the_skill_says(fixture))

    assert outcome.scored, (
        f"{fixture.label} supports no metric -- its case.json declares a truth "
        "the verifier does not read"
    )
    assert outcome.passed, outcome.summary()
