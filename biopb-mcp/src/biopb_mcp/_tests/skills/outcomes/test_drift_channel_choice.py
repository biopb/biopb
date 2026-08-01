"""Does not knowing which channel is structural actually cost anything?

`docs/skill-testing.md` §6 rests on one claim: design the fixture so the ground
truth is obtainable **only by asking**, and a numeric assertion tests the
interaction for free. That claim is not self-evidently true of any particular
fixture, and it is cheap to get wrong — an ambiguity every heuristic happens to
survive leaves a suite that certifies nothing while looking like it tests
conversation.

So this module settles it without a model in the loop. Three scripted subjects:
one that was told which channel is structural, and two using the choices a run
that never asked would make from the pixels alone. If the verifier cannot
separate them, `_drift_channels` is not an interaction fixture, whatever the
respondent is later told to say.

This is `test_drift_correction`'s calibration argument (§5b) applied one tier
up. There it was "a verifier that only ever sees correct runs cannot be told
apart from one that returns green"; here it is that a fixture nothing has ever
failed for want of asking is not known to test asking.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from . import _drift_channels
from ._drift import SKILL, save_artifacts, verify
from ._outcome import providers_for

pytestmark = pytest.mark.outcome

#: Every subject registers with pystackreg, so the module needs it whole.
SUBJECTS = {
    "told-which-channel": _drift_channels.told_which_channel,
    "the-brightest-channel": _drift_channels.the_brightest_channel,
    "the-mean-of-the-channels": _drift_channels.the_mean_of_the_channels,
}

#: (case, subject) -> (should it pass, which claim this row is)
EXPECTED = {
    ("two-channels-one-structural", "told-which-channel"): (
        True,
        "step 2, answered: the run that asked gets the structural channel and "
        "the trajectory is exact",
    ),
    ("two-channels-one-structural", "the-brightest-channel"): (
        False,
        "step 2's reason for asking: 'These look identical in a single frame "
        "and the correction for one destroys the other.' Brightness picks the "
        "channel whose objects move, and their common motion is added to the "
        "stage's without anything failing",
    ),
    ("two-channels-one-structural", "the-mean-of-the-channels"): (
        False,
        "the Parameters table, verbatim: 'Not a mean projection over channels "
        "-- that mixes in the very channel whose intensity is the "
        "measurement.' Averaging does not escape the choice, it makes it badly",
    ),
}

INTERACTION = providers_for(SKILL, tier="interaction")
BY_CASE = {p.case_id: p for p in INTERACTION}


def _case_of(provider) -> str:
    return getattr(provider, "case_id", "none")


@pytest.mark.parametrize("provider", INTERACTION, ids=_case_of)
def test_the_fixture_keeps_its_truth_out_of_the_data(provider):
    """The same invariant the outcome cases carry, and it is doing more work
    here: `structural_channel` is not just an answer key, it is the fact the
    whole tier is about withholding. If it ever appears in `data`, every
    subject below converges and the layer silently stops testing anything."""
    f = provider.build()
    assert f.truth.get("structural_channel") is not None
    assert not set(f.truth) & set(f.data), (
        f"{f.label} exposes {sorted(set(f.truth) & set(f.data))} to the run"
    )


@pytest.mark.parametrize("provider", INTERACTION, ids=_case_of)
def test_the_channels_are_not_separable_by_shape(provider):
    """Whatever tells the channels apart, it must not be the array. A run that
    could pick the structural channel off the metadata would never need to ask,
    and this fixture would be measuring something else."""
    f = provider.build()
    movie = f.data["movie"]
    assert movie.ndim == 4 and movie.shape[1] == 2, (
        f"{f.label} is shaped {movie.shape}, which is not a two-channel movie"
    )
    # And the tempting heuristic really does point the wrong way -- otherwise
    # `the_brightest_channel` below would be passing for the right reason.
    brightest = int(max(range(2), key=lambda c: movie[:, c].std()))
    assert brightest != int(f.truth["structural_channel"]), (
        "the structural channel is also the brightest, so registering on "
        "contrast would land on the right answer by accident"
    )


@pytest.mark.parametrize("provider", INTERACTION, ids=_case_of)
def test_the_objects_move_and_the_stage_does_too(provider):
    """Both motions have to be real. No stage drift and there is nothing to
    correct; no object motion and every channel is equally good to register
    on."""
    f = provider.build()
    offsets = np.asarray(f.truth["offsets"])
    stage = float(np.hypot(*offsets[-1]))
    assert stage > 10.0, f"{f.label} drifts only {stage:.1f} px"

    # The objects' own motion is what the reporter channel's centroid does over
    # and above the stage. Measured off the built fixture rather than off the
    # construction constants, so a change to either is caught here.
    reporter = f.data["movie"][:, 1 - int(f.truth["structural_channel"])]
    centroid = np.array(
        [ndimage.center_of_mass(frame - reporter.min()) for frame in reporter]
    )
    own = float(np.hypot(*(centroid[-1] - centroid[0] - offsets[-1])))
    assert own > 1.0, (
        f"{f.label}: the objects move only {own:.2f} px of their own, so "
        "registering on them would be nearly as good as registering on the "
        "structural channel"
    )


def test_the_expectation_table_covers_every_case():
    """A case or subject added without an expectation is one nobody decided the
    right answer for — the same anti-rot guard the outcome table carries."""
    want = {(p.case_id, s) for p in INTERACTION for s in SUBJECTS}
    assert not (want - set(EXPECTED)), (
        f"no expectation for {sorted(want - set(EXPECTED))}"
    )
    assert not (set(EXPECTED) - want), (
        f"EXPECTED names what is not registered: {sorted(set(EXPECTED) - want)}"
    )


@pytest.mark.parametrize(
    ("case_id", "subject"),
    sorted(EXPECTED),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_not_asking_costs_the_measurement(case_id, subject, record):
    """The layer's actual question. Every expected-to-fail row here is a run
    that did everything else right — `TRANSLATION`, `reference="previous"`,
    transforms applied to all channels — and lost the measurement on the one
    step it could not do without the user."""
    pytest.importorskip("pystackreg")
    should_pass, claim = EXPECTED[(case_id, subject)]

    fixture = BY_CASE[case_id].build()
    outcome = record(verify, save_artifacts, fixture, SUBJECTS[subject](fixture))

    assert outcome.scored, "nothing was measured, so nothing was tested"
    assert outcome.passed is should_pass, (
        f"{claim}\n\n"
        f"expected this run to {'pass' if should_pass else 'fail'}, and it did "
        f"not:\n{outcome.summary()}"
    )
