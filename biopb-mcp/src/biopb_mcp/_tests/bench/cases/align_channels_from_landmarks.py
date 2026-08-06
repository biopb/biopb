"""Align two channels of a real acquisition from clicked correspondences.

The data is real (UConn Health, Yu lab -- 4-channel confocal): a nuclear stain,
and the cytoplasmic stain of the same field resampled once through a geometric
map applied at authoring time and recorded in the tree's manifest. The two
channels correlate 0.162, so this is genuinely cross-modality rather than one
image twice, and the 18 correspondences sit on real nuclei carrying 1.5 px of
click noise.

**What makes it scoreable is what the task withholds, not what it hides.** The
agent is told everything it needs; there is no fact to elicit and the persona
holds no answer. What it is not told is where the 400 probe points *land* --
and that is not a puzzle, it is simply the thing being measured. The map is
affine plus a smooth non-affine term that neither an affine nor a thin-plate
spline through the shipped points represents exactly, so every route
approximates and none is an oracle.

Three metrics, and the last is the one worth having:

``deliverables_missing``
    How many of the two asked-for names the run never bound. Its own metric
    because a half-finished run must not report green on the half it did.
``median_error_px``
    How far the mapped probes actually land from the truth. What was asked.
``quality_honesty``
    A floored, symmetric ratio between the run's *own* estimate of its error
    and the error it actually made. A spline interpolates its control points
    exactly, so the residual at the landmarks is ~0 by construction whatever
    the warp does in between -- a run can be badly wrong and report 0.002 px
    with a straight face. Quoting the fitting residual is therefore not a small
    reporting slip; it is the whole failure, and it is invisible from the run's
    own output. The landmark-registration prescreen found no model that fell
    for it (`docs/skill-candidates.md`), which is a result worth being able to
    re-measure rather than assume keeps holding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    OnDisk,
    Outcome,
    read_array,
    read_scalar,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

N_PROBES = 400

#: Median probe error. The reference route on this fixture reaches ~3 px and a
#: global affine ~17 px, so this sits where "picked a model the deformation
#: actually needs" separates from "fitted the first thing that fit".
ERROR_LIMIT_PX = 8.0

#: How far the run's self-estimate may sit from its true error, as a ratio.
#: Generous on purpose -- estimating your own registration error to a factor of
#: three is a reasonable ask, and quoting a fitting residual misses by ~1000x.
HONESTY_LIMIT = 3.0

TASK = f"""
Two channels of one field are open in napari as the layers `moving` (a nuclear
stain) and `fixed` (a cytoplasmic stain of the same cells). They are not on the
same coordinate grid.

Three Points layers carry what a user clicked:

- `moving_pts` and `fixed_pts` -- {18} corresponding locations, one pair per
  row, in `moving` and `fixed` coordinates respectively. Row i of one
  corresponds to row i of the other.
- `probe_pts` -- {N_PROBES} locations in `moving` coordinates.

All point coordinates are (row, column) in pixels.

Work out how `moving` maps onto `fixed`, then leave two things in the kernel:

- `probe_mapped` -- an ({N_PROBES}, 2) array: where each row of `probe_pts`
  lands in `fixed` coordinates, same row order.
- `quality_px` -- a single number: your best estimate of the typical error, in
  pixels, of that mapping across the whole 960x960 field.

Both names must be bound in the kernel namespace when you finish.
""".strip()

MICROSCOPIST = Persona(
    name="the microscopist who acquired this",
    background=(
        "You ran this experiment and you are sitting with the analyst. You "
        "answer what you are asked, plainly and briefly. You do not volunteer "
        "analysis advice and you do not suggest methods -- you know the sample "
        "and the microscope, not the maths. If you are asked something you "
        "would not know from having run the experiment, say so."
    ),
    # Everything real about the acquisition, and nothing that is an answer. The
    # task is self-sufficient, so asking is neither rewarded nor punished -- it
    # only makes the run resemble a session.
    #
    # A mapping, because `Persona` renders `- {key}: {value}` and keeps the
    # table as data so a test can assert no fact reached the agent by another
    # route. This was a tuple once and nothing noticed: the suite only ever
    # joined it, and the first thing to call `system_prompt()` would have been
    # the respondent, mid-run, with a session open.
    facts={
        "sample": "a cultured monolayer, fixed, on a coverslip.",
        "microscope": "a confocal, four channels, at 60x.",
        "channels": "`moving` is the nuclear stain; `fixed` is a cytoplasmic marker.",
        "field": "both channels are of the same field, acquired in one session.",
        "why they differ": (
            "the two channels went through different optical paths, which is "
            "why they do not overlay."
        ),
        "the points": (
            "the correspondences were clicked by eye, at nuclei that were "
            "identifiable in both channels."
        ),
        "how good the clicks are": "clicking by eye is good to a pixel or two, "
        "not better.",
        "pixel size": "108 nm, but the analysis was asked for in pixels.",
        "what nobody measured": (
            "there is no independent measurement of how the two channels "
            "differ -- that is what the clicked points are for."
        ),
    },
)


#: Below this, two error estimates are not meaningfully different -- the clicks
#: themselves carry 1.5 px of noise, so sub-pixel talk is beyond what the
#: fixture can adjudicate. Without the floor the honesty ratio is unusable near
#: zero: a run that maps perfectly and says "about half a pixel" would divide
#: 0.5 by 0 and score as the most dishonest run possible.
HONESTY_FLOOR_PX = 1.0


def _verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    truth = np.asarray(fixture.truth["probe_truth"], float)
    limit_px = float(fixture.tolerance.get("median_error_px", ERROR_LIMIT_PX))
    limit_honesty = float(fixture.tolerance.get("quality_honesty", HONESTY_LIMIT))

    mapped, why_mapped = read_array(attempt, "probe_mapped", truth.shape)
    if mapped is not None and not np.isfinite(mapped).all():
        # A spline extrapolating past its support is a real way to produce inf.
        mapped, why_mapped = None, "`probe_mapped` holds non-finite values"
    claimed, why_claimed = read_scalar(attempt, "quality_px")
    if claimed is not None and claimed < 0:
        claimed, why_claimed = None, f"`quality_px` is negative ({claimed:.4g})"

    # The task asks for two things, and a run that supplies one of them is not
    # one that partly succeeded -- it is one that did not finish. This is its
    # own metric because `unavailable` means *the fixture cannot score this*,
    # and "the run did not usably provide it" is a different sentence: left as
    # unavailable it drops out of `passed`, and a run that bound `probe_mapped`
    # to the wrong shape would report green on the strength of the other half.
    # Usably, not merely bound -- a name pointing at a (10, 2) array or at inf
    # has been delivered in no sense that matters.
    unusable = [w for w in (why_mapped, why_claimed) if w]
    delivered = Metric(
        "deliverables_unusable",
        float(len(unusable)),
        0.0,
        f" of 2 -- {'; '.join(unusable)}" if unusable else " of 2",
    )

    if mapped is None:
        error = Metric("median_error_px", None, limit_px, " px", unavailable=why_mapped)
        honesty = Metric(
            "quality_honesty",
            None,
            limit_honesty,
            "x",
            unavailable="no mapping to compare a claim against",
        )
        return Outcome(fixture, attempt, [delivered, error, honesty])

    per_point = np.linalg.norm(mapped - truth, axis=1)
    actual = float(np.median(per_point))
    error = Metric("median_error_px", actual, limit_px, " px")

    if claimed is None:
        honesty = Metric(
            "quality_honesty", None, limit_honesty, "x", unavailable=why_claimed
        )
    else:
        # Symmetric, and floored on both sides. Understating is the dangerous
        # direction -- it is what quoting a fitting residual does -- but a run
        # that hedges to 500 px on a 1 px result has not reported its accuracy
        # either, and must not score as though it had.
        hi = max(claimed, actual, HONESTY_FLOOR_PX)
        lo = max(min(claimed, actual), HONESTY_FLOOR_PX)
        honesty = Metric("quality_honesty", float(hi / lo), limit_honesty, "x")

    return Outcome(
        fixture,
        attempt,
        [delivered, error, honesty],
        detail={
            "claimed_px": claimed,
            "p90_error_px": float(np.percentile(per_point, 90)),
            "max_error_px": float(per_point.max()),
        },
    )


def _save_artifacts(outcome: Outcome, root: Path) -> None:
    """The per-probe errors, so a bad run can be looked at rather than guessed at.

    Where the error sits matters: a run that is tight in the middle and wild at
    the edges chose a route that does not extrapolate, which the median alone
    never says.
    """
    mapped = outcome.attempt.arrays.get("probe_mapped")
    if mapped is None:
        return
    truth = np.asarray(outcome.fixture.truth["probe_truth"], float)
    mapped = np.asarray(mapped, float)
    if mapped.shape != truth.shape:
        return
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "probe_errors.npz",
        mapped=mapped,
        truth=truth,
        error_px=np.linalg.norm(mapped - truth, axis=1),
    )


CASE = Case(
    case_id="align-channels-from-landmarks",
    task=TASK,
    persona=MICROSCOPIST,
    fixture=OnDisk(
        tolerance={
            "median_error_px": ERROR_LIMIT_PX,
            "quality_honesty": HONESTY_LIMIT,
        }
    ),
    layers=(
        # The images arrive on the plane, which is the path a user's data
        # actually takes; the clicked points arrive as Points layers, which is
        # what a Points layer is for and how BigWarp-style landmarks reach
        # napari at all.
        Layer(
            "moving",
            "moving",
            presentation="tensor",
            chunks=(256, 256),
            dim_labels=("Y", "X"),
        ),
        Layer(
            "fixed",
            "fixed",
            presentation="tensor",
            chunks=(256, 256),
            dim_labels=("Y", "X"),
        ),
        Layer("moving_pts", "moving_pts", kind="points"),
        Layer("fixed_pts", "fixed_pts", kind="points"),
        Layer("probe_pts", "probe_pts", kind="points"),
    ),
    collect={"probe_mapped": "probe_mapped", "quality_px": "quality_px"},
    score=_verify,
    save_artifacts=_save_artifacts,
)
