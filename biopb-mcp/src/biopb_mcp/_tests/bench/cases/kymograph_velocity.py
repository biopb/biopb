"""Two transport speeds along a neurite the user already traced by hand.

Kymograph was **dropped without screening** as a skill candidate on 2026-08-06
(`docs/skill-candidates.md`), and the reason was structural: the resample is one
`skimage.measure.profile_line` call, so a straight-ROI kymograph is a
comprehension over frames with no decisions between the steps. Nothing in that
verdict says an agent gets the *number* right, and the entry records one live
trap on its way out:

    sampling equal *counts* per segment rather than equal *arc length* makes
    spacing vary with segment length, so a constant-speed particle reads faster
    on the short limbs. Measured at 1.21x on an even test polyline and worse on
    a hand-drawn one, and **nothing in the kymograph shows it**.

This case is that sentence made scoreable, on a fixture where three further
things a run does wrong are also live. The task asks for two speeds in um/s and
names no method: `profile_line`, kymographs, Radon transforms and tracking are
all routes to it, and the case scores where the run arrived.

**What the fixture withholds is not the procedure, it is the answer.** The
pixel size and frame interval are in the task, the person who acquired it will
confirm them, and the ROI they traced is on the viewer. There is no fact to
elicit. What nobody in the room knows is how fast the cargo moves.

Measured on the shipped fixture (seed 11), speeds in um/s against a truth of
2.35 forward and 0.95 backward:

  =============================================  ======  ======  =======  =======
  route                                          fwd     bwd     e_fwd    e_bwd
  =============================================  ======  ======  =======  =======
  reference: arc length, stationary removed        2.34    0.95     0.5%     0.5%
  the same, FFT angular power                      2.33    0.94     0.6%     0.8%
  the same, lag cross-correlation                  2.34    0.94     0.3%     1.2%
  its own path, traced from the data               2.33    0.94     0.8%     0.7%
  no perpendicular averaging (linewidth 1)         2.34    0.94     0.5%     0.6%
  ---------------------------------------------  ------  ------  -------  -------
  equal count per segment, 100 each                2.85    1.41    21.1%    48.4%
  equal count per segment, 60 each                 2.85    1.41    21.2%    48.4%
  the stationary component left in                 2.08    2.39    11.6%   151.8%
  one speed reported for both directions           2.34    2.34     0.5%   146.0%
  a straight line from first vertex to last        1.10    1.10    53.2%    15.7%
  rigid phase correlation, ROI never touched       0.00    0.00   100.0%   100.0%
  canonical guess, 1.0 and 0.5 um/s                1.00    0.50    57.4%    47.4%
  =============================================  ======  ======  =======  =======

The line divides the routes that measure from the routes that do not, and
:data:`SPEED_LIMIT` is set at 0.15 -- five times the worst reference variation
over seven seeds and three estimators (3.1%), and below every row under the line
on at least one direction.

Four things that table is load-bearing about:

**The trap reproduces.** ``<L><1/L>`` over the shipped ROI's segments is 1.207,
which is the prescreen's 1.21x arrived at independently -- and the ROI is not
drawn to produce it. Its vertices come from a rule (click again before the
chord bows 0.8 px off the filament, and at least every 120 px), which is what a
careful person does and which yields 16.5 px through the bends and 80.3 px along
the straights on its own. The measured damage is worse than 1.207 because a
distorted distance axis does not merely scale the slopes, it kinks every trace,
and a kinked trace has no single slope to find.

**Three estimators, not one.** A reference that only works under the estimator
it was tuned with has been tuned, so the arc-length route is measured three
ways that share no code -- shearing and summing (Radon), the angle of the 2-D
power spectrum, and plain cross-correlation at a lag. They agree to 1.2%.

**The ROI is a convenience, not the measurement.** A run that ignores it,
skeletonises the time average and traces its own path scores 0.8% -- so the
case is not a puzzle about reading `layer.data[0]`, and a run is free to
distrust the ROI it was handed.

**And one knob turned out not to matter.** Perpendicular averaging -- the
`linewidth`/`reduce_func` half of `profile_line`, and the thing that gives a
kymograph its SNR -- is worth nothing here: linewidth 1 scores the same as
linewidth 5. Sixty frames of coherent integration have already bought what
averaging five pixels would, so the fixture does not separate that decision and
this docstring should not imply it does.

The two speeds are 2.35 and 0.95 um/s, deliberately off the band a remembered
number lands on. Canonical fast anterograde transport is quoted at about 1 um/s
and retrograde at about half that; both have to be hit within 15% here, and the
row above shows what quoting them costs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ...agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_scalar,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

NAMESPACE = "kymograph-velocity"
CASE_ID = "two-directions-on-a-curve-the-user-drew"

# --- the acquisition --------------------------------------------------------

FIELD = 288
FRAMES = 60
DT_S = 0.2
PIXEL_UM = 0.11

#: The answer, and the reason it is these numbers rather than rounder ones: a
#: run that quotes canonical transport speeds from memory must fail, so neither
#: may sit where a remembered number lands and their ratio must not be tidy.
FORWARD_UM_S = 2.35
BACKWARD_UM_S = 0.95

#: Fraction of the true speed. Five times the worst spread across seven seeds and
#: three estimators, and under every wrong route on at least one direction.
SPEED_LIMIT = 0.15

TUBE_SIGMA = 2.2
CARGO_SIGMA = 1.5
TUBE_AMP = 1.0
CARGO_AMP = 0.28
BACKGROUND = 0.05
PHOTONS = 60.0
READ_NOISE = 1.2

#: Mean gap between cargo of each kind, in px along the filament.
FORWARD_SPACING = 55.0
BACKWARD_SPACING = 70.0
DOCKED_SPACING = 60.0

SEED = 11

#: The shape of the filament, at 448 px and scaled down -- long straight runs
#: and two tight bends, which is what makes an uneven hand-drawn ROI both
#: plausible and harmful.
WAYPOINTS = np.array(
    [
        [400.0, 40.0],
        [372.0, 190.0],
        [300.0, 262.0],
        [176.0, 250.0],
        [110.0, 300.0],
        [96.0, 404.0],
    ]
) * (FIELD / 448.0)

#: When the user clicks again: before the chord bows this far off the filament,
#: and at least this often regardless. The skew the equal-count trap needs is a
#: consequence of this rule, not an input to it -- which is the difference
#: between a fixture that reproduces a finding and one that stages it.
MAX_SAGITTA_PX = 0.8
MAX_SEGMENT_PX = 120.0


def _arclength(points: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def _resample(points: np.ndarray, n: int) -> np.ndarray:
    """*points* at *n* positions evenly spaced in arc length."""
    s = _arclength(points)
    want = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(want, s, points[:, k]) for k in range(2)])


def _centreline(n: int = 4000) -> np.ndarray:
    """A Catmull-Rom spline through the waypoints, evenly sampled in arc
    length -- so a cargo position in px along the curve is a lookup, and the
    truth is exact by construction rather than by fitting."""
    pad = np.vstack(
        [
            WAYPOINTS[0] + (WAYPOINTS[0] - WAYPOINTS[1]),
            WAYPOINTS,
            WAYPOINTS[-1] + (WAYPOINTS[-1] - WAYPOINTS[-2]),
        ]
    )
    dense = []
    for i in range(len(WAYPOINTS) - 1):
        p0, p1, p2, p3 = pad[i], pad[i + 1], pad[i + 2], pad[i + 3]
        t = np.linspace(0.0, 1.0, 400, endpoint=False)[:, None]
        dense.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    dense.append(WAYPOINTS[-1][None, :])
    return _resample(np.vstack(dense), n)


def _traced_roi(curve: np.ndarray) -> np.ndarray:
    """Where the clicks land: extend a segment until the curve bows too far off
    its chord, or the segment gets too long, then click."""
    idx = [0]
    i = 0
    while i < len(curve) - 1:
        best = i + 1
        j = i + 1
        while j < len(curve):
            chord = curve[j] - curve[i]
            span = float(np.linalg.norm(chord))
            direction = chord / max(span, 1e-9)
            normal = np.array([-direction[1], direction[0]])
            bow = float(np.abs((curve[i : j + 1] - curve[i]) @ normal).max())
            if bow > MAX_SAGITTA_PX or span > MAX_SEGMENT_PX:
                break
            best = j
            j += 1
        idx.append(best)
        i = best
    if idx[-1] != len(curve) - 1:
        idx.append(len(curve) - 1)
    return curve[idx].astype(float)


# --- the movie --------------------------------------------------------------


def _stamp(positions: np.ndarray, amps: np.ndarray, curve, s, sigma) -> np.ndarray:
    """Blobs of the given amplitudes at the given positions along the curve."""
    field = np.zeros((FIELD, FIELD), float)
    live = (positions >= 0.0) & (positions <= s[-1])
    if not live.any():
        return field
    xy = np.column_stack([np.interp(positions[live], s, curve[:, k]) for k in range(2)])
    rows = np.clip(np.round(xy[:, 0]).astype(int), 0, FIELD - 1)
    cols = np.clip(np.round(xy[:, 1]).astype(int), 0, FIELD - 1)
    np.add.at(field, (rows, cols), amps[live])
    return ndi.gaussian_filter(field, sigma) * (2 * np.pi * sigma**2)


class Transport:
    """Cargo moving both ways along a filament, in camera counts."""

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(SEED)
        curve = _centreline()
        s = _arclength(curve)
        total = float(s[-1])

        fwd_px = FORWARD_UM_S / PIXEL_UM * DT_S
        bwd_px = BACKWARD_UM_S / PIXEL_UM * DT_S

        # Irregular gaps, not `arange`. Evenly spaced cargo is a grating, and a
        # grating puts a false peak into any correlation-based estimator
        # wherever the travel over the lag and the cargo spacing differ by a
        # whole gap -- measured at -2.6 px/frame on 4 of 6 seeds before this,
        # which would have been scored as the fixture separating the routes.
        def spread(lo: float, hi: float, gap: float) -> np.ndarray:
            pos = [lo + rng.uniform(0.0, gap)]
            while pos[-1] < hi:
                pos.append(pos[-1] + gap * rng.uniform(0.45, 1.55))
            return np.array(pos)

        # Enough cargo off each end that new cargo keeps arriving all movie.
        forward = spread(-fwd_px * FRAMES, total, FORWARD_SPACING)
        backward = spread(0.0, total + bwd_px * FRAMES, BACKWARD_SPACING)
        docked = spread(0.0, total, DOCKED_SPACING)

        fwd_amp = CARGO_AMP * rng.uniform(0.7, 1.4, size=forward.shape)
        bwd_amp = CARGO_AMP * rng.uniform(0.7, 1.4, size=backward.shape)
        dock_amp = CARGO_AMP * rng.uniform(0.8, 1.6, size=docked.shape)

        # The filament itself, varicose rather than a flat ridge -- so removing
        # the stationary component is a subtraction along time and not a
        # constant, and a run that removes a constant is left with structure.
        thickness = (
            1.0
            + 0.35 * np.sin(2 * np.pi * s / 190.0 + 1.1)
            + 0.2 * np.sin(2 * np.pi * s / 71.0)
        )
        tube = _stamp(s, TUBE_AMP * thickness, curve, s, TUBE_SIGMA)
        tube *= float(np.max(TUBE_AMP * thickness)) / tube.max()

        movie = np.empty((FRAMES, FIELD, FIELD), np.uint16)
        for t in range(FRAMES):
            frame = tube.copy()
            frame += _stamp(forward + t * fwd_px, fwd_amp, curve, s, CARGO_SIGMA)
            frame += _stamp(backward - t * bwd_px, bwd_amp, curve, s, CARGO_SIGMA)
            frame += _stamp(docked, dock_amp, curve, s, CARGO_SIGMA)
            counts = np.clip(frame + BACKGROUND, 0.0, None) * PHOTONS
            counts = rng.poisson(counts) + rng.normal(
                scale=READ_NOISE, size=counts.shape
            )
            movie[t] = np.clip(np.round(counts), 0, 65535)

        roi = _traced_roi(curve)
        assert roi.shape[1] == 2 and len(roi) > 6, roi.shape
        # The ROI runs the same way the curve does, so "toward the last vertex"
        # and "the direction cargo labelled forward moves" are one direction.
        assert np.allclose(roi[0], curve[0]) and np.allclose(roi[-1], curve[-1])
        # And it stays on the filament: a ROI that wanders off the ridge would
        # be scoring the ROI, not the run.
        fine = _resample(roi, 6000)
        off = np.linalg.norm(fine[:, None, :] - curve[None, ::4, :], axis=2).min(1)
        assert off.max() < 1.2, off.max()

        return Fixture(
            provenance=(
                f"synthetic: {FRAMES} frames at {DT_S} s, {PIXEL_UM} um/px, cargo "
                f"at {FORWARD_UM_S} and {BACKWARD_UM_S} um/s along a "
                f"{total * PIXEL_UM:.1f} um filament, Poisson at {PHOTONS:g} "
                f"counts, seed {SEED}"
            ),
            about=(
                "Two constant transport speeds, in opposite directions, along a "
                "curved filament with a hand-traced ROI whose segments are 16.5 "
                "to 80.3 px long. What separates the routes is whether distance "
                "along the ROI is measured in arc length or in samples, and "
                "whether the stationary filament and its docked cargo are taken "
                "out before a slope is looked for."
            ),
            data={"transport": movie, "roi": roi},
            truth={
                "forward_um_per_s": FORWARD_UM_S,
                "backward_um_per_s": BACKWARD_UM_S,
            },
            tolerance={"speed_error": SPEED_LIMIT},
        )


# --- scoring ----------------------------------------------------------------


def _verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    limit = float(fixture.tolerance.get("speed_error", SPEED_LIMIT))
    wanted = {
        "speed_forward_um_per_s": float(fixture.truth["forward_um_per_s"]),
        "speed_backward_um_per_s": float(fixture.truth["backward_um_per_s"]),
    }

    got: dict[str, float | None] = {}
    why: dict[str, str] = {}
    for key in wanted:
        value, reason = read_scalar(attempt, key)
        if value is not None and value <= 0.0:
            value, reason = None, f"`{key}` is not a positive speed ({value:.4g})"
        got[key], why[key] = value, reason

    # Both or neither. Scored separately -- the two directions are two
    # measurements and a run can get one of them right for the wrong reason --
    # but a run that reports only the fast population has not answered, and
    # without this metric it would score green on the half it did.
    unusable = [reason for reason in why.values() if reason]
    nothing_at_all = not any(v is not None for v in got.values()) and not attempt.arrays
    metrics = [
        Metric(
            "deliverables_unusable",
            None if nothing_at_all else float(len(unusable)),
            0.5,
            f" of 2 -- {'; '.join(unusable)}" if unusable else " of 2",
            unavailable="the run left nothing to score" if nothing_at_all else "",
        )
    ]

    detail = {}
    for key, want in wanted.items():
        name = key.removeprefix("speed_").removesuffix("_um_per_s") + "_speed_error"
        value = got[key]
        metrics.append(
            Metric(
                name,
                None if value is None else abs(value - want) / want,
                limit,
                " of the true speed",
                unavailable=why[key],
            )
        )
        detail[key] = value

    return Outcome(fixture, attempt, metrics, detail=detail)


def _save_artifacts(outcome: Outcome, root: Path) -> None:
    """The kymograph the reference route would have built, so a run that
    reported the wrong speed can be looked at rather than guessed at.

    Which of the two failures happened is visible in it at a glance and in
    nothing else the harness keeps: traces that are kinked at the ROI vertices
    say the distance axis is wrong, and a field of horizontal stripes says the
    stationary component is still there.
    """
    movie = np.asarray(outcome.fixture.data["transport"], float)
    roi = np.asarray(outcome.fixture.data["roi"], float)
    path = _resample(roi, max(2, int(round(_arclength(roi)[-1]))))

    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(5) - 2.0

    kymo = np.zeros((len(movie), len(path)))
    for i, frame in enumerate(movie):
        acc = sum(
            ndi.map_coordinates(frame, (path + off * normal).T, order=1, mode="nearest")
            for off in offsets
        )
        kymo[i] = acc / len(offsets)

    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "kymograph.npz",
        kymograph=kymo,
        moving=kymo - np.median(kymo, axis=0, keepdims=True),
        path=path,
        roi=roi,
        px_per_frame=np.array(
            [
                FORWARD_UM_S / PIXEL_UM * DT_S,
                -BACKWARD_UM_S / PIXEL_UM * DT_S,
            ]
        ),
    )


# --- the task ---------------------------------------------------------------

TASK = f"""
A {FRAMES}-frame time lapse of one neurite is open in napari as the Image layer
`transport`. Cargo moves along it in both directions at the same time. Each
frame is {FIELD}x{FIELD} pixels of raw camera counts.

The acquisition:

- {PIXEL_UM} microns per pixel
- {DT_S} seconds between frames

A Shapes layer `traced_neurite` holds one path: the neurite, traced by hand in
`transport` coordinates. Its vertices are (row, column) in pixels, in order,
and they run from one end of the neurite to the other.

Call the direction from the *first* vertex of that path toward the *last* one
**forward**, and the other one **backward**.

Work out how fast the cargo moves, and leave two numbers in the kernel:

- `speed_forward_um_per_s` -- the speed of the cargo travelling forward, in
  microns per second, as a positive number.
- `speed_backward_um_per_s` -- the speed of the cargo travelling backward, also
  in microns per second and also positive.

Both are single speeds: within each direction the cargo all moves at the same
rate, and it does not stop or reverse.

The person who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

Both names must be bound in the kernel namespace when you finish.
""".strip()

#: What is *not* here: nothing about kymographs, line profiles, arc length,
#: slopes, Radon or correlation, and no number that is or bounds an answer.
#: This person ran the experiment.
NEUROBIOLOGIST = Persona(
    name="the neurobiologist who acquired this",
    background=(
        "You imaged this neurite yourself and traced the ROI, and you are "
        "sitting with the analyst. You answer what you are asked, plainly and "
        "briefly. You do not volunteer analysis advice and you do not suggest "
        "methods -- you know the sample and the microscope, not the maths. If "
        "you are asked something you would not know from having run the "
        "experiment, say so."
    ),
    facts={
        "pixel size": "0.11 microns per pixel, on a 60x objective.",
        "frame interval": (
            "0.2 seconds, and it is a stream acquisition so the interval is even."
        ),
        "the sample": (
            "a cultured neuron, live, with a fluorescent cargo marker. The "
            "neurite runs across most of the field."
        ),
        "the ROI": (
            "I traced it myself over the neurite, clicking along it. I clicked "
            "more often round the bends than on the straight stretches."
        ),
        "the bright line": (
            "the neurite itself is labelled too, so it is bright everywhere "
            "along its length, not only where the cargo is."
        ),
        "cargo that does not move": (
            "some of it is parked and stays put for the whole movie. That is "
            "normal and I am not asking about it."
        ),
        "which way is which": (
            "I traced from the cell body outward, so the first vertex is the "
            "proximal end."
        ),
        "how fast it goes": (
            "that is what I want measured -- I do not have a number for it, "
            "which is why we are doing this."
        ),
        "whether the movie was processed": (
            "no, it is raw camera counts straight off the detector."
        ),
        "bleaching": (
            "there is not much over a movie this short; I did not correct for it."
        ),
    },
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=NEUROBIOLOGIST,
    fixture=Procedural(Transport()),
    layers=(
        Layer("transport", "transport", "image"),
        # A Shapes path, because that is what a traced neurite is in napari and
        # what the vertex spacing this case turns on is a property of. Handed
        # the same vertices as a Points layer it would be a different route,
        # and handed a resampled array it would be no route at all.
        Layer("traced_neurite", "roi", "path"),
    ),
    collect={
        "speed_forward_um_per_s": "speed_forward_um_per_s",
        "speed_backward_um_per_s": "speed_backward_um_per_s",
    },
    score=_verify,
    save_artifacts=_save_artifacts,
    # The task is self-sufficient, so asking is neither rewarded nor punished.
    # These are here so that a run which does ask is not misled: the pixel size
    # and the interval must come back the same as the task states them.
    persona_must_know=("0.11 microns per pixel", "0.2 seconds", "parked"),
    persona_must_not_know=(
        "kymograph",
        "arc length",
        "arclength",
        "profile_line",
        "radon",
        "slope",
        "resample",
        "cross-correlation",
        "fourier",
        "micron per second",
        "microns per second",
        "um/s",
    ),
)
