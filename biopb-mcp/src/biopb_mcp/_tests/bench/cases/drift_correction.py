"""`drift-correction` as benchmark data: which channel is the structural one?

The skill takes a time-lapse whose field of view has moved and returns one where
it has not. That is a rare thing in this domain — an *exactly* knowable answer:
shift one image along a trajectory you chose, and both the trajectory and the
un-drifted image are ground truth to machine precision.

The withheld fact is step 2's question. The body asks the user two things in one
breath — did the *field* move or the *objects*, and which channel is
`REF_CHANNEL` — and here they are the same fact:

===========  ==========================================================
channel 0    bright puncta that carry the stage drift *and* crawl on
             their own — a vesicle or reporter channel
channel 1    a dim, structurally stable field that moves only with the
             stage — a membrane or transmitted-light channel
===========  ==========================================================

The movie arrives with no channel names, and every heuristic available to a run
that does not ask points the wrong way: channel 0 is five times the contrast, an
order of magnitude the peak intensity, and far the more obviously feature-rich
thing to register on. Register there and the objects' common motion is
indistinguishable from stage drift, so it is silently added to the trajectory —
the fit does not fail, it succeeds at measuring the wrong thing.

**A caveat this fixture has already earned** (`biopb-mcp/docs/skills.md` §10c). A
capable agent recovered the structural channel anyway, by registering on both
and keeping the self-consistent one. Designing a fixture so the *heuristics its
author thought of* point the wrong way is not the same as making the fact
unobtainable, and the strongest withheld facts are categorically absent from the
pixels — a unit, a scale, a provenance. Read this case's `skill+silent` arm with
that in mind; `calibrated-measurements` is the cleaner example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ...agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    save_png,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

SKILL = "drift-correction"

# Set from measurement, not from taste. Over six seeds of this construction, a
# run told which channel is structural lands at 0.003-0.010 px RMS and a
# residual ratio of 0.0005-0.0016, while registering on brightness -- or
# averaging the channels, which the Parameters table forbids by name -- lands at
# 2.8-4.1 px and 0.35-0.54. These sit in the gap, and the narrowest margin is
# the residual ratio, where the mildest failure still clears the limit by 3.5x.
#
# Measured with the skill's own recipe (StackReg TRANSLATION,
# reference="previous"), which is also why the passing numbers are not at
# machine precision: cropping a padded canvas means consecutive frames no longer
# share identical content, so registration has real work to do. That costs an
# order of magnitude of precision and buys a fixture with no fabricated pixels
# in it.
TOLERANCE = {
    "trajectory_rms_px": 0.5,
    "trajectory_max_err_px": 1.0,
    "residual_ratio": 0.10,
}

#: Shared by every object, in px/frame. This is what makes the mistake
#: systematic rather than a lucky draw: a random-only velocity field averages
#: toward zero over 60 objects, and at one point in tuning that let a seed slip
#: back inside tolerance. Cells crawling up a gradient is the ordinary reading.
COMMON_VELOCITY_PX = 0.25

#: Per-object scatter about that common motion, in px/frame.
SPREAD_VELOCITY_PX = 0.5

#: Channel 1's amplitude relative to the reporter channel. Dim enough that
#: "register on the bright one" is the tempting call.
STRUCTURAL_DIM = 0.35

BACKGROUND = 100.0

#: Canvas kept outside the field of view, on top of the largest offset, so that
#: what drifts into frame is real sample rather than something the renderer
#: invented. Covers the cubic-spline support of the shift and the reach of
#: `_blobby_field`'s Gaussian at the canvas edge;
#: `test_the_drifted_movie_invents_no_pixels` measures that it is enough.
PAD_MARGIN = 16


# --- the fixture -----------------------------------------------------------


def _blobby_field(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Sparse bright objects on a flat background — a fluorescence field, and
    structurally what registration is easy on: isolated high-frequency features
    with unambiguous positions."""
    rng = np.random.default_rng(seed)
    seeds = (rng.random(shape) < 0.006).astype(np.float32)
    return (BACKGROUND + 3000.0 * ndimage.gaussian_filter(seeds, 3.0)).astype(
        np.float32
    )


def _trajectory(n_frames: int, per_frame_px: float, seed: int) -> np.ndarray:
    """A smooth, slowly turning drift of `per_frame_px` per frame, plus jitter.

    Deliberately not a straight line. A pure ramp is separable in y and x and
    would let an estimator that had collapsed one axis still look plausible on
    the other; a turning trajectory couples them. The jitter stays small enough
    to be "smooth and slow" in the sense step 4 checks for — this is a
    *correctable* movie, not a pathological one.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    heading = 0.6 + 0.35 * np.sin(t / 7.0)
    dy = np.cumsum(per_frame_px * np.sin(heading))
    dx = np.cumsum(per_frame_px * np.cos(heading))
    offsets = np.stack([dy - dy[0], dx - dx[0]], axis=1)
    offsets += rng.normal(0.0, 0.02, offsets.shape)
    offsets[0] = 0.0  # frame 0 is the reference, exactly
    return offsets


def _puncta(positions: np.ndarray, amplitudes: np.ndarray, shape, sigma=2.0):
    """Render point objects as Gaussian spots. Positions may be sub-pixel; they
    are rounded to the nearest sample, which is well below the tolerances here
    and keeps the render cheap.

    An object off the canvas is dropped, not clamped to the edge. Clamping piles
    every escapee onto one border row, and since they all escape in the
    direction of the motion, that pile-up is a picture of the trajectory.
    """
    img = np.zeros(shape, dtype=np.float32)
    yy = np.round(positions[:, 0]).astype(int)
    xx = np.round(positions[:, 1]).astype(int)
    inside = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    np.add.at(img, (yy[inside], xx[inside]), amplitudes[inside])
    return ndimage.gaussian_filter(img, sigma)


@dataclass(frozen=True)
class AmbiguousChannels:
    """A two-channel movie where only a person knows which channel is which.

    `ndimage.shift(base, (dy, dx))` puts a feature at ``(y, x)`` in `base` at
    ``(y + dy, x + dx)``, so ``offsets[t]`` is the displacement of frame *t*
    relative to frame 0 — the same sense and sign the task asks for.

    Both channels are rendered on a canvas larger than `shape` and cropped to a
    fixed window in the middle of it, because a moving stage does not create
    pixels: it reveals sample that was outside the field of view. Shifting a
    frame-sized image instead leaves a border of pixels the renderer invented,
    and their width is the shift — the trajectory, legible without registering
    anything, and a band of flat correlated structure sitting in the data the
    run *does* register on. See `test_the_drifted_movie_invents_no_pixels`.
    """

    per_frame_px: float = 1.7
    n_frames: int = 24
    n_objects: int = 60
    #: The field of view, not the canvas — objects are drawn at this density
    #: over the whole padded canvas so the count in frame stays about `n_objects`.
    shape: tuple[int, int] = (192, 192)
    seed: int = 0

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed + 100)
        offsets = _trajectory(self.n_frames, self.per_frame_px, self.seed + 1)

        pad = int(np.ceil(np.abs(offsets).max())) + PAD_MARGIN
        canvas = (self.shape[0] + 2 * pad, self.shape[1] + 2 * pad)
        window = (slice(pad, pad + self.shape[0]), slice(pad, pad + self.shape[1]))

        # Channel 1 -- the structural one. A pure shift of one image, so its
        # un-drifted state is ground truth to machine precision.
        field = _blobby_field(self.seed, canvas)
        field = (field - BACKGROUND) * STRUCTURAL_DIM + BACKGROUND
        structural = np.array(
            [ndimage.shift(field, o, order=3, mode="nearest")[window] for o in offsets]
        )
        stable = field[window]

        # Channel 0 -- bright objects that both ride the stage and move. Drawn
        # across the whole canvas, so objects leaving the frame are replaced by
        # others arriving from outside it rather than by empty background.
        n_objects = int(round(self.n_objects * np.prod(canvas) / np.prod(self.shape)))
        start = rng.uniform(0.0, 1.0, size=(n_objects, 2)) * np.asarray(canvas, float)
        amplitudes = rng.uniform(4000.0, 9000.0, size=n_objects)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        common = COMMON_VELOCITY_PX * np.array([np.sin(heading), np.cos(heading)])
        velocity = common + rng.normal(0.0, SPREAD_VELOCITY_PX, size=(n_objects, 2))
        reporter = np.array(
            [
                _puncta(start + velocity * t + o, amplitudes, canvas)[window]
                + BACKGROUND
                for t, o in enumerate(offsets)
            ]
        )

        movie = np.stack([reporter, structural], axis=1).astype(np.float32)
        drift = float(np.hypot(*offsets[-1]))
        objects = float(np.hypot(*common) * (self.n_frames - 1))
        return Fixture(
            provenance=(
                f"procedural: 2 channels, seed {self.seed}, {self.n_frames} frames, "
                f"{drift:.1f} px of stage drift, {self.n_objects} objects also "
                f"moving {objects:.1f} px of their own"
            ),
            about=(
                f"The stage drifts {drift:.0f} px while the objects in the bright "
                f"channel crawl {objects:.0f} px of their own. Registering on "
                "brightness measures the sum of the two and reports success."
            ),
            data={"movie": movie},
            truth={
                "offsets": offsets,
                "stable": stable,
                # The private fact. It is stripped from `data`, the respondent
                # holds it, and the verifier reads it to know what to score.
                "structural_channel": 1,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _structural(fixture: Fixture, movie) -> np.ndarray | None:
    """The one channel this fixture's truth describes, out of a `(T, C, Y, X)`
    stack. Anything with fewer dimensions is already that channel.

    The verifier may read this and the run may not: *which* channel is
    structural is the fact the fixture strips, and reading it off the truth is
    exactly the asymmetry. A run that picked the wrong channel is still scored
    on this one, which is what makes the mistake visible — its corrected
    structural channel carries whatever the other channel's motion told it to do.
    """
    if movie is None:
        return None
    arr = np.asarray(movie)
    if arr.ndim < 4:
        return arr
    channel = fixture.truth.get("structural_channel")
    if channel is None:
        raise ValueError(
            f"{fixture.label} is multi-channel but its truth does not say which "
            "channel is structural, so nothing here can be scored"
        )
    return arr[:, int(channel)]


def _margin(fixture: Fixture) -> int:
    """Pixels to ignore at each border.

    Registration slides data off one side and invents nothing on the other, so
    the margins hold extrapolated pixels that belong to no frame — step 6's
    point. Scoring them would charge every run for the same artifact.
    """
    offsets = fixture.truth.get("offsets")
    if offsets is None:
        return 8
    return int(np.ceil(np.abs(np.asarray(offsets, float)).max())) + 4


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Two metric families, with different requirements on the truth — which is
    the whole reason the fixture is substitutable:

    ``trajectory_*`` need ``truth["offsets"]``. A curated movie can carry those,
    measured off a fiducial or a bead, so these survive the substitution.

    ``residual_ratio`` needs ``truth["stable"]``, the un-drifted image. It is
    the stronger measurement — it never touches a registration estimator, so it
    cannot share a systematic error with the run — and it is the one real data
    cannot supply, because no un-drifted acquisition exists. It reports as
    unavailable there rather than as a pass.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    truth_offsets = fixture.truth.get("offsets")
    got_offsets = attempt.arrays.get("offsets")
    # A run binding `offsets` to something of the wrong shape is a result that
    # cannot be scored, which is not the same as an error in the verifier.
    if (
        got_offsets is not None
        and truth_offsets is not None
        and np.asarray(got_offsets).shape != np.asarray(truth_offsets).shape
    ):
        got_offsets = None
    if truth_offsets is None or got_offsets is None:
        why = (
            "the fixture carries no annotated trajectory"
            if truth_offsets is None
            else "the run reported no per-frame offsets of the expected shape"
        )
        metrics += [
            Metric(
                "trajectory_rms_px", None, limits["trajectory_rms_px"], unavailable=why
            ),
            Metric(
                "trajectory_max_err_px",
                None,
                limits["trajectory_max_err_px"],
                unavailable=why,
            ),
        ]
    else:
        truth_offsets = np.asarray(truth_offsets, float)
        got_offsets = np.asarray(got_offsets, float)
        # Registration is defined only up to its reference frame, so compare
        # both series relative to frame 0. A constant offset between them is a
        # different origin, not an error.
        error = (got_offsets - got_offsets[0]) - (truth_offsets - truth_offsets[0])
        per_frame = np.hypot(error[:, 0], error[:, 1])
        metrics += [
            Metric(
                "trajectory_rms_px",
                float(np.sqrt((per_frame**2).mean())),
                limits["trajectory_rms_px"],
                unit=" px",
            ),
            # RMS hides a single lost frame in the average; the max is the
            # quantity step 4 actually stops the workflow on.
            Metric(
                "trajectory_max_err_px",
                float(per_frame.max()),
                limits["trajectory_max_err_px"],
                unit=" px",
            ),
        ]
        # The per-frame series itself goes to trajectory.csv, not here --
        # summary.json is meant to be read at a glance.
        detail["worst_frame"] = int(per_frame.argmax())

    stable = fixture.truth.get("stable")
    corrected = _structural(fixture, attempt.arrays.get("corrected"))
    movie = _structural(fixture, fixture.data.get("movie"))
    mismatched = (
        corrected is not None
        and movie is not None
        and (corrected.ndim != movie.ndim or corrected.shape[-2:] != movie.shape[-2:])
    )
    if mismatched:
        metrics.append(
            Metric(
                "residual_ratio",
                None,
                limits["residual_ratio"],
                unavailable=(
                    f"the run's corrected movie is {corrected.shape}, which "
                    f"cannot be compared with the input {movie.shape}"
                ),
            )
        )
        detail["corrected_shape"] = list(corrected.shape)
        return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)
    if stable is None or corrected is None or movie is None:
        why = (
            "no un-drifted reference exists for this fixture"
            if stable is None
            else "the run produced no corrected movie"
        )
        metrics.append(
            Metric("residual_ratio", None, limits["residual_ratio"], unavailable=why)
        )
    else:
        margin = _margin(fixture)
        inner = (slice(margin, -margin), slice(margin, -margin))
        ref = np.asarray(stable, float)[inner]
        before = np.median(
            [np.abs(np.asarray(f, float)[inner] - ref).mean() for f in movie]
        )
        after = np.median(
            [np.abs(np.asarray(f, float)[inner] - ref).mean() for f in corrected]
        )
        metrics.append(
            Metric(
                "residual_ratio",
                float(after / before) if before > 0 else float("inf"),
                limits["residual_ratio"],
                unit="x",
            )
        )
        detail["mean_abs_error_before"] = float(before)
        detail["mean_abs_error_after"] = float(after)
        detail["margin_px"] = margin

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half of §2: what the number means, for a human to page
    through. Never raises — an artifact explains a failure, it does not cause
    one."""
    fixture, attempt = outcome.fixture, outcome.attempt
    movie = _structural(fixture, fixture.data.get("movie"))
    corrected = _structural(fixture, attempt.arrays.get("corrected"))

    # The two images that carry the answer: last minus first, before and after.
    # A stabilised movie is near-flat; a failed one keeps the whole structure.
    # Both are scaled to the *raw* difference's range so the pair can be read
    # side by side -- see save_png.
    if movie is not None:
        raw_diff = np.abs(np.asarray(movie[-1], float) - np.asarray(movie[0], float))
        scale = float(raw_diff.max()) or 1.0
        save_png(movie[0], where / "raw-first.png")
        save_png(movie[-1], where / "raw-last.png")
        save_png(raw_diff, where / "raw-difference.png", vmax=scale)
        if corrected is not None and corrected.shape[-2:] == movie.shape[-2:]:
            save_png(corrected[-1], where / "corrected-last.png")
            save_png(
                np.abs(
                    np.asarray(corrected[-1], float) - np.asarray(corrected[0], float)
                ),
                where / "corrected-difference.png",
                vmax=scale,
            )

    truth_offsets = fixture.truth.get("offsets")
    got_offsets = attempt.arrays.get("offsets")
    if truth_offsets is None or got_offsets is None:
        return
    truth_offsets = np.asarray(truth_offsets, float)
    got_offsets = np.asarray(got_offsets, float)
    if got_offsets.shape != truth_offsets.shape:
        return
    rows = ["frame,truth_dy,truth_dx,got_dy,got_dx,error_px"]
    error = (got_offsets - got_offsets[0]) - (truth_offsets - truth_offsets[0])
    for i, (t, g, e) in enumerate(zip(truth_offsets, got_offsets, error, strict=True)):
        rows.append(
            f"{i},{t[0]:.4f},{t[1]:.4f},{g[0]:.4f},{g[1]:.4f},{np.hypot(*e):.4f}"
        )
    (where / "trajectory.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A two-channel time-lapse is open in the napari viewer as the layer `timelapse`.
Its axes are (T, C, Y, X) = (24, 2, 192, 192), float32. There is no tensor
server in this session, so that layer is the only source of pixels.

The field of view moved during the acquisition, and I need that corrected
before anything is measured off this movie.

The microscopist who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  offsets    a (24, 2) float array of (dy, dx), relative to frame 0.
             SIGN CONVENTION: offsets[t] is the displacement frame t *carries* —
             a feature at (y, x) in frame 0 appears at (y + dy, x + dx) in
             frame t. It is NOT the shift needed to correct frame t, which is
             the negative of this.
  corrected  the drift-corrected movie, the same shape as the input

Those two names are how your result is collected, so bind them exactly.
"""

#: Note what is *not* here: nothing about registration, nothing about
#: `reference="previous"`, no mention of a structural channel being the right
#: one to register on. This person knows their sample, not the procedure. A
#: persona that knew the skill could answer a question the agent never asked
#: properly, and the numeric result would stop meaning what it appears to.
MICROSCOPIST = Persona(
    name="microscopist-two-channel",
    facts={
        "what channel 0 is": (
            "the vesicle reporter — bright puncta, and they move around on "
            "their own inside the cells, that motion is the thing I am "
            "studying"
        ),
        "what channel 1 is": (
            "the membrane marker — dim, but it is just the cell outlines and "
            "they do not go anywhere"
        ),
        "did the field move or the objects": (
            "both, and that is my problem: the stage drifted over the run AND "
            "the vesicles are moving. The drift is the part I want gone"
        ),
        "how the movie was acquired": (
            "24 frames, one every 30 seconds, on a spinning disk. The stage "
            "was not touched during the run"
        ),
        "why it matters": (
            "I need to track individual vesicles, so the frame has to hold still first"
        ),
    },
    background=(
        "A two-channel time-lapse of cultured cells. You are happy to answer "
        "questions about the sample and the acquisition."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="two-channels-one-structural",
    task=TASK,
    persona=MICROSCOPIST,
    fixture=Procedural(AmbiguousChannels()),
    layers=(Layer("timelapse", "movie"),),
    collect={"offsets": "offsets", "corrected": "corrected"},
    score=verify,
    save_artifacts=save_artifacts,
    # It must be able to answer: the fixture withholds which channel is
    # structural, and this person knows both channels and that the stage moved.
    persona_must_know=("channel 0", "channel 1", "move", "drift"),
    # And it must not know the procedure — only the sample.
    persona_must_not_know=(
        "reference=",
        "stackreg",
        "register",
        "phase_cross_correlation",
        "structural channel",
    ),
)
