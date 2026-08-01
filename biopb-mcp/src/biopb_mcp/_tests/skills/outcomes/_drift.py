"""Outcome fixtures and verifier for the `drift-correction` skill.

The skill takes a time-lapse whose field of view has moved and returns one where
it has not. That is a rare thing in this domain: an *exactly* knowable answer.
Shift a single image by a trajectory you chose, and both the trajectory and the
un-drifted image are ground truth by construction, to machine precision.

**What this fixture deliberately does not cover.** The body's other load-bearing
instruction — estimate on one structural channel and apply the transforms to all
of them — needs a *choice* in order to be wrong, and a reference implementation
makes the right one by construction. Only an agent can get it wrong, so it
belongs to the interaction tier (§6), where which channel is structural is a
private fact the respondent holds. A second channel here would be scored data
that no subject could fail.

That tier's fixture lives in :mod:`._drift_channels` and is scored by the
verifier here, unchanged. All this module lends it is :func:`_structural`: a
multi-channel movie is collapsed to the channel the truth is about, and every
measurement below is then the same single-channel comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ._outcome import (
    Attempt,
    Fixture,
    Kind,
    Metric,
    Outcome,
    Tier,
    register,
    register_curated,
    save_png,
)

SKILL = "drift-correction"

# Set from measurement, not from taste. Across the three synthetic cases the
# procedures the body prescribes land at 0.000-0.24 px RMS and a residual ratio
# of 0.0001-0.042, while every mistake the body warns about lands at 5.4-53 px
# and 0.29-1.00. These sit in the gap, with roughly 2x headroom over the worst
# correct run and 5x under the best failure -- wide enough that a small upstream
# change is not a red suite, narrow enough that a lost registration cannot pass.
TOLERANCE = {
    "trajectory_rms_px": 0.5,
    "trajectory_max_err_px": 1.0,
    "residual_ratio": 0.10,
}


# --- the synthetic fixtures ------------------------------------------------
#
# Names without a leading underscore here are this skill's shared fixture
# vocabulary: `_drift_channels` builds its movies out of them. The underscored
# ones are private to this module because nothing else uses them, which is the
# convention meaning what it says rather than being tolerated — `_smooth_field`
# sitting beside a public `blobby_field` is that distinction, not an oversight.


def blobby_field(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Sparse bright objects on a flat background — a fluorescence field.

    Structurally what registration is easy on: isolated high-frequency features
    with unambiguous positions.
    """
    rng = np.random.default_rng(seed)
    seeds = (rng.random(shape) < 0.006).astype(np.float32)
    return (100.0 + 3000.0 * ndimage.gaussian_filter(seeds, 3.0)).astype(np.float32)


def _smooth_field(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Low-frequency, low-contrast texture — a brightfield or a badly exposed
    channel. The condition the skill's failure table names as where a
    frequency-domain estimator loses the true peak."""
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.random(shape).astype(np.float32), 12.0)
    base -= base.min()
    base /= base.max()
    return (1000.0 + 20.0 * base).astype(np.float32)


def trajectory(n_frames: int, per_frame_px: float, seed: int) -> np.ndarray:
    """A smooth, slowly turning drift of `per_frame_px` per frame, plus jitter.

    Deliberately not a straight line. A pure ramp is separable in y and x and
    would let an estimator that had collapsed one axis still look plausible on
    the other; a turning trajectory couples them. The jitter is small enough to
    stay "smooth and slow" in the sense step 4 checks for — this fixture is a
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


@dataclass(frozen=True)
class SyntheticDrift:
    """A movie built by shifting one image along a known trajectory.

    `ndimage.shift(base, (dy, dx))` puts a feature at ``(y, x)`` in `base` at
    ``(y + dy, x + dx)``, so ``offsets[t]`` is the displacement of frame *t*
    relative to frame 0 — the same sense and sign a subject must report.
    """

    case_id: str
    about: str
    per_frame_px: float
    texture: str = "blobby"
    n_frames: int = 24
    shape: tuple[int, int] = (192, 192)
    seed: int = 0
    skill_id: str = SKILL
    kind: Kind = "synthetic"
    tier: Tier = "outcome"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def build(self) -> Fixture:
        make = blobby_field if self.texture == "blobby" else _smooth_field
        base = make(self.seed, self.shape)
        offsets = trajectory(self.n_frames, self.per_frame_px, self.seed + 1)
        movie = np.array(
            [ndimage.shift(base, o, order=3, mode="nearest") for o in offsets],
            dtype=np.float32,
        )
        total = float(np.hypot(*offsets[-1]))
        return Fixture(
            skill_id=self.skill_id,
            case_id=self.case_id,
            kind="synthetic",
            provenance=(
                f"procedural: {self.texture} field, seed {self.seed}, "
                f"{self.n_frames} frames, {self.per_frame_px} px/frame, "
                f"{total:.1f} px total"
            ),
            about=self.about,
            data={"movie": movie},
            truth={"offsets": offsets, "stable": base},
            tolerance=dict(TOLERANCE),
        )


SLOW = register(
    SyntheticDrift(
        case_id="blobs-slow",
        per_frame_px=1.7,
        about=(
            "1.7 px/frame over 24 frames -- the rate at which the body reports "
            "reference='first' beginning to lose lock. The ordinary case."
        ),
    )
)

FAST = register(
    SyntheticDrift(
        case_id="blobs-fast",
        per_frame_px=4.0,
        about=(
            "4.0 px/frame, 91 px total. Past any local optimiser's capture "
            "range against frame 0, which is what makes step 3 a requirement "
            "rather than a preference."
        ),
    )
)

SMOOTH = register(
    SyntheticDrift(
        case_id="smooth-low-contrast",
        per_frame_px=1.7,
        texture="smooth",
        about=(
            "The same drift on a low-contrast, low-frequency field. Separates "
            "the two methods: a pyramid optimiser is unbothered, a "
            "frequency-domain one is not."
        ),
    )
)

# Real acquisitions, if this machine has any. Registers nothing when the tree is
# absent, which is the normal case -- see _outcome.register_curated.
CURATED = register_curated(SKILL)


# --- the subjects ----------------------------------------------------------
#
# Reference implementations of what the body says, and of what it warns
# against. Each returns the trajectory it recovered and the movie it produced,
# which is the same pair an agent run would be scraped for.


def undo_offsets(movie: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Undo `offsets` frame by frame — the transform-stack step, by hand."""
    return np.array(
        [
            ndimage.shift(f, -o, order=3, mode="nearest")
            for f, o in zip(movie, offsets, strict=True)
        ]
    )


def run_stackreg(movie: np.ndarray, reference: str) -> Attempt:
    from pystackreg import StackReg

    sr = StackReg(StackReg.TRANSLATION)
    with warnings.catch_warnings():
        # pystackreg guesses a time axis from per-axis variability and warns
        # when its guess differs from the axis given. On a square fixture that
        # guess is noise, and the axis here is not in doubt.
        warnings.simplefilter("ignore", UserWarning)
        tmats = sr.register_stack(movie, reference=reference)
        corrected = sr.transform_stack(movie, tmats=tmats)
    # Step 4's indexing claim, which the contract layer pins independently.
    offsets = np.array([(m[1, 2], m[0, 2]) for m in tmats])
    return Attempt(
        subject=f"stackreg-{reference}",
        arrays={"offsets": offsets, "corrected": np.asarray(corrected)},
        notes=f"pystackreg TRANSLATION, reference={reference!r}",
    )


def _cross_correlation(movie: np.ndarray, normalization: str | None) -> Attempt:
    from skimage.registration import phase_cross_correlation

    offsets = [np.zeros(2)]
    for frame in movie[1:]:
        # Returns the shift that registers `frame` onto `movie[0]`, i.e. the
        # negative of the displacement the frame carries.
        shift, _, _ = phase_cross_correlation(
            movie[0], frame, upsample_factor=20, normalization=normalization
        )
        offsets.append(-np.asarray(shift, dtype=float))
    offsets = np.asarray(offsets)
    return Attempt(
        subject=f"cross-correlation-{normalization or 'none'}",
        arrays={"offsets": offsets, "corrected": undo_offsets(movie, offsets)},
        notes=f"skimage phase_cross_correlation, normalization={normalization!r}",
    )


def as_the_skill_says(fixture: Fixture) -> Attempt:
    """Step 3 and step 5, verbatim: TRANSLATION, ``reference="previous"``."""
    return run_stackreg(fixture.data["movie"], "previous")


def the_degraded_path(fixture: Fixture) -> Attempt:
    """Step 1's fallback for a session without pystackreg, with the failure
    table's fix (``normalization=None``) already applied."""
    return _cross_correlation(fixture.data["movie"], None)


def against_the_first_frame(fixture: Fixture) -> Attempt:
    """The mistake step 3 exists to prevent: registering every frame to frame 0,
    so the displacement grows past the optimiser's capture range and the fit
    returns a confident, wrong transform rather than failing."""
    return run_stackreg(fixture.data["movie"], "first")


def with_default_normalization(fixture: Fixture) -> Attempt:
    """The first row of the failure table: leaving `normalization` at its
    ``"phase"`` default, which whitens bins holding only numerical noise and
    buries the true peak. Recovers ~0 drift and reports success."""
    return _cross_correlation(fixture.data["movie"], "phase")


# --- the verifier ----------------------------------------------------------


def _structural(fixture: Fixture, movie) -> np.ndarray | None:
    """The one channel this fixture's truth describes, out of a `(T, C, Y, X)`
    stack. Anything with fewer dimensions is already that channel.

    The verifier may read this and the run may not: *which* channel is
    structural is the fact §6 strips from the data, and reading it off the
    truth is exactly the asymmetry. A run that picked the wrong channel is
    still scored on this one, which is what makes the mistake visible — its
    corrected structural channel carries whatever the other channel's motion
    told it to do.
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


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Two metric families, with different requirements on the truth — which is
    the whole reason the fixture is substitutable:

    ``trajectory_*`` need ``truth["offsets"]``. A curated movie can carry those,
    measured off a fiducial or a bead, so these survive the substitution.

    ``residual_ratio`` needs ``truth["stable"]``, the un-drifted image. It is
    the stronger measurement — it never touches a registration estimator, so it
    cannot share a systematic error with the subject — and it is the one real
    data cannot supply, because no un-drifted acquisition exists. It reports as
    unavailable there rather than as a pass.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    truth_offsets = fixture.truth.get("offsets")
    got_offsets = attempt.arrays.get("offsets")
    if truth_offsets is None or got_offsets is None:
        why = (
            "the fixture carries no annotated trajectory"
            if truth_offsets is None
            else "the run reported no per-frame offsets"
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
        # summary.json is meant to be read at a glance. What it keeps is the
        # number step 4 stops on, and where.
        detail["worst_frame"] = int(per_frame.argmax())

    stable = fixture.truth.get("stable")
    corrected = _structural(fixture, attempt.arrays.get("corrected"))
    movie = _structural(fixture, fixture.data.get("movie"))
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


def _margin(fixture: Fixture) -> int:
    """Pixels to ignore at each border.

    Registration slides data off one side and invents nothing on the other, so
    the margins hold extrapolated pixels that belong to no frame — step 6's
    point. Scoring them would charge every subject for the same artifact.
    """
    offsets = fixture.truth.get("offsets")
    if offsets is None:
        return 8
    return int(np.ceil(np.abs(np.asarray(offsets, float)).max())) + 4


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
        if corrected is not None:
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
    rows = ["frame,truth_dy,truth_dx,got_dy,got_dx,error_px"]
    error = (got_offsets - got_offsets[0]) - (truth_offsets - truth_offsets[0])
    for i, (t, g, e) in enumerate(zip(truth_offsets, got_offsets, error, strict=True)):
        rows.append(
            f"{i},{t[0]:.4f},{t[1]:.4f},{g[0]:.4f},{g[1]:.4f},{np.hypot(*e):.4f}"
        )
    (where / "trajectory.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
