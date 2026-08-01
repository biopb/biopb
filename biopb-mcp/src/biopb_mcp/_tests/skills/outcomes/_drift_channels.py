"""The interaction-tier fixture for `drift-correction`: which channel is it?

`docs/skill-testing.md` §6. Step 2 of the body asks the user two things in one
question — did the *field* move or the *objects*, and which channel is
`REF_CHANNEL`. Both are the same fact here, and the fixture is built so that
**nothing in the pixels answers it**:

===========  ==========================================================
channel 0    bright puncta that carry the stage drift *and* crawl on
             their own — a vesicle or reporter channel
channel 1    a dim, structurally stable field that moves only with the
             stage — a membrane or transmitted-light channel
===========  ==========================================================

The movie arrives with no channel names. Every heuristic available to a run that
does not ask points the wrong way: channel 0 is five times the contrast, an
order of magnitude the peak intensity, and far the more obviously "feature-rich"
thing to register on. The body's own words for why a single frame cannot settle
it — "These look identical in a single frame and the correction for one destroys
the other."

Register on channel 0 and the objects' common motion is indistinguishable from
stage drift, so it is silently added to the trajectory: the fit does not fail,
it succeeds at measuring the wrong thing. That is the same shape as the failure
step 3 exists to prevent, arriving through a different door.

**Why this fixture is here and not in the interaction suite.** What §6 needs to
be true of it is a claim that can be settled without a model: that the ambiguity
actually costs something. So the subjects below are scripted — one told which
channel is structural, two using the heuristics a run that never asked would
reach for — and `test_drift_channel_choice` asserts the verifier separates them.
A fixture nothing has ever failed *for want of asking* is not known to test
asking, which is §5b's argument one tier up. Only once that holds is it worth
paying a model to demonstrate it.

**Measured**, over six seeds of the same construction, against the limits in
`_drift.TOLERANCE` (0.5 px RMS, 1.0 px max, 0.10 residual):

===================  ===================  ==================  ==============
subject              trajectory RMS       max error           residual ratio
===================  ===================  ==================  ==============
told which channel   0.0001 – 0.0006 px   0.0002 – 0.0010 px  0.0001
the bright channel   1.95 – 5.28 px       2.59 – 9.67 px      0.29 – 0.56
mean over channels   1.92 – 5.16 px       2.55 – 9.46 px      0.29 – 0.55
===================  ===================  ==================  ==============

The narrowest gap is on max error, where the mildest failure still clears the
limit by 2.6x while the correct run sits a thousand-fold under it. That is the
same shape of margin the outcome cases carry, and it is why `TOLERANCE` is
reused here rather than restated: a fixture needing its own looser limits to
separate would not be separating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ._drift import SKILL, blobby_field, run_stackreg, trajectory, undo_offsets
from ._outcome import Attempt, Fixture, Kind, Tier, register

#: Shared by every object, in px/frame. This is the part that makes the mistake
#: systematic rather than a lucky draw: a random-only velocity field averages
#: toward zero over 60 objects, and at one point in tuning that let a seed slip
#: back inside tolerance. Cells crawling up a gradient is the ordinary reading.
COMMON_VELOCITY_PX = 0.25

#: Per-object scatter about that common motion, in px/frame.
SPREAD_VELOCITY_PX = 0.5

#: Channel 1's amplitude, relative to the blobby field the outcome cases use.
#: Dim enough that "register on the bright one" is the tempting call.
STRUCTURAL_DIM = 0.35

BACKGROUND = 100.0


def _puncta(positions: np.ndarray, amplitudes: np.ndarray, shape, sigma=2.0):
    """Render point objects as Gaussian spots. Positions may be sub-pixel; they
    are rounded to the nearest sample, which is well below the tolerances here
    and keeps the render cheap."""
    img = np.zeros(shape, dtype=np.float32)
    yy = np.clip(np.round(positions[:, 0]).astype(int), 0, shape[0] - 1)
    xx = np.clip(np.round(positions[:, 1]).astype(int), 0, shape[1] - 1)
    np.add.at(img, (yy, xx), amplitudes)
    return ndimage.gaussian_filter(img, sigma)


@dataclass(frozen=True)
class AmbiguousChannels:
    """A two-channel movie where only a person knows which channel is which."""

    case_id: str
    about: str
    per_frame_px: float = 1.7
    n_frames: int = 24
    n_objects: int = 60
    shape: tuple[int, int] = (192, 192)
    seed: int = 0
    skill_id: str = SKILL
    kind: Kind = "synthetic"
    tier: Tier = "interaction"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def build(self) -> Fixture:
        rng = np.random.default_rng(self.seed + 100)
        offsets = trajectory(self.n_frames, self.per_frame_px, self.seed + 1)

        # Channel 1 -- the structural one. A pure shift of one image, so its
        # un-drifted state is ground truth to machine precision, exactly as in
        # the single-channel cases.
        stable = blobby_field(self.seed, self.shape)
        stable = (stable - BACKGROUND) * STRUCTURAL_DIM + BACKGROUND
        structural = np.array(
            [ndimage.shift(stable, o, order=3, mode="nearest") for o in offsets]
        )

        # Channel 0 -- bright objects that both ride the stage and move.
        start = rng.uniform(10, min(self.shape) - 10, size=(self.n_objects, 2))
        amplitudes = rng.uniform(4000.0, 9000.0, size=self.n_objects)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        common = COMMON_VELOCITY_PX * np.array([np.sin(heading), np.cos(heading)])
        velocity = common + rng.normal(
            0.0, SPREAD_VELOCITY_PX, size=(self.n_objects, 2)
        )
        reporter = np.array(
            [
                _puncta(start + velocity * t + o, amplitudes, self.shape) + BACKGROUND
                for t, o in enumerate(offsets)
            ]
        )

        movie = np.stack([reporter, structural], axis=1).astype(np.float32)
        drift = float(np.hypot(*offsets[-1]))
        objects = float(np.hypot(*common) * (self.n_frames - 1))
        return Fixture(
            skill_id=self.skill_id,
            case_id=self.case_id,
            kind="synthetic",
            provenance=(
                f"procedural: 2 channels, seed {self.seed}, {self.n_frames} frames, "
                f"{drift:.1f} px of stage drift, {self.n_objects} objects also "
                f"moving {objects:.1f} px of their own"
            ),
            about=self.about,
            data={"movie": movie},
            truth={
                "offsets": offsets,
                "stable": stable,
                # The private fact. §6 strips it from `data`; the respondent
                # holds it; the verifier reads it to know what to score.
                "structural_channel": 1,
            },
            tolerance={},
        )


CROWDED = register(
    AmbiguousChannels(
        case_id="two-channels-one-structural",
        about=(
            "The stage drifts 39 px while the objects in the bright channel "
            "crawl 6 px of their own. Registering on brightness measures the "
            "sum of the two and reports success."
        ),
    )
)


# --- the subjects ----------------------------------------------------------
#
# One run that was told which channel is structural, and two that were not.
# Neither of the latter is a straw man: both are what a competent run does when
# it has to choose from the pixels alone.


def _corrected_stack(movie: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Undo `offsets` on every channel — step 5's "apply to all of them"."""
    return np.stack(
        [undo_offsets(movie[:, c], offsets) for c in range(movie.shape[1])], axis=1
    )


def _on_channel(fixture: Fixture, plane: np.ndarray, subject: str, notes: str):
    attempt = run_stackreg(plane, "previous")
    offsets = np.asarray(attempt.arrays["offsets"])
    return Attempt(
        subject=subject,
        arrays={
            "offsets": offsets,
            "corrected": _corrected_stack(np.asarray(fixture.data["movie"]), offsets),
        },
        notes=notes,
    )


def told_which_channel(fixture: Fixture) -> Attempt:
    """Step 2, answered: register on the structural channel, apply to all."""
    channel = int(fixture.truth["structural_channel"])
    movie = np.asarray(fixture.data["movie"])
    return _on_channel(
        fixture,
        movie[:, channel],
        "told-which-channel",
        f"asked, and was told channel {channel} is the structural one",
    )


def the_brightest_channel(fixture: Fixture) -> Attempt:
    """The choice made from the pixels: register on whichever channel carries
    the most signal. Nothing about that reasoning is careless — it is simply
    answering a question the data cannot answer."""
    movie = np.asarray(fixture.data["movie"])
    channel = int(np.argmax([movie[:, c].std() for c in range(movie.shape[1])]))
    return _on_channel(
        fixture,
        movie[:, channel],
        "the-brightest-channel",
        f"never asked; picked channel {channel} for having the most contrast",
    )


def the_mean_of_the_channels(fixture: Fixture) -> Attempt:
    """What the Parameters table forbids by name: "Not a mean projection over
    channels — that mixes in the very channel whose intensity is the
    measurement." The dodge available to a run that will not choose."""
    movie = np.asarray(fixture.data["movie"])
    return _on_channel(
        fixture,
        movie.mean(axis=1),
        "the-mean-of-the-channels",
        "never asked; averaged the channels to avoid choosing between them",
    )
