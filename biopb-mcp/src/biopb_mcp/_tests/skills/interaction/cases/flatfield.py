"""`flatfield` as benchmark data: what is the camera offset?

The skill takes a collection of frames shot through the same optics and returns
the illumination field they share. The withheld fact is step 2's second
question — the camera's fixed offset — and it is the cleanest kind this layer
has (`biopb-mcp/docs/skill-testing.md` §5d): **categorically absent from the
data**, not merely hidden behind heuristics the fixture's author thought of.

The forward model is ``I_i = F * S_i * a_i + D``: an illumination field, a
specimen, a per-tile brightness, and the camera's offset. The specimen's own
background sits between the darkest pixel and ``D``, so every route through the
pixels overshoots by an amount the pixels do not reveal. Measured against the
true field, over the estimators an agent plausibly reaches for:

===============================================  ========
any sane estimator, offset obtained by asking    0.5-1.0%
the same estimators, offset assumed zero         4.0-4.8%
the same estimators, offset from a low quantile  6.9-10.0%
===============================================  ========

`TOLERANCE` sits in that gap with 2x on both sides. Note what that means about
what is being scored: **this case measures the asking, not the estimator.** A
median blurred with a well-chosen Gaussian — which the body's step 4 argues
against — still lands at 1.8% and passes here. That is not a hole: the two
questions have different answers to give, and the one this layer can answer is
whether the withheld fact was obtained.

Two constants set the regime, and both matter. `DARKFIELD` is a large enough
fraction of the signal that getting it wrong is expensive — a dim acquisition,
which is exactly when a microscopist's answer is worth having — while
`BACKGROUND` stays comfortably above it so the offset cannot be read off the
data. :meth:`VignettedTiles.__call__` asserts both before handing the fixture
over, since a fixture that drifted out of that regime would look like it tests
asking while handing the answer across.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ....agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_array,
    read_scalar,
    save_png,
)
from ....agentbench._respondent import Persona
from .._benchmark import Case, Layer

SKILL = "flatfield"

#: Set from measurement, not from taste -- see the module docstring for the two
#: populations these separate. The field error is deterministic to ~0.05% across
#: seeds, so the headroom is bounding an agent's implementation choices rather
#: than absorbing run-to-run noise in the fixture.
TOLERANCE = {
    "field_error_pct": 2.0,
    "darkfield_error": 40.0,
}

#: The camera offset baked into every pixel. A real number for a real sCMOS:
#: vendors ship a positive offset so read noise cannot clip at zero.
DARKFIELD = 200.0

#: Specimen background in counts, before the field is applied. Above
#: :data:`DARKFIELD` on purpose -- that gap is what puts the offset out of reach
#: of any quantile of the data.
BACKGROUND = 250.0

#: Lognormal sigma of per-tile brightness. Real (exposure drift, bleaching), and
#: the thing a two-way fit is there to absorb; small enough to be the ordinary
#: reading of an acquisition rather than a pathological one.
BRIGHTNESS_SPREAD = 0.15


# --- the fixture -----------------------------------------------------------


def _true_field(shape: tuple[int, int]) -> np.ndarray:
    """A vignette plus a tilt, normalised to mean 1.

    The tilt matters: a pure radial vignette is symmetric about both axes, so an
    estimator that had collapsed one axis would still look plausible. A tilt
    breaks that symmetry and makes the orientation of the answer checkable.

    The coefficients put the corner at ~59% of the brightest point — a strong
    but ordinary vignette. Gain and exposure are jointly unidentifiable, so only
    the *shape* of the field is a meaningful target, which is what mean-1 says.
    """
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]] / shape[0] - 0.5
    field = 1.0 - 0.8 * (y**2 + x**2) + 0.18 * x
    return field / field.mean()


def _specimen(rng: np.random.Generator, n: int, shape: tuple[int, int]) -> np.ndarray:
    """Blobby content with the same statistics in every tile — a confluent
    monolayer, which is what the persona says the sample is. Uniform statistics
    are the assumption every across-tile estimator makes, so the fixture grants
    it: the case is about the offset, and stacking a second difficulty on top
    would make a failure ambiguous."""
    out = np.empty((n, *shape), np.float32)
    for i in range(n):
        blobs = ndimage.gaussian_filter(
            (rng.random(shape) < 0.02).astype(np.float32), 4
        )
        out[i] = BACKGROUND + blobs * 900.0
    return out


def field_error_pct(estimate, truth) -> float:
    """Mean absolute deviation of *estimate* from *truth*, in % of the mean field.

    Both sides are renormalised to mean 1 first, so a run is scored on the shape
    of the field it recovered and never on how it chose to scale it — which is
    the only thing that can be scored, since gain and exposure are jointly
    unidentifiable.
    """
    estimate = np.asarray(estimate, float)
    truth = np.asarray(truth, float)
    estimate = estimate / estimate.mean()
    truth = truth / truth.mean()
    return float(100.0 * np.abs(estimate - truth).mean() / truth.mean())


@dataclass(frozen=True)
class VignettedTiles:
    """A tile collection whose camera offset only the microscopist knows."""

    n_tiles: int = 24
    shape: tuple[int, int] = (256, 256)
    seed: int = 11

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        field = _true_field(self.shape)
        content = _specimen(rng, self.n_tiles, self.shape)
        brightness = np.exp(rng.normal(0.0, BRIGHTNESS_SPREAD, self.n_tiles))
        images = (
            field[None] * content * brightness[:, None, None]
            + DARKFIELD
            + rng.normal(0.0, 3.0, (self.n_tiles, *self.shape))
        ).astype(np.float32)

        # The property the whole case rests on, checked before anyone pays for a
        # run (§5d's rule about a fixture whose truth is wrong). The darkest
        # pixel is the offset *plus* the specimen's background, so the stack's
        # low quantile is an upper bound that overshoots -- if these two ever
        # converged, the withheld fact would be readable off the data and every
        # arm would score on a question it could look up.
        quantile = float(np.quantile(images, 0.001))
        assert quantile > 1.5 * DARKFIELD, (
            f"the stack's 0.1% quantile is {quantile:.0f} against an offset of "
            f"{DARKFIELD:.0f} — close enough for a run to read the withheld "
            "fact off the pixels"
        )
        span = float(field.max() / field.min())
        assert 1.5 < span < 2.5, f"the vignette is {span:.1f}x, no longer ordinary"

        return Fixture(
            provenance=(
                f"procedural: {self.n_tiles} tiles of {self.shape[0]}x"
                f"{self.shape[1]}, seed {self.seed}, camera offset "
                f"{DARKFIELD:.0f} counts, per-tile brightness sigma "
                f"{BRIGHTNESS_SPREAD:g}"
            ),
            about=(
                f"A {span:.1f}x vignette over {self.n_tiles} fields, on top of a "
                f"camera offset of {DARKFIELD:.0f} counts. The darkest pixel in "
                f"the stack is {float(images.min()):.0f}, so the offset cannot "
                "be read off the data: a run that does not ask for it either "
                "ignores it or overshoots, and either way the recovered field "
                "is several times further off than the estimator itself would "
                "put it."
            ),
            data={"tiles": images},
            truth={
                "field": field,
                # The private fact. Stripped from `data`, held by the
                # respondent, and read by the verifier to know what to score.
                "darkfield": DARKFIELD,
                "brightness": brightness,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Two metrics with different requirements on the truth, which is what keeps
    the fixture substitutable (§5d):

    ``field_error_pct`` needs ``truth["field"]``. Real data can carry one — a
    field measured off a fluorescent slide or a uniform dye layer — so this
    survives a curated substitution, with the caveat that the reference is then
    someone's measurement rather than a construction.

    ``darkfield_error`` needs ``truth["darkfield"]``, which real data supplies
    *better* than this fixture does: an acquisition with dark frames has the
    offset exactly. It is the one number here a microscope hands over for free.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    truth_field = fixture.truth.get("field")
    if truth_field is None:
        flat, why = None, "the fixture carries no reference illumination field"
    else:
        flat, why = read_array(attempt, "flat", np.asarray(truth_field).shape)
    if flat is None:
        metrics.append(
            Metric("field_error_pct", None, limits["field_error_pct"], unavailable=why)
        )
    else:
        metrics.append(
            Metric(
                "field_error_pct",
                field_error_pct(flat, truth_field),
                limits["field_error_pct"],
                unit="%",
            )
        )
        detail["field_range"] = float(np.max(flat) / np.min(flat))
        detail["true_field_range"] = float(np.max(truth_field) / np.min(truth_field))

    truth_darkfield = fixture.truth.get("darkfield")
    if truth_darkfield is None:
        got, why = None, "the fixture does not record a camera offset"
    else:
        got, why = read_scalar(attempt, "darkfield")
    if got is None:
        metrics.append(
            Metric("darkfield_error", None, limits["darkfield_error"], unavailable=why)
        )
    else:
        metrics.append(
            Metric(
                "darkfield_error",
                abs(got - float(truth_darkfield)),
                limits["darkfield_error"],
                unit=" counts",
            )
        )
        detail["darkfield_reported"] = got
        detail["darkfield_true"] = float(truth_darkfield)

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means, for a human to page through.
    Never raises — an artifact explains a failure, it does not cause one."""
    fixture, attempt = outcome.fixture, outcome.attempt
    truth_field = fixture.truth.get("field")
    tiles = fixture.data.get("tiles")
    if truth_field is None or tiles is None:
        return
    truth_field = np.asarray(truth_field, float)

    # The true field and the run's, on one shared scale, so the pair reads side
    # by side instead of each being stretched to its own range (see save_png).
    scale = float(truth_field.max())
    save_png(truth_field, where / "field-true.png", vmax=scale)
    save_png(np.asarray(tiles[0], float), where / "tile-raw.png")

    flat = attempt.arrays.get("flat")
    if flat is None or np.asarray(flat).shape != truth_field.shape:
        return
    flat = np.asarray(flat, float)
    flat = flat / flat.mean()
    save_png(flat, where / "field-estimated.png", vmax=scale)
    # Where the estimate went wrong, on the same scale as the field itself: a
    # good fit is near-black and a vignette left in or invented is a visible
    # bowl. Scaling this one to its own range would make every run look equally
    # bad, which is exactly backwards.
    save_png(np.abs(flat - truth_field), where / "field-error.png", vmax=scale)

    # `read_scalar`, the same reader `verify` uses, rather than a bare reshape:
    # the most plausible wrong binding here is a 2-D darkfield *image* (what
    # BaSiC returns, against a task asking for one number), and reshaping that
    # raises. `verify` already scores it as unavailable, so the bare version
    # threw away a scored arm to fail on the picture explaining it. No offset is
    # the right stand-in: the correction then shows what the run's own `flat`
    # does on its own, which is the comparison the image is for.
    offset, _ = read_scalar(attempt, "darkfield")
    save_png(
        (np.asarray(tiles[0], float) - (offset or 0.0)) / np.maximum(flat, 1e-6),
        where / "tile-corrected.png",
    )

    # A horizontal cut through the middle: the shape of a field is much easier
    # to compare as a profile than as two grey squares.
    middle = truth_field.shape[0] // 2
    rows = ["x,true,estimated"]
    rows += [
        f"{x},{t:.5f},{g:.5f}"
        for x, (t, g) in enumerate(zip(truth_field[middle], flat[middle], strict=True))
    ]
    (where / "field-profile.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A stack of 24 fields from one acquisition is open in the napari viewer as the
layer `tiles`. Its axes are (N, Y, X) = (24, 256, 256), float32. There is no
tensor server in this session, so that layer is the only source of pixels.

The illumination is not even across the frame, and I need that characterised and
corrected before I compare intensities between these fields.

The microscopist who acquired them is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  flat       the illumination field you estimated, a (256, 256) float array.
             Its overall scale does not matter — it is compared up to a constant
             factor — but its shape across the frame does.
  darkfield  the camera offset you used, as a single number in counts.

Those two names are how your result is collected, so bind them exactly.
"""

#: Note what is *not* here: nothing about medians, DCTs, log space, or how to
#: estimate an offset from data. This person knows their camera and their
#: sample, not the procedure. A persona that knew the skill could answer a
#: question the agent never properly asked, and the numeric result would stop
#: meaning what it appears to.
MICROSCOPIST = Persona(
    name="microscopist-tile-scan",
    facts={
        "what the camera offset is": (
            "the sCMOS has a fixed offset of 200 counts baked into every pixel "
            "— that is just where the camera puts zero, and nothing has taken "
            "it out"
        ),
        "whether anything has been corrected already": (
            "no, these came straight off the camera. I exported the raw frames "
            "on purpose because the last person did some processing I could "
            "not reproduce"
        ),
        "how they were acquired": (
            "24 positions in one sitting, same objective, same lamp setting "
            "throughout. Nothing was touched between fields except the stage"
        ),
        "what the sample is": (
            "a confluent monolayer, so there is roughly the same amount of "
            "material in every field — none of them are mostly empty"
        ),
        "why it matters": (
            "I want to compare brightness between the fields, and right now the "
            "edges of every frame come out darker than the middle"
        ),
    },
    background=(
        "You acquired a set of fields on a widefield microscope. You are happy "
        "to answer questions about the camera, the sample and the acquisition."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="offset-known-only-to-the-operator",
    task=TASK,
    persona=MICROSCOPIST,
    fixture=Procedural(VignettedTiles()),
    layers=(Layer("tiles", "tiles"),),
    collect={"flat": "flat", "darkfield": "darkfield"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="illumination",
    # It must be able to answer: the fixture withholds the camera offset, and
    # this person knows it, knows the data is raw, and knows the symptom.
    persona_must_know=("offset", "200", "camera", "raw"),
    # And it must not know the method — only the instrument and the sample.
    persona_must_not_know=(
        "dct",
        "median polish",
        "log space",
        "low-order",
        "quantile",
        "flat field",
    ),
)
