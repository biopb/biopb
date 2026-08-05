"""`ratiometric-fret` as benchmark data: what did the control slides say?

The skill turns a donor channel and a FRET channel into a ratio that means
something. The withheld fact is step 2's gate — the two coefficients measured on
single-label control samples — and it is categorically absent from the data
(`biopb-mcp/docs/skill-testing.md` §5d) for a reason that is textbook rather than
constructed.

The forward model is the standard three-cube one::

    D = KD * cD * (1 - E)                       donor ex, donor em
    A = KA * cA                                 acceptor ex, acceptor em
    F = KF * cD * E  +  BT * D  +  DE * A       donor ex, acceptor em

Three observations per pixel, and the FRET channel is a sum of three terms. The
sensitized-emission term is what is wanted; the other two are the donor's own
emission leaking through the acceptor filter and the acceptor being excited
directly. Nothing in a doubly-labelled field separates them — that separation is
what a donor-only and an acceptor-only slide are *for*. The fixture asserts the
non-identifiability rather than assuming it: the least-squares fit an agent would
reach for, ``F ~ a*D + b*A`` over the mask, recovers an `a` far from ``BT``,
because the sensitized-emission term correlates with `D` too.

What the coefficients buy, measured over seeds 5, 17 and 31:

===========================================  ===========  ==============
route                                        level error  contrast error
===========================================  ===========  ==============
coefficients obtained by asking                0.1-0.4%       0.0-0.3%
no correction at all (the raw F/D ratio)         269-287%       45.0-50.6%
donor leak subtracted, direct excitation not      70-89%       20.1-33.1%
``F ~ a*D + b*A`` fitted on the field itself     203-210%          100.0%
correct, then renormalised for display              165%        0.0-0.3%
the acceptor channel over the donor, not FRET    459-644%       45.7-71.5%
===========================================  ===========  ==============

`TOLERANCE` sits in the gap: 20x above the reference on level and 8x below the
nearest failure, 18x and 4x on contrast. Note which column is which. **Level**
is what a missing constant does to the number; **contrast** is what it does to
the biology — the field holds two populations differing 3.05x in true ratio, and
the uncorrected ratio reports 1.64x. The two are not redundant, and the last two
rows are why: a correct ratio renormalised for display keeps every fold-change
and is still 165% off, while a fit on the field itself over-subtracts the
resting population through zero and loses the fold-change entirely.

The true ratio here is ``(KF/KD) * E/(1-E)`` — the donor concentration cancels,
so this fixture's answer is independent of donor brightness by construction. That
is a property of the construction and **not** a rule about real data: on a TIRF
tension-sensor field the correctly-corrected ratio was rank-correlated with donor
intensity at -0.41. The skill body says so; this case does not score it.
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

SKILL = "ratiometric-fret"

#: Set from the route table in the module docstring, not from taste. Every
#: failure route is 20x or more outside the level limit and 10x or more outside
#: the contrast limit, and the reference sits an order of magnitude inside both,
#: so these bound an agent's implementation choices rather than absorbing noise.
TOLERANCE = {
    "level_error_pct": 8.0,
    "contrast_error_pct": 5.0,
    "bleedthrough_error": 0.06,
}

#: Donor emission leaking through the acceptor filter, as a fraction of the donor
#: channel. 0.35 is an ordinary CFP->YFP number; the point is that it is a
#: property of the filter set, measured once on a donor-only slide.
BLEEDTHROUGH = 0.35

#: Acceptor excited directly by the donor's excitation line, as a fraction of the
#: acceptor channel. Measured on an acceptor-only slide, and the term an
#: incomplete correction leaves behind.
DIRECT_EXCITATION = 0.12

#: The two interaction levels the field holds. Chosen so the true ratio differs
#: 3.0x between them -- a fold-change an experiment would be built to detect, and
#: large enough that compressing it is unmistakable.
E_LOW, E_HIGH = 0.15, 0.35

#: Channel sensitivities, in counts per unit concentration. Equal, so the true
#: ratio is exactly ``E/(1-E)`` and every number in the docstring can be checked
#: by hand.
KD = KA = KF = 900.0


# --- the fixture -----------------------------------------------------------


def _blobs(rng: np.random.Generator, shape: tuple[int, int], sigma: float = 9.0):
    """A smooth positive field, mean ~1, for a concentration that varies."""
    field = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    field = field / (field.std() + 1e-12)
    return np.exp(0.45 * field)


def _two_cells(shape: tuple[int, int]) -> np.ndarray:
    """Two disjoint elliptical footprints, labelled 1 and 2.

    Two of them, not one: the case scores a *fold-change between populations*,
    which needs two populations. They are far apart and hard-edged because
    nothing here is about segmentation -- the verifier uses these labels
    directly rather than asking the run to find them.
    """
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    labels = np.zeros(shape, np.uint8)
    for value, (cy, cx, ry, rx) in enumerate(
        ((0.32, 0.30, 0.22, 0.16), (0.68, 0.70, 0.18, 0.24)), start=1
    ):
        inside = ((y / shape[0] - cy) / ry) ** 2 + ((x / shape[1] - cx) / rx) ** 2 <= 1
        labels[inside] = value
    return labels


@dataclass(frozen=True)
class BiosensorField:
    """A three-cube acquisition whose control slides are not in the session."""

    shape: tuple[int, int] = (256, 256)
    seed: int = 5

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        labels = _two_cells(self.shape)
        cell = labels > 0

        # Donor and acceptor abundance vary independently: a fixed ratio between
        # them would make the direct-excitation term proportional to the donor
        # channel, and then one coefficient could stand in for both.
        c_donor = _blobs(rng, self.shape) * cell
        c_acceptor = _blobs(rng, self.shape) * cell
        efficiency = np.where(labels == 2, E_HIGH, E_LOW) * cell

        donor = KD * c_donor * (1.0 - efficiency)
        acceptor = KA * c_acceptor
        sensitized = KF * c_donor * efficiency
        fret = sensitized + BLEEDTHROUGH * donor + DIRECT_EXCITATION * acceptor

        # Shot noise, so the counts and the noise are tied together the way a
        # camera ties them. The offset is zero and the task says so: this case is
        # about the coefficients, and a second withheld constant would make a
        # failure ambiguous (`flatfield` owns that one).
        observed = {
            "donor": rng.poisson(donor).astype(np.float32),
            "acceptor": rng.poisson(acceptor).astype(np.float32),
            "fret": rng.poisson(fret).astype(np.float32),
        }

        mask = cell & (donor > 0.15 * donor[cell].mean())
        ratio = np.zeros(self.shape, np.float32)
        ratio[mask] = (sensitized[mask] / donor[mask]).astype(np.float32)

        # The property the case rests on, checked before anyone pays for a run.
        # This is the fit an agent reaches for when told there are no controls;
        # it is biased by the sensitized-emission term, which correlates with the
        # donor channel. If it ever converged on BLEEDTHROUGH the withheld fact
        # would be readable off the pixels.
        design = np.stack([observed["donor"][mask], observed["acceptor"][mask]], axis=1)
        fitted, *_ = np.linalg.lstsq(design, observed["fret"][mask], rcond=None)
        assert abs(float(fitted[0]) - BLEEDTHROUGH) > 0.4 * BLEEDTHROUGH, (
            f"a plain F ~ aD + bA fit on the field recovers a={float(fitted[0]):.3f} "
            f"against a true {BLEEDTHROUGH} — close enough for a run to read the "
            "withheld fact off the data"
        )
        contrast = _contrast(ratio, [mask & (labels == v) for v in (1, 2)])
        assert 2.5 < contrast < 3.6, f"the two populations differ {contrast:.2f}x"

        return Fixture(
            provenance=(
                f"procedural: {self.shape[0]}x{self.shape[1]} three-cube field, "
                f"seed {self.seed}, donor bleedthrough {BLEEDTHROUGH}, acceptor "
                f"direct excitation {DIRECT_EXCITATION}, Poisson counts"
            ),
            about=(
                "Two populations of cells differing "
                f"{contrast:.1f}x in true FRET ratio, imaged in three cubes. The "
                f"FRET channel carries {BLEEDTHROUGH:g} of the donor channel and "
                f"{DIRECT_EXCITATION:g} of the acceptor channel on top of the "
                "sensitized emission, and no single-label control is in the "
                "session: the split is not identifiable from a doubly-labelled "
                "field, so a run that does not ask reports a ratio roughly four "
                "times too high and a fold-change roughly half what it is."
            ),
            data=observed,
            truth={
                "ratio": ratio,
                "mask": mask,
                "labels": labels,
                # The private facts. Held by the respondent, read by the verifier
                # to know what to score.
                "bleedthrough": BLEEDTHROUGH,
                "direct_excitation": DIRECT_EXCITATION,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _medians(ratio: np.ndarray, regions) -> list[float]:
    return [float(np.median(ratio[where])) for where in regions]


def _contrast(ratio: np.ndarray, regions) -> float:
    """Median ratio in the stimulated population over the resting one — the
    fold-change the experiment exists to measure.

    Zero when the resting median is not positive, which is a real outcome rather
    than a division to guard: an over-subtracted correction drives that
    population through zero, and a run that did it has not reported a
    fold-change. Scoring it as 0 (a 100% error) says that; `nan` would say the
    metric could not be computed, which is the one thing it must never be
    confused with.
    """
    low, high = _medians(ratio, regions)
    return high / low if low > 0 else 0.0


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Three metrics, ordered by what they need from the truth, which is what keeps
    the fixture substitutable (§5d):

    ``level_error_pct`` and ``contrast_error_pct`` need a reference ratio map and
    a two-population labelling. A curated acquisition can carry both — a
    calibrated biosensor with a saturating and a resting condition is an ordinary
    control experiment — with the caveat that the reference is then someone's
    measurement.

    ``bleedthrough_error`` needs the coefficient itself, which real data supplies
    *better* than this fixture does: an acquisition with single-label controls
    has it measured rather than constructed.

    The run's own masking is deliberately not scored. The verifier applies the
    fixture's mask to whatever full-frame map came back, so the number reports
    the correction and never the run's choice of threshold.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    truth_ratio = fixture.truth.get("ratio")
    mask = fixture.truth.get("mask")
    labels = fixture.truth.get("labels")
    if truth_ratio is None or mask is None or labels is None:
        got, why = None, "the fixture carries no reference ratio map for two groups"
    else:
        got, why = read_array(attempt, "fret_ratio", np.asarray(truth_ratio).shape)

    if got is None:
        metrics.append(
            Metric("level_error_pct", None, limits["level_error_pct"], unavailable=why)
        )
        metrics.append(
            Metric(
                "contrast_error_pct",
                None,
                limits["contrast_error_pct"],
                unavailable=why,
            )
        )
    else:
        truth_ratio = np.asarray(truth_ratio, float)
        mask = np.asarray(mask, bool)
        labels = np.asarray(labels)
        # Per population, never pooled. The two groups are the point of the
        # field, so their values are bimodal and the pooled median sits in one
        # group's tail: measured, the reference route scored 13% off a truth it
        # reproduces to 0.4% per group.
        regions = [mask & (labels == value) for value in (1, 2)]
        want = _medians(truth_ratio, regions)
        have = _medians(got, regions)
        metrics.append(
            Metric(
                "level_error_pct",
                max(
                    100.0 * abs(h - w) / abs(w) for h, w in zip(have, want, strict=True)
                ),
                limits["level_error_pct"],
                unit="%",
            )
        )
        detail["ratio_reported"] = have
        detail["ratio_true"] = want

        want_c = _contrast(truth_ratio, regions)
        have_c = _contrast(got, regions)
        metrics.append(
            Metric(
                "contrast_error_pct",
                100.0 * abs(have_c - want_c) / abs(want_c),
                limits["contrast_error_pct"],
                unit="%",
            )
        )
        detail["contrast_reported"] = have_c
        detail["contrast_true"] = want_c

    truth_bt = fixture.truth.get("bleedthrough")
    if truth_bt is None:
        coefficient, why = None, "the fixture records no bleedthrough coefficient"
    else:
        coefficient, why = read_scalar(attempt, "bleedthrough")
    if coefficient is None:
        metrics.append(
            Metric(
                "bleedthrough_error",
                None,
                limits["bleedthrough_error"],
                unavailable=why,
            )
        )
    else:
        metrics.append(
            Metric(
                "bleedthrough_error",
                abs(coefficient - float(truth_bt)),
                limits["bleedthrough_error"],
            )
        )
        detail["bleedthrough_reported"] = coefficient
        detail["bleedthrough_true"] = float(truth_bt)

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means, for a human to page through.
    Never raises — an artifact explains a failure, it does not cause one."""
    fixture, attempt = outcome.fixture, outcome.attempt
    truth_ratio = fixture.truth.get("ratio")
    if truth_ratio is None:
        return
    truth_ratio = np.asarray(truth_ratio, float)
    mask = np.asarray(fixture.truth.get("mask"), bool)

    # One shared scale for both ratio images, so the pair reads side by side. A
    # run that added a constant is then a visibly brighter cell, which is exactly
    # what an uncorrected ratio is.
    scale = float(np.quantile(truth_ratio[mask], 0.99)) * 2.0
    save_png(truth_ratio, where / "ratio-true.png", vmax=scale)
    for name in ("donor", "fret"):
        if (channel := fixture.data.get(name)) is not None:
            save_png(np.asarray(channel, float), where / f"channel-{name}.png")

    got = attempt.arrays.get("fret_ratio")
    if got is None or np.asarray(got).shape != truth_ratio.shape:
        return
    got = np.asarray(got, float)
    save_png(got, where / "ratio-estimated.png", vmax=scale)

    # The two populations as numbers, since a fold-change is not something two
    # grey squares show.
    labels = fixture.truth.get("labels")
    if labels is None:
        return
    labels = np.asarray(labels)
    rows = ["population,true_median,reported_median"]
    for value in (1, 2):
        here = labels == value
        rows.append(
            f"{value},{np.median(truth_ratio[here]):.5f},{np.median(got[here]):.5f}"
        )
    (where / "populations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Three channels from one three-cube FRET acquisition are open in the napari
viewer, each a (256, 256) float32 layer:

  donor      donor excitation, donor emission
  acceptor   acceptor excitation, acceptor emission
  fret       donor excitation, acceptor emission

There is no tensor server in this session, so those layers are the only source of
pixels. The camera offset has already been removed from all three, and the field
holds two groups of cells.

I need a FRET ratio I can compare between those two groups, not just a picture.

The person who ran the experiment is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will come
back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  fret_ratio    the corrected ratio, a (256, 256) float array over the whole
                frame: the sensitized-emission part of the FRET channel divided
                by the donor channel, pixel by pixel. Leave it at 0 wherever you
                decided there was nothing to measure. Its absolute scale is
                compared against the truth, so do not renormalise it.
  bleedthrough  the coefficient you used for the donor's contribution to the
                FRET channel, as a single number.

Those two names are how your result is collected, so bind them exactly.
"""

#: Note what is *not* here: nothing about how to combine the channels, no
#: formula, no name for either coefficient. This person ran control slides and
#: read two numbers off them; they do not know what the assistant is supposed to
#: do with those numbers.
BIOSENSOR_USER = Persona(
    name="biosensor-three-cube",
    facts={
        "whether single-label control samples exist": (
            "yes — I imaged a dish expressing only the donor and a dish "
            "expressing only the acceptor, on the same day with the same "
            "filters, exposure and laser power as the experiment"
        ),
        "what the donor-only dish gave": (
            "on the donor-only dish the FRET channel came out at 0.35 of the "
            "donor channel, everywhere in the cell"
        ),
        "what the acceptor-only dish gave": (
            "on the acceptor-only dish the FRET channel came out at 0.12 of the "
            "acceptor channel"
        ),
        "what the two groups of cells are": (
            "one group is the resting condition and the other was stimulated; I "
            "expect them to differ, and how much is the whole question"
        ),
        "what the sample is": (
            "cells co-expressing two tagged proteins that bind each other, so "
            "how much of each one a cell has varies from cell to cell"
        ),
        "why it matters": (
            "I need the fold-change between the two groups to be believable — "
            "the last analysis gave a difference so small it was not worth "
            "reporting"
        ),
    },
    background=(
        "You ran a three-cube FRET experiment on a widefield microscope and "
        "kept everything about the acquisition constant. You are happy to "
        "answer questions about the sample, the controls and the settings."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="controls-were-run-but-are-not-in-the-session",
    task=TASK,
    persona=BIOSENSOR_USER,
    fixture=Procedural(BiosensorField()),
    layers=(
        Layer("donor", "donor"),
        Layer("acceptor", "acceptor"),
        Layer("fret", "fret"),
    ),
    collect={"fret_ratio": "fret_ratio", "bleedthrough": "bleedthrough"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="fret",
    # It must be able to answer: the fixture withholds both coefficients, and
    # this person ran the slides they were measured on.
    persona_must_know=("0.35", "0.12", "only the donor", "only the acceptor"),
    # And it must not know the method. Fenced in the operator's own vocabulary --
    # a person who says "bleedthrough" has been handed the agent's half of the
    # conversation, and could answer a question that was never properly asked.
    persona_must_not_know=(
        "bleedthrough",
        "crosstalk",
        "sensitized emission",
        "spectral unmixing",
        "subtract",
    ),
)
