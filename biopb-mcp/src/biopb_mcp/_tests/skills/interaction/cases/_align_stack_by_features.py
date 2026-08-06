"""`align-stack-by-features` as benchmark data: real tissue, real placements.

The skill takes a stack whose sections were each placed independently and
returns one where the same structure sits at the same coordinates.

**This case was rewritten because its first fixture was wrong**, and the way it
was wrong is the reason `docs/fixtures.md` exists. That fixture rendered
every object as an identical isotropic Gaussian blob, and on such content
descriptor matching is not the strong method — a cold ablation chose it in 1 arm
of 9 on the blobs against 2 of 3 on real sections, i.e. **the fixture ranked two
method families in the opposite order from the data the skill is for.** Its
tolerances were calibrated where the reference scored 0.56 px; the same reference
scores 4.04 px here and would have failed its own gate. A synthetic stack was not
a hard version of this problem, it was a different one.

So the fixture is an acquisition: twelve adjacent slices of an Arabidopsis
ovule. The *placement* is still synthetic, and has to be — the correctly-aligned
stack is the ground truth, and no acquisition of independently mounted sections
can supply one. What is real is every pixel of tissue the matcher actually sees.

**What it is hard about is the tail, not the head.** Inliers against section 0
run 1296, 190, 62, 31 and then single digits from section 4 on, while consecutive
pairs hold 124–213 the whole way down. Registering everything to the first
section is *right* for the first third and silently catastrophic after it:
RANSAC keeps returning a confident model as the inliers fall toward
`min_samples`, because three points fit a three-parameter model exactly. Nothing
in that run's own output looks wrong — it ends up 109 px out.

The discriminator is therefore the **procedure** — the gate on inlier count, and
the fallback composed onto the neighbour's *resolved* position — rather than the
withheld fact. Step 2 does ask for that fact (were the sections *placed*
differently, or are they *deformed*?), but a rigid model is also the reasonable
default guess, so `skill+silent` can reach it without asking. `skill` versus
`no-skill` is the comparison this case informs.

**The skill is deferred** (`_`-prefixed, and so is this module): the ablation
showed Sonnet implements SIFT + RANSAC correctly unprompted on real sections. The
case is kept correct rather than deleted, so that promoting the skill for a lower
tier does not begin by rebuilding its benchmark.

## The fixture this case needs

`$BIOPB_SKILL_FIXTURES/align-stack-by-features/ovule-serial-sections/`, holding
`case.json` and `arrays.npz` (`stack`, `canonical`), with the tree's
`manifest.json` recording provenance, citation, licence and hash. The data is not
in git — it is 14 MB of somebody else's imaging — and `make.py` beside it rebuilds
it from the published source:

    PlantSeg Arabidopsis ovules, `N_425_ds2x.tif`; 60 slices about the middle of
    the volume, of which 10–21 are used; each cut into a 384 px centre window
    after an independently drawn rigid placement (≤90°, ≤40 px, seed 0).
    Wolny et al., eLife 2020;9:e57613, CC BY 4.0.

Without that tree the case reports unavailable and does not run. It is never
quietly replaced by something else — that is the whole point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    OnDisk,
    Outcome,
    read_array,
    save_png,
)
from ....agentbench._respondent import Persona
from .._benchmark import Case, Layer

SKILL = "align-stack-by-features"
CASE_ID = "ovule-serial-sections"

#: Sections, and the window each was cut into. Stated here because the task
#: prompt quotes them and the verifier's crop is a fraction of them.
N_SECTIONS = 12
SHAPE = (384, 384)

# Measured on this exact fixture, not chosen. Three routes over the same twelve
# sections, and the spread is two independent runs of each -- RANSAC samples
# randomly, so **the reference itself is not a fixed number**, which is the
# first thing a limit here has to survive:
#
#   route                              residual_ratio     worst section
#   the skill's own recipe             0.356 - 0.381    3.47 -   4.04 px
#   every section direct to section 0  0.446 - 0.563   109.49 - 143.50 px
#   untouched                                  1.000            151.05 px
#
# and the three cold Sonnet arms that passed landed at 3.05-4.70 px.
#
# **`worst_section_offset_px` is the only real discriminator**, and it is a good
# one: 12 px sits ~2.5x above the worst passing run observed and ~9x below the
# best failing one.
#
# **`residual_ratio` does not discriminate on real tissue, and the limit says
# so.** On the old blob fixture the two routes were 0.03 against 0.9 -- sparse
# content means a misplaced section overlaps almost nothing. Dense tissue still
# overlaps itself when 100 px out, so the same measurement compresses to 0.38
# against 0.45-0.56, whose ranges nearly touch -- any limit between them would
# be tuned to one run of one acquisition. 0.75 is above *both* routes: it catches
# a run that left the stack no better than it found it, and nothing finer. Read
# a pass on it as "something was done", never as "it worked".
TOLERANCE = {
    "residual_ratio": 0.75,
    "worst_section_offset_px": 12.0,
}

#: Fraction of the window trimmed before either metric is computed. A section
#: placed by up to 90 deg and 40 px has, after correct alignment, a border whose
#: content came from beyond its own window; every route pays that cost equally,
#: and charging for it would score the border rather than the alignment.
MARGIN_FRACTION = 0.12


# --- the verifier ----------------------------------------------------------


def _inner(shape: tuple[int, int]) -> tuple[slice, slice]:
    margin = int(np.ceil(shape[0] * MARGIN_FRACTION))
    return slice(margin, shape[0] - margin), slice(margin, shape[1] - margin)


def _residual_offset(reference: np.ndarray, got: np.ndarray):
    """How far `got` still sits from `reference`, in px, by phase correlation.

    `normalization=None` is the same point `drift-correction` and `stitch-tiles`
    both make: skimage's `"phase"` default whitens frequency bins holding only
    numerical noise and buries the peak on smooth microscopy content — which
    this fixture is, far more than a synthetic one was.
    """
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(
        np.asarray(reference, float),
        np.asarray(got, float),
        normalization=None,
        upsample_factor=4,
    )
    return shift


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Both metrics compare against `truth["canonical"]`, never against a transform
    the run reported. A run that mis-stated its own transforms but produced a
    correctly aligned stack passes, and one that reported beautiful transforms
    and warped with the wrong one fails — which is the right way round, and it is
    why the task asks only for the stack.

    Truth and data arrive as `ArrayRef` handles off disk; `np.asarray` is what
    reads them, so nothing here knows or cares.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    canonical = np.asarray(fixture.truth["canonical"], float)
    raw = np.asarray(fixture.data["stack"], float)
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    aligned, why = read_array(attempt, "aligned", canonical.shape)
    if aligned is None:
        return Outcome(
            fixture=fixture,
            attempt=attempt,
            metrics=[
                Metric(name, None, limits[name], unavailable=why)
                for name in ("residual_ratio", "worst_section_offset_px")
            ],
        )

    inner = _inner(canonical.shape[1:])
    ref = canonical[(slice(None), *inner)]
    before = np.median(
        [
            np.abs(a - b).mean()
            for a, b in zip(raw[(slice(None), *inner)], ref, strict=True)
        ]
    )
    after = np.median(
        [
            np.abs(a - b).mean()
            for a, b in zip(aligned[(slice(None), *inner)], ref, strict=True)
        ]
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

    # Alignment is defined only up to which section the run made its reference,
    # so every section is measured relative to section 0's own residual. A
    # constant offset across the stack is a different origin, not an error.
    shifts = np.stack(
        [
            _residual_offset(r, g)
            for r, g in zip(ref, aligned[(slice(None), *inner)], strict=True)
        ]
    )
    per_section = np.hypot(*(shifts - shifts[0]).T)
    metrics.append(
        Metric(
            "worst_section_offset_px",
            float(per_section.max()),
            limits["worst_section_offset_px"],
            unit=" px",
        )
    )
    detail["worst_section"] = int(per_section.argmax())
    detail["offset_px_per_section"] = [round(float(v), 3) for v in per_section]

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: an XZ cut, which is where a slipped section shows.

    Never raises — an artifact explains a failure, it does not cause one.
    """
    fixture, attempt = outcome.fixture, outcome.attempt
    canonical = np.asarray(fixture.truth["canonical"], float)
    raw = np.asarray(fixture.data["stack"], float)
    aligned = attempt.arrays.get("aligned")

    mid = canonical.shape[1] // 2
    scale = float(raw.max()) or 1.0
    save_png(raw[:, mid, :], where / "xz-raw.png", vmax=scale)
    save_png(canonical[:, mid, :], where / "xz-truth.png", vmax=scale)
    save_png(raw[-1], where / "raw-last-section.png", vmax=scale)
    save_png(canonical[-1], where / "truth-last-section.png", vmax=scale)

    if aligned is None:
        return
    aligned = np.asarray(aligned, float)
    if aligned.shape != canonical.shape:
        return
    save_png(aligned[:, mid, :], where / "xz-aligned.png", vmax=scale)
    save_png(aligned[-1], where / "aligned-last-section.png", vmax=scale)

    worst = outcome.detail.get("worst_section")
    if isinstance(worst, int):
        save_png(
            aligned[worst], where / f"aligned-worst-section-{worst}.png", vmax=scale
        )
        save_png(
            canonical[worst], where / f"truth-worst-section-{worst}.png", vmax=scale
        )


# --- the conversation ------------------------------------------------------

TASK = f"""\
A stack of serial sections is open in the napari viewer as the layer `sections`.
Its axes are (Z, Y, X) = ({N_SECTIONS}, {SHAPE[0]}, {SHAPE[1]}), float32. There
is no tensor server in this session, so that layer is the only source of pixels.

The sections do not line up with each other, and I need that fixed before
anything is measured off this stack.

The person who prepared and imaged the sections is here and can answer
questions. To ask them something, reply in plain text without calling a tool,
and their answer will come back as the next message.

When you are finished, leave one name bound in the kernel namespace:

  aligned    the aligned stack, the same shape and dtype order as the input,
             with every section carried into the coordinate frame of section 0

That name is how your result is collected, so bind it exactly.
"""

#: Note what is absent: nothing about features, inliers, a gate, or a fallback.
#: This person prepared the sections; they do not know the procedure, and a
#: persona that did could answer a question the agent never properly asked.
HISTOLOGIST = Persona(
    name="histologist-serial-sections",
    facts={
        "how the sections were prepared": (
            "cut them on the microtome one at a time and mounted each one on "
            "its own slide by hand, so every section sits at whatever angle and "
            "position it happened to land at"
        ),
        "whether the tissue is distorted": (
            "no — it is resin embedded and the sections are flat and intact, "
            "nothing is stretched or folded, they are just placed differently"
        ),
        "why the sections do not look alike": (
            "they are cut in sequence through the tissue, so you are looking at "
            "genuinely different structure as you go through the stack — that "
            "is expected, it is not a problem with the imaging"
        ),
        "how far apart the sections are": "consecutive, cut in the order given",
        "what it is for": (
            "I want to follow individual structures from one section to the "
            "next through the whole block"
        ),
    },
    background=(
        "A stack of serial sections through a resin-embedded block, imaged one "
        "slide at a time. You are happy to answer questions about how they were "
        "prepared and imaged."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id=CASE_ID,
    task=TASK,
    persona=HISTOLOGIST,
    # Real tissue, and no substitute. The blob version of this case reversed the
    # method ranking, so "generated or acquired" is not a setting here — it is
    # what the case is.
    fixture=OnDisk(tolerance=TOLERANCE),
    layers=(Layer("sections", "stack"),),
    collect={"aligned": "aligned"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="serial sections",
    # It must be able to answer step 2: were the sections placed differently, or
    # are they deformed, and is the changing content expected.
    persona_must_know=("mounted", "placed", "distorted", "sequence"),
    # And it must not know the procedure.
    persona_must_not_know=(
        "sift",
        "ransac",
        "inlier",
        "feature",
        "euclidean",
        "chained",
    ),
)
