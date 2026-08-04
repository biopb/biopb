"""`align-stack-by-features` as benchmark data: the reference match runs out.

The skill takes a stack whose sections were each placed independently and
returns one where the same structure sits at the same coordinates. Like
`drift_correction`, the answer is exactly knowable: place a section by a
transform you chose, and the correctly-placed section is ground truth.

**What this fixture is hard about is the tail, not the head.** Content turns
over along the stack, so features shared with section 0 decay away — over four
seeds of this construction the inliers against the reference run 202–270 at
section 1, 22–44 by section 3, and single digits from section 4 on, while the
*consecutive* pairs hold above 200 the whole way down. The obvious procedure,
register everything to the first section, is *right* for the first third and
silently catastrophic after it: RANSAC keeps returning a confident model as the
inliers fall toward `min_samples`, because three points fit a three-parameter
model exactly. Nothing in that run's own output looks wrong.

So the discriminator here is mostly the **procedure** — the gate on inlier count
and the neighbour fallback composed onto the neighbour's resolved position —
rather than the withheld fact, and the arms should be read that way. The
withheld fact is real and step 2 asks for it (were the sections *placed*
differently, or are they *deformed*?), but it is a weaker instrument than
`calibrated-measurements`'s missing unit: a rigid model is also the reasonable
default guess, so `skill+silent` can reach it without asking. `skill` versus
`no-skill` is the comparison this case is built to inform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .._benchmark import Case, Layer
from .._fixture import Attempt, Fixture, Metric, Outcome, read_array, save_png
from .._respondent import Persona

SKILL = "align-stack-by-features"

# Set from measurement, not from taste. Over four seeds of this construction,
# the skill's own recipe lands at a residual ratio of 0.026-0.054 with its worst
# section 0.50-0.56 px out; registering every section to the first one with no
# gate on the inlier count lands at 0.881-1.071 and 42-62 px, with more than
# half the sections tens of px adrift.
#
# `worst_section_offset_px` is the discriminating one and it is not delicately
# placed -- a factor of 5 above every passing run and 14 below every failing
# one. `residual_ratio` is the weaker of the pair, because it is a median over
# sections and so is diluted by the head of the stack that both routes get
# right; its limit sits ~4.7x above passing and ~3.5x below the narrowest
# failure.
TOLERANCE = {
    "residual_ratio": 0.25,
    "worst_section_offset_px": 3.0,
}

#: Sections, and the placement each one gets relative to the one before it.
#: Both are far outside what an intensity method tolerates — that is the point
#: of the skill, and `drift-correction`'s fixture deliberately sits below it.
N_SECTIONS = 12
STEP_PX = 12.0
STEP_DEG = 2.0

#: How many sections a feature survives. This is the dial that makes the
#: reference match die, and it is genuinely two-sided: at 3.0 the direct inliers
#: are gone by section 4 or 5 while consecutive pairs stay above 200 throughout,
#: which is the regime the skill is about. Raising it to 4.0 was measured to let
#: the ungated route succeed on all but the last two sections (residual ratio
#: 0.028 against the gated route's 0.027), i.e. to stop discriminating at all.
FEATURE_LIFETIME = 3.0

#: The field of view. Sections are cut from a canvas far larger, so every pixel
#: in every section is content the "specimen" really had there — see
#: `_canvas_halfwidth`.
SHAPE = (256, 256)
BACKGROUND = 100.0
BLOB_SIGMA = 3.0
#: Spread over the whole canvas, of which a section window is about a ninth --
#: so this is ~290 objects in frame, and SIFT finds ~540 keypoints per section.
N_BLOBS = 2600


def _canvas_halfwidth() -> int:
    """Half-width of the canvas sections are cut from.

    A section is a *window* onto a larger specimen, rotated and translated. If
    the canvas is only section-sized, the warp has to invent the pixels it
    rotates in from outside — and the width of what it invents is the placement,
    which is the withheld answer, readable off the borders with no registration
    at all. So the canvas has to reach past the furthest corner any window
    visits: the window's own half-diagonal, plus the total translation.
    """
    half_diagonal = np.hypot(*SHAPE) / 2.0
    total_translation = STEP_PX * (N_SECTIONS - 1) * np.sqrt(2.0)
    return int(np.ceil(half_diagonal + total_translation)) + 8


# --- the fixture -----------------------------------------------------------


def _specimen(seed: int, canvas: tuple[int, int]) -> np.ndarray:
    """Blobs that live for a stretch of the stack and then are gone.

    Rendered by splatting points and blurring once, rather than by summing a
    Gaussian per blob: the canvas is several times the field of view and this is
    what keeps building the fixture cheap enough to run in an ordinary suite.
    """
    rng = np.random.default_rng(seed)
    depth = rng.uniform(-1.0, N_SECTIONS + 1.0, N_BLOBS)
    extent = rng.uniform(0.6, 1.0, N_BLOBS) * FEATURE_LIFETIME
    yy = rng.integers(0, canvas[0], N_BLOBS)
    xx = rng.integers(0, canvas[1], N_BLOBS)
    amplitude = rng.uniform(3000.0, 9000.0, N_BLOBS)

    sections = []
    for k in range(N_SECTIONS):
        alive = np.exp(-0.5 * ((k - depth) / extent) ** 2)
        img = np.zeros(canvas, dtype=np.float32)
        np.add.at(img, (yy, xx), amplitude * alive)
        sections.append(ndimage.gaussian_filter(img, BLOB_SIGMA) + BACKGROUND)
    return np.stack(sections)


def _placements() -> list[np.ndarray]:
    """3x3 matrices carrying reference coordinates to section-k coordinates.

    A steadily turning walk rather than a straight line: a pure ramp is
    separable in y and x, so an estimator that had collapsed one axis could
    still look plausible on the other.
    """
    out = []
    for k in range(N_SECTIONS):
        heading = 0.5 + 0.3 * np.sin(k / 3.0)
        angle = np.deg2rad(STEP_DEG * k)
        ty = STEP_PX * k * np.sin(heading)
        tx = STEP_PX * k * np.cos(heading)
        cos, sin = np.cos(angle), np.sin(angle)
        out.append(np.array([[cos, -sin, tx], [sin, cos, ty], [0.0, 0.0, 1.0]]))
    return out


def _cut(plane: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """The `SHAPE` window at the canvas centre, after `matrix` is applied.

    `ndimage.affine_transform` takes an output -> input map, which is what the
    inverse of a placement is. Every sampled coordinate lands inside the canvas
    by `_canvas_halfwidth`, so nothing here is extrapolated.
    """
    half = np.array(plane.shape) / 2.0
    centre = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    centre[:2, 2] = half[::-1]
    # (x, y) matrix -> the (row, col) convention ndimage wants
    flip = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    full = flip @ (centre @ np.linalg.inv(matrix) @ np.linalg.inv(centre)) @ flip
    offset = np.array(
        [(plane.shape[0] - SHAPE[0]) // 2, (plane.shape[1] - SHAPE[1]) // 2],
        dtype=float,
    )
    shifted = full.copy()
    shifted[:2, 2] += full[:2, :2] @ offset
    return ndimage.affine_transform(
        plane,
        shifted[:2, :2],
        offset=shifted[:2, 2],
        output_shape=SHAPE,
        order=1,
        mode="constant",
        cval=BACKGROUND,
    ).astype(np.float32)


@dataclass(frozen=True)
class SerialSections:
    """A stack whose sections were each placed independently.

    `data["stack"]` is what the agent sees. `truth["canonical"]` is the same
    twelve sections cut without any placement applied — the aligned stack, to
    interpolation precision, and the thing no acquisition can supply.
    """

    case_id: str = "reference-match-runs-out"
    seed: int = 0

    def __call__(self) -> Fixture:
        half = _canvas_halfwidth()
        canvas = (2 * half, 2 * half)
        specimen = _specimen(self.seed, canvas)
        matrices = _placements()

        identity = np.eye(3)
        stack = np.stack([_cut(p, m) for p, m in zip(specimen, matrices, strict=True)])
        canonical = np.stack([_cut(p, identity) for p in specimen])

        turn = STEP_DEG * (N_SECTIONS - 1)
        shift = float(np.hypot(matrices[-1][1, 2], matrices[-1][0, 2]))
        return Fixture(
            skill_id=SKILL,
            case_id=self.case_id,
            kind="synthetic",
            provenance=(
                f"procedural: {N_SECTIONS} sections, seed {self.seed}, cut from a "
                f"{canvas[0]}x{canvas[1]} canvas, last section placed {shift:.0f} px "
                f"and {turn:.0f} deg off the first, features surviving "
                f"~{FEATURE_LIFETIME} sections"
            ),
            about=(
                f"Content turns over along the stack, so features shared with "
                f"section 0 are gone well before section {N_SECTIONS - 1}. "
                "Registering every section to the first one keeps returning a "
                "confident transform as the inliers fall to a handful, and the "
                "tail of the stack ends up hundreds of px out with nothing in "
                "the run's own output saying so."
            ),
            data={"stack": stack},
            truth={
                "canonical": canonical,
                # Step 2's question, and the fact the pixels cannot answer.
                "deformed": False,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _inner(shape: tuple[int, int]) -> tuple[slice, slice]:
    """The crop metrics are computed on.

    A section rotated into place has corners that came from outside its own
    window, so every run pays the same edge cost. Charging for it would score
    the border rather than the alignment.
    """
    margin = int(np.ceil(np.hypot(*shape) / 2.0 * np.deg2rad(STEP_DEG * N_SECTIONS)))
    return slice(margin, shape[0] - margin), slice(margin, shape[1] - margin)


def _residual_offset(reference: np.ndarray, got: np.ndarray) -> float:
    """How far `got` still sits from `reference`, in px, by phase correlation.

    `normalization=None` is the same point `drift-correction` and `stitch-tiles`
    both make: skimage's `"phase"` default whitens frequency bins holding only
    numerical noise and buries the peak on smooth microscopy content.
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

    Both metrics compare against `truth["canonical"]`, never against a
    transform the run reported. A run that mis-stated its own transforms but
    produced a correctly aligned stack passes, and one that reported beautiful
    transforms and warped with the wrong one fails — which is the right way
    round, and it is why the task asks only for the stack.
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
            "they are 15 microns apart in the block, so you are looking at "
            "genuinely different tissue as you go through the stack — that is "
            "expected, it is not a problem with the imaging"
        ),
        "how far apart the sections are": "15 microns, cut in the order given",
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
    task=TASK,
    persona=HISTOLOGIST,
    build=SerialSections(),
    layers=(Layer("sections", "stack"),),
    collect={"aligned": "aligned"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="serial sections",
    # It must be able to answer step 2: were the sections placed differently, or
    # are they deformed, and is the changing content expected.
    persona_must_know=("mounted", "placed", "distorted", "microns"),
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
