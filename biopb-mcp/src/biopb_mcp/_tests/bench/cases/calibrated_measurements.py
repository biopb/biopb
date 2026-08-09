"""`calibrated-measurements` as benchmark data: how big is a voxel?

The cleanest withheld fact in the catalogue, and the reason this case was
written second. `biopb-mcp/docs/skills.md` §10c records what the drift fixture
taught: designing data so the heuristics its author thought of point the wrong
way is not the same as making a fact **unobtainable**, and a capable agent
recovered drift's structural channel by trying both. A voxel size is not like
that. It is categorically absent from an array of numbers — no amount of looking
at the pixels yields µm — so a run either asks or invents, and inventing is
visible in the µm³ column by a factor of 200.

The layer arrives with `scale` at its napari default of all-ones, which is
exactly the condition step 3 of the body makes a blocking checkpoint. The
objects are ellipsoids, so a plausibility check *is* available to the agent
(step 5's "is this median plausible for a nucleus") — deliberately, since that
step is what catches a spacing wrong by 10³, and a fixture where it could not
fire would not exercise it.

**Truth is the voxel count, not the analytic volume.** `regionprops(spacing=)`
multiplies a discrete voxel count by `prod(spacing)`, so the exactly right
answer for a rasterised ellipsoid is `count × 0.1 × 0.1 × 0.5` — not
`4/3·π·abc`, which differs by the rasterisation error. Scoring against the
analytic figure would charge every correct run a few percent and force a
tolerance wide enough to hide real mistakes. The analytic volume is reported in
`detail` instead, where it is context rather than a limit.

**The silent arm is expected to score nothing, and that is the finding.** With
no answer available the body's own instruction is to report pixels and *label
them as pixels* — so a run that follows it leaves `volumes_um3` unbound and the
row reads `no-result`. A run that binds voxel counts under a µm name instead is
the failure the skill exists to prevent, and it lands as `wrong-answer` off by
the same factor. The two are worth telling apart in the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_array,
    relative_error,
    save_png,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

SKILL = "calibrated-measurements"

#: (Z, Y, X) µm per voxel — the fact the fixture strips and the persona holds.
#: Anisotropic on purpose: a run that assumes cubic voxels is wrong by the Z:XY
#: ratio alone, which is the first row of the body's failure table.
SPACING_UM = (0.5, 0.1, 0.1)

# Both tight, because both quantities are exact by construction: the volume is a
# voxel count times a constant, and the spacing is a number the run was told.
# There is no measurement noise here to leave headroom for, and a loose limit
# would let a Z-only mistake (a factor of 5) through as "close enough".
TOLERANCE = {"volume_rel_err": 0.02, "spacing_rel_err": 0.01}

BACKGROUND = 120.0


@dataclass(frozen=True)
class Ellipsoids:
    """Ellipsoids of known voxel extent on an anisotropic grid.

    Laid out on a jittered grid so nothing overlaps and nothing touches a face
    of the volume: a border-clipped object would make the count depend on
    whether the run excluded it, which is a second choice this case is not
    trying to measure.
    """

    shape: tuple[int, int, int] = (16, 256, 256)
    seed: int = 0

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        nz, ny, nx = self.shape
        labels = np.zeros(self.shape, dtype=np.int32)
        zz, yy, xx = np.ogrid[:nz, :ny, :nx]

        centres = [
            (6 if (i + j) % 2 else 10, y, x)
            for i, y in enumerate(np.linspace(40, ny - 40, 4))
            for j, x in enumerate(np.linspace(40, nx - 40, 3))
        ]
        for label, (cz, cy, cx) in enumerate(centres, start=1):
            rz = float(rng.uniform(3.0, 4.0))
            ry = float(rng.uniform(18.0, 26.0))
            rx = ry * float(rng.uniform(0.85, 1.15))
            inside = (
                ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            ) <= 1.0
            labels[inside] = label

        # An intensity image to measure against, and the layer the body says to
        # take the spacing from. Its scale is napari's default all-ones, which
        # is what makes step 3 a blocking checkpoint rather than a lookup.
        image = BACKGROUND + 900.0 * (labels > 0)
        image = image + rng.normal(0.0, 12.0, self.shape)

        counts = np.bincount(labels.ravel(), minlength=len(centres) + 1)[1:]
        voxel_um3 = float(np.prod(SPACING_UM))
        _agrees_with_regionprops(labels, counts * voxel_um3)
        return Fixture(
            provenance=(
                f"procedural: {len(centres)} ellipsoids, seed {self.seed}, "
                f"{self.shape} voxels at {SPACING_UM} µm (Z, Y, X)"
            ),
            about=(
                f"{len(centres)} nuclei-sized ellipsoids on {SPACING_UM[0]} µm "
                f"z-steps and {SPACING_UM[1]} µm pixels. Nothing in the arrays "
                "says so; measuring them as voxels is wrong by "
                f"{1.0 / voxel_um3:.0f}x."
            ),
            data={"image": image.astype(np.float32), "labels": labels},
            truth={
                "volumes_um3": counts * voxel_um3,
                "spacing_um": np.array(SPACING_UM, dtype=float),
                "voxel_counts": counts,
            },
            tolerance=dict(TOLERANCE),
        )


def _agrees_with_regionprops(labels, volumes_um3) -> None:
    """The declared truth, checked against the call the body prescribes.

    Not a reference implementation of the procedure — there is no choice to make
    here, and the choices are what §5 exists to measure. It settles one thing the
    truth silently depends on: that `spacing=` is a per-axis multiplier in the
    array's own axis order, which is the claim this case scores an agent on. If
    that ever stops holding, the truth is wrong and every arm scored against it
    is meaningless, so it raises at build time.
    """
    from skimage.measure import regionprops_table

    measured = regionprops_table(labels, properties=("area",), spacing=SPACING_UM)
    if not np.allclose(measured["area"], volumes_um3, rtol=1e-9):
        raise AssertionError(
            f"{SKILL} fixture: regionprops with spacing={SPACING_UM} does not "
            "agree with voxel count x voxel volume. The truth is wrong."
        )


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score the calibrated table and the spacing that produced it.

    Both are reported, because they fail differently and the pair says which
    mistake happened. A spacing read off the labels layer instead of the image
    is all-ones and lands in `spacing_rel_err`; a spacing correctly obtained and
    then applied to the wrong axis leaves `spacing_rel_err` clean and moves only
    the volumes.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    truth_volumes = np.asarray(fixture.truth["volumes_um3"], float)
    volumes, why = read_array(attempt, "volumes_um3", truth_volumes.shape)
    if volumes is None:
        metrics.append(
            Metric("volume_rel_err", None, limits["volume_rel_err"], unavailable=why)
        )
    else:
        metrics.append(
            Metric(
                "volume_rel_err",
                relative_error(volumes, truth_volumes),
                limits["volume_rel_err"],
            )
        )
        detail["median_volume_um3_reported"] = float(np.median(volumes))
        detail["median_volume_um3_truth"] = float(np.median(truth_volumes))

    truth_spacing = np.asarray(fixture.truth["spacing_um"], float)
    spacing, why = read_array(attempt, "spacing_um", truth_spacing.shape)
    if spacing is None:
        metrics.append(
            Metric("spacing_rel_err", None, limits["spacing_rel_err"], unavailable=why)
        )
    else:
        metrics.append(
            Metric(
                "spacing_rel_err",
                relative_error(spacing, truth_spacing),
                limits["spacing_rel_err"],
            )
        )
        detail["spacing_um_reported"] = [float(v) for v in spacing]

    # Context, never a limit: what the shapes would measure if the rasterisation
    # were exact. A run inside `volume_rel_err` but far from this is measuring
    # the right voxels; a run far from both is measuring the wrong thing.
    counts = np.asarray(fixture.truth["voxel_counts"], float)
    detail["analytic_check_voxels"] = float(counts.sum())
    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """A slice through the volume, and the table the number came from."""
    fixture, attempt = outcome.fixture, outcome.attempt
    image = np.asarray(fixture.data["image"])
    labels = np.asarray(fixture.data["labels"])
    mid = image.shape[0] // 2
    save_png(image[mid], where / "image-mid-z.png")
    save_png(labels[mid] > 0, where / "labels-mid-z.png")

    truth = np.asarray(fixture.truth["volumes_um3"], float)
    got = attempt.arrays.get("volumes_um3")
    got = None if got is None else np.asarray(got, float).ravel()
    rows = ["label,truth_um3,reported,ratio"]
    for i, want in enumerate(truth, start=1):
        if got is not None and got.size == truth.size:
            rows.append(f"{i},{want:.4f},{got[i - 1]:.4f},{got[i - 1] / want:.4f}")
        else:
            rows.append(f"{i},{want:.4f},,")
    (where / "volumes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Two layers are open in the napari viewer: `nuclei`, a (Z, Y, X) = (16, 256, 256)
float32 confocal stack, and `nuclei_labels`, the segmentation of it. There is no
tensor server in this session, so those layers are the only source of pixels.

I need the size of every labelled object reported in physical units, not voxels.

The person who acquired the stack is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  volumes_um3  a float array of one volume per object in cubic microns,
               ordered by increasing label id (1, 2, 3, ...)
  spacing_um   the per-axis physical size of one voxel in microns that you
               used, as three numbers in the array's own axis order (Z, Y, X)

Those two names are how your result is collected, so bind them exactly. If the
physical size of a voxel cannot be established at all, do not guess one: say so
and leave `volumes_um3` unbound rather than putting voxel counts under a name
that claims microns.
"""

#: What this person knows is their microscope, not `regionprops`. The spacing is
#: the whole fact: the arrays carry no physical size, `layer.scale` is all ones,
#: and there is no metadata anywhere in the session that says otherwise.
MICROSCOPIST = Persona(
    name="microscopist-confocal-stack",
    facts={
        "how big one pixel is in x and y": (
            "0.1 microns across, that is the 60x objective with the 1.5x tube lens"
        ),
        "how far apart the z slices are": (
            "0.5 microns per step — I under-sampled Z on purpose to save the sample"
        ),
        "what the objects are": (
            "nuclei of cultured epithelial cells, DAPI. They should come out a "
            "few microns across"
        ),
        "what I need the numbers for": (
            "comparing nuclear size against a second condition imaged on a "
            "different scope, so pixels are no use to me"
        ),
        "whether anything is cropped": (
            "no, every object I care about is fully inside the field"
        ),
    },
    background=(
        "A confocal stack of fixed cells with its segmentation already done. "
        "You are happy to answer questions about the acquisition."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="twelve-nuclei-anisotropic",
    task=TASK,
    persona=MICROSCOPIST,
    fixture=Procedural(Ellipsoids()),
    layers=(
        Layer("nuclei", "image"),
        Layer("nuclei_labels", "labels", kind="labels"),
    ),
    collect={"volumes_um3": "volumes_um3", "spacing_um": "spacing_um"},
    score=verify,
    save_artifacts=save_artifacts,
    # It must be able to answer: the fixture strips the voxel size, and this
    # person knows both the lateral pixel size and the z-step.
    persona_must_know=("0.1 micron", "0.5 micron", "nuclei"),
    # And it must not know the procedure — only the microscope.
    persona_must_not_know=(
        "regionprops",
        "spacing=",
        "skimage",
        "properties=",
        "layer.scale",
    ),
)
