"""Local thickness as benchmark data: how thick is this structure?

A deferred-tier case (`docs/skill-candidates.md`). Local thickness was
**prescreened and dropped** 2026-08-03 for the shipped catalog — both cold
Sonnet arms named Hildebrand & Ruegsegger, passed `sampling=spacing`, and swept
radii largest-first, matching or beating the reference. It is here anyway,
because the *work* is real whether or not a skill for it is served, and because
the rejection is conditional on the consuming tier: Haiku returned
**1.50 / 1.44 / 1.60**, bit-identical to the naive answer, against a truth of
5.00 / 3.00 / 2.40. This case is what makes that re-measurable rather than
remembered.

**The distinction it rests on.** Local thickness is the diameter of the largest
inscribed sphere *containing* a voxel (Hildebrand & Ruegsegger 1997), not twice
that voxel's distance to the boundary. The two agree only at a medial axis, and
averaged over an object they come apart by a factor of two or more — a voxel
just under the surface of a 5 um sphere sits inside the same 5 um sphere as the
centre does, while its own distance transform reads nearly zero.

Measured on this fixture, at 1.0 x 0.25 x 0.25 um voxels:

  ==========================================  ========  ========  ========
  route                                         sphere       rod      slab
  ==========================================  ========  ========  ========
  truth                                           5.00      3.00      2.40
  Hildebrand, spacing honoured                    5.02      3.18      2.48
  ..relative error                               0.004     0.059     0.033
  naive 2x distance transform                     1.54      1.52      1.36
  ..relative error                               0.692     0.495     0.434
  Hildebrand, voxels assumed cubic                1.31      0.97      2.46
  ..relative error                               0.739     0.678     0.026
  ==========================================  ========  ========  ========

`TOLERANCE` sits in that gap: the reference's worst object is 0.059 and the
nearest wrong route's best is 0.434, so 0.20 is about three times above a clean
run and less than half of the closest failure.

**Two wrong routes, and the second is why the voxels are anisotropic.** The
naive transform fails on all three objects. Assuming cubic voxels fails on the
sphere and the rod and is *right about the slab* — the slab's thin direction
runs along a fine axis, so the z step never enters its answer. One object
agreeing is exactly how a spacing bug survives a spot check, which is why the
case scores three shapes rather than one.

**What is not withheld, deliberately.** The voxel spacing is in the task text.
The screen this case reproduces disclosed it too (protocol §6: disclose the
environment, withhold only the skill), and the gap being measured is the
definition of local thickness rather than an elicitation. The persona is here
for realism and holds no part of the answer.

**A run that recognises the primitives may measure them directly**, and that is
a pass on the merits rather than a back door: the metric is the thickness, not
the route to it, and the failure this case catches — averaging a distance
transform — is not avoided by recognising a sphere. What the primitives do cost
is a fourth object with no closed form, which is worth adding if this case is
ever used to compare *methods* rather than to catch the naive one.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    save_png,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

NAMESPACE = "local-thickness"
CASE_ID = "three-primitives-on-anisotropic-voxels"

#: From the table in the module docstring, not from taste. One limit for all
#: three objects: the reference's worst is 0.059 and the nearest wrong route's
#: best is 0.434, so nothing is gained by tuning it per shape.
TOLERANCE = {
    "thickness_1_rel_err": 0.20,
    "thickness_2_rel_err": 0.20,
    "thickness_3_rel_err": 0.20,
}

SPACING = (1.0, 0.25, 0.25)  #: um per voxel, z:xy = 4:1
SHAPE = (30, 160, 200)  #: 30 x 40 x 50 um

#: Diameter in microns of the largest inscribed sphere, which for each of these
#: primitives is the same everywhere inside it. Exact by construction, which is
#: what lets the tolerance be read off the routes rather than off a reference.
TRUTH_UM = {1: 5.0, 2: 3.0, 3: 2.4}

SPHERE_CENTRE = (15.0, 8.0, 12.0)
ROD_AXIS_Z, ROD_AXIS_Y = 15.0, 20.0
ROD_X = (6.0, 44.0)
SLAB_Y = 32.0
SLAB_Z = (8.0, 22.0)
SLAB_X = (8.0, 42.0)

NOISE_SEED = 17


@dataclass(frozen=True)
class ThreePrimitives:
    """Three structures whose thickness only the voxel size turns into microns."""

    shape: tuple[int, int, int] = SHAPE
    spacing: tuple[float, float, float] = SPACING

    def _positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zz, yy, xx = np.mgrid[0 : self.shape[0], 0 : self.shape[1], 0 : self.shape[2]]
        return (
            zz * self.spacing[0],
            yy * self.spacing[1],
            xx * self.spacing[2],
        )

    def _labels(self) -> np.ndarray:
        z, y, x = self._positions()
        labels = np.zeros(self.shape, np.uint8)

        cz, cy, cx = SPHERE_CENTRE
        radius = TRUTH_UM[1] / 2
        labels[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius**2] = 1

        radius = TRUTH_UM[2] / 2
        labels[
            ((z - ROD_AXIS_Z) ** 2 + (y - ROD_AXIS_Y) ** 2 <= radius**2)
            & (x >= ROD_X[0])
            & (x <= ROD_X[1])
        ] = 2

        # Normal along y -- a fine axis, so this is the object a run that
        # ignored the z step still gets right. See the module docstring.
        labels[
            (np.abs(y - SLAB_Y) <= TRUTH_UM[3] / 2)
            & (z >= SLAB_Z[0])
            & (z <= SLAB_Z[1])
            & (x >= SLAB_X[0])
            & (x <= SLAB_X[1])
        ] = 3
        return labels

    def __call__(self) -> Fixture:
        labels = self._labels()
        mask = labels > 0

        # The stack the segmentation came from: blurred by a PSF wider between
        # planes than within one, so the picture agrees with the voxel size the
        # prompt quotes. Here to be looked at, not measured.
        rng = np.random.default_rng(NOISE_SEED)
        signal = ndi.gaussian_filter(mask.astype(np.float32), sigma=(0.7, 1.4, 1.4))
        image = 100.0 + 800.0 * signal
        image = image + rng.normal(0, np.sqrt(np.maximum(image, 1.0)) * 0.5)
        image = (image + rng.normal(0, 3.0, self.shape)).astype(np.float32)

        counts = {label: int((labels == label).sum()) for label in TRUTH_UM}
        # The properties the case rests on, checked before anyone pays for a
        # run. None of them is visible from the arrays alone.
        for label, count in counts.items():
            assert count > 500, (
                f"object {label} is {count} voxels, too few for a mean thickness "
                "to be about the shape rather than about quantisation"
            )
        assert ndi.label(mask)[1] == 3, (
            "the three structures are not three components, so a run could "
            "report one object's thickness for another's"
        )
        assert self.spacing[0] / self.spacing[2] >= 3, (
            "the z step is not coarse enough for cubic voxels to be a wrong "
            "answer, and that is one of the two routes this case separates"
        )
        thinnest = min(TRUTH_UM.values())
        assert thinnest > 2 * self.spacing[0], (
            f"the thinnest object is {thinnest} um against a {self.spacing[0]} um "
            "z step, so quantisation alone would swamp the measurement"
        )

        return Fixture(
            provenance=(
                f"procedural: a sphere, a rod and a slab of diameter "
                f"{TRUTH_UM[1]}/{TRUTH_UM[2]}/{TRUTH_UM[3]} um in "
                f"{self.shape[0]}x{self.shape[1]}x{self.shape[2]} voxels at "
                f"{self.spacing} um, noise seed {NOISE_SEED}"
            ),
            about=(
                "Three segmented structures of exactly known thickness — "
                f"{TRUTH_UM[1]}, {TRUTH_UM[2]} and {TRUTH_UM[3]} um — sampled at "
                f"{self.spacing[1]} um laterally and {self.spacing[0]} um between "
                "planes. Local thickness is the diameter of the largest inscribed "
                "sphere containing a voxel, and twice the distance transform is "
                "not it: averaged over these objects the naive quantity reads "
                "1.54/1.52/1.36 um, 43-69% low. Assuming cubic voxels reads "
                "1.31/0.97/2.46 — wrong on two objects and right on the third, "
                "whose thin direction runs along a fine axis."
            ),
            data={"structures": image, "segmentation": labels},
            truth={
                "thickness_um": np.array(
                    [TRUTH_UM[1], TRUTH_UM[2], TRUTH_UM[3]], float
                ),
                "spacing_um": np.array(self.spacing, float),
                "voxel_counts": np.array([counts[i] for i in sorted(counts)], int),
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    One metric per object, relative rather than absolute: a thickness in voxels
    against one in microns is wrong by a factor, not by an amount, and the
    factor is exactly what the spacing route gets wrong.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    want_all = fixture.truth.get("thickness_um")
    for index, label in enumerate(sorted(TRUTH_UM)):
        name = f"thickness_{label}_rel_err"
        if want_all is None:
            metrics.append(
                Metric(
                    name,
                    None,
                    limits[name],
                    unavailable="the fixture carries no thickness truth",
                )
            )
            continue
        want = float(np.asarray(want_all, float).reshape(-1)[index])
        got, why = read_scalar(attempt, f"thickness_{label}_um")
        if got is None:
            metrics.append(Metric(name, None, limits[name], unavailable=why))
            continue
        metrics.append(
            Metric(name, abs(got - want) / max(abs(want), 1e-12), limits[name])
        )
        detail[f"thickness_{label}_reported_um"] = round(got, 3)
        detail[f"thickness_{label}_true_um"] = want

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    image = fixture.data.get("structures")
    labels = fixture.data.get("segmentation")
    if image is None or labels is None:
        return
    image = np.asarray(image, float)
    labels = np.asarray(labels, float)

    # Through z, and through x. The second is the view that shows how few planes
    # the objects span, which is where the anisotropy does its damage.
    save_png(image.max(axis=0), where / "structures-xy.png")
    save_png(labels.max(axis=0), where / "segmentation-xy.png")
    save_png(np.repeat(labels.max(axis=2), 4, axis=1), where / "segmentation-zy.png")

    rows = ["object,truth_um,reported_um"]
    for label in sorted(TRUTH_UM):
        got = attempt.arrays.get(f"thickness_{label}_um")
        got = "" if got is None else f"{np.asarray(got, float).reshape(-1)[0]:g}"
        rows.append(f"{label},{TRUTH_UM[label]:g},{got}")
    (where / "thickness.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A segmented 3D stack is open in the napari viewer:

  structures    float32 (30, 160, 200) — the raw stack
  segmentation  uint8   (30, 160, 200) — three structures, labelled 1, 2 and 3

The voxels are 1.0 micron between planes and 0.25 microns in y and x. They are
not cubic.

There is no tensor server in this session, so those layers are the only source
of pixels. The segmentation is done; do not re-segment it.

For each of the three labelled structures I need its mean local thickness, in
microns — the thickness of the material itself, averaged over the structure,
not a bounding size and not a surface-to-surface distance along an axis.

The person who acquired and segmented this is here and can answer questions. To
ask them something, reply in plain text without calling a tool, and their answer
will come back as the next message.

When you are finished, leave three names bound in the kernel namespace:

  thickness_1_um   mean local thickness of structure 1, in microns
  thickness_2_um   mean local thickness of structure 2, in microns
  thickness_3_um   mean local thickness of structure 3, in microns

Those names are how your result is collected, so bind them exactly.
"""

#: Self-sufficient: the spacing is in the prompt and the definition is the thing
#: under test, so this person holds no part of the answer. Note what is *not*
#: here — nothing about inscribed spheres, distance transforms or which shapes
#: these are. The last one matters: naming the primitives would let a run quote
#: a diameter it never measured.
MATERIALS_SCIENTIST = Persona(
    name="operator-local-thickness",
    facts={
        "what the sample is": (
            "a porous scaffold we print and then image, to check the struts "
            "came out the size we asked for"
        ),
        "how the segmentation was made": (
            "a threshold and then a manual tidy-up. I am confident about the "
            "boundaries — that part is not in question"
        ),
        "whether the stack was resampled": (
            "no. That is how it came off the microscope, and the planes really "
            "are further apart than the pixels within one"
        ),
        "what the numbers are for": (
            "comparing print runs against each other and against what the "
            "design file asked for, so they have to be in real units"
        ),
        "why there are three of them": (
            "they came off three different runs. I keep them in one stack so "
            "nothing about the imaging can differ between them"
        ),
    },
    background=(
        "You print porous scaffolds, image them on a confocal microscope and "
        "segment the result yourself. You are happy to answer questions about "
        "the sample, the microscope and the segmentation."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=MATERIALS_SCIENTIST,
    fixture=Procedural(ThreePrimitives()),
    layers=(
        Layer("structures", "structures"),
        Layer("segmentation", "segmentation", kind="labels"),
    ),
    collect={
        "thickness_1_um": "thickness_1_um",
        "thickness_2_um": "thickness_2_um",
        "thickness_3_um": "thickness_3_um",
    },
    score=verify,
    save_artifacts=save_artifacts,
)
