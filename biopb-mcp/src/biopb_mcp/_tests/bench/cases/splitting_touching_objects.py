"""Touching nuclei as benchmark data: how many are there, and where do they part?

A deferred-tier case (`docs/skill-candidates.md`). Splitting touching objects was
**ablated and dropped** 2026-08-03 — three cold arms all passed
`sampling=spacing` to the distance transform, all used marker-controlled
watershed masked to the input, and every one matched or beat the skill-informed
reference. It is here anyway, because that screen ran two thirds at Opus and the
entry sits in the *untested at the low tier* remainder of the queue: nobody has
measured whether the spacing argument survives one tier down.

**This case is the instrument for that measurement, not a claim about it.**
Unlike `fibre-orientation` or `strahler-ordering`, there is no low-tier number
here to reproduce. What the module can honestly promise is the other half of the
screening protocol's rule 4 — that the fixture separates the named routes,
measured, before anyone pays for a run.

Measured on this fixture, F1 at IoU 0.5 via `segmentation_qc.match_labels`:

  ==========================================  ===========  ===========
  route                                        cluster_a    cluster_b
                                                 (7:1)      (isotropic)
  ==========================================  ===========  ===========
  physical EDT, `sampling=spacing`                 1.000        1.000
  the same code with `sampling=None`               0.809        1.000
  ..the same, suppression radius 1.5 um            0.333        1.000
  voxel EDT + `min_distance=5` px                  0.016        1.000
  voxel EDT + `min_distance=10` px                 0.255        1.000
  watershed on the raw intensity                   0.704        0.571
  every local maximum a seed                       0.980        1.000
  no split at all (connected components)           0.649        0.706
  ==========================================  ===========  ===========

Three things that table is chosen to show.

* **The spacing is the whole of it.** Rows 1 and 2 are the same dozen lines of
  code differing in one keyword, and they are 0.19 apart on `cluster_a` and
  identical on `cluster_b`. `cluster_b` is the control: its voxels are cubic, so
  `sampling=spacing` and `sampling=None` differ there by a scale factor a
  watershed cannot see. A run that fails a and passes b failed *at the
  anisotropy* — a stronger statement than one scene can make.
* **The correct route is not a parameter lottery.** The reference scores 1.000 at
  every seed-suppression radius from 1.5 to 5.0 um. The naive one scores 0.333 to
  0.809 over that same sweep, so the wrong keyword does not merely shift the
  answer: it makes the answer depend on a parameter that should not matter.
* **Two rows are here because they did *not* separate**, and dropping them would
  misrepresent what was screened. Seeding on every local maximum costs 0.020
  (0.980) rather than the collapse the original entry's "seed suppression" claim
  implies — the peak mask's connected components already collapse each plateau to
  one seed — so that half of the entry is not armed here and nothing in this case
  tests it. Watershed on the raw intensity fails *both* scenes (0.704 / 0.571),
  so it is a real failure but not an anisotropy failure; the control is what
  tells the two apart.

`TOLERANCE` is 0.10 of F1 shortfall on each scene. The reference is exact, so the
whole margin faces the trap: 1.9x under the naive route's best score across the
radius sweep, and far under everything else in the table. It is the tightest
two-sided gap among these deferred-tier cases, and it is tight for a reason no
tuning fixes — the naive route still gets every xy-neighbouring pair right, so
its ceiling is set by how many pairs are stacked axially, not by a parameter.

**Most touching pairs are stacked along z, and that is the fixture's one
deliberate bias.** A pair side by side in xy splits correctly whether or not the
spacing was passed, so a scene that mixed the two evenly would spend most of its
objects diluting the thing it exists to measure. The mix is stated in the
provenance line rather than left implicit.

**The truth is a nearest-centre partition**, which is also what a distance
watershed approximates for equal spheres — so the radii are held within a few
percent of each other on purpose. Where radii differ the two definitions part
company (an EDT splits equidistant from the *surfaces*, not from the centres) and
the case would be scoring that discrepancy instead of the spacing.

**Nothing is withheld.** The prompt carries both voxel sizes, the definition of
the score and the fact that the mask merges whatever touches. The persona is here
for realism and holds no part of the answer.
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
    save_png,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

NAMESPACE = "splitting-touching-objects"
CASE_ID = "a-spheroid-at-seven-to-one"

#: F1 shortfall (``1 - f1``) at IoU 0.5, one per scene. From the table in the
#: module docstring: the reference is exact on both, so the entire allowance
#: faces the naive route's 0.191 on `cluster_a`.
TOLERANCE = {"f1_shortfall_a": 0.10, "f1_shortfall_b": 0.10}

#: 1.4 um between planes against 0.2 um in the plane. The ratio the dropped
#: entry screened at, kept so the two are comparable.
SPACING_A = (1.4, 0.2, 0.2)
BOX_A = (39.2, 50.0, 50.0)
#: The control. Cubic voxels, where the spacing argument cannot change anything.
SPACING_B = (0.5, 0.5, 0.5)
BOX_B = (30.0, 30.0, 30.0)

RADIUS_UM = 4.0
N_A = 24
N_B = 10
#: How often a touching pair is stacked along the axial direction. See the
#: docstring: an xy pair splits correctly either way, so it measures nothing.
AXIAL_FRACTION = 0.85

SEED_PACK_A, SEED_RENDER_A = 4, 1
SEED_PACK_B, SEED_RENDER_B = 9, 2


def _shape(box_um, spacing) -> tuple[int, ...]:
    return tuple(int(round(b / s)) for b, s in zip(box_um, spacing, strict=True))


SHAPE_A = _shape(BOX_A, SPACING_A)
SHAPE_B = _shape(BOX_B, SPACING_B)


def _pack(rng, n, box_um, axial_fraction) -> tuple[list, list, int]:
    """Sphere centres in micrometres, deliberately in touching pairs.

    A pair is placed at 1.8 radii, which is contact with a shallow neck — the
    configuration a segmentation merges. Anything closer would put the neck below
    the axial sampling and make the pair genuinely unresolvable, which is a
    different experiment.
    """
    centres: list[np.ndarray] = []
    radii: list[float] = []
    axial = 0
    for _ in range(20000):
        if len(centres) >= n:
            break
        base = np.array(
            [rng.uniform(RADIUS_UM * 1.3, b - RADIUS_UM * 1.3) for b in box_um]
        )
        if any(np.linalg.norm(base - c) < 2.6 * RADIUS_UM for c in centres):
            continue
        radius = RADIUS_UM * rng.uniform(0.97, 1.03)
        centres.append(base)
        radii.append(radius)
        if len(centres) >= n or rng.random() >= 0.85:
            continue
        direction = rng.normal(size=3)
        stacked = rng.random() < axial_fraction
        if stacked:
            direction = np.array([1.0, 0.0, 0.0]) * np.sign(direction[0])
        direction /= np.linalg.norm(direction)
        mate = base + direction * radius * 1.8
        inside = all(
            RADIUS_UM * 1.1 < mate[i] < box_um[i] - RADIUS_UM * 1.1 for i in range(3)
        )
        if inside and all(np.linalg.norm(mate - c) > 1.7 * RADIUS_UM for c in centres):
            centres.append(mate)
            radii.append(RADIUS_UM * rng.uniform(0.97, 1.03))
            axial += int(stacked)
    return centres[:n], radii[:n], axial


def _draw(shape, spacing, centres, radii) -> tuple[np.ndarray, np.ndarray]:
    """``(truth labels, mask)`` for spheres at *centres*, nearest centre wins.

    Each sphere is drawn inside its own bounding box: the whole point of the
    anisotropic scene is that a sphere occupies a few planes and hundreds of
    columns, so a full-volume distance per object would be most of the build
    cost for a few percent of the voxels.
    """
    spacing = np.asarray(spacing, float)
    truth = np.zeros(shape, np.int32)
    nearest = np.full(shape, np.inf, np.float32)
    for label, (centre, radius) in enumerate(zip(centres, radii, strict=True), start=1):
        lo = np.maximum(((centre - radius) / spacing).astype(int) - 1, 0)
        hi = np.minimum(((centre + radius) / spacing).astype(int) + 2, shape)
        grids = np.meshgrid(
            *[np.arange(a, b) * s for a, b, s in zip(lo, hi, spacing, strict=True)],
            indexing="ij",
        )
        box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        distance = np.linalg.norm(
            np.stack(grids, -1) - np.asarray(centre, float), axis=-1
        ).astype(np.float32)
        closer = (distance <= radius) & (distance < nearest[box])
        nearest[box] = np.where(closer, distance, nearest[box])
        truth[box] = np.where(closer, label, truth[box])
    return truth, truth > 0


def _render(mask, spacing, seed) -> np.ndarray:
    """A nuclear stain over *mask*: chromatin texture, then shot noise.

    The texture is what stops the intensity from being a smoothed copy of the
    mask. Without it, seeding on the raw intensity is as good as seeding on the
    distance transform and the entry's classic wrong answer is not wrong here.
    """
    rng = np.random.default_rng(seed)
    speckle = rng.normal(0, 1, mask.shape).astype(np.float32)
    speckle = ndi.gaussian_filter(speckle, np.maximum(0.8, 1.2 / np.asarray(spacing)))
    speckle /= speckle.std()
    body = ndi.gaussian_filter(mask.astype(np.float32), 1.0)
    image = body * (800.0 + 260.0 * speckle) + 100.0
    return (image + rng.normal(0, np.sqrt(np.maximum(image, 1.0)))).astype(np.float32)


@dataclass(frozen=True)
class TwoClusters:
    """Two clusters of touching nuclei: one at 7:1, one with cubic voxels."""

    def _scene(self, shape, spacing, n, box, seed_pack, seed_render):
        rng = np.random.default_rng(seed_pack)
        centres, radii, axial = _pack(rng, n, box, AXIAL_FRACTION)
        truth, mask = _draw(shape, spacing, centres, radii)
        clusters = ndi.label(mask, structure=np.ones((3, 3, 3)))[0].astype(np.int32)
        return _render(mask, spacing, seed_render), clusters, truth, axial

    def __call__(self) -> Fixture:
        image_a, clusters_a, truth_a, axial_a = self._scene(
            SHAPE_A, SPACING_A, N_A, BOX_A, SEED_PACK_A, SEED_RENDER_A
        )
        image_b, clusters_b, truth_b, axial_b = self._scene(
            SHAPE_B, SPACING_B, N_B, BOX_B, SEED_PACK_B, SEED_RENDER_B
        )

        # The two properties the case rests on, checked before anyone pays for a
        # run. Neither is visible from the images.
        assert int(truth_a.max()) == N_A and int(truth_b.max()) == N_B, (
            f"the packing lost objects: {int(truth_a.max())}/{N_A} and "
            f"{int(truth_b.max())}/{N_B} drawn"
        )
        assert int(clusters_a.max()) < N_A, (
            "no nuclei touch in cluster_a, so there is nothing to split and the "
            "case measures a connected-component labelling"
        )
        assert axial_a >= 5, (
            f"only {axial_a} pairs in cluster_a are stacked along z — an xy pair "
            "splits correctly with or without the spacing, so the trap is not armed"
        )

        return Fixture(
            provenance=(
                f"procedural: {N_A} nuclei of radius {RADIUS_UM:g} um in "
                f"{SHAPE_A} at {SPACING_A} um ({axial_a} axially stacked pairs), "
                f"{N_B} in {SHAPE_B} at {SPACING_B} um ({axial_b} axial), seeds "
                f"{(SEED_PACK_A, SEED_RENDER_A)} and {(SEED_PACK_B, SEED_RENDER_B)}"
            ),
            about=(
                f"Two clusters of touching nuclei, {int(clusters_a.max())} "
                f"connected components covering {N_A} objects in cluster_a and "
                f"{int(clusters_b.max())} covering {N_B} in cluster_b. Voxels are "
                f"{SPACING_A} um in a and cubic {SPACING_B[0]:g} um in b. A "
                "distance watershed that is told the spacing scores F1 1.000 on "
                "both; the same code without it scores 0.809 on a and 1.000 on b."
            ),
            data={
                "nuclei_a": image_a,
                "mask_a": clusters_a,
                "nuclei_b": image_b,
                "mask_b": clusters_b,
            },
            truth={"labels_a": truth_a, "labels_b": truth_b},
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _read_labels(attempt: Attempt, key: str, shape) -> tuple[np.ndarray | None, str]:
    """A label volume the run left behind, or why it cannot be scored.

    Deliberately tolerant about dtype — a run that built its labels as float64
    has answered the question — and strict about shape, which is the one way a
    plausible-looking array is silently about something else.
    """
    got = attempt.arrays.get(key)
    if got is None:
        return None, f"the run left no `{key}`"
    got = np.asarray(got)
    if got.shape != tuple(shape):
        return None, f"the run's `{key}` is {got.shape}, not {tuple(shape)}"
    if not np.isfinite(np.asarray(got, float)).all():
        return None, f"the run's `{key}` is not finite everywhere"
    return np.rint(np.asarray(got, float)).astype(np.int32), ""


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score each scene's label volume against its nearest-centre truth.

    F1 at IoU 0.5 is the operating point the dropped entry was screened at, and
    `segmentation_qc` is the same matcher — so a number here is comparable to the
    table on that page rather than merely internally consistent.
    """
    from biopb_mcp.plugins import segmentation_qc

    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    for scene in ("a", "b"):
        name = f"f1_shortfall_{scene}"
        truth = np.asarray(fixture.truth[f"labels_{scene}"])
        got, why = _read_labels(attempt, f"labels_{scene}", truth.shape)
        if got is None:
            metrics.append(Metric(name, None, limits[name], unavailable=why))
            continue
        score = segmentation_qc.match_labels(truth, got, iou_threshold=0.5)
        if not np.isfinite(score.f1):
            metrics.append(
                Metric(
                    name,
                    None,
                    limits[name],
                    unavailable=f"the run's `labels_{scene}` holds no objects",
                )
            )
            continue
        metrics.append(Metric(name, 1.0 - float(score.f1), limits[name]))
        # What kind of wrong, not just how much: the anisotropy trap merges, and
        # a merge count is what says so when the F1 alone could be either.
        detail[f"scene_{scene}"] = {
            "f1": round(float(score.f1), 4),
            "objects_found": int(got.max()),
            "objects_true": int(truth.max()),
            "tp": int(score.tp),
            "fp": int(score.fp),
            "fn": int(score.fn),
            "splits": int(score.splits),
            "merges": int(score.merges),
        }

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """A mid-plane of each scene, truth beside answer. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    for scene in ("a", "b"):
        image = np.asarray(fixture.data[f"nuclei_{scene}"])
        middle = image.shape[0] // 2
        save_png(image[middle], where / f"nuclei-{scene}.png")
        truth = np.asarray(fixture.truth[f"labels_{scene}"])
        save_png(truth[middle] > 0, where / f"truth-{scene}.png")
        got = attempt.arrays.get(f"labels_{scene}")
        if got is not None and np.asarray(got).shape == truth.shape:
            save_png(np.asarray(got)[middle] > 0, where / f"answer-{scene}.png")

    rows = ["scene,f1,objects_found,objects_true,merges,splits"]
    for scene in ("a", "b"):
        record = outcome.detail.get(f"scene_{scene}")
        if isinstance(record, dict):
            rows.append(
                f"{scene},{record['f1']},{record['objects_found']},"
                f"{record['objects_true']},{record['merges']},{record['splits']}"
            )
        else:
            rows.append(f"{scene},,,,,")
    (where / "split.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = f"""\
Two 3D fields of a nuclear stain are open in the napari viewer, from two
acquisitions of the same spheroid line:

  nuclei_a  float32 {SHAPE_A}   voxels {SPACING_A[0]:g} x {SPACING_A[1]:g} x \
{SPACING_A[2]:g} microns (z, y, x)
  mask_a    labels  {SHAPE_A}
  nuclei_b  float32 {SHAPE_B}   voxels {SPACING_B[0]:g} x {SPACING_B[1]:g} x \
{SPACING_B[2]:g} microns (z, y, x)
  mask_b    labels  {SHAPE_B}

There is no tensor server in this session, so those layers are the only source
of voxels.

The two mask layers are my thresholding of the two stains. They are right about
which voxels are nuclei and wrong about how many nuclei there are: wherever two
nuclei touch, the mask gives them one label between them. I need them separated,
one label per nucleus.

The person who acquired these is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  labels_a   an integer array of shape {SHAPE_A}, one label per nucleus in
             field a, 0 for background
  labels_b   an integer array of shape {SHAPE_B}, the same for field b

The labels themselves need not be in any particular order — they will be matched
to mine by overlap, and scored as F1 at an IoU of 0.5. Those two names are how
your result is collected, so bind them exactly.
"""

#: Self-sufficient: the prompt carries both voxel sizes, what the masks are, and
#: how the answer is scored. Note what is *not* here — nothing about distance
#: transforms, seeds, watershed or anisotropy. This person knows their microscope
#: and their cells, and could not tell you how to split a mask.
IMAGING_SCIENTIST = Persona(
    name="operator-spheroid-nuclei",
    facts={
        "what the sample is": (
            "spheroids of the same line, fixed and stained for DNA. Field a is "
            "one I imaged deep, field b is a small one near the coverslip"
        ),
        "why the two acquisitions differ": (
            "field a is a big spheroid and I had to get through it, so I took "
            "coarse steps in z to keep the exposure down. Field b is small "
            "enough that I could sample it evenly in all three directions"
        ),
        "how the masks were made": (
            "a threshold and a bit of hole filling, nothing clever. It gets the "
            "outline right and it cannot tell two nuclei apart when they touch"
        ),
        "how big a nucleus is": (
            "about eight microns across, and they do not vary much in this line — "
            "that is one of the reasons we use it"
        ),
        "whether nuclei are cut off at the edge": (
            "no, I cropped both fields so that whole nuclei are inside"
        ),
        "what the count is for": (
            "we are comparing growth between treatments, so the number of nuclei "
            "and their individual volumes are what I am after"
        ),
    },
    background=(
        "You imaged two spheroids on a confocal, one deep and one shallow, and "
        "thresholded both. You are happy to answer questions about the sample, "
        "the microscope and how the masks were made."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=IMAGING_SCIENTIST,
    fixture=Procedural(TwoClusters()),
    layers=(
        Layer("nuclei_a", "nuclei_a"),
        Layer("mask_a", "mask_a", kind="labels"),
        Layer("nuclei_b", "nuclei_b"),
        Layer("mask_b", "mask_b", kind="labels"),
    ),
    collect={"labels_a": "labels_a", "labels_b": "labels_b"},
    score=verify,
    save_artifacts=save_artifacts,
    # No `plugins`: the verifier imports `segmentation_qc` itself, and the
    # session needs nothing seeded into it. The matcher is the one the dropped
    # entry was screened with, so a number here is comparable to that page.
)
