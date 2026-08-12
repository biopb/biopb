"""Four light-sheet views of one bead field, and no correspondence between them.

BigStitcher / multiview-reconstruction sits in the watch list
(`docs/skill-candidates.md`) as **a new skill gated on a data modality we
cannot assume**, and the entry is explicit about which half is the gap:

    What BigStitcher has that `stitch-tiles` does not is the multiview case: 3D
    tiles at multiple angles, registered on interest points (beads) under
    transforms richer than translation.

**This case is deliberately not about the solver.** That question is closed:
#690 measured a maximum spanning tree over NCC-weighted pairwise offsets
against least squares and the tree won -- 100% of tiles on the correct pixel
against 52-92% -- and outlier link rejection bought nothing at all. So nothing
here composes pairwise anything. Every view registers straight to view 0, there
is no link graph, and re-running that argument on this fixture is impossible
rather than merely discouraged.

What is new is everything the modality adds:

**Correspondence is not given.** Four lists of bead positions, in four
different coordinate systems, in no particular order, and no run can index row
i of one against row i of another. Each view sees a different subset -- one
list is 214 detections and another 187 -- because detection falls off with
depth into the sample, which is the reason to acquire four views at all. About
5% of each list is spurious.

**The sampling is anisotropic, so a rotation is only rigid in microns.** The
voxel is 0.40 um along z and 0.13 across, and the views differ by rotations
about y, which is exactly the axis that trades z for x. Fitting the rotation to
voxel indices is not a small units slip on the way out; it is fitting a
different geometry, and it lands 319 um away. A 2D tile grid cannot have this
failure, which is part of why it belongs to this entry and not to
`stitch-tiles`.

**The stage angle is known and is not the answer.** It is recorded to the
degree -- 0, 90, 180, 270 -- but where the sample settled is not, and using it
unrefined lands 6.07 um out.

Measured on the shipped fixture, against a truth of 0 um and 247 beads:

  ==========================================  ==========  =========  ========
  route                                       worst view    n_beads   n error
  ==========================================  ==========  =========  ========
  the truth transforms                           0.00 um        247      0.0%
  reference: microns, nominal, centroid, ICP     0.03 um        247      0.0%
  ------------------------------------------  ----------  ---------  --------
  ICP without the centroid start                 4.32 um        216     12.6%
  nominal plus centroid, no ICP                  2.20 um         96     61.1%
  nominal stage transform only                   6.07 um         15     93.9%
  registered in voxel indices                  319.22 um         20     91.9%
  principal axes, no correspondence             26.82 um         19     92.3%
  every detection kept, none merged              0.03 um        778    215.0%
  view 0's list alone                            0.03 um        190     23.1%
  ==========================================  ==========  =========  ========

Two metrics because the last two rows exist. A run that registers perfectly and
never works out which detections are the same bead has done the hard half and
not the task, and it reports a bead count 3.1x too high while scoring 0.03 um
on the transforms.

Three things the table is load-bearing about:

**The worst view, not the median over all of them.** Scored pooled, one failed
view out of three sits above the median and disappears: the first reference
implementation here scored 0.03 um pooled while view 3 was 4.32 um and 8.7
degrees out. A run has to register every view, so the score is the worst one.

**What made that view fail is worth the fixture existing.** The nominal
transform is a good start for three views and not for the fourth, whose stage
shift is 6.1 um -- outside ICP's basin, where it converges to a confident wrong
answer rather than diverging. Putting the two clouds' centroids together first
fixes it, and that step is bought by nothing else in the pipeline.

**And the reference is exact rather than merely good** -- 0.015 to 0.035 um
across six seeds, with the bead count right every time. :data:`POSITION_LIMIT_UM`
at 0.5 um is therefore not a negotiation with the estimator: it is fourteen
times the worst reference run and an eighth of the nearest wrong route.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_array,
    read_scalar,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

NAMESPACE = "multiview-reconstruction"
CASE_ID = "four-angles-one-bead-field"

# --- the acquisition --------------------------------------------------------

#: (z, y, x). z is 3.1x coarser, which is what makes a rotation about y a
#: question about units before it is a question about geometry.
VOXEL_UM = np.array([0.40, 0.13, 0.13])
VIEW_SHAPE = np.array([160, 480, 480])
NOMINAL_DEG = np.array([0.0, 90.0, 180.0, 270.0])
N_VIEWS = 4

N_BEADS = 260
BEAD_MIN_GAP_UM = 2.2
#: Localisation is worse along the coarse axis, as it is on a real detection.
LOCALISE_SIGMA_UM = np.array([0.11, 0.035, 0.035])

#: Detection falls off with depth into the sample. This is the reason to
#: acquire four views, and it is what makes each list a different subset --
#: without it every view would see every bead and correspondence would be a
#: matter of sorting.
DETECT_NEAR = 0.98
DETECT_FAR = 0.42
SPURIOUS_FRACTION = 0.05

#: Stage repeatability: the angle is written down, the rest is not. View 3's
#: shift is the largest at 6.1 um, which is what puts it outside ICP's basin
#: unless the centroids are matched first.
ANGLE_ERROR_DEG = np.array([0.0, 2.6, -3.4, 1.9])
TILT_ERROR_DEG = np.array([0.0, 1.4, -1.1, 0.8])
SHIFT_UM = np.array(
    [[0.0, 0.0, 0.0], [2.9, -4.1, 3.3], [-3.6, 2.2, -2.8], [4.4, 3.7, -2.1]]
)

N_PROBES_PER_VIEW = 100
N_PROBES = 3 * N_PROBES_PER_VIEW

#: Microns, and the worst view's median rather than the pooled one. Fourteen
#: times the worst reference run over six seeds, an eighth of the nearest wrong
#: route, and a shade over one coarse voxel -- which is the loosest reading of
#: "registered" that means anything.
POSITION_LIMIT_UM = 0.5

#: Fraction of the true count. The reference is exact on every seed, so this is
#: sized against the wrong routes: the nearest is view 0's list alone at 23%.
COUNT_LIMIT = 0.10

SEED = 5


def _rotation(deg_y: float, deg_tilt: float) -> np.ndarray:
    """About y, then a small tilt about z's partner. Axes are (z, y, x)."""
    a = np.deg2rad(deg_y)
    about_y = np.array(
        [[np.cos(a), 0.0, -np.sin(a)], [0.0, 1.0, 0.0], [np.sin(a), 0.0, np.cos(a)]]
    )
    b = np.deg2rad(deg_tilt)
    tilt = np.array(
        [[np.cos(b), np.sin(b), 0.0], [-np.sin(b), np.cos(b), 0.0], [0.0, 0.0, 1.0]]
    )
    return tilt @ about_y


def _transforms(exact: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(R, t)`` taking a point in view 0's physical frame to view *k*, in um.

    With ``exact=False`` this is what the stage recorded: the angle and nothing
    else. It is the only starting point a run is given, and it is wrong.
    """
    centre = VIEW_SHAPE * VOXEL_UM / 2.0
    out = []
    for k in range(N_VIEWS):
        r = _rotation(
            NOMINAL_DEG[k] + (ANGLE_ERROR_DEG[k] if exact else 0.0),
            TILT_ERROR_DEG[k] if exact else 0.0,
        )
        # Rotation about the middle of the field, so the specimen stays in it.
        t = centre - r @ centre + (SHIFT_UM[k] if exact else 0.0)
        out.append((r, t))
    return out


class BeadField:
    """One bead field, seen four times from four angles."""

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(SEED)
        span = VIEW_SHAPE * VOXEL_UM
        centre = span / 2.0
        radius = 0.42 * span.min()

        # Inside a sphere, so no view's field of view clips a different part of
        # the specimen -- the subsets have to differ by depth alone, or the
        # case would be measuring which corner each view can see.
        beads: list[np.ndarray] = []
        while len(beads) < N_BEADS:
            p = centre + rng.normal(size=3) * radius * 0.45
            if np.linalg.norm(p - centre) > radius:
                continue
            if beads and np.linalg.norm(np.array(beads) - p, axis=1).min() < (
                BEAD_MIN_GAP_UM
            ):
                continue
            beads.append(p)
        specimen = np.array(beads)

        truth = _transforms(exact=True)
        views, seen_by = [], np.zeros(len(specimen), int)
        for r, t in truth:
            voxels = (specimen @ r.T + t) / VOXEL_UM
            inside = np.all((voxels > 2) & (voxels < VIEW_SHAPE - 2), axis=1)
            depth = np.clip(voxels[:, 0] / VIEW_SHAPE[0], 0.0, 1.0)
            detected = inside & (
                rng.random(len(specimen))
                < DETECT_NEAR + (DETECT_FAR - DETECT_NEAR) * depth
            )
            seen_by += detected
            found = voxels[detected] + rng.normal(size=(int(detected.sum()), 3)) * (
                LOCALISE_SIGMA_UM / VOXEL_UM
            )
            spurious = rng.uniform(
                4, VIEW_SHAPE - 4, size=(round(SPURIOUS_FRACTION * len(found)), 3)
            )
            both = np.vstack([found, spurious])
            # Shuffled, because a list that arrives in bead order would hand
            # over the correspondence this case is about.
            views.append(both[rng.permutation(len(both))])

        # Probes: real positions expressed in each non-reference view's voxel
        # grid. Not beads -- a probe that landed on a bead would let a run read
        # its answer off the nearest detection instead of transforming it.
        probes_voxel, probes_truth = [], []
        for k in (1, 2, 3):
            r, t = truth[k]
            idx = rng.choice(len(specimen), N_PROBES_PER_VIEW, replace=False)
            here = specimen[idx] + rng.normal(size=(N_PROBES_PER_VIEW, 3)) * 1.5
            probes_voxel.append((here @ r.T + t) / VOXEL_UM)
            probes_truth.append(here)

        confirmed = int((seen_by >= 2).sum())
        assert 200 < confirmed < len(specimen), confirmed
        assert len({len(v) for v in views}) > 1, "every view saw the same number"

        return Fixture(
            provenance=(
                f"synthetic: {N_BEADS} beads, {N_VIEWS} views at "
                f"{'/'.join(str(int(d)) for d in NOMINAL_DEG)} deg about y, "
                f"voxel {tuple(VOXEL_UM)} um (z, y, x), detection {DETECT_NEAR} "
                f"to {DETECT_FAR} with depth, {SPURIOUS_FRACTION:.0%} spurious, "
                f"seed {SEED}"
            ),
            about=(
                "Four bead lists in four coordinate systems with no "
                "correspondence between them, on an anisotropic grid, related "
                "by rotations the stage recorded only approximately. What "
                "separates the routes is whether the rotation is fitted in "
                "microns or in voxel indices, whether the nominal transform is "
                "refined, and whether the run ever works out which detections "
                "are the same bead."
            ),
            data={
                **{f"view{k}": views[k] for k in range(N_VIEWS)},
                "probes": np.vstack(probes_voxel),
            },
            truth={
                "probe_um": np.vstack(probes_truth),
                "n_confirmed": confirmed,
            },
            tolerance={
                "worst_view_median_um": POSITION_LIMIT_UM,
                "bead_count_error": COUNT_LIMIT,
            },
        )


# --- scoring ----------------------------------------------------------------


def _verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    truth = np.asarray(fixture.truth["probe_um"], float)
    wanted_count = float(fixture.truth["n_confirmed"])
    limit_um = float(fixture.tolerance.get("worst_view_median_um", POSITION_LIMIT_UM))
    limit_count = float(fixture.tolerance.get("bead_count_error", COUNT_LIMIT))

    mapped, why_mapped = read_array(attempt, "probe_um", truth.shape)
    if mapped is not None and not np.isfinite(mapped).all():
        mapped, why_mapped = None, "`probe_um` holds non-finite values"
    count, why_count = read_scalar(attempt, "n_beads_confirmed")
    if count is not None and count <= 0:
        count, why_count = None, f"`n_beads_confirmed` is not a count ({count:.4g})"

    unusable = [why for why in (why_mapped, why_count) if why]
    nothing_at_all = mapped is None and count is None and not attempt.arrays
    metrics = [
        Metric(
            "deliverables_unusable",
            None if nothing_at_all else float(len(unusable)),
            0.5,
            f" of 2 -- {'; '.join(unusable)}" if unusable else " of 2",
            unavailable="the run left nothing to score" if nothing_at_all else "",
        )
    ]

    detail: dict[str, object] = {}
    if mapped is None:
        metrics.append(
            Metric(
                "worst_view_median_um", None, limit_um, " um", unavailable=why_mapped
            )
        )
    else:
        # Per view, and the worst of them. Pooled, a run that registered two
        # views out of three keeps its failure below the median and reports a
        # number that is not about the third view at all.
        per_view = []
        for i in range(3):
            rows = slice(i * N_PROBES_PER_VIEW, (i + 1) * N_PROBES_PER_VIEW)
            distance = np.linalg.norm(mapped[rows] - truth[rows], axis=1)
            per_view.append(float(np.median(distance)))
        metrics.append(Metric("worst_view_median_um", max(per_view), limit_um, " um"))
        detail["median_um_per_view"] = [round(v, 4) for v in per_view]

    if count is None:
        metrics.append(
            Metric("bead_count_error", None, limit_count, "", unavailable=why_count)
        )
    else:
        metrics.append(
            Metric(
                "bead_count_error",
                abs(count - wanted_count) / wanted_count,
                limit_count,
                " of the true count",
            )
        )
        detail["n_beads_confirmed"] = count

    return Outcome(fixture, attempt, metrics, detail=detail)


def _save_artifacts(outcome: Outcome, root: Path) -> None:
    """Where each probe landed, per view.

    Which view failed is the first thing worth knowing about a bad run here,
    and a single worst-view number does not say it. Neither does it say whether
    a view is off by a translation -- a stage shift the run never refined -- or
    by a rotation, which the spread of the residuals across the field does.
    """
    mapped = outcome.attempt.arrays.get("probe_um")
    if mapped is None:
        return
    truth = np.asarray(outcome.fixture.truth["probe_um"], float)
    mapped = np.asarray(mapped, float)
    if mapped.shape != truth.shape:
        return
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "probe_residuals.npz",
        mapped=mapped,
        truth=truth,
        residual_um=mapped - truth,
        view=np.repeat([1, 2, 3], N_PROBES_PER_VIEW),
    )


# --- the task ---------------------------------------------------------------

TASK = f"""
One field of fluorescent beads was imaged four times, from four angles, on a
light-sheet microscope. The detected bead positions from each view are open in
napari as four Points layers:

  `view0_beads`, `view1_beads`, `view2_beads`, `view3_beads`

Each layer holds that view's detections in **that view's own voxel
coordinates**, as (z, y, x) indices. The lists are in no particular order and
there is no correspondence between them: row i of one is not row i of another,
and the four lists are not even the same length, because a view misses beads
that are deep in the sample and picks up a few detections that are not beads.

The acquisition:

- the voxel is 0.40 microns along z and 0.13 microns along y and x, in every
  view
- the four views differ by a rotation about the specimen's y axis. The stage
  recorded 0, 90, 180 and 270 degrees for views 0 to 3. That is what the stage
  was told to do, not a measurement of what the sample did.

Take **view 0's physical frame** -- its voxel grid scaled into microns -- as
the reference.

A fifth Points layer `probe_pts` holds {N_PROBES} locations, also (z, y, x) in
voxel indices, but *not* in view 0:

- rows 0 to {N_PROBES_PER_VIEW - 1} are in `view1_beads` coordinates
- rows {N_PROBES_PER_VIEW} to {2 * N_PROBES_PER_VIEW - 1} are in `view2_beads`
  coordinates
- rows {2 * N_PROBES_PER_VIEW} to {N_PROBES - 1} are in `view3_beads`
  coordinates

Work out how the four views relate to one another, then leave two things in the
kernel:

- `probe_um` -- a ({N_PROBES}, 3) array: where each row of `probe_pts` sits in
  the reference frame, in **microns**, as (z, y, x), same row order.
- `n_beads_confirmed` -- a single number: how many distinct beads the four
  lists represent between them, counting a bead only when **at least two** of
  the four views detected it.

The person who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

Both names must be bound in the kernel namespace when you finish.
""".strip()

#: What is *not* here: nothing about rigid transforms, ICP, Procrustes, RANSAC,
#: centroids or clustering, and no transform this person could not have known.
#: They ran the microscope.
LIGHT_SHEET = Persona(
    name="the imaging specialist who acquired this",
    background=(
        "You ran this light-sheet acquisition and detected the beads, and you "
        "are sitting with the analyst. You answer what you are asked, plainly "
        "and briefly. You do not volunteer analysis advice and you do not "
        "suggest methods -- you know the microscope and the sample, not the "
        "maths. If you are asked something you would not know from having run "
        "the experiment, say so."
    ),
    facts={
        "voxel size": (
            "0.40 microns between planes and 0.13 microns in the plane, the "
            "same in every view."
        ),
        "the angles": (
            "the stage went to 0, 90, 180 and 270 degrees about the vertical "
            "axis, in that order, one view at each."
        ),
        "how good the angles are": (
            "the stage is repeatable to a couple of degrees, not better, and "
            "the sample settles a few microns between rotations. I would not "
            "trust the numbers past that."
        ),
        "the beads": (
            "sub-resolution fluorescent beads mixed into the mounting medium. "
            "They are sparse -- a couple of microns apart at the closest."
        ),
        "why the lists differ in length": (
            "the deeper into the sample, the fewer beads come through, so each "
            "view sees a different set. That is why there are four of them."
        ),
        "bad detections": (
            "a few per view are not beads, just noise the detector picked up. "
            "It is a small fraction."
        ),
        "how well a bead is localised": (
            "much better in the plane than between planes -- the planes are "
            "coarse and the point spread function is long that way."
        ),
        "what the sample is": (
            "a cleared specimen in a gel, mounted in a capillary so it can be rotated."
        ),
        "whether anything was registered already": (
            "no, these are the raw detections from each view on its own. "
            "Putting them together is what I am asking for."
        ),
    },
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=LIGHT_SHEET,
    fixture=Procedural(BeadField()),
    layers=(
        # Points layers, because detections are what a multiview pipeline
        # registers on and a Points layer is how they reach napari. Four of
        # them rather than one array with a view column: the coordinate systems
        # are genuinely separate, and stacking them would suggest otherwise.
        Layer("view0_beads", "view0", kind="points"),
        Layer("view1_beads", "view1", kind="points"),
        Layer("view2_beads", "view2", kind="points"),
        Layer("view3_beads", "view3", kind="points"),
        Layer("probe_pts", "probes", kind="points"),
    ),
    collect={"probe_um": "probe_um", "n_beads_confirmed": "n_beads_confirmed"},
    score=_verify,
    save_artifacts=_save_artifacts,
    # The task states the voxel size and the nominal angles, so nothing has to
    # be elicited. These are here so a run that does ask is not misled.
    persona_must_know=("0.40 microns", "0.13 microns", "couple of degrees"),
    persona_must_not_know=(
        "rigid transform",
        "icp",
        "iterative closest",
        "procrustes",
        "kabsch",
        "ransac",
        "correspondence",
        "centroid",
        "cluster",
        "least squares",
        "spanning tree",
    ),
)
