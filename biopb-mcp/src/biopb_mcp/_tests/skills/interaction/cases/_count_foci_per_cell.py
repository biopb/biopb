"""`count-foci-per-cell` as benchmark data: which bright things are foci?

The skill takes a punctate channel and a parent segmentation and returns how many
foci sit in each parent. The withheld fact is step 2's question — **what the
counted population is** — and it is the categorical kind (`docs/skills.md`
§10): some nuclei carry a few much larger blobs, overlapping the foci in
brightness, and nothing in the pixels says whether those are foci, aggregate or
an artefact. Only the person who stained the sample knows.

The forward model, per cell, is ``pool_i * thickness + foci + aggregate``, on a
camera offset, under a gentle illumination gradient. Two properties do the work:

* ``pool_i`` spans an order of magnitude across cells, so the dimmest focus is
  well below the brightest cell's own background and **no global threshold on
  the raw channel exists**. Measured over 7 fields: ``median + 5*MAD`` on the raw
  channel finds 0 to 5 of the 50-56 foci in each, and nearly everything it does
  find is an aggregate.
* the aggregates are ~5x wider than a focus and **overlap it in brightness**, so
  every reasonable detector finds them and no intensity cut removes them.
  Rejecting them needs the expected *size*, which is the withheld fact. Measured
  over the same 7 fields, against a reference that scores MAE 0.02:

  ==========================================  =========  =============
  route                                       count MAE  total / truth
  ==========================================  =========  =============
  the whole procedure                              0.02           1.01
  the same, without the width filter               1.11           1.53
  the same, cutting on amplitude instead           1.11           1.53
  threshold on the raw channel instead             2.02           0.19
  ==========================================  =========  =============

`TOLERANCE` sits between those populations. `cells_reported` is the second
metric and it is not about detection at all: a run that tabulates only the labels
its spots landed on drops the foci-negative cells, which is 20-36% of this field
and the quantity most foci assays are actually after.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ....agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    save_png,
)
from ....agentbench._respondent import Persona
from .._benchmark import Case, Layer

SKILL = "count-foci-per-cell"

#: From measurement, not from taste. The reference scores 0.02 and the two
#: routes this case is built to separate score 1.11 and 2.02, so the limit sits
#: fifty times above a clean run and well under a third of a run that counted
#: the aggregates. `cells_reported` is an exact requirement — every cell
#: in the field, or the table is the wrong table — written as half a cell
#: because counts are integers and the engine's limits are strictly positive.
TOLERANCE = {
    "count_mae": 0.35,
    "cells_reported": 0.5,
}

OFFSET = 100.0  #: camera offset, counts
SIGMA_SPOT = 1.3  #: px — a 0.3 um diffraction-limited spot at 0.1 um/px
SIGMA_AGGREGATE = 6.0  #: px — antibody aggregate, ~1 um across
FOCUS_AMP = (320.0, 480.0)  #: counts above the local pool
#: Deliberately **overlapping** :data:`FOCUS_AMP` at the bottom. Drawn from a
#: disjoint range instead, the two populations separate on peak brightness alone
#: — which a cold run found and used — and the case would then be scoring a
#: property of this generator rather than the one it is about. Real aggregate is
#: a clump of a few stain molecules up to a large one, so it spans the range.
AGGREGATE_AMP = (400.0, 1400.0)
PIXEL_UM = 0.1


def _gauss(canvas: np.ndarray, y: int, x: int, amp: float, sigma: float) -> None:
    r = int(np.ceil(4 * sigma))
    y0, y1 = max(0, y - r), min(canvas.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(canvas.shape[1], x + r + 1)
    if y1 <= y0 or x1 <= x0:
        return
    dy = (np.arange(y0, y1)[:, None] - y) ** 2
    dx = (np.arange(x0, x1)[None, :] - x) ** 2
    canvas[y0:y1, x0:x1] += amp * np.exp(-(dy + dx) / (2 * sigma**2))


@dataclass(frozen=True)
class FociField:
    """One field of nuclei whose bright blobs only the operator can classify."""

    shape: tuple[int, int] = (512, 512)
    grid: tuple[int, int] = (5, 5)
    seed: int = 7

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        h, w = self.shape
        ny, nx = self.grid
        labels = np.zeros(self.shape, np.int32)
        # Optical thickness of the ellipsoid, so the nucleoplasmic pool tapers
        # to nothing at the rim. A nucleus that ended on a step edge would put a
        # residual as big as a focus all round its border, and the case would be
        # about segmentation edges rather than about detection.
        thickness = np.zeros(self.shape, np.float64)
        yy, xx = np.mgrid[0:h, 0:w]

        n = ny * nx
        for k in range(1, n + 1):
            i, j = divmod(k - 1, nx)
            cy = (i + 0.5) * (h / ny) + rng.uniform(-8, 8)
            cx = (j + 0.5) * (w / nx) + rng.uniform(-8, 8)
            a, b = rng.uniform(26, 34), rng.uniform(26, 34)
            th = rng.uniform(0, np.pi)
            dy, dx = yy - cy, xx - cx
            u = dy * np.cos(th) + dx * np.sin(th)
            v = -dy * np.sin(th) + dx * np.cos(th)
            rr = (u / a) ** 2 + (v / b) ** 2
            inside = rr <= 1.0
            labels[inside] = k
            thickness[inside] = np.sqrt(1.0 - rr[inside])

        # A gentle illumination gradient: real, and it moves a global raw
        # threshold around without being what the case is about.
        field = 1.0 - ((yy / h - 0.5) ** 2 + (xx / w - 0.5) ** 2)

        nuclei = (
            OFFSET + 2600 * thickness * field + rng.normal(0, 12, self.shape)
        ).astype(np.float32)

        # The diffuse nucleoplasmic pool. An order of magnitude of spread across
        # cells is what makes one threshold for the whole field hopeless.
        pool = np.exp(rng.uniform(np.log(80), np.log(900), n))
        image = np.zeros(self.shape, np.float64)
        for k in range(1, n + 1):
            m = labels == k
            image[m] = pool[k - 1] * thickness[m]

        counts = np.zeros(n, int)
        spots: list[tuple[int, int, int]] = []
        aggregates: list[tuple[int, int, int]] = []
        for k in range(1, n + 1):
            # Off the rim, where the pool has tapered away: a focus half outside
            # its own mask is a segmentation question, not this one.
            inside = np.argwhere((labels == k) & (thickness > 0.45))
            wanted = 0 if rng.random() < 0.20 else int(rng.poisson(2.6))
            placed: list[tuple[int, int]] = []
            for _ in range(wanted):
                for _try in range(80):
                    y, x = inside[rng.integers(len(inside))]
                    if all((y - py) ** 2 + (x - px) ** 2 > 10**2 for py, px in placed):
                        placed.append((int(y), int(x)))
                        break
            counts[k - 1] = len(placed)
            for y, x in placed:
                _gauss(image, y, x, rng.uniform(*FOCUS_AMP) * field[y, x], SIGMA_SPOT)
                spots.append((k, y, x))
            for _ in range(int(rng.integers(0, 3))):
                for _try in range(80):
                    y, x = inside[rng.integers(len(inside))]
                    if all((y - py) ** 2 + (x - px) ** 2 > 24**2 for py, px in placed):
                        placed.append((int(y), int(x)))
                        _gauss(
                            image,
                            int(y),
                            int(x),
                            rng.uniform(*AGGREGATE_AMP) * field[y, x],
                            SIGMA_AGGREGATE,
                        )
                        aggregates.append((k, int(y), int(x)))
                        break

        # Sharp specks on the coverslip, outside every nucleus: the reason spot
        # -> parent assignment has to drop what it cannot assign to anything.
        outside = np.argwhere(labels == 0)
        dirt = []
        for _ in range(12):
            y, x = outside[rng.integers(len(outside))]
            _gauss(image, int(y), int(x), rng.uniform(*FOCUS_AMP), SIGMA_SPOT)
            dirt.append((int(y), int(x)))

        image = image + OFFSET
        image = image + rng.normal(0, np.sqrt(np.maximum(image, 1.0)) * 0.35)
        image = (image + rng.normal(0, 3.0, self.shape)).astype(np.float32)

        # The two properties the case rests on, checked before anyone pays for a
        # run (§5d's rule about a fixture whose truth is wrong).
        cell_background = np.array(
            [np.median(image[labels == k]) for k in range(1, n + 1)]
        )
        focus_peak = np.array(
            [image[y - 1 : y + 2, x - 1 : x + 2].max() for _, y, x in spots]
        )
        assert focus_peak.min() < cell_background.max(), (
            f"the dimmest focus peaks at {focus_peak.min():.0f} and the brightest "
            f"cell's background sits at {cell_background.max():.0f} — one global "
            "threshold on the raw channel would separate them, and the case is "
            "built on there being no such threshold"
        )
        assert (counts == 0).sum() >= 3, (
            "too few foci-negative cells to score dropping them"
        )
        assert len(aggregates) >= 8, (
            "too few aggregates to separate asking from not asking"
        )

        return Fixture(
            provenance=(
                f"procedural: {n} nuclei in {self.shape[0]}x{self.shape[1]}, seed "
                f"{self.seed}, {int(counts.sum())} foci at sigma {SIGMA_SPOT} px, "
                f"{len(aggregates)} aggregates at sigma {SIGMA_AGGREGATE} px, "
                f"pixel {PIXEL_UM} um"
            ),
            about=(
                f"{int(counts.sum())} foci across {n} nuclei, {int((counts == 0).sum())} "
                f"of which have none. Each nucleus carries its own diffuse pool "
                f"({cell_background.min():.0f} to {cell_background.max():.0f} counts), so the "
                f"dimmest focus ({focus_peak.min():.0f}) is well below the brightest cell's "
                f"background: no threshold on the raw channel finds foci in the dim "
                f"cells without drowning the bright ones. {len(aggregates)} antibody "
                "aggregates are brighter and about five times wider than a focus, and "
                "nothing in the pixels says they are not foci — a run that does not ask "
                "counts them and overstates every treated cell."
            ),
            data={"foci": image, "nuclei": nuclei, "labels": labels},
            truth={
                # The private facts. Stripped from `data`, held by the
                # respondent, and read by the verifier to know what to score.
                "counts": counts,
                "spots": np.array(spots, int),
                "aggregates": np.array(aggregates, int),
                "dirt": np.array(dirt, int),
                "pixel_um": PIXEL_UM,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _read_counts(attempt: Attempt, n: int) -> tuple[np.ndarray | None, str]:
    """The run's `counts`, as a 1-D integer vector of any length.

    Deliberately *not* `read_array(..., (n,))`: the length is the second metric,
    so a run that reported 21 of 25 cells has to arrive as a wrong length rather
    than as nothing to score.
    """
    got = attempt.arrays.get("counts")
    if got is None:
        return None, "the run left no `counts`"
    got = np.asarray(got, float).squeeze()
    if got.ndim != 1 or got.size == 0:
        return (
            None,
            f"the run's `counts` is {np.shape(got)}, not a vector of per-cell counts",
        )
    if not np.isfinite(got).all():
        return None, "the run's `counts` holds non-finite values"
    return got, ""


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Both metrics read ``truth["counts"]`` alone, which is exactly what an
    annotated real field carries — a human scoring foci per nucleus produces
    this vector and nothing else — so the case survives a curated substitution
    without changing what is measured.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = fixture.truth.get("counts")
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    if truth is None:
        why = "the fixture carries no per-cell foci counts"
        return Outcome(
            fixture=fixture,
            attempt=attempt,
            metrics=[
                Metric("count_mae", None, limits["count_mae"], unavailable=why),
                Metric(
                    "cells_reported", None, limits["cells_reported"], unavailable=why
                ),
            ],
        )

    truth = np.asarray(truth, float)
    got, why = _read_counts(attempt, truth.size)

    if got is None:
        metrics.append(
            Metric("cells_reported", None, limits["cells_reported"], unavailable=why)
        )
        metrics.append(Metric("count_mae", None, limits["count_mae"], unavailable=why))
        return Outcome(fixture=fixture, attempt=attempt, metrics=metrics)

    metrics.append(
        Metric(
            "cells_reported",
            abs(float(got.size) - float(truth.size)),
            limits["cells_reported"],
            unit=" cells",
        )
    )
    detail["cells_reported"] = int(got.size)
    detail["cells_in_field"] = int(truth.size)
    detail["foci_negative_truth"] = float((truth == 0).mean())

    if got.size != truth.size:
        metrics.append(
            Metric(
                "count_mae",
                None,
                limits["count_mae"],
                unavailable=(
                    f"the run reported {got.size} cells against {truth.size} in the "
                    "field, so its counts cannot be lined up with the truth"
                ),
            )
        )
    else:
        metrics.append(
            Metric("count_mae", float(np.abs(got - truth).mean()), limits["count_mae"])
        )
        detail["total_reported"] = int(got.sum())
        detail["total_true"] = int(truth.sum())
        detail["foci_negative_reported"] = float((got == 0).mean())
        detail["exact_cells_pct"] = float(100.0 * (got == truth).mean())

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    image = fixture.data.get("foci")
    labels = fixture.data.get("labels")
    truth = fixture.truth.get("counts")
    if image is None or labels is None or truth is None:
        return
    image = np.asarray(image, float)
    truth = np.asarray(truth, int)

    save_png(image, where / "foci-raw.png")
    # The same channel on a hard ceiling just above a focus: what the operator
    # sees, and why the aggregates are the first thing any detector finds.
    save_png(np.minimum(image, 1200.0), where / "foci-clipped.png", vmax=1200.0)
    save_png(np.asarray(labels, float), where / "parents.png")

    rows = ["label,truth,reported"]
    got = attempt.arrays.get("counts")
    got = np.asarray(got, float).squeeze() if got is not None else None
    for i, t in enumerate(truth):
        r = "" if got is None or got.ndim != 1 or i >= got.size else f"{got[i]:g}"
        rows.append(f"{i + 1},{t},{r}")
    (where / "counts.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A field of stained nuclei is open in the napari viewer:

  foci          float32 (512, 512) — the punctate channel, raw off the camera
  nuclei        float32 (512, 512) — the nuclear counterstain
  nuclei_labels int32   (512, 512) — those nuclei already segmented, 25 objects

There is no tensor server in this session, so those layers are the only source
of pixels. The nuclei are segmented already; do not re-segment them.

I need to know how many foci are in each nucleus.

The person who stained and imaged this is here and can answer questions. To ask
them something, reply in plain text without calling a tool, and their answer
will come back as the next message.

When you are finished, leave one name bound in the kernel namespace:

  counts   an integer array of length 25, ordered by label id, where counts[i]
           is the number of foci in the nucleus with label i+1.

That name is how your result is collected, so bind it exactly.
"""

#: Note what is *not* here: nothing about band-pass filters, thresholds,
#: residuals or peak finding. This person knows their stain, their microscope
#: and what they are counting — not the procedure. A persona that knew the
#: method could answer a question the agent never properly asked, and the
#: numeric result would stop meaning what it appears to.
OPERATOR = Persona(
    name="operator-foci-assay",
    facts={
        "what the big bright blobs are": (
            "those are antibody aggregate — clumps of the stain itself. They "
            "are about a micron across and they are the brightest things in the "
            "picture. They are not foci and I never count them"
        ),
        "how big a focus is": (
            "a focus is a single diffraction-limited dot, about 0.3 microns "
            "across. They all look the same size"
        ),
        "what one pixel is": "0.1 microns",
        "what to do with nuclei that have no foci": (
            "report them as zero. The fraction of cells with none is the whole "
            "point of the experiment — the untreated cells should mostly have "
            "none"
        ),
        "what the specks outside the nuclei are": (
            "dirt on the coverslip. Nothing outside a nucleus counts"
        ),
        "whether anything has been corrected already": (
            "no, that is straight off the camera. The nuclei are very different "
            "brightnesses because the stain took differently in each one"
        ),
        "how bright a focus is": (
            "I could not tell you in numbers. In a dim cell you can see them "
            "easily and in a bright cell they barely stand out — that is why I "
            "stopped scoring these by eye"
        ),
    },
    background=(
        "You stained and imaged a field of cells for a foci assay. You are happy "
        "to answer questions about the stain, the sample and the microscope."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="aggregates-only-the-operator-can-name",
    task=TASK,
    persona=OPERATOR,
    fixture=Procedural(FociField()),
    layers=(
        Layer("foci", "foci"),
        Layer("nuclei", "nuclei"),
        Layer("nuclei_labels", "labels", kind="labels"),
    ),
    collect={"counts": "counts"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="foci",
    # It must be able to answer: the fixture withholds what the counted
    # population is, and this person knows it, knows the spot size, and knows
    # that empty cells are the readout.
    persona_must_know=("aggregate", "0.3", "0.1", "diffraction", "zero"),
    # And it must not know the method — only the stain and the microscope.
    persona_must_not_know=(
        "difference of gaussians",
        "tophat",
        "band-pass",
        "mad",
        "peak_local_max",
        "background subtraction",
        "residual",
    ),
)
