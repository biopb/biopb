"""`measure-smlm-resolution` as benchmark data: which reconstruction is finer?

**Deliberately partial, and the boundary is the point.** The skill covers
localization, rendering and resolution; this case scores only the last, because
only the last has a number a cold run gets wrong. The prescreen measured the
rest: detection, Gaussian fitting, the ADU-to-photon conversion and the choice of
Mortensen over Thompson were all reproduced correctly and unprompted, so
benchmarking them would score agreement with a procedure the model already has.
What it got wrong was **the split**, 4 arms out of 4.

A localization list is not a bag of independent samples. One fluorophore blinks
several times and is localized once per blink, so a split that can put two blinks
of the *same* molecule into opposite halves correlates those halves by something
that is not structure. The fixture is two acquisitions of one structure differing
only in labelling density, nine-fold, and the contrast is stark:

===============  ===========  ===========
split            dense        sparse
===============  ===========  ===========
``"blocks"``     ~219 nm      ~343 nm
``"random"``     ~23 nm       ~23 nm
===============  ===========  ===========

A random split reports ~23 nm for **both** — roughly the localization precision,
five times finer than the sparse list's density floor, and with the density
difference erased entirely.

**Truth is the density floor, which the generator fixes exactly.** With *N*
molecules over structure area *A* the mean label spacing is ``sqrt(A/N)`` and
nothing below about twice that is resolvable however well each molecule was
localized (Shroff 2008). That is arithmetic on the emitter count, so it does not
depend on this repository's FRC being right — which matters, because scoring
against a number this repository computed would measure agreement with my own
implementation rather than with the sample.

**What is deliberately not scored is the ratio.** The two reconstructions come
back 1.57x apart, not the ``sqrt(9) = 3x`` a purely density-limited pair would
give, because FRC also sees the structure's own spectrum. A tolerance around 1.57
would be a tolerance around this implementation's output, so the ratio is
reported in `detail` as context and the scored bound is only that the sparser
list must measure *coarser* — a direction, not a level.

**Emitters have to sample a continuous structure, or the case measures nothing.**
An earlier draft scattered them uniformly, so with a blocks split each molecule
landed wholly in one half and the two halves shared no structure at all; FRC
returned ~1000 nm for both and the build-time check below caught it. The
structure here is a smoothed random field thresholded at its median — blobs
covering about half the field, which both halves sample independently.

**A run that reaches the answer through the density floor has also got it
right**, and that is not a shortcut tolerated by accident: the floor is the
correct physics for a density-limited reconstruction, and computing it needs the
molecule count, which needs blinks merged — the other thing the body warns about.
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
    read_scalar,
    save_png,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

SKILL = "measure-smlm-resolution"

FIELD_NM = 4000.0
#: Molecules in the dense acquisition, and the factor the sparse one is thinned
#: by. Nine-fold so the two density floors are a clear 3x apart even though the
#: measured resolutions are not.
N_EMITTERS = 20_000
DILUTION = 9
#: Structure scale of the thresholded random field. Well above the floors below,
#: so the blobs are resolved in both reconstructions and what separates them is
#: sampling density rather than the structure running out of detail.
FEATURE_NM = 400.0
MASK_GRID = 256
#: Localization jitter, well under either floor, so neither reconstruction is
#: precision-limited. It is also what a random split mistakes for the answer.
SIGMA_NM = 6.0
#: Blinks per molecule, and how long one burst lasts. The burst is far shorter
#: than a split block, so a blocks-split keeps a molecule's blinks together and
#: a random split does not. That contrast is the measurement.
BLINKS = 6
BURST_FRAMES = 6
N_FRAMES = 2000
BLOCK_FRAMES = 500
RENDER_NM = 5.0

#: `floor / reported - 1`, so 0 is honest and the random split scores ~4.2 on the
#: sparse list. The allowance covers where the 1/7 crossing lands on a finite
#: ring grid, not a real claim below the floor.
#: `ranking`: how much coarser the sparse list must measure. 1.15 against a
#: correct split's 1.57 and a random split's 1.00 — placed between them, and
#: nearer the failure, because the 1.57 is not itself a target.
TOLERANCE = {"floor_violation": 0.25, "ranking_error": 0.10}
RANKING_MIN = 1.15


@dataclass(frozen=True)
class BlinkingLists:
    """Two localization lists over one structure, differing only in density."""

    seed: int = 0

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        mask = _structure(self.seed)
        coverage = float(mask.mean())

        # One structure, sampled twice. The sparse acquisition is a subset of the
        # same molecules, so nothing about the sample changes between the two.
        pos = _emitters(N_EMITTERS, mask, rng)
        keep = rng.permutation(N_EMITTERS)[: N_EMITTERS // DILUTION]

        dense = _localize(pos, rng)
        sparse = _localize(pos[keep], rng)

        area = FIELD_NM**2 * coverage
        floors = {
            "floor_dense_nm": 2.0 * float(np.sqrt(area / N_EMITTERS)),
            "floor_sparse_nm": 2.0 * float(np.sqrt(area / len(keep))),
        }
        _the_case_can_be_passed_and_failed(dense, sparse, floors)

        return Fixture(
            provenance=(
                f"procedural: {N_EMITTERS} emitters on a {FEATURE_NM:.0f} nm "
                f"blob field over {FIELD_NM:.0f} nm square, thinned {DILUTION}x, "
                f"{BLINKS} blinks each at sigma {SIGMA_NM} nm, seed {self.seed}"
            ),
            about=(
                "Two localization lists of one structure at densities differing "
                f"{DILUTION}x, so their density floors are "
                f"{floors['floor_dense_nm']:.0f} nm and "
                f"{floors['floor_sparse_nm']:.0f} nm. Splitting the lists at "
                "random instead of in time reports ~23 nm for both."
            ),
            data={"dense": dense, "sparse": sparse},
            truth={**floors, "coverage": coverage},
            tolerance=dict(TOLERANCE),
        )


def _structure(seed: int) -> np.ndarray:
    """Blobs covering about half the field: continuous, and isotropic.

    Isotropic on purpose — FRC ring-averages, so an oriented structure would put
    the answer's dependence on where the features happen to point.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    field = gaussian_filter(
        rng.random((MASK_GRID, MASK_GRID)),
        FEATURE_NM / (FIELD_NM / MASK_GRID),
        mode="wrap",
    )
    return field > np.median(field)


def _emitters(n: int, mask: np.ndarray, rng) -> np.ndarray:
    """*n* positions uniform inside the mask, by rejection."""
    out: list[np.ndarray] = []
    while len(out) < n:
        p = rng.uniform(0.0, FIELD_NM, size=(n, 2))
        idx = np.clip((p / FIELD_NM * MASK_GRID).astype(int), 0, MASK_GRID - 1)
        out.extend(p[mask[idx[:, 1], idx[:, 0]]])
    return np.asarray(out[:n])


def _localize(pos: np.ndarray, rng) -> np.ndarray:
    """(N, 3) float32 of frame, x_nm, y_nm — one row per blink."""
    n = len(pos)
    starts = rng.integers(0, N_FRAMES - BURST_FRAMES, size=n)
    frames = (starts[:, None] + rng.integers(0, BURST_FRAMES, size=(n, BLINKS))).ravel()
    xy = np.repeat(pos, BLINKS, axis=0) + rng.normal(
        0.0, SIGMA_NM, size=(n * BLINKS, 2)
    )
    order = np.argsort(frames, kind="stable")
    out = np.empty((n * BLINKS, 3), dtype=np.float32)
    out[:, 0] = frames[order]
    out[:, 1:] = np.clip(xy[order], 0.0, FIELD_NM)
    return out


def _measure(table: np.ndarray, split: str) -> float:
    from ....plugins import image_resolution as ir

    return float(
        ir.frc_from_localizations(
            table[:, 1],
            table[:, 2],
            table[:, 0],
            render_pixel_size=RENDER_NM,
            split=split,
            block_frames=BLOCK_FRAMES,
            n_emitters=len(table) // BLINKS,
            extent=(0.0, FIELD_NM, 0.0, FIELD_NM),  # (ymin, ymax, xmin, xmax)
        ).resolution
    )


def _the_case_can_be_passed_and_failed(dense, sparse, floors) -> None:
    """Both halves of a benchmark's contract, asserted where they can be seen.

    A case is worth running only if a correct procedure passes it *and* the
    failure it targets fails it. Neither is a reference implementation of the
    choice being scored — the split is named in both calls — but a fixture that
    has drifted out of the density-limited regime would silently start scoring
    something else, so it raises at build time rather than in a report.
    """
    ok = {s: _measure(t, "blocks") for s, t in (("dense", dense), ("sparse", sparse))}
    if ok["sparse"] < floors["floor_sparse_nm"] * (1 - TOLERANCE["floor_violation"]):
        raise AssertionError(
            f"{SKILL} fixture: a correct blocks-split measures the sparse list at "
            f"{ok['sparse']:.0f} nm, under its own {floors['floor_sparse_nm']:.0f} nm "
            "density floor. The fixture is unwinnable."
        )
    if ok["sparse"] / ok["dense"] < RANKING_MIN:
        raise AssertionError(
            f"{SKILL} fixture: a correct blocks-split ranks the two only "
            f"{ok['sparse'] / ok['dense']:.2f}x apart, under the {RANKING_MIN} the "
            "scorer demands. The fixture is unwinnable."
        )
    bad = _measure(sparse, "random")
    if bad >= floors["floor_sparse_nm"] * (1 - TOLERANCE["floor_violation"]):
        raise AssertionError(
            f"{SKILL} fixture: a random split measures {bad:.0f} nm, which the "
            "scorer would accept. The failure this case exists to catch is no "
            "longer detectable on this data."
        )


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score both reported resolutions against the density floor, and their order.

    Two metrics because they fail differently and the pair names the mistake. A
    split that correlates repeat blinks drives both reported numbers down to the
    localization precision: the floor violation goes large *and* the ranking
    collapses to 1.0. A run that simply reported one number twice collapses the
    ranking while leaving the floor alone.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    dense, why_d = read_scalar(attempt, "resolution_dense_nm")
    sparse, why_s = read_scalar(attempt, "resolution_sparse_nm")
    missing = why_d or why_s
    if dense is None or sparse is None or dense <= 0 or sparse <= 0:
        missing = missing or "the run reported a non-positive resolution"
        for name in ("floor_violation", "ranking_error"):
            metrics.append(Metric(name, None, limits[name], unavailable=missing))
    else:
        worst = max(
            float(fixture.truth["floor_dense_nm"]) / dense - 1.0,
            float(fixture.truth["floor_sparse_nm"]) / sparse - 1.0,
            0.0,
        )
        metrics.append(Metric("floor_violation", worst, limits["floor_violation"]))
        metrics.append(
            Metric(
                "ranking_error",
                max(0.0, RANKING_MIN - sparse / dense),
                limits["ranking_error"],
            )
        )
        detail["resolution_dense_nm"] = float(dense)
        detail["resolution_sparse_nm"] = float(sparse)
        detail["ratio_reported"] = float(sparse / dense)
        # Context, never a limit: what a correct split measures on this fixture.
        # A run far from it but inside both bounds is answering honestly by
        # another route, which is allowed.
        detail["ratio_reference"] = 1.57

    detail["floor_dense_nm"] = float(fixture.truth["floor_dense_nm"])
    detail["floor_sparse_nm"] = float(fixture.truth["floor_sparse_nm"])
    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """Both reconstructions as plain counts renders, at the scored bin size."""
    bins = int(round(FIELD_NM / RENDER_NM))
    for name in ("dense", "sparse"):
        table = np.asarray(outcome.fixture.data[name])
        hist, _, _ = np.histogram2d(
            table[:, 2], table[:, 1], bins=bins, range=[[0, FIELD_NM], [0, FIELD_NM]]
        )
        save_png(hist, where / f"render-{name}.png")


# --- the conversation ------------------------------------------------------

TASK = f"""\
Two layers are open in the napari viewer, `locs_dense` and `locs_sparse`. Each is
a localization table from a single-molecule localization experiment, not an
image: an (N, 3) float32 array whose columns are

  0  frame index the localization came from
  1  x position in nanometres
  2  y position in nanometres

Both cover the same {FIELD_NM:.0f} nm x {FIELD_NM:.0f} nm field of the same
structure. There is no tensor server in this session, so those layers are the
only source of data.

I need to know what spatial resolution each of the two reconstructions actually
achieves, in nanometres.

The person who ran the experiment is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will come
back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  resolution_dense_nm   the resolution of `locs_dense`, in nm, as one number
  resolution_sparse_nm  the resolution of `locs_sparse`, in nm, as one number

Those two names are how your result is collected, so bind them exactly.
"""

#: Knows the sample and the acquisition, not the analysis. Says the two differ in
#: labelling density without quantifying it — the factor is most of the answer,
#: and a persona that hands it over turns the case into arithmetic.
IMAGER = Persona(
    name="smlm-two-densities",
    facts={
        "what the two datasets are": (
            "the same structure imaged twice — the second one I deliberately "
            "labelled more sparsely, but I could not tell you by how much"
        ),
        "what the structure is": (
            "a dense synthetic test pattern of blobs a few hundred nanometres "
            "across, covering roughly half the field — not filaments, there is "
            "no large empty space"
        ),
        "how long the acquisitions were": (
            f"{N_FRAMES} frames each, same camera, same laser power"
        ),
        "whether the molecules blink more than once": (
            "yes, each one comes back several times over a few consecutive "
            "frames — that is just how the dye behaves"
        ),
        "how precisely each molecule was located": (
            "a few nanometres — the fitting was very good, these were bright"
        ),
        "whether there is drift": "no, this was a short acquisition on a stable stage",
        "what I want the number for": (
            "deciding whether the sparser labelling was good enough to keep "
            "using, so I need the two compared on the same footing"
        ),
    },
    background=(
        "An SMLM experiment already localized — you have the tables, not the raw "
        "frames. You are happy to answer questions about the sample and the run."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="two-densities-blocks-split",
    task=TASK,
    persona=IMAGER,
    fixture=Procedural(BlinkingLists()),
    layers=(Layer("locs_dense", "dense"), Layer("locs_sparse", "sparse")),
    collect={
        "resolution_dense_nm": "resolution_dense_nm",
        "resolution_sparse_nm": "resolution_sparse_nm",
    },
    score=verify,
    save_artifacts=save_artifacts,
    plugins=("image_resolution",),
    # It must be able to say the molecules repeat, which is what makes the split
    # a choice at all, and that the structure fills the field, which is what
    # makes the areal density floor the right one to compare against.
    persona_must_know=("blink", "sparsely", "half the field"),
    # And it must not know the procedure, or the factor.
    persona_must_not_know=(
        "Fourier",
        "FRC",
        "split",
        "nine",
        "9x",
        "density floor",
        "Nyquist",
    ),
)
