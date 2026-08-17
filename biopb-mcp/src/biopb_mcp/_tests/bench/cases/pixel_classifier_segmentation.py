"""`pixel-classifier-segmentation` as benchmark data: does the run know it broke?

**The skill is banked, not served.** The `_` prefix on
`_pixel-classifier-segmentation.md` keeps it out of the catalog; this case
names no `skill` for that reason, and runs the shipped corner rather than an
ablation over an entry that is not there. The skill is banked because a
Sonnet-class model derives nearly all of it unaided, and the
catalog is consumed at that tier; it is kept because a Haiku-class one derives
none of it, and lower-tier usability is named as real backlog in
`skill-candidates.md`. The measurement, four cold arms per tier, identical
prompts and fixed signature, no skill and no repo:

    tier     macro IoU a   macro IoU b   collapse        overstatement
    Haiku    0.60-0.64     0.33-0.64     0.000-0.474     0.103-0.270  (0/4 pass)
    Sonnet   0.70-0.79     0.62-0.82     0.000-0.139     0.034-0.090  (4/4 pass)
    reference    0.708         0.709     0.000           0.043

Every Sonnet arm recovered the withheld fact **from the histograms**, without
asking — one wrote "median 902 vs 1348, MAD 56 vs 104 ... almost certainly a
lamp/exposure change between acquisitions ... any feature that is a bare
function of intensity has to be normalized per-field" as a code comment. All
four then normalised each field on its own statistics (step 3), diagnosed
pixel-wise CV as leaking across contiguous strokes and built stroke-grouped CV
instead (step 6), reported resubstitution accuracy explicitly labelled "not as
evidence of generalization" (step 9), counted components and compared class
balance between fields (step 7). Two added a spatial pass (step 5). That is the
body, derived — and two arms **beat the reference**: 0.7225 on the scribbled
field, and 0.795/0.815 on both from the arm that reached for CLAHE, whose
per-tile equalisation absorbs the illumination gradient as well as the level
shift and takes the rim to 0.744/0.797 where the reference gets 0.543.

The four Haiku arms did none of it: all four quoted training-pixel accuracy as
their quality number, all four named the 4 px rim as their best class where its
true IoU was 0.44-0.59, none counted components (2128-12688 bodies on a field
with 7 cells), and all four read the second field's balance shift as biology.

So the information asymmetry this case is built on does not survive at the
consuming tier, which is what keeps the skill banked rather than shipped.
Nothing below is weakened by that: the fixture, the verifier and the tolerances
are checked exactly as hard, and the case is *run* — the work is real whether or
not a skill for it is served, and promoting the skill later does not begin by
rebuilding its benchmark.

One arm found a property of this fixture its author had not measured, and it is
recorded here because it is real. Field b is brighter *and relatively noisier* —
its 16-84 spread is 22.6% of its median against field a's 18.6% — so per-field
normalisation by that spread leaves field b's texture systematically smaller.
Since local-variance features carry most of the model's importance, that is a
second-order domain shift on top of the level shift, and it is why an oracle
trained on field a's own truth still loses 0.114 here. The arm diagnosed it from
its own output ("several cells predicted as pure background, only the edge ring
correct") and rescaled each texture feature by its own image's median absolute
value. Worth knowing before this is ever promoted: `second_field_collapse` is
not measuring one effect.

Two acquisitions of one three-class field, scribbles on the first only. The two
large classes have the **same mean and the same standard deviation** and differ
only in correlation length, so no threshold separates them and a texture
classifier is the honest route. The third class is a 4 px rim, 2.6% of the
field — thin enough that a scribble on it lands in its middle, never at its
edges, which is what makes every holdout over the scribbles optimistic about it.

The second field is the same sample imaged again at a **1.35x exposure**, its
own illumination gradient and a different camera offset. That is the whole
construction: a classifier whose features were built on raw counts learns this
exposure and collapses on it, losing 0.341 of the first field's macro IoU.

The defence that *looks* like a defence and is not is feature scaling. Fitting a
scaler on the training pixels is invariant for a forest — trees split on
per-feature thresholds — and measured as a 2x2 it moves the second field's score
by nothing to four decimals, with or without normalisation underneath. So a run
can hold a completely reasonable belief that it has handled scale.

The withheld fact is **whether the second field came off the same acquisition**.
It is the §10 kind — categorically absent, not merely hard. A field that is
brighter is equally well explained by a longer exposure and by a brighter
sample, and no amount of looking at either array distinguishes them. The four
cold Haiku runs all read the shift as biology ("different cellular density",
"fewer or smaller cells", "different depth plane"); three were right by luck and
the fourth had lost half its answer. The Sonnet arms did not — they read the
level shift straight off the histograms, which is the finding that deferred this.

**There is a back door and it should be named** (`skills.md` §10). Per-field
min-max normalisation defends against this shift without anyone asking, and
three of the four Haiku arms reached for it reflexively (scoring 0.000-0.080 on
`second_field_collapse` while the fourth scored 0.474). So a green
`second_field_collapse` does not prove the run interacted — it proves the run
was not damaged, by whichever route, and a run that walked through the back door
has done something defensible rather than something lucky.

`accuracy_overstatement` is the metric no reflex covers: it asks what the run
*believed*, and all four Haiku arms believed something wrong whether or not their
labels came out — 0.981 to 0.995 claimed against true accuracies of 0.72 to 0.89.
It is also the metric that separates the tiers most cleanly, 0/4 against 4/4.

`accuracy_overstatement` is one-sided on purpose. Understating is a run that was
careful; overstating is a run that told the user a number it had no basis for,
which is the failure this skill exists to prevent and the one all four Haiku arms
hit.

The reference implementation these tolerances come from is in the pull request
that added this case, per `biopb-mcp/docs/skills.md` §11b.
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
    read_array,
    read_scalar,
    save_png,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

NAMESPACE = "pixel-classifier-segmentation"

MEDIUM, INTERIOR, EDGE = 1, 2, 3
CLASSES = {MEDIUM: "medium", INTERIOR: "interior", EDGE: "edge"}

#: The two large classes share this. Only their correlation length differs, so
#: an intensity cut between them does not exist to be found.
TEXTURE_STD = 0.09
SIGMA_MEDIUM, SIGMA_INTERIOR = 6.0, 1.8

#: The rim: 4 px wide, 1.45x brighter. Thin enough that a scribble on it cannot
#: reach its boundary, which is the whole point of it being here.
EDGE_LEVEL, EDGE_WIDTH = 1.45, 4

#: The withheld fact, in numbers. Field B is the same sample, imaged again.
ACQUISITION = {
    "a": {"photons": 800.0, "offset": 100.0, "vignette": 0.08, "angle": 0.6},
    "b": {"photons": 1080.0, "offset": 260.0, "vignette": 0.25, "angle": 2.4},
}

#: Set from the reference implementation (per-field robust normalisation,
#: `multiscale_basic_features` at the library defaults, a 200-tree forest,
#: stroke-wise holdout) against the failures it has to be separated from, every
#: route run on this construction:
#:
#:                                            field_a_miss  collapse
#:   oracle, 60k true pixels of field a ..... 0.039         0.114
#:   reference .............................. 0.292         0.000
#:   reference + class_weight="balanced" .... 0.296         0.000
#:   per-field min-max (the reflex) ......... 0.292         0.013
#:   reference at sigma_max 32 .............. 0.328         0.049
#:   image never normalised per field ....... 0.292         0.341
#:   four cold Haiku arms ................... 0.363-0.398   0.000-0.474
#:   four cold Sonnet arms .................. 0.205-0.304   0.000-0.139
#:
#: `second_field_collapse` is the measurement and 0.25 sits in a real gap: the
#: worst surviving route scores 0.139 — a sound Sonnet arm, and the reason this
#: row is here rather than the 0.049 of the ablations — and the broken ones
#: 0.341 and 0.474. Narrower than it first measured, but still a gap. The
#: oracle's 0.114 is the *irreducible* part — trained on the first field's own
#: truth, the second is still harder — so a limit under that would fail a run
#: for the fixture's own physics. It is a ratio rather than the second field's
#: score for the same reason.
#:
#: **Feature scaling is not one of the rows because it is not a factor.** Fitting
#: a scaler on the training pixels changes nothing against a forest — measured
#: as a 2x2, macro IoU on the second field is 0.7093/0.7100 normalised and
#: 0.4664/0.4664 on raw counts, i.e. identical to four decimals. What breaks the
#: second field is only ever whether the *image* was normalised per field.
#:
#: `field_a_miss` is a floor, not a discriminator: every route above passes it,
#: including the ones that fail everything else. It is here so a run cannot buy
#: a clean `collapse` ratio by being equally poor on both fields.
#:
#: `accuracy_overstatement` is the one all four Haiku arms failed. Measured on
#: this fixture: stroke-wise holdout says 0.9296 and the second field really
#: scores 0.8870, so an honest run overstates by **0.043**. Training-pixel
#: accuracy is exactly **1.000**, so quoting it overstates by 0.113 even for a
#: run whose labels came out as well as the reference's. The limit sits between
#: those two, and the gap is genuinely that narrow — a run good enough on the
#: second field can quote the wrong number and still pass, which is the right
#: outcome: it got the answer right.
TOLERANCE = {
    "field_a_miss": 0.45,
    "second_field_collapse": 0.25,
    "accuracy_overstatement": 0.10,
}


# --- the field -------------------------------------------------------------


def _correlated(shape, sigma, rng):
    """Zero-mean, unit-std field with the given correlation length."""
    field = ndi.gaussian_filter(rng.normal(size=shape), sigma)
    return field / field.std()


def _cells(shape, rng, n):
    """Random non-touching ellipses -- the count is visible in the picture."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    mask = np.zeros(shape, bool)
    placed = 0
    for _ in range(400):
        if placed == n:
            break
        cy, cx = rng.uniform(80, shape[0] - 80), rng.uniform(80, shape[1] - 80)
        ra, rb = rng.uniform(48, 84), rng.uniform(48, 84)
        theta = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dy * np.cos(theta) + dx * np.sin(theta)
        v = -dy * np.sin(theta) + dx * np.cos(theta)
        this = (u / ra) ** 2 + (v / rb) ** 2 <= 1.0
        if (this & ndi.binary_dilation(mask, iterations=12)).any():
            continue
        mask |= this
        placed += 1
    assert placed == n, f"cell packing failed: {placed} of {n}"
    return mask


def _truth_from(mask):
    """1 medium / 2 interior / 3 edge, the rim straddling the boundary."""
    grown = ndi.binary_dilation(mask, iterations=EDGE_WIDTH // 2)
    shrunk = ndi.binary_erosion(mask, iterations=EDGE_WIDTH // 2)
    truth = np.full(mask.shape, MEDIUM, np.uint8)
    truth[shrunk] = INTERIOR
    truth[grown & ~shrunk] = EDGE
    return truth


def _illumination(shape, vignette, angle):
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(float)
    yy = (yy - shape[0] / 2) / (shape[0] / 2)
    xx = (xx - shape[1] / 2) / (shape[1] / 2)
    ramp = np.cos(angle) * xx + np.sin(angle) * yy
    radial = 1.0 - 0.4 * (yy**2 + xx**2)
    return 1.0 + vignette * (0.6 * ramp + radial - radial.mean())


def _render(truth, acq, rng):
    """Level field -> photons -> counts. The exposure is the withheld fact."""
    shape = truth.shape
    level = 1.0 + TEXTURE_STD * _correlated(shape, SIGMA_MEDIUM, rng)
    fine = TEXTURE_STD * _correlated(shape, SIGMA_INTERIOR, rng)
    inside = truth != MEDIUM
    level[inside] = 1.0 + fine[inside]
    level[truth == EDGE] *= EDGE_LEVEL

    signal = (
        acq["photons"] * level * _illumination(shape, acq["vignette"], acq["angle"])
    )
    counts = rng.poisson(np.clip(signal, 0, None)) + rng.normal(0, 8.0, shape)
    return np.clip(counts + acq["offset"], 0, 65535).astype(np.uint16)


def _strokes(truth, label, n, length, rng, restrict=None):
    """Short hand-drawn strokes strictly inside one true class."""
    room = ndi.binary_erosion(truth == label, iterations=2)
    if restrict is not None:
        room &= restrict
    out = np.zeros(truth.shape, bool)
    ys, xs = np.nonzero(room)
    for _ in range(n):
        for _attempt in range(200):
            i = rng.integers(len(ys))
            theta = rng.uniform(0, 2 * np.pi)
            py = ys[i] + np.arange(length) * np.sin(theta)
            px = xs[i] + np.arange(length) * np.cos(theta)
            if not (
                (py >= 0).all()
                and (py < truth.shape[0]).all()
                and (px >= 0).all()
                and (px < truth.shape[1]).all()
            ):
                continue
            py, px = py.astype(int), px.astype(int)
            if not room[py, px].all():
                continue
            out[py, px] = True
            break
    return ndi.binary_dilation(out, iterations=1) & room


def _dabs(truth, label, n, radius, rng, restrict=None):
    """Short patches following a thin class. A straight stroke cannot stay
    inside a 4 px ring, and a person dragging along a halo leaves these."""
    room = truth == label
    if restrict is not None:
        room &= restrict
    ys, xs = np.nonzero(room)
    seeds = np.zeros(truth.shape, bool)
    for i in rng.choice(len(ys), n, replace=False):
        seeds[ys[i], xs[i]] = True
    return (
        ndi.binary_dilation(seeds, ndi.generate_binary_structure(2, 2), radius) & room
    )


def _best_single_cut(field, truth, n=400):
    """Balanced accuracy of the best global threshold separating medium from
    interior. 0.5 is chance; this construction has to stay near it."""
    m = field[truth == MEDIUM].astype(float)
    i = field[truth == INTERIOR].astype(float)
    cuts = np.linspace(*np.percentile(field, [0.5, 99.5]), n)[:, None]
    above = 0.5 * ((i[None, :] > cuts).mean(1) + (m[None, :] <= cuts).mean(1))
    return float(np.maximum(above, 1.0 - above).max())


@dataclass(frozen=True)
class ScribbledField:
    """Two acquisitions of one three-class field; scribbles on the first."""

    shape: tuple[int, int] = (640, 640)
    n_cells: int = 7
    seed_a: int = 20260805
    seed_b: int = 771
    seed_scribbles: int = 4242

    def __call__(self) -> Fixture:
        fields, truths = {}, {}
        for name, seed in (("a", self.seed_a), ("b", self.seed_b)):
            rng = np.random.default_rng(seed)
            truths[name] = _truth_from(_cells(self.shape, rng, self.n_cells))
            fields[name] = _render(truths[name], ACQUISITION[name], rng)

        rng = np.random.default_rng(self.seed_scribbles)
        truth_a = truths["a"]
        labelled, n = ndi.label(truth_a == INTERIOR)
        chosen = np.isin(labelled, rng.choice(np.arange(1, n + 1), 3, replace=False))
        near = ndi.binary_dilation(chosen, iterations=EDGE_WIDTH + 3)

        scribbles = np.zeros(self.shape, np.uint8)
        scribbles[_strokes(truth_a, MEDIUM, 9, 70, rng)] = MEDIUM
        scribbles[_strokes(truth_a, INTERIOR, 9, 60, rng, restrict=chosen)] = INTERIOR
        scribbles[_dabs(truth_a, EDGE, 14, 6, rng, restrict=near)] = EDGE

        drawn = scribbles > 0
        # A mislabelled scribble is a different experiment: it would measure
        # robustness to bad annotation, which is not what this case is about.
        assert not (drawn & (scribbles != truth_a)).any(), "mislabelled scribble"
        for k in CLASSES:
            assert (scribbles == k).sum() >= 400, f"class {k} barely scribbled"

        # The two large classes must not be separable by a level, or the whole
        # construction is a threshold problem wearing a classifier's clothes.
        # Asserted operationally rather than on the class means: the means do
        # drift apart by ~3% on field b, but that is its illumination gradient
        # picking up where the cells happen to sit, not an intrinsic level. What
        # has to be false is that *some* global cut separates them.
        for name, truth in truths.items():
            got = _best_single_cut(fields[name], truth)
            assert got < 0.65, (
                f"field {name}: a single global cut separates medium from "
                f"interior at balanced accuracy {got:.3f} -- this is a "
                f"threshold problem, not a texture one"
            )

        return Fixture(
            provenance=(
                f"procedural: seeds {self.seed_a}/{self.seed_b}, two "
                f"{self.shape[0]}x{self.shape[1]} acquisitions of one field, "
                f"{self.n_cells} cells, three classes (medium/interior share a "
                f"mean and an s.d. and differ only in correlation length "
                f"{SIGMA_MEDIUM}/{SIGMA_INTERIOR} px; edge is a {EDGE_WIDTH} px "
                f"rim at {EDGE_LEVEL}x). Field b is the same sample at "
                f"{ACQUISITION['b']['photons'] / ACQUISITION['a']['photons']:.2f}x "
                f"exposure, its own illumination gradient and offset "
                f"{ACQUISITION['b']['offset']:.0f} against "
                f"{ACQUISITION['a']['offset']:.0f}. Scribbles on field a only"
            ),
            about=(
                "Two fields of one sample, scribbles on the first. The second "
                "was acquired at a different exposure, which is not in either "
                "array: a brighter field is equally well explained by a "
                "brighter sample. A classifier that encodes the first field's "
                "counts labels the second one wrongly and reports no error."
            ),
            data={
                "field_a": fields["a"],
                "field_b": fields["b"],
                "scribbles": scribbles,
            },
            truth={
                "truth_a": truths["a"],
                "truth_b": truths["b"],
                "scribbled": drawn,
                "n_cells": self.n_cells,
            },
            tolerance=dict(TOLERANCE),
        )


# --- truth-side arithmetic, shared by the verifier and the artifacts --------


def _macro_iou(pred, truth, valid):
    """Mean per-class IoU, and the per-class breakdown."""
    per = {}
    for k, name in CLASSES.items():
        p, t = (pred == k) & valid, (truth == k) & valid
        union = (p | t).sum()
        per[name] = float((p & t).sum() / union) if union else float("nan")
    return float(np.nanmean(list(per.values()))), per


def _labels(attempt: Attempt, key, shape) -> tuple[np.ndarray | None, str]:
    """The run's class map for one field, or why it cannot be scored.

    `read_array` returns floats, so the labels are rounded rather than cast --
    a run that leaves `2.0` meant class 2, and truncation would only bite on
    the value that arrived slightly under an integer."""
    got, why = read_array(attempt, key, tuple(shape))
    if got is None:
        return None, why
    got = np.rint(got).astype(np.int64)
    present = set(np.unique(got).tolist()) - {0}
    if not present:
        return None, f"the run's `{key}` is empty -- nothing to score"
    if not present <= set(CLASSES):
        return None, (
            f"the run's `{key}` holds labels {sorted(present)}, "
            f"outside the three classes {sorted(CLASSES)}"
        )
    return got.astype(np.uint8), ""


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    `second_field_collapse` is a *ratio* rather than the second field's score on
    its own: the fixture's second field is intrinsically harder (an oracle loses
    0.182 on it), so an absolute limit would charge every run for that. What is
    being measured is whether the classifier held up, relative to the field it
    was trained on.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = fixture.truth
    truth_a = np.asarray(truth["truth_a"], np.uint8)
    truth_b = np.asarray(truth["truth_b"], np.uint8)
    # Field A is scored off the scribbles: they are the training set.
    held_out = ~np.asarray(truth["scribbled"], bool)
    everywhere = np.ones(truth_b.shape, bool)

    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    got_a, why_a = _labels(attempt, "labels_a", truth_a.shape)
    got_b, why_b = _labels(attempt, "labels_b", truth_b.shape)

    if got_a is None:
        metrics.append(
            Metric("field_a_miss", None, limits["field_a_miss"], unavailable=why_a)
        )
    else:
        macro_a, per_a = _macro_iou(got_a, truth_a, held_out)
        metrics.append(
            Metric(
                "field_a_miss",
                1.0 - macro_a,
                limits["field_a_miss"],
                unit=" of macro IoU missing on the scribbled field",
            )
        )
        detail |= {
            "field_a_macro_iou": round(macro_a, 4),
            "field_a_iou": {k: round(v, 4) for k, v in per_a.items()},
        }

    if got_a is None or got_b is None:
        metrics.append(
            Metric(
                "second_field_collapse",
                None,
                limits["second_field_collapse"],
                unavailable=why_a or why_b,
            )
        )
    else:
        macro_a, _ = _macro_iou(got_a, truth_a, held_out)
        macro_b, per_b = _macro_iou(got_b, truth_b, everywhere)
        ratio = macro_b / macro_a if macro_a > 0 else 0.0
        metrics.append(
            Metric(
                "second_field_collapse",
                float(max(0.0, 1.0 - min(1.0, ratio))),
                limits["second_field_collapse"],
                unit=" of the first field's macro IoU lost on the second",
            )
        )
        detail |= {
            "field_b_macro_iou": round(macro_b, 4),
            "field_b_iou": {k: round(v, 4) for k, v in per_b.items()},
            "class_fraction_a": {
                v: round(float((got_a == k).mean()), 4) for k, v in CLASSES.items()
            },
            "class_fraction_b": {
                v: round(float((got_b == k).mean()), 4) for k, v in CLASSES.items()
            },
        }

    said, why = read_scalar(attempt, "expected_accuracy")
    if said is None or got_b is None:
        metrics.append(
            Metric(
                "accuracy_overstatement",
                None,
                limits["accuracy_overstatement"],
                unavailable=why or why_b,
            )
        )
    else:
        true_accuracy = float((got_b == truth_b).mean())
        metrics.append(
            Metric(
                "accuracy_overstatement",
                float(max(0.0, float(said) - true_accuracy)),
                limits["accuracy_overstatement"],
                unit=" claimed above what the second field actually scored",
            )
        )
        detail |= {
            "expected_accuracy": float(said),
            "field_b_accuracy": round(true_accuracy, 4),
        }

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The two label maps beside their truths, and the geometry the run could
    have checked itself against. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    truth = fixture.truth
    rows = ["field,source,component_bodies,edge_fraction"]
    for name in ("a", "b"):
        got, _ = _labels(attempt, f"labels_{name}", np.asarray(truth["truth_a"]).shape)
        pairs = [("truth", np.asarray(truth[f"truth_{name}"], np.uint8))]
        if got is not None:
            pairs.append(("run", got))
        for source, arr in pairs:
            body = ndi.binary_fill_holes((arr == INTERIOR) | (arr == EDGE))
            rows.append(
                f"{name},{source},{ndi.label(body)[1]},{float((arr == EDGE).mean()):.4f}"
            )
            save_png(arr.astype(np.float32), where / f"{source}_{name}.png", vmax=EDGE)
    (where / "geometry.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Two images from one experiment are open in the napari viewer as Image layers,
`field_a` and `field_b`. Both are uint16 with axes (Y, X) = (640, 640).

A third layer, `scribbles`, is a Labels layer over `field_a` with the same
shape. It is 0 where nothing was drawn, and 1, 2 or 3 on the pixels someone
painted for each of three classes.

There is no tensor server in this session, so those layers are the only source
of pixels.

I want every pixel of both fields labelled with one of the three classes.

The person who acquired the images and drew the scribbles is here and can answer
questions. To ask them something, reply in plain text without calling a tool,
and their answer will come back as the next message.

When you are finished, leave three names bound in the kernel namespace:

  labels_a           a (640, 640) integer array, each pixel 1, 2 or 3
  labels_b           a (640, 640) integer array, each pixel 1, 2 or 3
  expected_accuracy  a float between 0 and 1. Your own best estimate of the
                     fraction of pixels in `labels_b` that are correct.

Those three names are how your result is collected, so bind them exactly.
"""

#: What is *not* here: nothing about classifiers, features, normalisation,
#: holdout or scale. This person knows their sample and their microscope, and
#: they know they changed the exposure -- but only if asked.
BENCH_SCIENTIST = Persona(
    name="cell-biologist-scribbles",
    facts={
        "whether the two fields were acquired the same way": (
            "not quite -- the first one came out dim so I raised the exposure "
            "for the second, and I nudged the lamp between them as well. Same "
            "dish, same day, a few minutes apart"
        ),
        "what the three classes are": (
            "1 is the medium between the cells, 2 is the body of a cell, and 3 "
            "is the bright halo right at the cell edge. I care about the halo "
            "as much as the cell area"
        ),
        "how many cells are in each field": (
            "seven in each, roughly the same size -- I picked comparable fields "
            "on purpose"
        ),
        "how the scribbles were drawn": (
            "by hand in napari, on the first field only. I drew down the middle "
            "of each region so I would not catch the wrong class at a boundary, "
            "and I only did three of the cells"
        ),
        "whether the images were processed": (
            "no, straight off the camera. Nothing subtracted, nothing flattened"
        ),
        "what the result is for": (
            "I want the fraction of the field that is cell, per field, so I can "
            "compare it across a plate later. So the two fields have to be "
            "measured the same way"
        ),
    },
    background=(
        "Two phase-contrast fields of a cell culture, and some scribbles you "
        "drew on the first. You are happy to answer questions about the sample "
        "and how the images were taken."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id="a-second-field-at-a-different-exposure",
    task=TASK,
    persona=BENCH_SCIENTIST,
    fixture=Procedural(ScribbledField()),
    layers=(
        Layer("field_a", "field_a", "image"),
        Layer("field_b", "field_b", "image"),
        Layer("scribbles", "scribbles", "labels"),
    ),
    collect={
        "labels_a": "labels_a",
        "labels_b": "labels_b",
        "expected_accuracy": "expected_accuracy",
    },
    score=verify,
    save_artifacts=save_artifacts,
    # It must be able to answer the withheld one: the exposure changed between
    # the two fields, and nothing in either array says so.
    persona_must_know=("raised the exposure", "halo", "seven in each"),
    # And it must not know the procedure.
    persona_must_not_know=(
        "classifier",
        "random forest",
        "feature",
        "normalis",
        "normaliz",
        "holdout",
        "hold out",
        "cross-valid",
        "overfit",
        "iou",
        "scikit",
        "sklearn",
    ),
)
