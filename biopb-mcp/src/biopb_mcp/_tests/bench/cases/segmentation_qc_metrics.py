"""`segmentation-qc-metrics` as benchmark data: which of these two is the truth?

The body's step 2 is a blocking confirm-input for exactly one reason, and it
says so: *"This is not derivable from the data, and getting it backwards swaps
precision with recall and inverts the split/merge diagnosis."* So the fixture is
two label layers of the same field, named `labels_run_a` and `labels_run_b`,
with nothing in either saying which one a person drew.

**The available heuristic points the wrong way, on purpose.** `labels_run_a` has
more objects and misses nothing, which is what an annotation is usually assumed
to look like — and it is the model output. The hand annotation is
`labels_run_b`, the smaller set, because the annotator skipped the faint cells
the model happily detected. A run that guesses instead of asking has a plausible
reason to guess, and gets it backwards.

**F1 alone cannot catch the mistake**, which is why three numbers are collected
rather than one. `f1 = 2·TP / (n_gt + n_pred)` is symmetric under swapping the
two layers, so a backwards run reports the *right* F1 while precision and recall
trade places. That is the shape of the failure the body's table calls
"precision and recall swapped versus expectation", and it is invisible in the
headline number a report would normally quote.

The counts are known in closed form by construction — 32 matched pairs, 8
predictions with nothing under them, 2 annotated cells the model missed — and
the builder asserts that `segmentation_qc` agrees before handing the fixture
over. That check is the cheap half of proving the fixture measures what it
claims: if the plugin and the bookkeeping ever disagree, the truth is wrong and
every arm scored against it is meaningless.

Nothing touches a border, so `EXCLUDE_BORDER` cannot move the numbers. That is
deliberate: the body asks for a border policy in the same breath as the ground
truth, and leaving it live would put a second uncontrolled choice inside one
measurement.
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

SKILL = "segmentation-qc-metrics"

#: Absolute error on a 0-1 score. Every number here is exact by construction, so
#: this is a "did you compute the same thing" limit, not a measurement band: it
#: passes a run that rounded to three decimals and fails one that swapped the
#: layers (which moves precision by 0.14).
TOLERANCE = {"precision_err": 0.01, "recall_err": 0.01, "f1_err": 0.01}


@dataclass(frozen=True)
class TwoRuns:
    """One field segmented twice: a hand annotation and a model's output.

    Laid out on a coarse grid, one object per cell, so every object is isolated:
    no splits, no merges, no border contact, and no ambiguity about which
    prediction belongs to which annotation.
    """

    shape: tuple[int, int] = (256, 256)
    rows: int = 6
    cols: int = 7
    n_gt: int = 34
    n_missed: int = 2
    n_spurious: int = 8
    seed: int = 0

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        cell_y = self.shape[0] / self.rows
        cell_x = self.shape[1] / self.cols
        centres = np.array(
            [
                (cell_y * (i + 0.5), cell_x * (j + 0.5))
                for i in range(self.rows)
                for j in range(self.cols)
            ]
        )
        rng.shuffle(centres, axis=0)
        annotated = centres[: self.n_gt]
        spurious = centres[self.n_gt : self.n_gt + self.n_spurious]

        gt = np.zeros(self.shape, dtype=np.int32)
        pred = np.zeros(self.shape, dtype=np.int32)
        yy, xx = np.ogrid[: self.shape[0], : self.shape[1]]

        def disk(into, label, centre, radius):
            cy, cx = centre
            into[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2] = label

        # The matched pairs, offset by a pixel: identical masks would score a
        # mean IoU of exactly 1, which is a degenerate field that says nothing
        # about whether a run matched the objects or copied one layer.
        matched = self.n_gt - self.n_missed
        pred_label = 0
        for label, centre in enumerate(annotated, start=1):
            radius = float(rng.uniform(7.0, 11.0))
            disk(gt, label, centre, radius)
            if label > matched:
                continue  # the two the model missed
            pred_label += 1
            step = rng.choice([-1.0, 1.0], size=2)
            disk(pred, pred_label, (centre[0] + step[0], centre[1] + step[1]), radius)

        # Detections with nothing under them, in cells the annotator left empty.
        for centre in spurious:
            pred_label += 1
            disk(pred, pred_label, centre, float(rng.uniform(6.0, 9.0)))

        n_pred = matched + self.n_spurious
        truth = {
            "precision": matched / n_pred,
            "recall": matched / self.n_gt,
            "f1": 2 * matched / (self.n_gt + n_pred),
            "tp": matched,
            "fp": self.n_spurious,
            "fn": self.n_missed,
            "gt_layer": "labels_run_b",
        }
        _agrees_with_the_plugin(gt, pred, truth)

        return Fixture(
            provenance=(
                f"procedural: seed {self.seed}, {self.n_gt} annotated objects, "
                f"{n_pred} predicted — {matched} matched, {self.n_spurious} "
                f"spurious, {self.n_missed} missed"
            ),
            about=(
                "Two label layers of one field. The larger, more complete set is "
                "the model's; the hand annotation is the smaller one, so the "
                "obvious guess is backwards and swaps precision with recall."
            ),
            # `labels_run_a` is the prediction and `labels_run_b` the truth.
            # Which is which lives in `truth`, not here.
            data={"labels_run_a": pred, "labels_run_b": gt},
            truth=truth,
            tolerance=dict(TOLERANCE),
        )


def _agrees_with_the_plugin(gt, pred, truth) -> None:
    """The closed-form counts, checked against the plugin the skill calls.

    Two independent derivations of the same numbers: the construction knows how
    many objects it matched, and `segmentation_qc` matches them by IoU without
    being told. A disagreement means the fixture's truth is wrong — objects that
    overlap when they should not, a shift that dropped a pair below the
    threshold — and every arm scored against it would be meaningless, so this
    raises at build time rather than reporting a quiet zero later.
    """
    from biopb_mcp.plugins import segmentation_qc

    got = segmentation_qc.match_labels(gt, pred, iou_threshold=0.5)
    counted = (got.tp, got.fp, got.fn)
    declared = (truth["tp"], truth["fp"], truth["fn"])
    if counted != declared:
        raise AssertionError(
            f"{SKILL} fixture: segmentation_qc counts {counted} (tp, fp, fn) but "
            f"the construction declares {declared}. The truth is wrong, not the "
            "plugin."
        )
    if got.splits or got.merges:
        raise AssertionError(
            f"{SKILL} fixture: objects are meant to be isolated, but the plugin "
            f"reports {got.splits} splits and {got.merges} merges"
        )


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score the three numbers at IoU 0.5.

    `precision_err` and `recall_err` are what carry the withheld fact: swapping
    the two layers exchanges them. `f1_err` is reported beside them precisely
    because it does *not* — a run can have the headline number right and the
    diagnosis backwards, and the pair is what shows it.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    for name in ("precision", "recall", "f1"):
        limit = limits[f"{name}_err"]
        got, why = read_scalar(attempt, name)
        if got is None:
            metrics.append(Metric(f"{name}_err", None, limit, unavailable=why))
            continue
        metrics.append(
            Metric(f"{name}_err", abs(got - float(fixture.truth[name])), limit)
        )
        detail[f"{name}_reported"] = got

    detail["truth"] = {
        k: fixture.truth[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")
    }
    # The one line that names the mistake when it happened, since the metrics
    # can only say "0.14 out": a swapped run's precision *is* the truth's recall.
    swapped = (
        detail.get("precision_reported") is not None
        and abs(detail["precision_reported"] - float(fixture.truth["recall"])) < 0.01
    )
    detail["reads_as_layers_swapped"] = bool(swapped)
    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The two label fields, and the numbers beside the truth."""
    fixture = outcome.fixture
    save_png(np.asarray(fixture.data["labels_run_a"]) > 0, where / "run-a.png")
    save_png(np.asarray(fixture.data["labels_run_b"]) > 0, where / "run-b.png")
    rows = ["metric,truth,reported"]
    for name in ("precision", "recall", "f1"):
        got = outcome.detail.get(f"{name}_reported")
        rows.append(
            f"{name},{float(fixture.truth[name]):.4f},"
            f"{'' if got is None else f'{got:.4f}'}"
        )
    (where / "scores.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Two Labels layers of the same 256x256 field are open in the napari viewer:
`labels_run_a` and `labels_run_b`. There is no tensor server in this session,
so those layers are the only source of pixels.

I want to know how good the segmentation is.

The person who produced them is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave three names bound in the kernel namespace, all at
an IoU threshold of 0.5:

  precision   the precision, as a float
  recall      the recall, as a float
  f1          the F1 score, as a float

Those three names are how your result is collected, so bind them exactly.
"""

#: This person knows which layer they drew by hand and which came out of a
#: model. They do not know what F1 is, and must not: a respondent that can talk
#: about precision and recall could answer a question the agent never properly
#: asked, and the numbers would stop meaning what they appear to.
ANNOTATOR = Persona(
    name="annotator-two-runs",
    facts={
        "where labels_run_b came from": (
            "I drew that one by hand, cell by cell, in the napari labels tool"
        ),
        "where labels_run_a came from": (
            "that is the model's output — I ran the pretrained segmentation on "
            "the same field and saved what came back"
        ),
        "why the two differ in count": (
            "I skipped the very faint cells I was not sure about; the model "
            "picked up quite a few things I did not count"
        ),
        "whether anything is cut off at the edge": (
            "no, nothing runs off the edge of this crop, I chose it for that"
        ),
        "what I want to decide": (
            "whether the model is good enough that I can stop drawing these by hand"
        ),
    },
    background=(
        "One field of cultured cells, segmented twice. You are happy to answer "
        "questions about how each of the two layers was produced."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="hand-annotation-versus-model",
    task=TASK,
    persona=ANNOTATOR,
    fixture=Procedural(TwoRuns()),
    layers=(
        Layer("labels_run_a", "labels_run_a", kind="labels"),
        Layer("labels_run_b", "labels_run_b", kind="labels"),
    ),
    collect={"precision": "precision", "recall": "recall", "f1": "f1"},
    score=verify,
    save_artifacts=save_artifacts,
    # `checklist: plugin:segmentation_qc`, and the body says there is no degraded
    # path. Without this the session has no such plugin and the run would be
    # scoring an environment the skill declares it cannot work in.
    plugins=("segmentation_qc",),
    catalog_query="segmentation quality",
    # It must be able to answer which layer is which.
    persona_must_know=("labels_run_a", "labels_run_b", "by hand"),
    # And it must not know the metric — only which layer it drew.
    persona_must_not_know=(
        "f1",
        "iou threshold",
        "precision",
        "recall",
        "match_labels",
        "segmentation_qc",
    ),
)
