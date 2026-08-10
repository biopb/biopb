"""`track-objects` as benchmark data: how big is a pixel, and how long is a frame?


The skill takes a label movie whose ids mean nothing across frames and returns
one identity per object. The construction knows the answer exactly, because it
chose the trajectories before it drew anything.

The withheld facts are the two in step 2 that no pixel carries — **0.5 µm per
pixel and 90 s per frame**. They are load-bearing twice over. They set
`MAX_STEP_PX`, the one parameter the linker cannot be run without; and they are
the whole of the conversion from px/frame to µm/min, so a run that assumes one
pixel is one micron and one frame is one minute reports a migration speed
**3.1x** too high with tracks that are otherwise perfectly good. That is a
scale, which is the kind of fact §5d asks for: unlike a "which channel is
structural", there is no back door in the pixels through which it can be
recovered.

What makes the tracking itself non-trivial, and is *not* withheld — it is
visible in the data and an unaided run may well find it:

  * cells sit in colonies, so the local spacing (median 17 px) is only ~3x the
    per-frame step (median 5.3 px) and the nearest blob is often the wrong one
  * 7% of detections are missing, so a linker that only looks at consecutive
    frames fragments every track that blinks
  * cells divide, so `track_id` (which ends at a division) and `tree_id` (which
    does not) are different answers to "how many cells"

The reference implementation these tolerances come from is in the pull request
that added this case, per `biopb-mcp/docs/skills.md` §11b.
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
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

SKILL = "track-objects"

#: The withheld pair. A 20x objective on a large sensor and a 90 s interval --
#: ordinary numbers, deliberately not round ones, so a run cannot land on them
#: by assuming the convenient thing.
PIXEL_UM = 0.5
INTERVAL_S = 90.0

# Set from the reference implementation over 5 seeds of this construction, not
# from taste. The reference lands at link_accuracy 0.89-0.91, false_link_rate
# 0.053-0.073, speed error 1.6-4.0% and lineage error 1.5-6.2%; the failures
# these limits have to separate it from are measured in the same run:
#
#   cutoff given as a distance, not squared   link_error 0.54, speed 0.44
#   frame-to-frame linking only               ............... lineage 0.40
#   splitting left off (the library default)  ............... lineage 0.54
#   `track_id` reported as the lineage column ............... lineage 2.06
#   px/frame reported as um/min ............................... speed 1.88
#
# Each limit sits between the reference and the mildest thing it has to
# reject, and the tightest of those gaps is `lineage_error`: 0.03 against 0.40.
#
# `link_accuracy` is deliberately loose enough to admit a competent improvised
# tracker (greedy nearest neighbour measures 0.84, a global assignment per
# frame pair 0.87): the arm's *number* is the finding, and a limit tight enough
# to fail those would be reporting the skill's delta as a pass/fail instead of
# measuring it.
TOLERANCE = {
    "link_error": 0.25,
    "false_link_rate": 0.15,
    "speed_error": 0.15,
    "lineage_error": 0.25,
}


# --- the fixture -----------------------------------------------------------


@dataclass(frozen=True)
class Colonies:
    """Migrating cells in colonies, segmented independently in every frame.

    Every frame's label ids are a fresh random permutation. That is not
    decoration: identity across frames is exactly what the run has to produce,
    so leaving the ids stable would put the answer in the data.
    """

    shape: tuple[int, int] = (600, 600)
    n_frames: int = 24
    n_colonies: int = 5
    per_colony: int = 13
    colony_sigma: float = 40.0
    radius: float = 7.0
    #: Per-frame step of the persistent random walk, px. Against a median
    #: nearest-neighbour distance of ~17 px this is what makes the linking
    #: ambiguous rather than obvious.
    step_px: float = 4.5
    persistence: float = 0.6
    #: Chance a present cell is missed in a given frame -- a segmentation that
    #: is good but not perfect, which is what gap closing is for.
    dropout: float = 0.07
    #: Chance a founder cell divides once during the movie.
    division_rate: float = 0.30
    seed: int = 0

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        h, w = self.shape
        centres = rng.uniform(0.18, 0.82, size=(self.n_colonies, 2)) * np.array(
            [h, w], float
        )

        # One entry per cell *segment*: a founder, or a daughter appended when
        # its mother divides. `lineage` is shared by a whole family.
        cells = [
            {
                "t0": 0,
                "t1": self.n_frames - 1,
                "lineage": c * self.per_colony + k,
                "pos0": centres[c] + rng.normal(0.0, self.colony_sigma, 2),
                "vel0": rng.normal(0.0, self.step_px, 2),
            }
            for c in range(self.n_colonies)
            for k in range(self.per_colony)
        ]
        n_founders = len(cells)

        paths: dict[int, np.ndarray] = {}
        i = 0
        while i < len(cells):
            cell = cells[i]
            n = cell["t1"] - cell["t0"] + 1
            pos = np.empty((n, 2))
            p = np.array(cell["pos0"], float)
            v = np.array(cell["vel0"], float)
            lo = [self.radius, self.radius]
            hi = [h - self.radius - 1, w - self.radius - 1]
            for k in range(n):
                pos[k] = p
                v = self.persistence * v + np.sqrt(
                    1.0 - self.persistence**2
                ) * rng.normal(0.0, self.step_px, 2)
                p = np.clip(p + v, lo, hi)
            paths[i] = pos

            # Only founders divide, and never within 5 frames of either end, so
            # every daughter is long enough to be a track rather than a blip.
            if cell["t0"] == 0 and rng.random() < self.division_rate:
                at_frame = int(rng.integers(5, self.n_frames - 5))
                cell["t1"] = at_frame
                paths[i] = pos[: at_frame + 1]
                mother = paths[i][-1]
                for sign in (+1, -1):
                    off = sign * rng.normal(0.0, 1.0, 2)
                    off = 3.0 * off / (np.linalg.norm(off) + 1e-9)
                    cells.append(
                        {
                            "t0": at_frame + 1,
                            "t1": self.n_frames - 1,
                            "lineage": cell["lineage"],
                            "pos0": mother + off,
                            "vel0": rng.normal(0.0, self.step_px, 2),
                        }
                    )
            i += 1

        # Which detections exist. Never the first or last frame of a cell: a
        # dropout there is indistinguishable from a shorter track, which makes
        # the truth ambiguous rather than the problem hard.
        present: dict[tuple[int, int], np.ndarray] = {}
        for idx, cell in enumerate(cells):
            path = paths[idx]
            for k, p in enumerate(path):
                if 0 < k < len(path) - 1 and rng.random() < self.dropout:
                    continue
                present[(idx, cell["t0"] + k)] = p

        yy, xx = np.mgrid[0:h, 0:w]
        labels = np.zeros((self.n_frames, h, w), np.uint16)
        det: list[tuple[int, int, int, int, float, float]] = []
        for t in range(self.n_frames):
            here = [(idx, p) for (idx, tt), p in present.items() if tt == t]
            order = rng.permutation(len(here))
            frame = labels[t]
            for lab, k in enumerate(order, start=1):
                _, p = here[k]
                frame[(yy - p[0]) ** 2 + (xx - p[1]) ** 2 <= self.radius**2] = lab
            for lab, k in enumerate(order, start=1):
                idx, _ = here[k]
                mask = frame == lab
                # A cell drawn over by a later one is not a detection: the run
                # cannot see it, so the truth must not claim it is there.
                if not mask.any():
                    continue
                py, px = np.nonzero(mask)
                det.append((t, lab, idx, cells[idx]["lineage"], py.mean(), px.mean()))

        det_arr = np.array(det, float)
        truth = {
            "det_frame": det_arr[:, 0].astype(int),
            "det_label": det_arr[:, 1].astype(int),
            "det_cell": det_arr[:, 2].astype(int),
            "det_lineage": det_arr[:, 3].astype(int),
            "det_y": det_arr[:, 4],
            "det_x": det_arr[:, 5],
            "n_founders": n_founders,
            "pixel_um": PIXEL_UM,
            "interval_s": INTERVAL_S,
        }
        truth["speed_um_per_min"] = _speed(truth, _links(truth))

        n_div = len(cells) - n_founders
        return Fixture(
            provenance=(
                f"procedural: seed {self.seed}, {self.n_frames} frames, "
                f"{len(det)} detections of {len(cells)} cell segments from "
                f"{n_founders} founders, {n_div} divisions, {self.dropout:.0%} "
                f"of detections dropped, {PIXEL_UM} um/px at {INTERVAL_S:.0f} s"
            ),
            about=(
                "Label ids are a fresh permutation in every frame, so identity "
                "has to be recovered rather than read. The pixel size and the "
                "frame interval are not in the data: without them the tracks "
                "can be right and the migration speed still 3x wrong."
            ),
            data={"labels": labels},
            truth=truth,
            tolerance=dict(TOLERANCE),
        )


# --- truth-side arithmetic, shared by the builder and the verifier ----------


def _links(truth) -> list[tuple[int, int]]:
    """Consecutive detections of one cell, as row pairs.

    Consecutive among the detections that *exist*, so a link spans a dropout --
    which is what makes gap closing something this can score.
    """
    order = np.lexsort((truth["det_frame"], truth["det_cell"]))
    cell = truth["det_cell"]
    return [
        (int(a), int(b))
        for a, b in zip(order[:-1], order[1:], strict=True)
        if cell[a] == cell[b]
    ]


def _speed(truth, links) -> float:
    """Mean instantaneous speed over *links*, in um/min. A link n frames long
    is divided by n, so spanning a dropout does not read as a fast cell."""
    y, x, f = truth["det_y"], truth["det_x"], truth["det_frame"]
    if not links:
        return float("nan")
    per_frame = np.mean(
        [np.hypot(y[b] - y[a], x[b] - x[a]) / (f[b] - f[a]) for a, b in links]
    )
    return float(per_frame) * truth["pixel_um"] / (truth["interval_s"] / 60.0)


def _rows(attempt: Attempt, truth) -> tuple[np.ndarray | None, str]:
    """The run's `tracks` table, aligned onto the truth's detection order.

    Deliberately not `read_array(..., shape)`: a table with the wrong number of
    rows is the most interesting way to get this wrong -- a run that dropped
    every object it could not link, or one that emitted a row per *track*
    rather than per detection, looks fine until something counts. So the shape
    is checked in pieces and what is wrong is said out loud.
    """
    got = attempt.arrays.get("tracks")
    if got is None:
        return None, "the run left no `tracks`"
    got = np.asarray(got)
    if got.ndim != 2 or got.shape[1] != 4:
        return None, f"the run's `tracks` is {got.shape}, not (N, 4)"
    want = {
        (int(f), int(lab)): i
        for i, (f, lab) in enumerate(
            zip(truth["det_frame"], truth["det_label"], strict=True)
        )
    }
    out = np.full((len(want), 2), -1, dtype=np.int64)
    seen = 0
    for frame, label, track_id, lineage_id in np.asarray(got, np.int64):
        i = want.get((int(frame), int(label)))
        if i is None or out[i, 0] >= 0:
            continue
        out[i] = (track_id, lineage_id)
        seen += 1
    if seen < len(want):
        return None, (
            f"the run's `tracks` covers {seen} of the {len(want)} segmented "
            "objects; a table missing rows cannot be scored against the whole "
            "movie"
        )
    return out, ""


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Four metrics, two families. The link metrics are a pair on purpose: a run
    that links nothing scores no false links, and a run that puts every
    detection in one track recovers every true link. Neither passes both.

    ``speed_error`` is the one the withheld facts control, and it is scored
    from the run's *own* number rather than recomputed from its table -- what
    is being measured is whether the answer reached the user in the units the
    user's question was in.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = fixture.truth
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    rows, why = _rows(attempt, truth)
    if rows is None:
        for name in ("link_error", "false_link_rate", "lineage_error"):
            metrics.append(Metric(name, None, limits[name], unavailable=why))
    else:
        track_id, lineage_id = rows[:, 0], rows[:, 1]
        truth_links = set(_links(truth))
        order = np.lexsort((truth["det_frame"], track_id))
        predicted = [
            (int(a), int(b))
            for a, b in zip(order[:-1], order[1:], strict=True)
            if track_id[a] == track_id[b]
        ]
        cell = truth["det_cell"]
        recovered = sum(1 for p in predicted if p in truth_links)
        wrong = sum(1 for a, b in predicted if cell[a] != cell[b])
        n_lineages = len(np.unique(lineage_id))
        metrics += [
            Metric(
                "link_error",
                1.0 - recovered / len(truth_links),
                limits["link_error"],
                unit=" of true links missed",
            ),
            Metric(
                "false_link_rate",
                wrong / max(len(predicted), 1),
                limits["false_link_rate"],
                unit=" of made links wrong",
            ),
            Metric(
                "lineage_error",
                abs(n_lineages - truth["n_founders"]) / truth["n_founders"],
                limits["lineage_error"],
            ),
        ]
        detail |= {
            "true_links": len(truth_links),
            "predicted_links": len(predicted),
            "n_tracks": int(len(np.unique(track_id))),
            "n_lineages": int(n_lineages),
            "n_founders": int(truth["n_founders"]),
        }

    speed, why = read_scalar(attempt, "mean_speed_um_per_min")
    want = truth["speed_um_per_min"]
    if speed is None:
        metrics.append(
            Metric("speed_error", None, limits["speed_error"], unavailable=why)
        )
    else:
        metrics.append(
            Metric("speed_error", abs(speed - want) / want, limits["speed_error"])
        )
        detail |= {"speed_um_per_min": speed, "true_speed_um_per_min": float(want)}

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The table a human needs to see why a number came out the way it did:
    every true link, and whether the run made it. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    truth = fixture.truth
    rows, _ = _rows(attempt, truth)
    if rows is None:
        return
    track_id = rows[:, 0]
    y, x, f = truth["det_y"], truth["det_x"], truth["det_frame"]
    lines = ["frame_from,frame_to,step_px,same_track,true_cell"]
    for a, b in _links(truth):
        step = np.hypot(y[b] - y[a], x[b] - x[a]) / (f[b] - f[a])
        lines.append(
            f"{f[a]},{f[b]},{step:.2f},"
            f"{int(track_id[a] == track_id[b])},{truth['det_cell'][a]}"
        )
    (where / "links.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A segmented time-lapse is open in the napari viewer as the Labels layer
`cells`. Its axes are (T, Y, X) = (24, 600, 600), uint16. Each frame was
segmented on its own, so the label ids do NOT carry over between frames. There
is no tensor server in this session, so that layer is the only source of pixels.

I need each cell followed through the movie, and I need to know how fast they
are migrating.

The person who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  tracks                  an (N, 4) integer array, one row per segmented
                          object in the movie, columns
                          [frame, label, track_id, lineage_id]
                            frame, label  identify the object in `cells`
                            track_id      the same for one object across frames
                            lineage_id    the same for a cell and all of its
                                          descendants
                          Every object present in `cells` must appear exactly
                          once.
  mean_speed_um_per_min   a float. The mean instantaneous speed over every
                          linked pair of consecutive detections of the same
                          object -- for a pair n frames apart, the distance
                          divided by n frames -- in microns per minute.

Those two names are how your result is collected, so bind them exactly.
"""

#: What is *not* here: no cutoff, nothing about squared distances, no mention
#: of gap closing or of two rounds, and no hint that lineage and track are
#: different columns of anything. This person knows their cells and their
#: microscope. A persona that had absorbed the procedure could answer a
#: question the agent never properly asked, and the numbers would stop meaning
#: what they appear to.
OPERATOR = Persona(
    name="operator-migration-assay",
    facts={
        "what the pixel size is": (
            "0.5 microns per pixel -- 20x objective, and I checked it against "
            "a stage micrometer"
        ),
        "how often the frames were taken": (
            "every 90 seconds, for 24 frames, so a little over half an hour"
        ),
        "how fast these cells move": (
            "they are quite slow -- I have never seen one do more than about "
            "5 microns a minute, and most are well under that"
        ),
        "whether the cells divide": (
            "yes, a good fraction of them divide once during the run. I care "
            "about the families -- how many cells I started with, and which "
            "ones came from which"
        ),
        "how good the segmentation is": (
            "mostly fine, but it loses a cell here and there for a frame or "
            "two when one drifts out of focus, and then it comes back"
        ),
        "whether the stage moved": (
            "no, the field is steady -- I checked that before you started"
        ),
        "what the experiment is": (
            "a migration assay. What I want out of it is how fast they move "
            "and how the families grow"
        ),
    },
    background=(
        "A time-lapse of cultured cells migrating in a dish, already "
        "segmented. You are happy to answer questions about the sample, the "
        "microscope and the acquisition."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="colonies-no-scale-in-the-pixels",
    task=TASK,
    persona=OPERATOR,
    fixture=Procedural(Colonies()),
    layers=(Layer("cells", "labels", kind="labels"),),
    collect={"tracks": "tracks", "mean_speed_um_per_min": "mean_speed_um_per_min"},
    score=verify,
    save_artifacts=save_artifacts,
    # It must be able to answer: the fixture strips the scale, and this person
    # knows the pixel size, the interval, how fast the cells go and that they
    # divide.
    persona_must_know=("0.5", "90 seconds", "5 microns", "divide"),
    # And it must not know the procedure.
    persona_must_not_know=(
        "laptrack",
        "cutoff",
        "gap clos",
        "track_id",
        "tree_id",
        "squared",
        "linear_sum_assignment",
    ),
)
