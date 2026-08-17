"""Strahler ordering as benchmark data: how branched is this arbor, really?

A deferred-tier case (`docs/skill-candidates.md`). SNT arbor morphometrics was
**prescreened and dropped** 2026-08-06 for the shipped catalog — every Sonnet
arm collapsed degree-2 runs into branches unprompted and said so. It is here
anyway, because the rejection is conditional on the consuming tier: every Haiku
arm computed Strahler orders per *node*, reported **1379 terminals against
422**, and narrated the resulting bifurcation ratio as "fractal-like". This case
is what makes that re-measurable rather than remembered.

**It is an integration test, not a screen of one claim.** The arbor arrives as a
segmented volume, so the run has to skeletonize it, cluster its junctions,
collapse its degree-2 runs and only then order it. That is more than the
candidate was nominated for — the candidate assumes tracing is done — and it is
deliberate: this case names no skill, so there is nothing to ablate and nothing
to isolate, and what is worth knowing is whether the whole pipeline lands on the
right number.

**The volume route had to be earned.** A first arbor grown without a clearance
constraint lost 14.3% of its terminals to skeletonization before Strahler was
reached, which would have made the benchmark unwinnable (protocol §4) and scored
the run on the wrong step (§8). The loss was not skeletonization — it was the
arbor crossing itself, which merges two branches into a junction the generating
tree does not have. Rejection-sampling every segment against every non-incident
one at `CLEARANCE_PX` removed it entirely, the same fix
`skeleton_network_metrics` makes for the same reason.

Measured on this fixture, against a tree of 52 terminals at max Strahler order 5
with a Horton bifurcation ratio of 3.07:

  ===============================================  =========  =====  ======
  route                                            terminals  order      Rb
  ===============================================  =========  =====  ======
  tree truth                                              52      5    3.07
  skeletonize, cluster junctions, collapse, order         53      5    3.16
  ..the same, without junction clustering                 71      5    3.28
  Strahler order per skeleton voxel                     2138      5   16.06
  ===============================================  =========  =====  ======

**The terminal count carries the discrimination, and that is on purpose.** It is
convention-free — the doc's protocol §8 exists because the SNT run nearly failed
a correct arm over whether the soma counts as a branch, 835 against 834, both
self-consistent — while a terminal is a terminal. The reference lands within
1.9%; the two wrong routes are 36.5% and 4011% away, so `TOLERANCE` at 0.15 sits
about eight times above a clean run and well under half the nearest failure.

**Why the per-voxel number is so much larger than the doc's.** The Haiku arms
ordered per *traced node* and got 1379 against 422, a factor of three. A voxel
skeleton is an order of magnitude denser than a tracing, so the same mistake
made here costs a factor of forty. The failure is identical; only the sampling
of the thing being mis-ordered differs, and that is worth knowing before this
number is compared against the doc's.

**The bifurcation ratio is scored anyway, loosely, and it is not the gate.** It
is the number that actually reaches a figure, and the doc's whole point about
this candidate is that a wrong one does not read as wrong: neuronal arbors are
normally reported at Rb 2-3, so the Haiku arms' 1.95 read as an interesting cell
rather than as an error. On *this* fixture it separates the per-voxel route
(424%) and not the un-clustered one — 3.1% for the reference against 7.1% — so
its limit catches a gross miss and nothing finer. A run that gets the terminals
right and the ratio wrong has still built a different tree, and the report
should say so.

**Max Strahler order is collected and never scored.** It is 5 for the truth, for
the reference, for the un-clustered route and for the per-voxel reading alike —
invariant to exactly the collapse this case is about, so it cannot serve as the
check. The doc found the same thing (order 7 for every arm at both tiers), and
collecting it without scoring it is how that stays visible instead of being
rediscovered.
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

NAMESPACE = "strahler-ordering"
CASE_ID = "traced-arbor-as-a-segmented-volume"

#: From the table in the module docstring, not from taste. The terminal limit
#: sits between the reference's 1.9% and the un-clustered route's 36.5%; the
#: bifurcation limit is deliberately loose — see the docstring for why it is not
#: the gate.
TOLERANCE = {
    "terminal_error_frac": 0.15,
    "bifurcation_error_frac": 0.25,
}

SHAPE = (96, 448, 448)
#: Between non-incident segments. A crossing is a junction the mask has and the
#: generating tree does not, and it is what made the first version of this
#: fixture unwinnable.
CLEARANCE_PX = 4.5
DEPTH = 8
N_PRIMARY = 7
SOMA = (48.0, 224.0, 224.0)
SEED = 23


def _segment_distance(p0, p1, q0, q1) -> float:
    """Closest approach of two line segments."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    den = a * c - b * b
    if den < 1e-9:
        sc, tc = 0.0, (e / c if c > 1e-9 else 0.0)
    else:
        tc = float(np.clip((a * e - b * d) / den, 0, 1))
        sc = float(np.clip((b * tc - d) / a, 0, 1))
    return float(np.linalg.norm(w + sc * u - tc * v))


def strahler(children: dict[int, list[int]], root: int = 0) -> dict[int, int]:
    """Strahler order per **branch**, collapsing degree-2 runs on the way down.

    The collapse is the whole distinction this case exists to measure: a node
    with one child is a point along a branch, not a branch of its own, and
    ordering per node instead inflates the terminal count — by three on the
    doc's traced arbor and by forty on this fixture's denser voxel skeleton —
    while leaving the maximum order untouched.
    """
    order: dict[int, int] = {}

    def walk(node: int) -> int:
        chain = children[node]
        while len(chain) == 1:
            node = chain[0]
            chain = children[node]
        if not chain:
            order[node] = 1
            return 1
        sub = sorted(walk(child) for child in chain)
        order[node] = sub[-1] + 1 if len(sub) > 1 and sub[-1] == sub[-2] else sub[-1]
        return order[node]

    walk(root)
    return order


def bifurcation_ratio(order: dict[int, int]) -> float:
    """Horton's Rb: the mean of consecutive order-count ratios."""
    counts: dict[int, int] = {}
    for value in order.values():
        counts[value] = counts.get(value, 0) + 1
    top = max(counts)
    ratios = [counts[k] / counts[k + 1] for k in range(1, top) if counts.get(k + 1)]
    return float(np.mean(ratios)) if ratios else float("nan")


@dataclass(frozen=True)
class TracedArbor:
    """A branching arbor that never touches itself, as a segmented volume."""

    shape: tuple[int, int, int] = SHAPE

    def _grow(self) -> list[list[float]]:
        """Rows of ``(id, z, y, x, radius, parent)``."""
        rng = np.random.default_rng(SEED)
        rows: list[list[float]] = [[0, *SOMA, 3.0, -1]]
        drawn: list[tuple[int, int, np.ndarray, np.ndarray]] = []

        def clear(parent: int, p0, p1) -> bool:
            return all(
                _segment_distance(p0, p1, q0, q1) >= CLEARANCE_PX
                for a, b, q0, q1 in drawn
                if parent not in (a, b)
            )

        frontier: list[tuple[int, np.ndarray, int]] = []
        for k in range(N_PRIMARY):
            theta = np.pi * (0.30 + 0.40 * rng.random())
            phi = 2 * np.pi * k / N_PRIMARY + rng.normal(0, 0.12)
            frontier.append(
                (
                    0,
                    np.array(
                        [
                            np.cos(theta),
                            np.sin(theta) * np.sin(phi),
                            np.sin(theta) * np.cos(phi),
                        ]
                    ),
                    1,
                )
            )

        while frontier:
            parent, direction, level = frontier.pop()
            if level > DEPTH:
                continue
            # Node spacing widens with depth, so a node-based reading is
            # distorted rather than merely scaled -- the doc's second trap.
            step = 5.0 + 2.5 * level
            here, grew = parent, 0
            for _ in range(int(rng.integers(3, 7))):
                direction = direction + rng.normal(0, 0.10, 3)
                direction /= np.linalg.norm(direction)
                p0 = np.array(rows[here][1:4], float)
                p1 = p0 + step * direction
                if not all(8 < p1[i] < self.shape[i] - 8 for i in range(3)):
                    break
                if not clear(here, p0, p1):
                    break
                rows.append([len(rows), *p1, float(max(0.9, 2.6 - 0.28 * level)), here])
                drawn.append((here, len(rows) - 1, p0, p1))
                here = len(rows) - 1
                grew += 1
            if grew >= 2 and level < DEPTH and rng.random() < 0.95:
                axis = np.cross(direction, rng.normal(0, 1, 3))
                axis /= max(float(np.linalg.norm(axis)), 1e-9)
                for sign in (+1, -1):
                    spread = rng.uniform(0.40, 0.80)
                    new = direction * np.cos(spread) + sign * axis * np.sin(spread)
                    frontier.append((here, new / np.linalg.norm(new), level + 1))
        return rows

    def _rasterize(self, rows) -> np.ndarray:
        """Per-segment, in a local box: a full-volume distance per segment is
        two orders of magnitude more work than the arbor is worth."""
        mask = np.zeros(self.shape, bool)
        for node in rows:
            if node[5] < 0:
                continue
            p0 = np.array(rows[int(node[5])][1:4], np.float32)
            p1 = np.array(node[1:4], np.float32)
            along = p1 - p0
            length = float(np.linalg.norm(along))
            if length < 1e-6:
                continue
            unit = along / length
            radius = float(node[4])
            lo = np.maximum(np.floor(np.minimum(p0, p1) - radius - 1).astype(int), 0)
            hi = np.minimum(
                np.ceil(np.maximum(p0, p1) + radius + 2).astype(int), self.shape
            )
            zz, yy, xx = np.mgrid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            off = np.stack([zz, yy, xx], -1).astype(np.float32) - p0
            t = np.clip(off @ unit, 0, length)
            near = np.linalg.norm(off - t[..., None] * unit, axis=-1) <= radius
            mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] |= near
        return mask

    def __call__(self) -> Fixture:
        rows = self._grow()
        children: dict[int, list[int]] = {int(r[0]): [] for r in rows}
        for row in rows:
            if row[5] >= 0:
                children[int(row[5])].append(int(row[0]))
        order = strahler(children)
        n_terminals = sum(1 for value in order.values() if value == 1)
        rb = bifurcation_ratio(order)
        mask = self._rasterize(rows)

        # The properties the case rests on, checked before anyone pays for a
        # run. None of them is visible from the volume alone.
        assert n_terminals >= 30, (
            f"{n_terminals} terminals is too few for a bifurcation ratio to be "
            "about the arbor rather than about one branch"
        )
        assert max(order.values()) >= 4, (
            f"max Strahler order is {max(order.values())}, so the arbor is not "
            "deep enough for an ordering to say anything"
        )
        chain = sum(1 for kids in children.values() if len(kids) == 1)
        assert chain >= 3 * n_terminals, (
            f"only {chain} of {len(rows)} nodes lie along a branch rather than "
            f"ending or splitting one, against {n_terminals} terminals — the "
            "arbor is too sparsely sampled for collapsing degree-2 runs to be "
            "load-bearing, which is the whole of what this case measures"
        )
        pieces = ndi.label(mask, structure=np.ones((3, 3, 3)))[1]
        assert pieces == 1, (
            f"the arbor is in {pieces} pieces, so a run could order one of them "
            "and the truth would not say so"
        )

        return Fixture(
            provenance=(
                f"procedural: an arbor of {len(rows)} nodes and {n_terminals} "
                f"terminals grown to depth {DEPTH} from {N_PRIMARY} primary "
                f"neurites, rejection-sampled at {CLEARANCE_PX} px clearance, "
                f"rasterized into {self.shape[0]}x{self.shape[1]}x{self.shape[2]} "
                f"voxels, seed {SEED}"
            ),
            about=(
                f"A traced arbor as a segmented volume: {n_terminals} terminals "
                f"at max Strahler order {max(order.values())}, with a Horton "
                f"bifurcation ratio of {rb:.2f}. Ordering the skeleton per voxel "
                "instead of collapsing degree-2 runs into branches reports 2138 "
                "terminals, and leaves the maximum order unchanged — so the "
                "maximum cannot be the check. Failing to cluster the junction "
                "voxels reports 71, because a thick junction is many "
                "degree-3 voxels and not many junctions."
            ),
            data={"arbor": mask.astype(np.uint8)},
            truth={
                "n_terminals": n_terminals,
                "bifurcation_ratio": rb,
                "max_strahler_order": max(order.values()),
                "n_nodes": len(rows),
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Both metrics are fractional rather than absolute: a terminal count is a
    count of a thing the run defined for itself, and the failure this case
    catches is a factor, not an offset.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    for key, limit_name in (
        ("n_terminals", "terminal_error_frac"),
        ("bifurcation_ratio", "bifurcation_error_frac"),
    ):
        want = fixture.truth.get(key)
        if want is None:
            metrics.append(
                Metric(
                    limit_name,
                    None,
                    limits[limit_name],
                    unavailable=f"the fixture carries no {key}",
                )
            )
            continue
        got, why = read_scalar(attempt, key)
        if got is None:
            metrics.append(
                Metric(limit_name, None, limits[limit_name], unavailable=why)
            )
            continue
        want = float(want)
        metrics.append(
            Metric(
                limit_name,
                abs(got - want) / max(abs(want), 1e-12),
                limits[limit_name],
            )
        )
        detail[f"{key}_reported"] = round(got, 3)
        detail[f"{key}_true"] = round(want, 3)

    # Collected and never scored: invariant to the collapse this case is about,
    # so it cannot be the check. See the module docstring.
    got, _why = read_scalar(attempt, "max_strahler_order")
    if got is not None:
        detail["max_strahler_order_reported"] = int(round(got))
        detail["max_strahler_order_true"] = int(
            fixture.truth.get("max_strahler_order", -1)
        )

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    mask = fixture.data.get("arbor")
    if mask is None:
        return
    mask = np.asarray(mask, float)
    save_png(mask.max(axis=0), where / "arbor-xy.png")
    save_png(mask.max(axis=1), where / "arbor-xz.png")

    rows = ["quantity,truth,reported"]
    for key in ("n_terminals", "bifurcation_ratio", "max_strahler_order"):
        want = fixture.truth.get(key)
        got = attempt.arrays.get(key)
        got = "" if got is None else f"{np.asarray(got, float).reshape(-1)[0]:g}"
        rows.append(f"{key},{'' if want is None else f'{float(want):g}'},{got}")
    (where / "arbor.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
A traced neuron is open in the napari viewer as a segmented volume:

  arbor  uint8 (96, 448, 448) — 1 inside the neuron, 0 outside

There is no tensor server in this session, so that layer is the only source of
pixels. The tracing is done; do not re-segment it, and do not clean it up beyond
what your measurement needs. The voxels are cubic.

I need this arbor described the way an anatomist would: how many branch tips it
has, how deep the hierarchy goes, and how fast branch counts fall as you go up
that hierarchy.

The person who prepared and traced this is here and can answer questions. To ask
them something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave three names bound in the kernel namespace:

  n_terminals         how many terminal branches the arbor has
  max_strahler_order  the Strahler order of the whole arbor
  bifurcation_ratio   the mean Horton bifurcation ratio over consecutive orders

Those names are how your result is collected, so bind them exactly.
"""

#: Self-sufficient: the prompt says what to report and the volume carries the
#: rest, so this person holds no part of the answer. Note what is *not* here —
#: nothing about skeletons, degree-2 runs, junction clustering or what counts as
#: a branch, which is the whole of what the case measures.
NEUROANATOMIST = Persona(
    name="operator-traced-arbor",
    facts={
        "what the cell is": (
            "a pyramidal neuron from a mouse cortical slice, filled with "
            "biocytin and imaged as a confocal stack"
        ),
        "how the tracing was made": (
            "semi-automatic, then checked plane by plane. I am confident it "
            "follows the real processes and does not join anything up wrongly"
        ),
        "whether the cell is complete": (
            "for the part in the slice, yes. Anything that left the slice was "
            "cut, and those endings are real endings as far as this stack goes"
        ),
        "why the processes look thicker near the middle": (
            "they are. Processes taper as they go out, and that is the cell "
            "rather than the imaging"
        ),
        "what the numbers are for": (
            "comparing cells between genotypes. We report the same handful of "
            "numbers for every cell so they can go in one table"
        ),
        "whether the stack was resampled": (
            "no, and the voxels were already cubic when it was acquired, so "
            "nothing needed rescaling"
        ),
    },
    background=(
        "You fill and image cortical neurons and trace them yourself. You are "
        "happy to answer questions about the cell, the microscope and how the "
        "tracing was done."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=NEUROANATOMIST,
    fixture=Procedural(TracedArbor()),
    layers=(Layer("arbor", "arbor", kind="labels"),),
    collect={
        "n_terminals": "n_terminals",
        "max_strahler_order": "max_strahler_order",
        "bifurcation_ratio": "bifurcation_ratio",
    },
    score=verify,
    save_artifacts=save_artifacts,
)
