"""`stitch-tiles` as benchmark data: which way did the stage travel?

The skill takes a grid of overlapping tiles and returns where each one belongs.
The withheld fact is step 2's: the **grid shape and the acquisition path**. A
stack of 24 tiles is 4x6 or 6x4 or 3x8, and each of those was collected either
row by row or in a serpentine, reversing every second row. None of it is in the
pixels, none of it is in the array's shape, and reading a snake acquisition as
row-major mirrors every other row — which is glaring in the mosaic and invisible
in every number a run prints about itself.

**This fixture is heuristic-defeating, not categorically absent, and that is a
weaker construction** (`biopb-mcp/docs/skill-testing.md` §5d). Unlike
`flatfield`'s camera offset, this answer *is* recoverable from the data: the
twelve shape-and-path combinations are enumerable, and registering under each
and keeping the one whose accepted pairs form a single connected component would
find it without asking anybody. That back door is real and is left open on
purpose — it is the same shape as the one `drift-correction`'s movie has, and a
run that walks through it has done something defensible rather than something
lucky. What the case then measures is whether the run got the layout *right*,
by whichever of the two routes.

The regime is deliberately generous: 18% overlap, an ordinary specimen, mild
stage error. Run end to end on this fixture, the skill's own method scores
**1.0 px** rms with the order obtained by asking, and **450 px** with the order
assumed row-major. The tolerance sits in that gap with wide margin on both
sides, because it is not resolving near-misses — it is separating a correct
layout from a mirrored one, and those differ by most of a tile row.
:meth:`SnakeGrid.__call__` asserts that separation before a run is paid for.

Worth knowing when reading a red arm: the wrong ordering is *also* caught by the
skill's own step-5 gate, which sees the accepted pairs fall into 15 disconnected
pieces rather than one. A run that follows the body therefore has a route to
noticing its mistake without the truth — so a failure here is usually a run that
skipped the gate, not one the fixture ambushed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ....agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_array,
    save_png,
)
from ....agentbench._respondent import Persona
from .._benchmark import Case, Layer

SKILL = "stitch-tiles"

#: Set from measurement (see the module docstring), not from taste. A correct
#: layout comes back at 1.0 px rms / 2.4 px worst; the failure this case exists
#: to catch — a serpentine acquisition read as row-major — misplaces half the
#: tiles by most of a row, some 450 px. Nothing lands in between, so the limits
#: are loose on purpose: they bound an agent's implementation choices rather
#: than discriminating between near-misses.
TOLERANCE = {
    "placement_error_px": 3.0,
    "worst_tile_error_px": 6.0,
}

#: 4 rows x 6 columns. Not square, so the shape is a real question rather than
#: one a run can shrug off, and `n_tiles` alone does not answer it.
GRID = (4, 6)

#: Percent of the tile. Comfortably inside the regime the skill reports as
#: reliable — this case is about the layout, and stacking a marginal overlap on
#: top would make a failure ambiguous.
OVERLAP_PCT = 18

#: Peak stage error in pixels, in each axis. Large enough that nominal placement
#: fails the tolerance on its own, so a run cannot pass by skipping registration.
JITTER = 10

TILE = (256, 256)


def _acquisition_order(rows: int, cols: int) -> list[tuple[int, int]]:
    """Serpentine: left to right along row 0, right to left along row 1, ...

    This is the fact the fixture withholds. It is applied when the stack is
    built and never recorded in anything the agent receives.
    """
    order = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else reversed(range(cols))
        order.extend((r, c) for c in cs)
    return order


def _canvas(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Specimen texture with structure at more than one scale.

    Both scales earn their place: the fine blobs are what phase correlation
    locks onto, and the slow component keeps a featureless tile from being
    impossible rather than merely hard.
    """
    fine = ndimage.gaussian_filter((rng.random(shape) < 0.003).astype(np.float32), 3.0)
    coarse = ndimage.gaussian_filter(rng.random(shape).astype(np.float32), 14.0)
    img = fine / max(fine.max(), 1e-6) + 0.35 * coarse
    return (300.0 + 2400.0 * img / img.max()).astype(np.float32)


def placement_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-tile distance between a layout and the truth, in pixels.

    Both sides are recentred first: where the mosaic's origin sits is a choice,
    not a result, so only the *relative* layout is scorable. Everything this
    case discriminates survives that — a mirrored row is wrong by hundreds of
    pixels relative to its neighbours, not merely offset from them.
    """
    estimate = np.asarray(estimate, float)
    truth = np.asarray(truth, float)
    delta = (estimate - estimate.mean(0)) - (truth - truth.mean(0))
    return np.hypot(delta[:, 0], delta[:, 1])


@dataclass(frozen=True)
class SnakeGrid:
    """A tile grid whose shape and acquisition path only the microscopist knows."""

    seed: int = 23

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        rows, cols = GRID
        H, W = TILE
        step_y = H - int(H * OVERLAP_PCT / 100)
        step_x = W - int(W * OVERLAP_PCT / 100)
        ground = _canvas(
            rng,
            (
                (rows - 1) * step_y + H + 4 * JITTER,
                (cols - 1) * step_x + W + 4 * JITTER,
            ),
        )

        tiles, positions = [], []
        for r, c in _acquisition_order(rows, cols):
            y = r * step_y + 2 * JITTER + int(rng.integers(-JITTER, JITTER + 1))
            x = c * step_x + 2 * JITTER + int(rng.integers(-JITTER, JITTER + 1))
            patch = ground[y : y + H, x : x + W]
            tiles.append(patch + rng.normal(0.0, np.sqrt(np.maximum(patch, 0.0))))
            positions.append((y, x))
        images = np.stack(tiles).astype(np.float32)
        truth = np.array(positions, float)

        # The properties the case rests on, checked before anyone pays for a run
        # (§5d's rule about a fixture whose truth is wrong).
        #
        # First: nominal placement must *fail*. Otherwise a run that never
        # registered anything would score as a success, and the case would be
        # measuring nothing at all.
        nominal = np.array(
            [
                (r * step_y + 2 * JITTER, c * step_x + 2 * JITTER)
                for r, c in _acquisition_order(rows, cols)
            ],
            float,
        )
        nominal_error = float(np.sqrt((placement_error(nominal, truth) ** 2).mean()))
        assert nominal_error > 2 * TOLERANCE["placement_error_px"], (
            f"nominal placement scores {nominal_error:.1f} px against a limit of "
            f"{TOLERANCE['placement_error_px']} — a run could pass this case "
            "without registering anything"
        )

        # Second: the failure this case exists to catch must be unambiguous. A
        # serpentine stack read as row-major is what an agent that does not ask
        # produces, and it has to land far outside the limit, not near it.
        row_major = np.array(
            [
                (r * step_y + 2 * JITTER, c * step_x + 2 * JITTER)
                for r in range(rows)
                for c in range(cols)
            ],
            float,
        )
        mirrored = float(np.sqrt((placement_error(row_major, truth) ** 2).mean()))
        assert mirrored > 20 * TOLERANCE["placement_error_px"], (
            f"reading the snake as row-major scores {mirrored:.1f} px, which is "
            "not clearly separated from a correct layout"
        )

        return Fixture(
            provenance=(
                f"procedural: {rows * cols} tiles of {H}x{W}, seed {self.seed}, "
                f"{rows}x{cols} grid acquired serpentine at {OVERLAP_PCT}% "
                f"overlap with +-{JITTER} px stage error"
            ),
            about=(
                f"{rows * cols} tiles stacked in acquisition order. The stack "
                f"says nothing about whether that order is {rows}x{cols} or "
                f"{cols}x{rows}, nor whether the stage travelled row by row or "
                "in a serpentine — and it travelled in a serpentine. Read as "
                f"row-major the layout is wrong by {mirrored:.0f} px; placed at "
                f"the nominal grid it is wrong by {nominal_error:.1f} px."
            ),
            data={"tiles": images},
            truth={
                "positions": truth,
                # The private facts. Stripped from `data`, held by the
                # respondent, and read by the verifier to know what to score.
                "grid": GRID,
                "order": "snake",
                "canvas": ground,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Both metrics come off one array — the layout the run left in `pos` — read
    two ways, because they fail differently. ``placement_error_px`` is the rms
    and answers "is this mosaic right"; ``worst_tile_error_px`` is the maximum
    and catches the run that placed twenty-three tiles correctly and dropped one
    somewhere else, which the rms over 24 tiles dilutes into looking fine.

    Both need only ``truth["positions"]``, which curated data can carry: a stage
    that logged its positions supplies it directly, and better than this fixture
    does.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    truth_positions = fixture.truth.get("positions")
    if truth_positions is None:
        pos, why = None, "the fixture carries no reference tile positions"
    else:
        pos, why = read_array(attempt, "pos", np.asarray(truth_positions).shape)

    if pos is None:
        for name in ("placement_error_px", "worst_tile_error_px"):
            metrics.append(Metric(name, None, limits[name], unavailable=why))
        return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)

    error = placement_error(pos, truth_positions)
    metrics.append(
        Metric(
            "placement_error_px",
            float(np.sqrt((error**2).mean())),
            limits["placement_error_px"],
            unit=" px",
        )
    )
    metrics.append(
        Metric(
            "worst_tile_error_px",
            float(error.max()),
            limits["worst_tile_error_px"],
            unit=" px",
        )
    )
    detail["tiles_within_2px"] = int((error <= 2.0).sum())
    detail["n_tiles"] = int(error.size)
    # The signature of the failure this case is built around: if the run read
    # the serpentine as row-major, the badly placed tiles are the odd rows, and
    # saying so turns a number into a diagnosis.
    detail["worst_tile_index"] = int(np.argmax(error))
    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means, for a human to page through.
    Never raises — an artifact explains a failure, it does not cause one."""
    fixture, attempt = outcome.fixture, outcome.attempt
    canvas = fixture.truth.get("canvas")
    if canvas is not None:
        save_png(np.asarray(canvas, float), where / "canvas-true.png")

    mosaic = attempt.arrays.get("mosaic")
    if mosaic is not None and np.asarray(mosaic).ndim == 2:
        save_png(np.asarray(mosaic, float), where / "mosaic-run.png")

    # The layout itself, as a picture: each tile's footprint drawn where the run
    # put it. A mirrored serpentine is instantly legible here and nowhere else —
    # the mosaic above only shows that *something* is wrong.
    truth_positions = fixture.truth.get("positions")
    pos = attempt.arrays.get("pos")
    if truth_positions is None or pos is None:
        return
    truth_positions = np.asarray(truth_positions, float)
    pos = np.asarray(pos, float)
    if pos.shape != truth_positions.shape:
        return
    H, W = TILE
    for name, layout in (("layout-true", truth_positions), ("layout-run", pos)):
        anchored = layout - layout.min(0)
        sketch = np.zeros(
            (int(anchored[:, 0].max()) + H, int(anchored[:, 1].max()) + W), np.float32
        )
        for index, (y, x) in enumerate(anchored.astype(int)):
            sketch[y : y + H, x : x + W] += 1.0 + index  # index shades the order
        save_png(sketch, where / f"{name}.png")


# --- the conversation ------------------------------------------------------

TASK = (
    "The layer 'tiles' holds every tile from one stage scan, stacked in the "
    "order they were acquired, one channel. Register them into a single "
    "continuous mosaic.\n\n"
    "Leave two names bound in the kernel when you are done:\n"
    "  pos    -- an (N, 2) array of each tile's (row, column) top-left position "
    "in the mosaic, in the same order as the stack\n"
    "  mosaic -- the blended mosaic as a 2-D array"
)

MICROSCOPIST = Persona(
    name="the microscopist who ran the scan",
    facts={
        "how the tiles are arranged": (
            "4 rows of 6, so 24 tiles altogether — it is wider than it is tall"
        ),
        "how the stage travelled": (
            "back and forth — it goes left to right along a row, then drops down "
            "and comes back right to left along the next one, so it never has to "
            "travel all the way back across"
        ),
        "how much the tiles overlap": "about 18%, that is what I set in the software",
        "whether the stage positions were saved": (
            "no, the export only kept the images, not the coordinates"
        ),
        "why it matters": (
            "I want to count cells across the whole region without counting the "
            "ones at the tile boundaries twice"
        ),
    },
    background=(
        "You acquired a tile scan on a motorised stage. You are happy to answer "
        "questions about how the scan was set up and how the sample was mounted."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="path-known-only-to-the-operator",
    task=TASK,
    persona=MICROSCOPIST,
    fixture=Procedural(SnakeGrid()),
    layers=(Layer("tiles", "tiles"),),
    collect={"pos": "pos", "mosaic": "mosaic"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="stitch",
    # It must be able to answer: the fixture withholds the grid shape and the
    # acquisition path, and this person knows both, plus the overlap.
    persona_must_know=("4", "6", "left to right", "18"),
    # And it must not know the method — only the instrument and the scan.
    persona_must_not_know=(
        "phase correlation",
        "spanning tree",
        "cross-correlation",
        "feather",
        "connected component",
        "serpentine",
    ),
)
