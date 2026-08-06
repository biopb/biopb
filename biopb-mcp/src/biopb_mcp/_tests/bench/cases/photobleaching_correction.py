"""A bleaching time series as benchmark data: which cells actually changed?

A deferred-tier case (`docs/skill-candidates.md`). Photobleaching correction was
**prescreened and dropped** 2026-08-03: the claimed non-trivial part — choosing
between ratio, exponential fit and histogram matching according to what is being
measured — did not survive contact, because both cold arms picked an
exponential-plus-offset fit unprompted and the method choice barely moved the
answer. It is here anyway for the reason the entry itself gives: what separated
the two arms was not the method at all.

  *Subtract the camera offset before a multiplicative correction.* One arm
  subtracted a per-frame background median first and scored 1.01 against a truth
  of 1.00; the other rescaled whole frames, offset included, and scored **0.59**
  — a 41 % error in the reported trend, from a stack that still looks correctly
  flattened.

That is one number in a page of prose, and this case is what makes it a
measurement again. The entry is also in the *untested at the low tier* remainder
of the queue, so — unlike `fibre-orientation` or `strahler-ordering` — there is
no low-tier result here to reproduce, only a fixture built to separate the routes
if one is ever run.

Measured on this fixture, mean reported fold change over each population:

  ==========================================  ============  =========
  route                                        responders    static
  ==========================================  ============  =========
  truth                                            1.400       1.000
  per-frame background median, then a fit          1.402       1.001
  the offset alone, then a fit                     1.372       1.001
  the offset alone, then a frame-mean ratio        1.293       0.943
  a frame-mean ratio with the offset left in       0.937       0.753
  no correction at all                             0.668       0.537
  ==========================================  ============  =========

Rows 2-4 are the same three procedures a competent run picks between, and on the
responders they span 0.11. Rows 4 and 5 are *the same procedure* with and without
the pedestal removed, and they are 0.36 apart. The method is not what the numbers
are about, which is exactly what the dropped entry concluded — so `TOLERANCE` is
set to admit all of rows 2-4 and refuse row 5: 0.20 on the responders (1.9x above
the loosest passing route at 0.107, 2.3x below the trap at 0.463) and 0.12 on the
static population (2.1x above 0.057, 2.1x below 0.247).

**Two populations, because one cannot see the failure.** A run that reports 1.00
for every cell has the static cells exactly right and has measured nothing; a run
that reports 1.40 for every cell has the responders right the same way. The pair
is mutually constraining, and no constant passes both.

**The regime is deliberately the recoverable one.** The entry's second finding is
that bleaching and a real global signal change are *not separable from one trace*
— when every object carried the same 40 % rise, no method recovered it, the
reference included. So seven of thirty cells rise here and twenty-three do not,
which is what leaves the decay estimable from the static remainder. A fixture in
the unrecoverable regime would have no right answer to score, which is a finding
about experiments and not a benchmark.

**The rise is flat, then a ramp, then flat.** The reported fold change is
therefore exactly 1.400 for any head and tail window inside the plateaus — the
prompt names five frames, and one frame or eight would give the same truth. That
is deliberate: the quantity being scored should not depend on a convention the
run had to guess.

**The offset is not in the prompt, and the case is still winnable without
asking** — a per-frame background median carries the pedestal and the bleaching
autofluorescence away together, and that route is exact here (1.402 / 1.001). The
person holds the number for a run that thinks to ask about the camera, which is
the right question rather than a shortcut past one.
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

NAMESPACE = "photobleaching-correction"
CASE_ID = "a-minority-that-really-rises"

#: Absolute error on a fold change. From the table in the module docstring: wide
#: enough for every method a competent run picks between, and closed against the
#: one thing that actually separated the screened arms.
TOLERANCE = {"responder_ratio_err": 0.20, "static_ratio_err": 0.12}

N_FRAMES = 30
SHAPE = (256, 256)
N_CELLS = 30
N_RESPONDERS = 7
#: The fold change the responders really undergo, between the two plateaus.
RISE = 1.40
#: Frames averaged at each end. Any window inside a plateau gives the same truth.
WINDOW = 5

#: A fixed pedestal the camera adds to every pixel, dark or not. Not in the task
#: prompt: this is the quantity the whole case is about.
OFFSET = 100.0
CELL_SIGNAL = 500.0
BACKGROUND = 40.0
#: Bleaching: down to 35 % with a 9-frame time constant, so the series loses
#: nearly two thirds of its signal — enough that no run can ignore it.
FLOOR = 0.35
TAU = 9.0
RAMP_START, RAMP_END = 9, 21

SEED = 0


def bleach_curve() -> np.ndarray:
    t = np.arange(N_FRAMES)
    return FLOOR + (1.0 - FLOOR) * np.exp(-t / TAU)


def biology_curve() -> np.ndarray:
    """Flat, a smooth ramp, then flat. See the docstring: the truth is exact."""
    t = np.arange(N_FRAMES)
    ramp = np.clip((t - RAMP_START) / (RAMP_END - RAMP_START), 0.0, 1.0)
    return 1.0 + (RISE - 1.0) * ramp * ramp * (3.0 - 2.0 * ramp)


@dataclass(frozen=True)
class BleachingSeries:
    """One field of cells imaged until it bleaches, seven of them responding."""

    def _cells(self, rng) -> np.ndarray:
        cells = np.zeros(SHAPE, np.int32)
        yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1]]
        rows = cols = 6
        centres = [
            (SHAPE[0] * (i + 0.5) / rows, SHAPE[1] * (j + 0.5) / cols)
            for i in range(rows)
            for j in range(cols)
        ]
        rng.shuffle(centres)
        for label, (cy, cx) in enumerate(centres[:N_CELLS], start=1):
            cy += rng.uniform(-4, 4)
            cx += rng.uniform(-4, 4)
            radius = rng.uniform(9, 13)
            cells[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = label
        return cells

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(SEED)
        cells = self._cells(rng)

        responders = np.zeros(N_CELLS + 1, bool)
        responders[
            rng.choice(np.arange(1, N_CELLS + 1), N_RESPONDERS, replace=False)
        ] = True
        # Per-cell brightness, so the field is not thirty identical disks and a
        # run cannot read the populations off the intensities.
        brightness = np.concatenate([[1.0], rng.uniform(0.7, 1.3, N_CELLS)])

        bleach = bleach_curve()
        rise = biology_curve()
        movie = np.empty((N_FRAMES, *SHAPE), np.float32)
        for frame in range(N_FRAMES):
            clean = np.full(SHAPE, BACKGROUND * bleach[frame], np.float32)
            for label in range(1, N_CELLS + 1):
                factor = rise[frame] if responders[label] else 1.0
                clean[cells == label] += (
                    CELL_SIGNAL * brightness[label] * bleach[frame] * factor
                )
            photons = np.maximum(clean, 0.0)
            movie[frame] = (
                OFFSET + photons + rng.normal(0, np.sqrt(photons + 4.0))
            ).astype(np.float32)

        truth = {
            "responders": responders[1:].copy(),
            "responder_ratio": float(RISE),
            "static_ratio": 1.0,
            "bleach": bleach,
        }
        _the_traps_are_armed(movie, cells, responders)

        return Fixture(
            provenance=(
                f"procedural: seed {SEED}, {N_FRAMES} frames of {SHAPE}, "
                f"{N_CELLS} cells of which {N_RESPONDERS} rise {RISE:g}x between "
                f"frames {RAMP_START} and {RAMP_END}, bleaching to {FLOOR:g} with "
                f"tau {TAU:g} frames over a {OFFSET:g}-count camera pedestal"
            ),
            about=(
                f"A field of {N_CELLS} cells imaged until it has lost "
                f"{100 * (1 - bleach[-1]):.0f} % of its signal. "
                f"{N_RESPONDERS} of them genuinely brighten {RISE:g}-fold; the "
                "rest do not change. Correcting the decay after removing the "
                "camera pedestal recovers both populations; correcting it with "
                "the pedestal still in reports 0.94 and 0.75."
            ),
            data={"movie": movie, "cells": cells},
            truth=truth,
            tolerance=dict(TOLERANCE),
        )


def _cell_traces(movie: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``(frames, cells)`` mean intensity inside each label."""
    index = np.arange(1, int(cells.max()) + 1)
    return np.stack([ndi.mean(frame, cells, index=index) for frame in movie])


def _fold_change(traces: np.ndarray) -> np.ndarray:
    return traces[-WINDOW:].mean(0) / traces[:WINDOW].mean(0)


def _the_traps_are_armed(movie, cells, responders) -> None:
    """The three properties the case rests on, before anyone pays for a run.

    All three are cheap — one pass over the traces, no fitting — and none of them
    is visible from the movie itself. A fixture whose trap has quietly stopped
    biting reads as a clean run, which is the failure this exists to prevent.
    """
    mask = responders[1:]

    # 1. The trap: a frame-mean rescale with the pedestal still in is wrong by
    #    more than the tolerance allows, on both populations.
    scale = movie.reshape(N_FRAMES, -1).mean(1)
    trapped = _fold_change(
        _cell_traces(movie / (scale / scale[0])[:, None, None], cells)
    )
    if abs(trapped[mask].mean() - RISE) <= TOLERANCE["responder_ratio_err"]:
        raise AssertionError(
            f"the offset trap reports {trapped[mask].mean():.3f} on the "
            f"responders, inside the {TOLERANCE['responder_ratio_err']} limit — "
            "the case would pass a run that never removed the pedestal"
        )
    if abs(trapped[~mask].mean() - 1.0) <= TOLERANCE["static_ratio_err"]:
        raise AssertionError(
            f"the offset trap reports {trapped[~mask].mean():.3f} on the static "
            f"cells, inside the {TOLERANCE['static_ratio_err']} limit"
        )

    # 2. Winnable: removing the per-frame background and dividing by the static
    #    population's own decay recovers both numbers.
    background = np.array([np.median(frame[cells == 0]) for frame in movie])
    clean = _cell_traces(movie - background[:, None, None], cells)
    decay = clean[:, ~mask].mean(1)
    recovered = _fold_change(clean / (decay / decay[0])[:, None])
    if (
        abs(recovered[mask].mean() - RISE) > 0.05
        or abs(recovered[~mask].mean() - 1) > 0.05
    ):
        raise AssertionError(
            f"the reference route recovers {recovered[mask].mean():.3f} and "
            f"{recovered[~mask].mean():.3f} against {RISE} and 1.0 — the case is "
            "not winnable as built"
        )

    # 3. The regime is the recoverable one: a minority responds.
    if mask.sum() * 2 >= mask.size:
        raise AssertionError(
            f"{mask.sum()} of {mask.size} cells respond, which is not a minority "
            "— with no static remainder the decay is not estimable and the case "
            "has no right answer"
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score the two populations' mean reported fold change.

    Both metrics come off the same array, and neither is redundant: a constant
    answer satisfies exactly one of them, whichever constant it is.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    responders = np.asarray(fixture.truth["responders"], bool)
    names = ("responder_ratio_err", "static_ratio_err")

    got = attempt.arrays.get("ratio_per_cell")
    if got is None:
        why = "the run left no `ratio_per_cell`"
    else:
        got = np.asarray(got, float).reshape(-1)
        if got.size != responders.size:
            why = f"the run's `ratio_per_cell` has {got.size} entries, not {responders.size}"
        elif not np.isfinite(got).all():
            why = "the run's `ratio_per_cell` is not finite everywhere"
        else:
            why = ""
    if why:
        return Outcome(
            fixture=fixture,
            attempt=attempt,
            metrics=[Metric(n, None, limits[n], unavailable=why) for n in names],
        )

    reported = {
        "responder_ratio_err": (
            float(got[responders].mean()),
            float(fixture.truth["responder_ratio"]),
        ),
        "static_ratio_err": (
            float(got[~responders].mean()),
            float(fixture.truth["static_ratio"]),
        ),
    }
    metrics = [
        Metric(name, abs(mean - want), limits[name])
        for name, (mean, want) in reported.items()
    ]
    detail: dict[str, object] = {
        "responder_mean": round(reported["responder_ratio_err"][0], 4),
        "static_mean": round(reported["static_ratio_err"][0], 4),
        "responder_spread": round(float(got[responders].std()), 4),
        "static_spread": round(float(got[~responders].std()), 4),
        "n_responders": int(responders.sum()),
    }
    # The one line that names the mistake when the metrics can only say "0.4
    # out": a run that reported one number for every cell measured nothing.
    detail["reads_as_one_number_for_every_cell"] = bool(got.std() < 0.02)
    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """First and last frame on one scale, and the answer beside the truth."""
    fixture, attempt = outcome.fixture, outcome.attempt
    movie = np.asarray(fixture.data["movie"], float)
    # A shared vmax, or the last frame is stretched back to looking like the
    # first and the artifact hides the very thing it was saved to show.
    vmax = float(movie[0].max())
    save_png(movie[0], where / "frame-first.png", vmax=vmax)
    save_png(movie[-1], where / "frame-last.png", vmax=vmax)
    save_png(np.asarray(fixture.data["cells"]) > 0, where / "cells.png")

    responders = np.asarray(fixture.truth["responders"], bool)
    got = attempt.arrays.get("ratio_per_cell")
    got = None if got is None else np.asarray(got, float).reshape(-1)
    rows = ["label,population,truth,reported"]
    for index, is_responder in enumerate(responders):
        want = RISE if is_responder else 1.0
        value = (
            "" if got is None or got.size != responders.size else f"{got[index]:.4f}"
        )
        rows.append(
            f"{index + 1},{'responder' if is_responder else 'static'},{want},{value}"
        )
    (where / "fold-change.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = f"""\
A time series is open in the napari viewer:

  movie  float32 {(N_FRAMES, *SHAPE)}   ({N_FRAMES} frames, y, x)
  cells  labels  {SHAPE}                {N_CELLS} cells, label i is cell i

There is no tensor server in this session, so those layers are the only source
of pixels. The cells do not move, so one segmentation covers the whole series.

The field bleaches badly over these {N_FRAMES} frames — by the end it is much
dimmer than it started. Some of these cells are also responding to a drug I added
before the first frame, and that is what I am trying to see through the bleaching.

For each cell I need its fold change over the series: the mean intensity of its
last {WINDOW} frames divided by the mean intensity of its first {WINDOW}, as it
would read if the sample had not bleached. A cell that did not respond should
come out at about 1.

The person who ran the experiment is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will come
back as the next message.

When you are finished, leave one name bound in the kernel namespace:

  ratio_per_cell   a float array of {N_CELLS} entries, in label order — entry i
                   is the fold change of the cell labelled i + 1

That name is how your result is collected, so bind it exactly.
"""

#: This person knows their instrument. The camera pedestal is a fact about the
#: hardware, and they will give the number to a run that asks about the camera —
#: which is the right question, not a shortcut past one. What they cannot tell
#: you is what to do about it: nothing here is a procedure.
EXPERIMENTER = Persona(
    name="operator-bleaching-series",
    facts={
        "what the experiment is": (
            "a live field of cells carrying a fluorescent reporter, imaged every "
            "ten seconds after I added the drug. I want to know which cells "
            "responded and by how much"
        ),
        "about the camera": (
            "an sCMOS. It puts a fixed pedestal of about 100 counts on every "
            "pixel whether or not any light arrives — that is just how the chip "
            "reads out, and it is in the spec sheet"
        ),
        "why the field gets dimmer": (
            "the dye does not survive this much illumination. I turned the laser "
            "down as far as I could and it still fades badly by the end"
        ),
        "whether the cells move": (
            "no, they are well attached and they stayed put — that segmentation "
            "is good for the whole series"
        ),
        "how many cells respond": (
            "I do not know, that is what I am asking you. From what I can see by "
            "eye it is a handful of them, not the whole field"
        ),
        "what the background is": (
            "there is some autofluorescence in the medium. It fades along with "
            "everything else"
        ),
        "whether anything changed during the acquisition": (
            "nothing. Same laser, same exposure, same focus, and I did not touch "
            "the stage once it started"
        ),
    },
    background=(
        "You imaged one field of live cells after adding a drug, on a confocal "
        "with an sCMOS camera. You are happy to answer questions about the "
        "sample, the instrument and how the series was acquired."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=EXPERIMENTER,
    fixture=Procedural(BleachingSeries()),
    layers=(
        Layer("movie", "movie"),
        Layer("cells", "cells", kind="labels"),
    ),
    collect={"ratio_per_cell": "ratio_per_cell"},
    score=verify,
    save_artifacts=save_artifacts,
    # It must be able to answer what the camera does, for a run that asks.
    persona_must_know=("100 counts", "pedestal"),
    # And it must not know what to do about it, or which cells responded.
    persona_must_not_know=(
        "exponential",
        "curve_fit",
        "subtract",
        "rescale",
        "fold change",
        "histogram matching",
    ),
)
