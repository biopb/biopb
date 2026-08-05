"""`detect-filaments` as benchmark data: did the detection find the dim ones?

The field holds twelve filaments whose peak amplitudes span a **decade**, log-
spaced so the range is populated rather than bimodal, with the dimmest at a few
times the noise. That span is the whole construction. It is still winnable —
every filament is above the noise for a scale-matched ridge filter — but it is
well past where a single global cut on the ridge response puts its threshold:
measured over three seeds, one Otsu cut finds 6-7 of the 12 **at precision
0.99-1.00**, so nothing in a run's own output says the other half is missing.

The withheld fact is the **pixel size in µm**. It is the §5d kind: a scale is
categorically absent from an array of numbers, so unlike "which layer is the
truth" there is no back door in the pixels. It is load-bearing exactly once and
completely — every width reported is a pixel count times this number.

What is *not* withheld, and is visible in the data:

  * that the structures are filaments, and roughly how wide they are in pixels
  * that the field is dim in places, if anyone stretches the contrast

`missed_filaments` is scored **per filament**, not pooled. A pooled recall of
0.58 and "five filaments are entirely absent" are different findings and this
case exists to tell them apart: the first reads as a slightly conservative
threshold, the second as half the answer missing.

`false_ridge_fraction` is the other side of the same knob and is why it is
here — the failure of thresholding too low is not a missing filament but a
field of spurious ones, and a case that only scored recall would rate
"threshold at nothing" a perfect run.

The reference implementation these tolerances come from is in the pull request
that added this case, per `biopb-mcp/docs/skills.md` §11b.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ....agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_scalar,
)
from ....agentbench._respondent import Persona
from .._benchmark import Case, Layer

SKILL = "detect-filaments"

#: The withheld number. Every width the run reports is a pixel count times this.
PX_UM = 0.1

#: How far a detected centreline pixel may sit from a true one, and vice versa,
#: and still count. Two pixels: a skeleton wanders by about that much inside a
#: mask whose edge the threshold placed, and scoring one-pixel skeletons for
#: exact coincidence would measure the skeletoniser rather than the detection.
TOL_PX = 2.0

#: A filament counts as *found* when this much of its centreline was recovered.
#: Half, deliberately far from both ends: a filament the run traced for a
#: quarter of its length is not found, and requiring 0.9 would charge the run
#: for the ends, where every ridge filter fades out.
FOUND_AT = 0.5

#: Set from the reference implementation (multiscale `sato`, hysteresis at
#: 0.25x the Otsu seed, 8-connected pruning, width from the transverse profile)
#: over **seven** seeds of this construction, against the failures it has to be
#: separated from, all measured on the same fields:
#:
#:                                            missed        false   width_error
#:   reference ............................ 0.000-0.167  0.005-0.039  0.006-0.157
#:   one global Otsu cut on the response .. 0.417-0.500  0.002-0.009  0.059-0.233
#:   4-connected skeleton pruning ......... 0.917-1.000  0.000-0.009  0.197-0.287
#:   low cut at 0.05x instead of 0.25x .... 0.000        0.633-0.770
#:   width read off the mask's EDT ........ (as reference)             0.128-0.349
#:   pixel size guessed at 1.0 not 0.1 .... (as reference)             7.43-8.33
#:
#: `missed_filaments` is the measurement, and 0.25 -- three of twelve -- sits in
#: the gap between the worst reference seed (0.167) and the best single-cut one
#: (0.417) on every seed tried. `false_ridge_fraction` is not symmetric with it
#: by accident: it exists to stop a run buying recall by thresholding at
#: nothing, which scores 0.63-0.77.
#:
#: **`width_error` is the withheld-scale test and nothing more.** A run that
#: never obtained the pixel size is out by the ratio (7.4-8.3), which no limit
#: in this range fails to catch. It is *not* a test of the width estimator:
#: reading the width off the mask -- the thing the skill's step 7 exists to
#: forbid -- scores 0.13-0.35 across seeds and would pass on three of the seven.
#: That is the skill's own claim about pooled widths coming back around: a
#: single mean width is a weak discriminator, which is why the body argues the
#: estimator per filament and this case does not try to. 0.25 also leaves room
#: for an honest disagreement about what "width" means, though the task prompt
#: asks for an FWHM specifically -- FWHM against Gaussian 2-sigma is 0.18 by
#: definition, and would otherwise dominate everything else here.
TOLERANCE = {
    "missed_filaments": 0.25,
    "false_ridge_fraction": 0.25,
    "width_error": 0.25,
}


# --- the field -------------------------------------------------------------


def _filament(shape, y0, x0, ang, curv, sigma_px, amp, half_len=200.0):
    """A gently curved Gaussian-profile ridge; returns (image, centreline).

    Normalised by `sigma_px` so the ridge *peak* is `amp` whatever its width --
    otherwise wide filaments would be bright by construction and the brightness
    ladder would be a width ladder wearing its clothes.
    """
    line = np.zeros(shape, np.float32)
    centre = np.zeros(shape, bool)
    for t in np.linspace(-half_len, half_len, int(20 * half_len)):
        v = curv * t**2
        y = y0 + t * np.sin(ang) + v * np.cos(ang)
        x = x0 + t * np.cos(ang) - v * np.sin(ang)
        yi, xi = int(round(y)), int(round(x))
        if 2 <= yi < shape[0] - 2 and 2 <= xi < shape[1] - 2:
            line[yi, xi] = 1.0
            centre[yi, xi] = True
    prof = ndi.gaussian_filter(line, sigma_px) * (sigma_px * np.sqrt(2 * np.pi))
    return prof * amp, centre


@dataclass(frozen=True)
class FilamentField:
    """Twelve curved filaments over a decade of brightness."""

    shape: tuple[int, int] = (512, 512)
    n: int = 12
    #: Full width at 2-sigma, µm. The lower end is 3 px at `PX_UM` -- resolvable
    #: by a matched scale and not by an unmatched one, which is what makes the
    #: `sigmas` range in the skill's step 3 a real choice.
    width_um: tuple[float, float] = (0.30, 0.90)
    brightest: float = 0.60
    dynamic_range: float = 10.0
    background: float = 0.08
    noise_sd: float = 0.02
    seed: int = 7

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        img = np.zeros(self.shape, np.float32)
        centres, amps, widths = [], [], []

        dim = self.brightest / self.dynamic_range
        # log-spaced, so the range is populated rather than two clusters
        ladder = np.geomspace(dim, self.brightest, self.n)

        placed, guard = 0, 0
        while placed < self.n and guard < 4000:
            guard += 1
            w_um = float(rng.uniform(*self.width_um))
            y0, x0 = rng.uniform(60, self.shape[0] - 60, 2)
            ang = float(rng.uniform(0, np.pi))
            curv = float(rng.uniform(-4e-4, 4e-4))
            prof, centre = _filament(
                self.shape,
                y0,
                x0,
                ang,
                curv,
                (w_um / 2.0) / PX_UM,
                float(ladder[placed]),
            )
            if centre.sum() < 200:  # ran off the field
                continue
            img += prof
            centres.append(centre)
            amps.append(float(ladder[placed]))
            widths.append(w_um)
            placed += 1
        assert placed == self.n, f"filament packing failed: {placed}"

        img = (
            img
            + self.background
            + rng.normal(0, self.noise_sd, self.shape).astype(np.float32)
        )

        order = np.argsort(amps)  # dimmest first, so a per-filament row reads
        per = np.stack([centres[i] for i in order])
        # The reported width is an FWHM, and the field is drawn to 2-sigma.
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) / 2.0  # = 1.1775, 2-sigma -> FWHM
        assert min(amps) / self.noise_sd >= 2.5, "the dimmest filament is unwinnable"

        return Fixture(
            provenance=(
                f"procedural: seed {self.seed}, {self.shape[0]}x{self.shape[1]} "
                f"field, {self.n} curved filaments {self.width_um[0]}-"
                f"{self.width_um[1]} µm wide at {PX_UM} µm/px, peak amplitudes "
                f"log-spaced over {self.dynamic_range:g}x "
                f"(SNR {min(amps) / self.noise_sd:.1f}-"
                f"{max(amps) / self.noise_sd:.1f})"
            ),
            about=(
                "Filament brightness spans a decade, so a single global cut on "
                "a ridge response keeps the bright ones at near-perfect "
                "precision and silently drops the rest. The pixel size is not "
                "in the data: without it a width in microns cannot be right."
            ),
            data={"filaments": img.astype(np.float32)},
            truth={
                "per_filament": per,
                "centre": np.any(per, axis=0),
                "widths_um": np.asarray(widths, float),
                "mean_fwhm_um": float(np.mean(widths) * fwhm),
                "px_um": PX_UM,
            },
            tolerance=dict(TOLERANCE),
        )


# --- truth-side arithmetic, shared by the verifier and the artifacts --------


def _distances(got: np.ndarray, truth: np.ndarray):
    """``(nearest truth per pixel, nearest detection per pixel)``, in pixels."""
    d_truth = ndi.distance_transform_edt(~truth)
    d_got = (
        ndi.distance_transform_edt(~got)
        if got.any()
        else np.full(got.shape, np.inf, float)
    )
    return d_truth, d_got


def _centrelines(attempt: Attempt, shape) -> tuple[np.ndarray | None, str]:
    got = attempt.arrays.get("centrelines")
    if got is None:
        return None, "the run left no `centrelines`"
    got = np.asarray(got)
    if got.shape != tuple(shape):
        return None, f"the run's `centrelines` is {got.shape}, not {tuple(shape)}"
    got = got.astype(bool)
    if not got.any():
        return None, "the run's `centrelines` is empty -- nothing to score"
    # A filled mask is not a centreline, and scoring one as if it were would
    # reward the run that skipped step 6 with a recall of 1.
    if got.mean() > 0.10:
        return None, (
            f"the run's `centrelines` covers {got.mean():.0%} of the field -- "
            "that is a mask, not a set of centrelines"
        )
    return got, ""


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    `missed_filaments` is the one this case exists for, and it is deliberately
    not recall: it counts filaments of which less than `FOUND_AT` was recovered,
    which is the number a pooled score hides.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = fixture.truth
    per = np.asarray(truth["per_filament"], bool)
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    got, why = _centrelines(attempt, per.shape[1:])
    if got is None:
        for name in ("missed_filaments", "false_ridge_fraction"):
            metrics.append(Metric(name, None, limits[name], unavailable=why))
    else:
        d_truth, d_got = _distances(got, np.asarray(truth["centre"], bool))
        recalls = [float((d_got[c] <= TOL_PX).mean()) for c in per]
        missed = [i for i, r in enumerate(recalls) if r < FOUND_AT]
        metrics.append(
            Metric(
                "missed_filaments",
                len(missed) / len(recalls),
                limits["missed_filaments"],
                unit=f" of {len(recalls)} filaments recovered under "
                f"{FOUND_AT:.0%} of their length",
            )
        )
        metrics.append(
            Metric(
                "false_ridge_fraction",
                float((d_truth[got] > TOL_PX).mean()),
                limits["false_ridge_fraction"],
                unit=" of detected centreline is not on a filament",
            )
        )
        detail |= {
            "per_filament_recall_dimmest_first": [round(r, 3) for r in recalls],
            "filaments_found": len(recalls) - len(missed),
            "filaments_total": len(recalls),
            "centreline_px": int(got.sum()),
        }

    said, why = read_scalar(attempt, "mean_width_um")
    want = float(truth["mean_fwhm_um"])
    if said is None:
        metrics.append(
            Metric("width_error", None, limits["width_error"], unavailable=why)
        )
    else:
        metrics.append(
            Metric("width_error", abs(said - want) / want, limits["width_error"])
        )
        detail |= {"mean_width_um": float(said), "mean_width_um_truth": want}

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """Per-filament recall against that filament's brightness and width, so a
    human can see whether what was missed was the dim end. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    truth = fixture.truth
    per = np.asarray(truth["per_filament"], bool)
    got, _ = _centrelines(attempt, per.shape[1:])
    if got is None:
        return
    _, d_got = _distances(got, np.asarray(truth["centre"], bool))
    lines = ["rank_dimmest_first,width_um,centreline_px,recall"]
    for i, c in enumerate(per):
        lines.append(
            f"{i},{float(np.asarray(truth['widths_um'])[i]):.3f},"
            f"{int(c.sum())},{float((d_got[c] <= TOL_PX).mean()):.3f}"
        )
    (where / "per_filament_recall.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# --- the conversation ------------------------------------------------------

TASK = """\
A 2D fluorescence image is open in the napari viewer as an Image layer called
`filaments`. It is float32 with axes (Y, X) = (512, 512).

There is no tensor server in this session, so that layer is the only source of
pixels.

It shows filamentous structures. I want their centrelines traced, and I want to
know how wide they are.

The person who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  centrelines     a boolean array with the same (512, 512) shape as
                  `filaments`, True on the one-pixel centreline of every
                  filament you detected
  mean_width_um   a float. The mean full width at half maximum of the
                  filaments, across them, in microns.

Those two names are how your result is collected, so bind them exactly.
"""

#: What is *not* here: nothing about ridge filters, scales, thresholds or
#: skeletons, and no hint that the field is dim in places. This person knows
#: their sample and their microscope.
CYTOSKELETON = Persona(
    name="cell-biologist-filaments",
    facts={
        "what the pixel size is": (
            "0.1 microns per pixel -- it is a 100x objective with the standard "
            "camera, I checked it against a stage micrometer"
        ),
        "what the filaments are": (
            "labelled cytoskeletal bundles in a fixed cell, stained with a "
            "phalloidin conjugate"
        ),
        "how wide the filaments should be": (
            "somewhere between about 0.3 and 0.9 microns -- they are bundles, "
            "not single fibres, and the thin ones are near the resolution limit"
        ),
        "why some are dim": (
            "the staining is uneven across the cell and the thinner bundles "
            "take up less dye, so brightness varies a lot between them. The "
            "illumination is flat, I checked that separately"
        ),
        "what the result is for": (
            "I want to compare bundle thickness between treated and untreated "
            "cells, so the widths need to be in real units and I need all the "
            "bundles, not just the obvious ones"
        ),
        "whether the image was processed": (
            "no, it is straight off the camera apart from the usual background offset"
        ),
    },
    background=(
        "A 2D fluorescence image of the cytoskeleton in a fixed cell. You are "
        "happy to answer questions about the sample, the stain and the "
        "microscope."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="a-decade-of-brightness-no-scale-in-the-pixels",
    task=TASK,
    persona=CYTOSKELETON,
    fixture=Procedural(FilamentField()),
    layers=(Layer("filaments", "filaments", "image"),),
    collect={"centrelines": "centrelines", "mean_width_um": "mean_width_um"},
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="filament",
    # It must be able to answer: the fixture strips the pixel size, and this
    # person knows it, along with what the filaments are and how wide.
    persona_must_know=("0.1 microns per pixel", "0.3 and 0.9", "phalloidin"),
    # And it must not know the procedure.
    persona_must_not_know=(
        "ridge",
        "sato",
        "frangi",
        "hessian",
        "hysteresis",
        "otsu",
        # not "skeleton" -- it is a substring of "cytoskeleton", which is what
        # this person's sample actually is
        "skeletonize",
        "skeletonise",
        "threshold",
    ),
)
