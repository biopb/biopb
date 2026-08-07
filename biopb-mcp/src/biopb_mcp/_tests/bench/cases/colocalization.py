"""Colocalization as benchmark data: how much of one channel sits on the other?

A deferred-tier case (`docs/skill-candidates.md`). Colocalization coefficients
were **prescreened and dropped** 2026-08-03 for the shipped catalog — both cold
Sonnet arms produced Costes thresholding, Manders above threshold and a
randomization test unprompted, and agreed with a reference to three significant
figures. It is here anyway, because the *work* is real whether or not a skill
for it is served, and because a rejection is conditional on the consuming tier.

**What that screen could not build, and this fixture is.** The entry records a
second finding: *no scene separated the correct procedure from the naive one*.
The intended trap — independent puncta that should read as uncolocalized — held
genuinely correlated shared autofluorescence, so a high correlation was the
right answer there and reference and arm both reported it. Being unable to
construct the trap was itself the finding, and it is why nothing here rests on
correlation between the channels and why the two backgrounds are independent
realisations rather than one shared field.

Measured on this fixture:

  ====================================  ========  ========  ========  ========
  route                                  M1 sp.1   M2 sp.1   M1 sp.2   M2 sp.2
  ====================================  ========  ========  ========  ========
  truth                                    0.711     0.371     0.194     0.780
  reference: background, then 5 sigma      0.710     0.373     0.194     0.781
  presence ten times stricter              0.687     0.362     0.188     0.764
  Otsu instead of a background estimate    0.756     0.357     0.197     0.604
  overlapping *area*, not signal           0.482     0.286     0.144     0.656
  no presence threshold at all             1.000     1.000     1.000     1.000
  raw frames, whole-field denominator      0.341     0.240     0.088     0.159
  ====================================  ========  ========  ========  ========

`TOLERANCE` sits in that gap: the reference's worst coefficient is 0.002 out and
a threshold ten times stricter is 0.024 out, while the nearest wrong route is
0.176 out. 0.08 is three times the spread between defensible thresholds and less
than half the distance to the nearest route that got the quantity wrong.

**Manders' coefficient is threshold-dependent by nature, so the fixture is what
takes the convention out of the score** (protocol §8). Two constructions do it,
and both are asserted rather than hoped for:

* the puncta are **flat-topped disks**, hundreds of noise widths above
  background. A Gaussian spot shrinks as the threshold rises, and it shrinks
  faster when it is dim — so on Gaussians the coefficient walks with the
  threshold and the score measures a choice rather than an answer. On disks the
  truth moves by less than 0.02 between a presence cut of 20 counts and one of
  300, and the builder refuses to hand out a fixture where it moves by more.
* every red partner is **bright**. When half the partners were dim, a stricter
  threshold ate them and a defensible route drifted 0.23; with bright partners
  the same route drifts 0.024.

**Brightness and pairing are correlated, and that is the triviality screen**
(protocol §11). Ninety-five of the hundred bright green puncta have a red
partner and five of the hundred dim ones do, so the *intensity-weighted*
fraction and the *count* of overlapping objects are different numbers — 0.711
against an area overlap of 0.482. Without that, counting overlapping area would
answer the question and the coefficient's definition would be decoration.

**The back door screen** (protocol §11): the two specimens are built so that no
constant can pass. M1 runs 0.711 against 0.194 and M2 runs 0.371 against 0.780,
so the best single constant fitted to the answer key is still 0.205 out — well
above the tolerance. The reversal is also what stops one number reported twice:
M1 is the larger coefficient on specimen 1 and the smaller on specimen 2.

**Otsu is a wrong route here, at 0.176**, and it is the near miss worth naming:
specimen 2's red channel occupies 0.7% of the field, and a two-class histogram
split of an image that is 99% background is not a background estimate. It costs
a fifth of that specimen's M2.

**What is not withheld, deliberately.** The definition of the two coefficients
is in the task text, because a coefficient quoted without its definition cannot
be compared to anything and the gap being measured is not whether the run can
guess a convention. The persona is here for realism and holds no part of the
answer.
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

NAMESPACE = "colocalization"
CASE_ID = "two-specimens-manders-above-background"

#: From the table in the module docstring, not from taste. One limit for all
#: four coefficients: they are the same quantity measured four times, and the
#: routes that miss do not miss differently on one of them.
TOLERANCE = {
    "m1_error_1": 0.08,
    "m2_error_1": 0.08,
    "m1_error_2": 0.08,
    "m2_error_2": 0.08,
}

SHAPE = (640, 640)
BLUR = 0.7  #: only enough to take the staircase off a disk edge
RADIUS = 3.0  #: puncta are disks, so geometry does not move with the threshold
JITTER = 1.0  #: how far a red partner sits from its green one, in pixels
BRIGHT, DIM = 9000.0, 3500.0  #: counts on the plateau
SPREAD = 0.12  #: lognormal sigma of the plateau within a class
OFFSET = 100.0  #: camera pedestal
HAZE = 25.0  #: autofluorescence, one realisation per channel and never shared
READ_NOISE = 3.0

#: Where the truth's presence mask is cut, in counts above background — about
#: five noise widths. The answer is invariant across this choice by
#: construction, and the builder asserts it.
PRESENCE = 55.0

#: ``specimen -> (seed, bright green, dim green, paired bright, paired dim,
#: unpaired red)``. Specimen 2 reverses which coefficient is the larger one.
SPECIMENS: dict[str, tuple[int, int, int, int, int, int]] = {
    "1": (11, 100, 100, 95, 5, 250),
    "2": (23, 100, 100, 25, 5, 10),
}


def manders(a: np.ndarray, b: np.ndarray, t_a: float, t_b: float) -> float:
    """The fraction of *a*'s signal that lies where *b* also has signal.

    Both sums are over background-subtracted intensity, and both are restricted
    to where *a* itself is present — which is the definition the task states and
    the one the truth is computed with.
    """
    signal = np.clip(a, 0, None)
    here, there = a > t_a, b > t_b
    total = signal[here].sum()
    return float(signal[here & there].sum() / total) if total > 0 else float("nan")


@dataclass(frozen=True)
class TwoSpecimens:
    """Two fields of punctate green and red, paired at two different rates."""

    shape: tuple[int, int] = SHAPE

    def _spots(self, rng, n: int) -> np.ndarray:
        return np.stack(
            [rng.uniform(0, self.shape[0], n), rng.uniform(0, self.shape[1], n)], 1
        )

    def _render(self, centres: np.ndarray, plateaus: np.ndarray) -> np.ndarray:
        field = np.zeros(self.shape, np.float64)
        span = int(np.ceil(RADIUS)) + 2
        for (cy, cx), plateau in zip(centres, plateaus, strict=True):
            y0, y1 = max(int(cy) - span, 0), min(int(cy) + span + 1, self.shape[0])
            x0, x1 = max(int(cx) - span, 0), min(int(cx) + span + 1, self.shape[1])
            if y0 >= y1 or x0 >= x1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            field[y0:y1, x0:x1] += plateau * (
                ((yy - cy) ** 2 + (xx - cx) ** 2) <= RADIUS**2
            )
        return ndi.gaussian_filter(field, BLUR)

    def _specimen(self, spec: tuple[int, int, int, int, int, int]):
        seed, n_bright, n_dim, pair_bright, pair_dim, n_unpaired = spec
        rng = np.random.default_rng(seed)
        bright, dim = self._spots(rng, n_bright), self._spots(rng, n_dim)
        green = self._render(
            np.concatenate([bright, dim]),
            rng.lognormal(
                np.log(
                    np.concatenate([np.full(n_bright, BRIGHT), np.full(n_dim, DIM)])
                ),
                SPREAD,
            ),
        )

        partners = np.concatenate([bright[:pair_bright], dim[:pair_dim]])
        partners = partners + rng.uniform(-JITTER, JITTER, partners.shape)
        red = self._render(
            np.concatenate([partners, self._spots(rng, n_unpaired)]),
            rng.lognormal(
                np.log(
                    np.concatenate(
                        [
                            np.full(len(partners), BRIGHT),
                            rng.choice([BRIGHT, DIM], n_unpaired),
                        ]
                    )
                ),
                SPREAD,
            ),
        )

        frames = {}
        for name, clean in (("green", green), ("red", red)):
            haze = ndi.gaussian_filter(rng.normal(0, 1, self.shape), 40.0)
            haze = HAZE * (haze - haze.min()) / (haze.max() - haze.min())
            frame = OFFSET + haze + clean
            frame = frame + rng.normal(0, np.sqrt(np.maximum(frame, 1.0)))
            frame = frame + rng.normal(0, READ_NOISE, self.shape)
            frames[name] = frame.astype(np.float32)
        return frames, {"green": green, "red": red}

    def __call__(self) -> Fixture:
        data: dict[str, np.ndarray] = {}
        truth: dict[str, object] = {}
        clean: dict[str, dict[str, np.ndarray]] = {}
        for name, spec in SPECIMENS.items():
            frames, signal = self._specimen(spec)
            data[f"specimen_{name}_green"] = frames["green"]
            data[f"specimen_{name}_red"] = frames["red"]
            clean[name] = signal
            truth[f"m1_{name}"] = manders(
                signal["green"], signal["red"], PRESENCE, PRESENCE
            )
            truth[f"m2_{name}"] = manders(
                signal["red"], signal["green"], PRESENCE, PRESENCE
            )
        self._check(clean, truth)

        return Fixture(
            provenance=(
                f"procedural: two {self.shape[0]}x{self.shape[1]} two-colour fields "
                f"of disk puncta radius {RADIUS:g} px on plateaus {DIM:g}/{BRIGHT:g} "
                f"counts over a {OFFSET:g}-count pedestal, seeds "
                f"{[spec[0] for spec in SPECIMENS.values()]}"
            ),
            about=(
                "Two two-colour fields whose overlap fractions are known exactly. "
                f"Specimen 1 runs M1 {truth['m1_1']:.3f} against M2 "
                f"{truth['m2_1']:.3f} and specimen 2 reverses it, "
                f"{truth['m1_2']:.3f} against {truth['m2_2']:.3f}. The bright green "
                "puncta are nearly all paired and the dim ones nearly none, so "
                "overlapping *area* is a different number (0.482 where M1 is "
                "0.711) and the intensity weighting in the definition is doing "
                "work. Counting every pixel as present gives 1.000 everywhere; "
                "leaving the background in the denominator gives 0.341."
            ),
            data=data,
            truth=truth,
            tolerance=dict(TOLERANCE),
        )

    def _check(self, clean, truth) -> None:
        """The properties the case rests on, before anyone pays for a run. None
        of them is visible from the frames alone, and each is one of the
        shortcut screens the protocol asks for in executable form."""
        limit = min(TOLERANCE.values())

        for name, signal in clean.items():
            # protocol 8: the answer must not move with a defensible threshold
            band = [
                (
                    manders(signal["green"], signal["red"], cut, cut),
                    manders(signal["red"], signal["green"], cut, cut),
                )
                for cut in (20.0, 55.0, 110.0, 300.0)
            ]
            drift = max(
                max(abs(m1 - band[0][0]) for m1, _ in band),
                max(abs(m2 - band[0][1]) for _, m2 in band),
            )
            assert drift < 0.5 * limit, (
                f"specimen {name}: the coefficients move {drift:.3f} between a "
                "presence cut of 20 counts and one of 300, so the score is "
                "about where the threshold went, not about the overlap"
            )

            # protocol 11, triviality: is the intensity weighting load-bearing?
            green, red = signal["green"] > PRESENCE, signal["red"] > PRESENCE
            area = float((green & red).sum() / green.sum())
            if name == "1":
                assert abs(area - truth[f"m1_{name}"]) > 2 * limit, (
                    f"specimen {name}: overlapping area is {area:.3f} against an "
                    f"M1 of {truth[f'm1_{name}']:.3f}, so counting area answers "
                    "the question and the definition is decoration"
                )

        # protocol 11, back door: no constant fitted to the answer key may pass
        for coefficient in ("m1", "m2"):
            values = [float(truth[f"{coefficient}_{n}"]) for n in SPECIMENS]
            best = float(np.mean(values))
            assert max(abs(best - v) for v in values) > limit, (
                f"a constant {coefficient} of {best:.3f} is within tolerance of "
                "both specimens, so a run that never looked at the pixels passes"
            )

        # one number reported twice must not pass either
        for name in SPECIMENS:
            gap = abs(float(truth[f"m1_{name}"]) - float(truth[f"m2_{name}"]))
            assert gap > limit, (
                f"specimen {name}: M1 and M2 differ by {gap:.3f}, so reporting "
                "one of them for both would score as correct"
            )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Absolute error, not relative: these are fractions bounded at 1, so being
    0.05 out means the same thing wherever it happens — and a relative error
    would make the smallest coefficient the hardest to pass for no reason but
    its size.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    for name in SPECIMENS:
        for coefficient in ("m1", "m2"):
            metric = f"{coefficient}_error_{name}"
            want = fixture.truth.get(f"{coefficient}_{name}")
            if want is None:
                metrics.append(
                    Metric(
                        metric,
                        None,
                        limits[metric],
                        unavailable=(
                            f"the fixture carries no {coefficient.upper()} for "
                            f"specimen {name}"
                        ),
                    )
                )
                continue
            got, why = read_scalar(attempt, f"{coefficient}_specimen_{name}")
            if got is None:
                metrics.append(Metric(metric, None, limits[metric], unavailable=why))
                continue
            metrics.append(Metric(metric, abs(got - float(want)), limits[metric]))
            detail[f"{coefficient}_{name}_reported"] = round(got, 3)
            detail[f"{coefficient}_{name}_true"] = round(float(want), 3)

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    for name in SPECIMENS:
        for channel in ("green", "red"):
            frame = fixture.data.get(f"specimen_{name}_{channel}")
            if frame is not None:
                save_png(np.asarray(frame, float), where / f"{name}-{channel}.png")

    rows = ["specimen,coefficient,truth,reported"]
    for name in SPECIMENS:
        for coefficient in ("m1", "m2"):
            want = fixture.truth.get(f"{coefficient}_{name}")
            got = attempt.arrays.get(f"{coefficient}_specimen_{name}")
            got = "" if got is None else f"{np.asarray(got, float).reshape(-1)[0]:g}"
            want = "" if want is None else f"{float(want):.3f}"
            rows.append(f"{name},{coefficient.upper()},{want},{got}")
    (where / "coefficients.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Four images are open in the napari viewer — a green and a red channel from each
of two specimens, taken on the same microscope in the same session:

  specimen_1_green  float32 (640, 640)
  specimen_1_red    float32 (640, 640)
  specimen_2_green  float32 (640, 640)
  specimen_2_red    float32 (640, 640)

There is no tensor server in this session, so those layers are the only source
of pixels.

For each specimen I need the two overlap fractions, in the Manders sense:

  M1 = the fraction of that specimen's total *green* signal that lies in pixels
       where the red channel also has signal
  M2 = the same the other way round — the fraction of the total red signal that
       lies where green also has signal

One thing about how those are defined, because it is a convention rather than a
finding: signal means intensity above that channel's own background, and
background is not signal — it belongs in neither the numerator nor the
denominator of either fraction. The two channels have separate backgrounds.

The person who prepared and imaged these is here and can answer questions. To
ask them something, reply in plain text without calling a tool, and their answer
will come back as the next message.

When you are finished, leave four names bound in the kernel namespace:

  m1_specimen_1   M1 for specimen 1
  m2_specimen_1   M2 for specimen 1
  m1_specimen_2   M1 for specimen 2
  m2_specimen_2   M2 for specimen 2

Those names are how your result is collected, so bind them exactly.
"""

#: Self-sufficient: the prompt carries the definition of both coefficients and
#: the fact that background is excluded, so this person holds no part of the
#: answer. Note what is *not* here — nothing about thresholds, auto-thresholds,
#: how bright the puncta are relative to the background, or that the two
#: specimens differ in how much pairing there is.
CELL_BIOLOGIST = Persona(
    name="operator-two-colour-puncta",
    facts={
        "what the specimens are": (
            "cultured cells stained for two proteins we think travel together. "
            "Two coverslips from two separate preparations"
        ),
        "how the images were taken": (
            "sequentially, one channel then the other, on a spinning disk. Same "
            "exposure and same laser power for every image here"
        ),
        "whether anything has been done to the images": (
            "no. No flat-fielding, no background subtraction, nothing "
            "deconvolved — that is straight off the camera"
        ),
        "whether the two channels are aligned": (
            "yes. We check the registration with beads every week and it has "
            "never been more than a pixel"
        ),
        "what the numbers are for": (
            "a figure comparing the two preparations. What I care about is "
            "whether they came out the same, so the two have to be measured "
            "identically"
        ),
        "how the two specimens differ": (
            "nothing deliberate. Same protocol, same antibodies, different day"
        ),
    },
    background=(
        "You are a cell biologist who prepared two coverslips of cultured cells, "
        "stained each for two proteins, and imaged them yourself. You are happy "
        "to answer questions about the cells, the staining and the microscope."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=CELL_BIOLOGIST,
    fixture=Procedural(TwoSpecimens()),
    layers=(
        Layer("specimen_1_green", "specimen_1_green"),
        Layer("specimen_1_red", "specimen_1_red"),
        Layer("specimen_2_green", "specimen_2_green"),
        Layer("specimen_2_red", "specimen_2_red"),
    ),
    collect={
        "m1_specimen_1": "m1_specimen_1",
        "m2_specimen_1": "m2_specimen_1",
        "m1_specimen_2": "m1_specimen_2",
        "m2_specimen_2": "m2_specimen_2",
    },
    score=verify,
    save_artifacts=save_artifacts,
)
