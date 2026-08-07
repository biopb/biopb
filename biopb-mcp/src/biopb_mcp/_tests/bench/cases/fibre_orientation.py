"""Fibre orientation as benchmark data: which way do these run, and how much?

A deferred-tier case (`docs/skill-candidates.md`). Fibre orientation was
**prescreened and dropped** 2026-08-03 for the shipped catalog — both cold
Sonnet arms built a global structure tensor, whose double-angle form handles the
wrap by construction, and scored 0.1 and 0.7 degrees. It is here anyway, because
the rejection is conditional on the consuming tier: Haiku returned **89.1
degrees** on the wraparound population — the arithmetic mean of angles — and
called an isotropic control **0.82 coherent**. This case is what makes that
re-measurable rather than remembered.

**Three fields, because one cannot see the failure.** An orientation is
undirected: 175 degrees and 5 degrees are ten degrees apart, not a hundred and
seventy. A mean that does not know this is *correct* on a population that avoids
the wrap and catastrophically wrong on one that straddles it, so a fixture with
only the first would pass the naive route and a fixture with only the second
could not tell a wrap bug from a fitting bug.

Measured on this fixture:

  =====================================  =========  =========  ==========
  route                                    field a    field b     field c
  =====================================  =========  =========  ==========
  truth                                    30.1 deg  179.7 deg  no direction
  global structure tensor                  31.3       0.4       coh 0.052
  ..angular error                           1.2       0.7       —
  arithmetic mean of fibre angles          30.1      97.9       —
  ..angular error                           0.0      81.8       —
  tensor without the 90 degree rotation   121.3      90.4       —
  ..angular error                          88.8      89.3       —
  mean of *local* coherence                 —          —        coh 0.544
  =====================================  =========  =========  ==========

Three separate failures, and each needs its own field:

* **the wrap.** Field b's population is centred on 0/180 and straddles it. The
  arithmetic mean lands at 97.9 — the average of numbers near 175 and numbers
  near 5 — while being exactly right on field a. This is the sharp one.
* **the 90 degrees.** The principal eigenvector of the *gradient* structure
  tensor is perpendicular to the fibre, so a run that reports it unrotated is
  wrong by 90 degrees on every field at once, which is why both populated fields
  score it and neither is decisive alone.
* **coherence of what.** Field c is uniform over 0-180, so it has no preferred
  direction and the global tensor says so (0.052). Averaging *per-pixel*
  coherence instead reports **0.544**, because every fibre is locally aligned
  with itself no matter how the population is distributed. The quantity is not
  wrong; the thing it is averaged over is.

`TOLERANCE` sits in those gaps. The angular limit of 10 degrees is eight times
above a clean run (1.2) and eight times under the nearest failure (81.8). The
coherence limit of 0.25 is the tightest of the three — 4.8x above the correct
answer and 2.2x under the trap — because a coherence is bounded at 1 and there
is only so much room between them.

**`alignment_shortfall_a` is the positive control**, and it is not decoration: a
run that reported coherence 0 for every field would pass field c's limit while
having measured nothing. Scoring field a's alignment from the same numbers makes
that unrepresentable.

**Nothing is withheld.** The prompt carries the pixel size, the definition of
the coherence being asked for, and the fact that a fibre is undirected — the
screen this case reproduces disclosed all three (protocol §6). What is being
measured is whether the run's *statistics* respect the wrap, not whether it
knows to ask. The persona is here for realism and holds no part of the answer.
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

NAMESPACE = "fibre-orientation"
CASE_ID = "three-fields-one-across-the-wrap"

#: From the table in the module docstring, not from taste. See the paragraph
#: above it for why the coherence limit is the tight one.
TOLERANCE = {
    "angle_error_a_deg": 10.0,
    "angle_error_b_deg": 10.0,
    "coherence_c": 0.25,
    "alignment_shortfall_a": 0.35,
}

SHAPE = (256, 256)
N_FIBRES = 220
LENGTH_PX = 70.0
HALF_WIDTH_PX = 1.6
PIXEL_UM = 0.2

#: ``field -> (centre degrees, spread degrees, seed)``. `None` is uniform over
#: the half-circle. Field b is centred on the wrap; field a deliberately is not,
#: so the naive route is right about one of them.
FIELDS: dict[str, tuple[float | None, float, int]] = {
    "a": (30.0, 8.0, 3),
    "b": (0.0, 8.0, 5),
    "c": (None, 0.0, 7),
}


def circular_mean_deg(angles: np.ndarray) -> float:
    """The mean of an *undirected* orientation, in degrees on [0, 180).

    Doubling maps the half-circle onto the full one, where an ordinary vector
    mean is meaningful, and halving maps it back. This is the whole of what the
    case measures, which is why the truth is computed this way and never from a
    fitted image.
    """
    return float(np.rad2deg(0.5 * np.angle(np.mean(np.exp(2j * angles)))) % 180.0)


def angular_error_deg(got: float, want: float) -> float:
    """Separation of two undirected orientations: at most 90 degrees."""
    return float(abs((got - want + 90.0) % 180.0 - 90.0))


@dataclass(frozen=True)
class FibreFields:
    """Three fields of straight fibres, one population straddling the wrap."""

    shape: tuple[int, int] = SHAPE

    def _draw(self, centre, spread, seed) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        if centre is None:
            angles = rng.uniform(0.0, np.pi, N_FIBRES)
        else:
            # von Mises on the *doubled* angle: the correct distribution for an
            # undirected orientation, and it wraps at 180 by construction rather
            # than by a clip that would pile density at the ends.
            kappa = 1.0 / np.deg2rad(spread) ** 2
            angles = (rng.vonmises(2 * np.deg2rad(centre), kappa, N_FIBRES) / 2.0) % (
                np.pi
            )
        mask = np.zeros(self.shape, np.float32)
        yy, xx = np.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        for angle in angles:
            cy = rng.uniform(0, self.shape[0])
            cx = rng.uniform(0, self.shape[1])
            uy, ux = np.sin(angle), np.cos(angle)
            dy, dx = yy - cy, xx - cx
            along = dy * uy + dx * ux
            across = np.abs(-dy * ux + dx * uy)
            mask[(np.abs(along) <= LENGTH_PX / 2) & (across <= HALF_WIDTH_PX)] = 1.0
        return angles, mask

    def _render(self, mask: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 100)
        image = ndi.gaussian_filter(mask, 1.2) * 700.0 + 100.0
        image = image + rng.normal(0, np.sqrt(np.maximum(image, 1.0)))
        return image.astype(np.float32)

    def __call__(self) -> Fixture:
        data: dict[str, np.ndarray] = {}
        truth: dict[str, object] = {}
        for name, (centre, spread, seed) in FIELDS.items():
            angles, mask = self._draw(centre, spread, seed)
            data[f"field_{name}"] = self._render(mask, seed)
            truth[f"angles_{name}_rad"] = angles
            if centre is not None:
                truth[f"angle_{name}_deg"] = circular_mean_deg(angles)

        # The three properties the case rests on, checked before anyone pays for
        # a run. None of them is visible from the images alone.
        wrapped = np.rad2deg(truth["angles_b_rad"])
        assert (wrapped > 170).any() and (wrapped < 10).any(), (
            "field b's population does not straddle the wrap, so the arithmetic "
            "mean is not a wrong answer and the case measures nothing"
        )
        naive_b = float(wrapped.mean() % 180.0)
        assert angular_error_deg(naive_b, truth["angle_b_deg"]) > 45.0, (
            f"the arithmetic mean of field b lands at {naive_b:.1f} deg, within "
            "45 deg of the truth — the trap did not arm"
        )
        naive_a = float(np.rad2deg(truth["angles_a_rad"]).mean() % 180.0)
        assert angular_error_deg(naive_a, truth["angle_a_deg"]) < 5.0, (
            "the arithmetic mean is already wrong on field a, so a run failing "
            "field b would not be evidence about the wrap specifically"
        )
        spread_c = np.abs(np.mean(np.exp(2j * truth["angles_c_rad"])))
        assert spread_c < 0.15, (
            f"field c has a resultant of {spread_c:.3f}, so it is not isotropic "
            "and its coherence is legitimately non-zero"
        )

        return Fixture(
            provenance=(
                f"procedural: three {self.shape[0]}x{self.shape[1]} fields of "
                f"{N_FIBRES} fibres {LENGTH_PX:g}x{2 * HALF_WIDTH_PX:g} px at "
                f"{PIXEL_UM} um/px, von Mises on the doubled angle, seeds "
                f"{[seed for _, _, seed in FIELDS.values()]}"
            ),
            about=(
                f"Three fields of straight fibres. Field a is a population at "
                f"{truth['angle_a_deg']:.1f} deg, field b one at "
                f"{truth['angle_b_deg']:.1f} deg — straddling the 180/0 wrap, "
                "where the arithmetic mean of the angles lands at 97.9 deg — and "
                "field c is uniform over the half-circle, with no preferred "
                "direction at all. The global structure tensor gives 1.2 and 0.7 "
                "deg of error and coherence 0.052 on field c; averaging per-pixel "
                "coherence instead gives 0.544 there, because every fibre is "
                "locally aligned with itself."
            ),
            data=data,
            truth=truth,
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    Angular error is taken modulo the half-circle, which is the same statement
    the case is testing the run for — scoring it any other way would penalise a
    correct answer expressed as 179.9 instead of -0.1.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    for field in ("a", "b"):
        name = f"angle_error_{field}_deg"
        want = fixture.truth.get(f"angle_{field}_deg")
        if want is None:
            metrics.append(
                Metric(
                    name,
                    None,
                    limits[name],
                    unavailable=f"the fixture carries no orientation for field {field}",
                )
            )
            continue
        got, why = read_scalar(attempt, f"angle_{field}_deg")
        if got is None:
            metrics.append(Metric(name, None, limits[name], unavailable=why))
            continue
        metrics.append(
            Metric(name, angular_error_deg(got, float(want)), limits[name], unit=" deg")
        )
        detail[f"angle_{field}_reported_deg"] = round(got, 2)
        detail[f"angle_{field}_true_deg"] = round(float(want), 2)

    # Field c has no orientation to be right about, so what is scored is the
    # claim of alignment itself.
    got_c, why_c = read_scalar(attempt, "coherence_c")
    if got_c is None:
        metrics.append(
            Metric("coherence_c", None, limits["coherence_c"], unavailable=why_c)
        )
    else:
        metrics.append(Metric("coherence_c", abs(got_c), limits["coherence_c"]))
        detail["coherence_c_reported"] = round(got_c, 3)

    # The positive control. Without it, "0 everywhere" passes the line above.
    got_a, why_a = read_scalar(attempt, "coherence_a")
    if got_a is None:
        metrics.append(
            Metric(
                "alignment_shortfall_a",
                None,
                limits["alignment_shortfall_a"],
                unavailable=why_a,
            )
        )
    else:
        metrics.append(
            Metric(
                "alignment_shortfall_a",
                max(0.0, 1.0 - got_a),
                limits["alignment_shortfall_a"],
            )
        )
        detail["coherence_a_reported"] = round(got_a, 3)

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    for field in FIELDS:
        image = fixture.data.get(f"field_{field}")
        if image is not None:
            save_png(np.asarray(image, float), where / f"field-{field}.png")

    rows = ["quantity,truth,reported"]
    for field in ("a", "b"):
        want = fixture.truth.get(f"angle_{field}_deg")
        got = attempt.arrays.get(f"angle_{field}_deg")
        got = "" if got is None else f"{np.asarray(got, float).reshape(-1)[0]:g}"
        rows.append(
            f"angle_{field}_deg,{'' if want is None else f'{float(want):.2f}'},{got}"
        )
    for field in ("a", "c"):
        got = attempt.arrays.get(f"coherence_{field}")
        got = "" if got is None else f"{np.asarray(got, float).reshape(-1)[0]:g}"
        rows.append(f"coherence_{field},,{got}")
    (where / "orientation.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Three images of collagen fibres are open in the napari viewer, from three
different specimens:

  field_a  float32 (256, 256)
  field_b  float32 (256, 256)
  field_c  float32 (256, 256)

The pixels are 0.2 microns square. There is no tensor server in this session,
so those layers are the only source of pixels.

For each field I need to know which way the fibres run and how consistently.

Two things about how to report it, because they are conventions rather than
findings. A fibre has no head and no tail, so an orientation is only defined up
to 180 degrees: report it in degrees on [0, 180), measured anticlockwise from
the x axis. And by how consistently, I mean the coherence of the field as a
whole — (L1 - L2) / (L1 + L2) for the two eigenvalues of the orientation
distribution, so that 0 means no preferred direction and 1 means every fibre
parallel.

The person who prepared and imaged these is here and can answer questions. To
ask them something, reply in plain text without calling a tool, and their answer
will come back as the next message.

When you are finished, leave four names bound in the kernel namespace:

  angle_a_deg   the mean fibre orientation of field_a, in degrees on [0, 180)
  angle_b_deg   the mean fibre orientation of field_b, in degrees on [0, 180)
  coherence_a   the coherence of field_a
  coherence_c   the coherence of field_c

Those names are how your result is collected, so bind them exactly.
"""

#: Self-sufficient: the prompt carries the pixel size, the reporting convention
#: and the fact that a fibre is undirected, so this person holds no part of the
#: answer. Note what is *not* here — nothing about structure tensors, doubled
#: angles, circular statistics or which field is which.
HISTOLOGIST = Persona(
    name="operator-collagen-fibres",
    facts={
        "what the specimens are": (
            "second-harmonic images of collagen in dermis, from three donors. "
            "Same microscope, same objective, same day"
        ),
        "how the images were prepared": (
            "nothing has been done to them. No flattening, no rotation, no "
            "background subtraction — that is what came off the instrument"
        ),
        "whether the specimens were mounted the same way up": (
            "they were cut and mounted by hand, so no. Nothing about the frame "
            "means the same thing from one to the next"
        ),
        "what the measurement is for": (
            "comparing scarred tissue against healthy. In a scar the fibres "
            "pull into one direction, and in healthy dermis they do not"
        ),
        "how bright the fibres are": (
            "brighter than the background everywhere, and I have never had to "
            "threshold them by hand"
        ),
    },
    background=(
        "You prepared three skin biopsies and imaged their collagen on a "
        "second-harmonic microscope. You are happy to answer questions about "
        "the specimens, the microscope and how the images were prepared."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=HISTOLOGIST,
    fixture=Procedural(FibreFields()),
    layers=(
        Layer("field_a", "field_a"),
        Layer("field_b", "field_b"),
        Layer("field_c", "field_c"),
    ),
    collect={
        "angle_a_deg": "angle_a_deg",
        "angle_b_deg": "angle_b_deg",
        "coherence_a": "coherence_a",
        "coherence_c": "coherence_c",
    },
    score=verify,
    save_artifacts=save_artifacts,
)
