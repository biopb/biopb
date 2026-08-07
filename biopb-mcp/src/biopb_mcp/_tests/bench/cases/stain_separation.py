"""H&E stain separation as benchmark data: how much of each stain is here?

A deferred-tier case (`docs/skill-candidates.md`). Colour deconvolution was
**prescreened and dropped** 2026-08-03 for the shipped catalog — both cold
Sonnet arms produced Ruifrok & Johnston unprompted, cited the paper, and
cross-checked themselves against `skimage.color.rgb2hed`. It is one call plus a
matrix, not a procedure with decisions between the steps.

**The entry kept one finding, and this case is that finding made scoreable: the
white point is not 255.** On a scanner whose blank field does not read 255,
assuming it does puts a constant optical-density floor under every pixel. What
makes it worth a benchmark rather than a footnote is that *the obvious way to
check the answer cannot see it*: every route below that unmixes at all — the
ones that are right and the ones that are a third of the signal out — correlates
with the true hematoxylin map at r >= 0.99. The error is close to a constant,
and a constant is exactly what a correlation coefficient is blind to.

So nothing here is scored by correlation. All three metrics are **ratios
between two regions**, which no scaling convention can move (protocol §8), and
the run never learns where those regions are — it hands back two concentration
maps and the verifier does the rest.

Measured on this fixture, whose blank glass reads 190/185/178:

  ======================================  =========  =========  =========  ======
  route                                    blank H   E in nuc   dark:pale    r(H)
  ======================================  =========  =========  =========  ======
  white point from the mode                  0.005      0.000      0.004   0.9978
  white point from a bright percentile       0.027      0.006      0.026   0.9978
  one scalar white point for all channels    0.003      0.026      0.003   0.9978
  white point = the brightest pixel          0.045      0.010      0.042   0.9978
  transmittance unmixed, Beer-Lambert skipped 0.036     0.069      0.153   0.9921
  ``skimage.color.rgb2hed``                  0.233      0.000      0.188   0.9966
  white point assumed to be 255              0.354      0.119      0.261   0.9978
  no unmixing: inverted R, G, B channels     0.625      1.112      0.395   0.8028
  ======================================  =========  =========  =========  ======

The top four pass and the bottom four fail, which is the whole claim: `r(H)` is
the last column and it separates none of them.

`TOLERANCE` sits in those gaps. Four ways of estimating the white point *from
the image* span 0.003–0.045 on the blank-glass leak, and the nearest route that
does not estimate it is 0.233 — so 0.12 is 2.7x above the worst defensible
answer and half the distance to the nearest wrong one.

**`skimage.color.rgb2hed` is the row that matters**, because it is the call
anybody reaches for and it bakes the assumption in: it normalises against a
white of 1.0, i.e. 255. Using it unexamined is 0.233 — nearly the whole 255
failure — while its correlation with the truth is 0.998. That is the finding the
survey wanted kept, stated as a number a run can be scored against.

**Three metrics, because each catches something the others do not.**

* *blank glass* — hematoxylin recovered where there is no tissue at all, over
  what is recovered in the darkest nuclei. This is the white point, and nothing
  else moves it much.
* *eosin in nuclei* — eosin recovered in nuclei that have none, over eosin in
  the stroma. This is whether an unmixing happened: reading the stains off the
  raw channels scores 1.112. It is also the one metric a *single scalar* white
  point fails worst (0.026 against 0.006), because a grey white point on a
  non-grey lamp leaks into the other stain rather than into the glass.
* *dark:pale* — the ratio of the two nucleus classes, which is 2.00 by
  construction. This is the positive control that stops "return zeros" passing
  the first two, and it is the only metric that catches unmixing the
  transmittance instead of the optical density (0.153): skipping Beer-Lambert
  leaves the map monotonic in concentration but not proportional to it, so the
  ratio is wrong while the leaks are fine.

**The route to the white point is in the data, not in the metadata** (protocol
§7). There is real blank glass in the field — about a quarter of it — so the
illumination is measurable, and the persona can confirm the frame contains bare
slide if asked. A fixture that only *stated* the white point would be testing
whether a run reads a number off a prompt.

**A lymphoid aggregate is what makes "eosin in nuclei" meaningful**: nuclei
elsewhere in the section sit on stroma that genuinely carries eosin, so
recovering some there is correct. The scored nuclei are the ones inside the
aggregate, where there is no cytoplasm and the true eosin concentration is
exactly zero.
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

NAMESPACE = "stain-separation"
CASE_ID = "he-section-on-a-scanner-whose-white-is-not-255"

#: From the table in the module docstring, not from taste. Three different
#: quantities, so three different limits: the blank-glass leak has the widest
#: gap to work in and the crosstalk the narrowest.
TOLERANCE = {
    "blank_glass_leak": 0.12,
    "eosin_in_nuclei": 0.05,
    "nucleus_ratio_error": 0.09,
}

SHAPE = (512, 512)

#: Ruifrok & Johnston, as `skimage.color.rgb_from_hed` carries them. Already
#: unit vectors, which is why "forgot to normalise" is not a route below — on
#: the published numbers it is a no-op.
H_VECTOR = (0.65, 0.70, 0.29)
E_VECTOR = (0.07, 0.99, 0.11)

#: This scanner's blank field. Not 255, and not grey — the lamp is warmer than
#: the sensor's blue channel, which is the ordinary case.
WHITE = (190.0, 185.0, 178.0)

C_DARK, C_PALE = 0.45, 0.225  #: hematoxylin OD of the two nucleus classes, 2:1
C_STROMA = 0.25  #: mean eosin OD of the stroma
READ_NOISE = 0.5
SHOT = 0.08  #: a brightfield scan is photon-rich; quantisation dominates

NUCLEI_IN_AGGREGATE = 110
NUCLEI_IN_STROMA = 70
SEED = 5

#: What the two nucleus classes are meant to come out as. Named because it is
#: the positive control, not because a run is told it.
TRUE_RATIO = C_DARK / C_PALE


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, float)
    return vector / np.linalg.norm(vector)


def stain_matrix() -> np.ndarray:
    """Rows: hematoxylin, eosin, and their cross product.

    The third row is not a stain. It is the direction neither stain can explain,
    and it is there so the 3x3 is invertible — two stains in three channels is
    otherwise an underdetermined system dressed up as a square one.
    """
    h, e = _unit(H_VECTOR), _unit(E_VECTOR)
    return np.stack([h, e, _unit(np.cross(h, e))])


def region_ratios(
    hematoxylin: np.ndarray, eosin: np.ndarray, masks
) -> tuple[float, float, float]:
    """The three scored numbers, from two maps and the truth's regions.

    Every one is a ratio, so a run may report concentrations in whatever units
    it likes — and a run that reports them negated scores the same, which is
    the point of taking absolute values here.
    """
    dark = float(np.mean(hematoxylin[masks["dark"]]))
    pale = float(np.mean(hematoxylin[masks["pale"]]))
    stroma_e = float(np.mean(eosin[masks["stroma"]]))
    blank = abs(float(np.mean(hematoxylin[masks["blank"]])) / dark) if dark else np.nan
    crosstalk = (
        abs(float(np.mean(eosin[masks["dark"]])) / stroma_e) if stroma_e else np.nan
    )
    ratio = abs(dark / pale) if pale else np.nan
    return blank, crosstalk, abs(ratio - TRUE_RATIO) / TRUE_RATIO


@dataclass(frozen=True)
class StainedSection:
    """One H&E field: tissue, blank glass, and an aggregate of bare nuclei."""

    shape: tuple[int, int] = SHAPE
    seed: int = SEED

    def _geometry(self, rng):
        yy, xx = np.mgrid[0 : self.shape[0], 0 : self.shape[1]].astype(float)
        blob = np.zeros(self.shape, bool)
        for cy, cx, ry, rx in (
            (250, 210, 210, 170),
            (150, 300, 120, 110),
            (370, 300, 110, 120),
        ):
            blob |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        ragged = ndi.gaussian_filter(rng.normal(0, 1, self.shape), 12.0)
        ragged = (ragged - ragged.mean()) / ragged.std()
        blob = ndi.binary_fill_holes(blob & (ragged > -1.4)) | (blob & (ragged > 0.2))
        tissue = ndi.binary_closing(blob, np.ones((7, 7)))

        aggregate = (((yy - 150) / 85) ** 2 + ((xx - 300) / 80) ** 2 <= 1.0) & tissue
        return yy, xx, tissue, aggregate

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        yy, xx, tissue, aggregate = self._geometry(rng)

        smooth = ndi.gaussian_filter(rng.normal(0, 1, self.shape), 25.0)
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
        c_eosin = np.where(tissue & ~aggregate, C_STROMA * (0.5 + smooth), 0.0)

        c_hema = np.zeros(self.shape)
        dark = np.zeros(self.shape, bool)
        pale = np.zeros(self.shape, bool)
        for inside in (True, False):
            pool = aggregate if inside else (tissue & ~aggregate)
            ys, xs = np.nonzero(pool)
            count = NUCLEI_IN_AGGREGATE if inside else NUCLEI_IN_STROMA
            for k, index in enumerate(rng.choice(len(ys), count, replace=False)):
                cy, cx = ys[index], xs[index]
                ry, rx = rng.uniform(3.5, 5.5), rng.uniform(3.5, 5.5)
                spot = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
                is_dark = k % 2 == 0
                c_hema[spot] = C_DARK if is_dark else C_PALE
                if inside:
                    (dark if is_dark else pale)[spot] = True

        # A later nucleus may have painted over an earlier one, so the classes
        # are read back off the map rather than trusted from the loop.
        midpoint = 0.5 * (C_DARK + C_PALE)
        dark &= (c_hema > midpoint) & (c_eosin == 0)
        pale &= (c_hema > 0) & (c_hema < midpoint) & (c_eosin == 0)
        masks = {
            "blank": ndi.binary_erosion(~tissue, np.ones((15, 15))),
            "dark": dark,
            "pale": pale,
            "stroma": (c_eosin > 0.2) & (c_hema == 0),
        }

        matrix = stain_matrix()
        od = c_hema[..., None] * matrix[0] + c_eosin[..., None] * matrix[1]
        rgb = np.asarray(WHITE, float) * np.power(10.0, -od)
        rgb = rgb + rng.normal(0, np.sqrt(np.maximum(rgb, 1.0)) * SHOT, rgb.shape)
        rgb = rgb + rng.normal(0, READ_NOISE, rgb.shape)
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

        self._check(rgb, c_hema, c_eosin, masks)

        return Fixture(
            provenance=(
                f"procedural: a {self.shape[0]}x{self.shape[1]} H&E field, "
                f"Beer-Lambert from Ruifrok & Johnston vectors on a white point "
                f"of {WHITE[0]:g}/{WHITE[1]:g}/{WHITE[2]:g}, hematoxylin "
                f"{C_PALE:g} and {C_DARK:g} OD, eosin about {C_STROMA:g} OD, "
                f"seed {self.seed}"
            ),
            about=(
                "One H&E field on a scanner whose blank glass reads "
                f"{WHITE[0]:g}/{WHITE[1]:g}/{WHITE[2]:g}, not 255. Assuming 255 "
                "puts a constant optical-density floor under every pixel: "
                "hematoxylin appears in bare glass at 0.354 of its concentration "
                "in the darkest nuclei, and the 2:1 ratio between the two "
                "nucleus classes reads 26% low. `skimage.color.rgb2hed` bakes "
                "that assumption in and scores 0.233. None of it is visible in "
                "the correlation with the truth, which stays above 0.994 for "
                "every route including the wrong ones."
            ),
            data={"stained_section": rgb},
            truth={
                "hematoxylin_od": c_hema.astype(np.float32),
                "eosin_od": c_eosin.astype(np.float32),
                "blank_mask": masks["blank"],
                "dark_mask": masks["dark"],
                "pale_mask": masks["pale"],
                "stroma_mask": masks["stroma"],
                "nucleus_ratio": TRUE_RATIO,
            },
            tolerance=dict(TOLERANCE),
        )

    def _check(self, rgb, c_hema, c_eosin, masks) -> None:
        """The properties the case rests on, before anyone pays for a run."""
        for name, mask in masks.items():
            assert mask.sum() > 800, (
                f"the {name} region is {int(mask.sum())} pixels, too few for a "
                "mean over it to be about the section rather than about noise"
            )
        assert (c_eosin[masks["dark"]] == 0).all(), (
            "the scored nuclei carry eosin, so recovering some there would be "
            "correct and the crosstalk metric measures nothing"
        )
        assert (c_hema[masks["blank"]] == 0).all(), (
            "the blank region carries hematoxylin, so it is not blank"
        )

        blank_rgb = rgb[masks["blank"]].mean(axis=0)
        assert np.allclose(blank_rgb, WHITE, atol=1.5), (
            f"the blank glass reads {blank_rgb.round(1)} against a declared "
            f"white point of {WHITE} — the illumination is not measurable from "
            "the data, and protocol 7 says that is where the route to it lives"
        )
        assert (blank_rgb < 235).all(), (
            f"the blank glass reads {blank_rgb.round(1)}, close enough to 255 "
            "that assuming 255 is nearly right and the case measures nothing"
        )

        # protocol 11, back door: the metrics are ratios, so a run that reports
        # a constant map scores nothing -- but a run that reports *the input*
        # should not pass either.
        inverted = 255.0 - np.asarray(rgb, float)
        blank, crosstalk, ratio = region_ratios(
            inverted[..., 2], inverted[..., 1], masks
        )
        assert (
            blank > TOLERANCE["blank_glass_leak"]
            or crosstalk > TOLERANCE["eosin_in_nuclei"]
            or ratio > TOLERANCE["nucleus_ratio_error"]
        ), "reading the stains straight off the blue and green channels passes"


# --- the verifier ----------------------------------------------------------


def _maps(fixture: Fixture, attempt: Attempt):
    """The run's two maps, or why they cannot be scored."""
    wanted = np.asarray(fixture.truth["hematoxylin_od"]).shape
    got = {}
    for key in ("hematoxylin", "eosin"):
        array = attempt.arrays.get(key)
        if array is None:
            return None, f"the run left no `{key}`"
        array = np.asarray(array, float)
        if array.shape != wanted:
            return None, (
                f"the run's `{key}` is {array.shape}, not a {wanted} map of "
                "concentrations"
            )
        if not np.isfinite(array).all():
            return None, f"the run's `{key}` is not finite everywhere"
        got[key] = array
    return got, ""


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    The run hands back two maps and nothing else; every region the score is
    taken over is the fixture's, so there is no convention for the run to get
    right or wrong and nothing about the regions leaks into the prompt.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    names = ("blank_glass_leak", "eosin_in_nuclei", "nucleus_ratio_error")

    maps, why = _maps(fixture, attempt)
    if maps is None:
        return Outcome(
            fixture=fixture,
            attempt=attempt,
            metrics=[
                Metric(name, None, limits[name], unavailable=why) for name in names
            ],
        )

    masks = {
        key: np.asarray(fixture.truth[f"{key}_mask"], bool)
        for key in ("blank", "dark", "pale", "stroma")
    }
    values = region_ratios(maps["hematoxylin"], maps["eosin"], masks)

    metrics: list[Metric] = []
    detail: dict[str, object] = {}
    for name, value in zip(names, values, strict=True):
        if not np.isfinite(value):
            metrics.append(
                Metric(
                    name,
                    None,
                    limits[name],
                    unavailable="the run's maps are flat over a region this needs",
                )
            )
            continue
        metrics.append(Metric(name, float(value), limits[name]))
        detail[name] = round(float(value), 4)

    dark = float(np.mean(maps["hematoxylin"][masks["dark"]]))
    pale = float(np.mean(maps["hematoxylin"][masks["pale"]]))
    detail["nucleus_ratio_reported"] = round(abs(dark / pale), 3) if pale else None
    detail["nucleus_ratio_true"] = round(float(fixture.truth["nucleus_ratio"]), 3)
    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: what the number means. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    section = fixture.data.get("stained_section")
    if section is not None:
        save_png(np.asarray(section, float).mean(axis=2), where / "section.png")
    for key in ("hematoxylin_od", "eosin_od"):
        truth = fixture.truth.get(key)
        if truth is not None:
            save_png(np.asarray(truth, float), where / f"truth-{key}.png")
    for key in ("hematoxylin", "eosin"):
        got = attempt.arrays.get(key)
        if got is not None and np.asarray(got).ndim == 2:
            save_png(np.asarray(got, float), where / f"reported-{key}.png")

    rows = ["quantity,truth,reported"]
    rows.append(f"blank_glass_leak,0,{outcome.detail.get('blank_glass_leak', '')}")
    rows.append(f"eosin_in_nuclei,0,{outcome.detail.get('eosin_in_nuclei', '')}")
    rows.append(
        f"nucleus_ratio,{outcome.detail.get('nucleus_ratio_true', '')},"
        f"{outcome.detail.get('nucleus_ratio_reported', '') or ''}"
    )
    (where / "stains.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
One image is open in the napari viewer:

  stained_section  uint8 (512, 512, 3) — an H&E section, RGB as it came off the
                   slide scanner

There is no tensor server in this session, so that layer is the only source of
pixels.

I need the two stains separated: a map of how much hematoxylin is at each pixel
and a map of how much eosin, in whatever concentration units fall out of your
method. What I care about is that the numbers are proportional to how much stain
is really there — I am going to compare them between regions of this section, so
a constant added to one of the maps would be worse for me than a noisy one.

The person who cut, stained and scanned this section is here and can answer
questions. To ask them something, reply in plain text without calling a tool,
and their answer will come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  hematoxylin   a (512, 512) map of hematoxylin concentration
  eosin         a (512, 512) map of eosin concentration

Those names are how your result is collected, so bind them exactly.
"""

#: Holds one fact worth asking for and no part of the answer: what the scanner's
#: blank field does, which is also measurable from the glass in the frame
#: (protocol 7 — the route to the truth is in the data, and the persona only
#: confirms it). Note what is *not* here — nothing about optical density,
#: Beer-Lambert, stain vectors, or what number a white point should be.
PATHOLOGIST = Persona(
    name="operator-he-section",
    facts={
        "what the section is": (
            "a routine H&E of a lymph node, four microns, cut and stained the "
            "same morning it was scanned"
        ),
        "how it was scanned": (
            "a brightfield slide scanner, 20x. We set the exposure once for the "
            "batch and it is not near the top of the range — nothing in these "
            "images is clipped"
        ),
        "whether the whole frame is tissue": (
            "no, and that is deliberate. I always leave bare slide in the field "
            "so I can see what the empty part looks like"
        ),
        "whether the staining is heavy or light": (
            "light. This batch went through the haematoxylin quickly and it "
            "shows — the nuclei are paler than I would like"
        ),
        "whether anything has been done to the image": (
            "nothing. No colour correction, no white balance, no shading "
            "correction — that is the file the scanner wrote"
        ),
        "what the maps are for": (
            "comparing how darkly the nuclei took up stain in one part of the "
            "section against another, so what matters is that the numbers "
            "scale with the real thing"
        ),
    },
    background=(
        "You are a histopathologist who cut, stained and scanned this section "
        "yourself. You are happy to answer questions about the tissue, the "
        "staining and the scanner."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=PATHOLOGIST,
    fixture=Procedural(StainedSection()),
    layers=(Layer("stained_section", "stained_section"),),
    collect={"hematoxylin": "hematoxylin", "eosin": "eosin"},
    score=verify,
    save_artifacts=save_artifacts,
    persona_must_know=("bare slide", "not near the top of the range"),
    persona_must_not_know=(
        "optical density",
        "beer",
        "lambert",
        "ruifrok",
        "deconvolution",
        "unmix",
        "255",
        "logarithm",
    ),
)
