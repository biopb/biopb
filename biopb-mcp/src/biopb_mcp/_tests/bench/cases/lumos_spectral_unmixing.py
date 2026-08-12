"""Blind spectral unmixing of a lambda stack: which dye is where?

**Scope note.** This case scores whether a run checked the background it cut,
not whether it chose a good thresholding method. The correct cut is a 3x-wide
empty gap in the histogram that any inspection finds, and the persona holds two
approximate numbers -- coverage and patch count -- that pin it for a run that
asks. So a failure means the run neither looked nor asked. See "does not ask a
run to pick the right thresholding method" below before reading a failure as a
claim about method choice.

A deferred-tier case (`docs/skill-candidates.md`). LUMoS was **prescreened and
dropped** 2026-08-06 as a *decomposition* reject — five of six arms L2-normalised
each pixel's spectrum unprompted, which is the one thing LUMoS is about, and the
candidate decomposes into a generic auto-threshold plus a normalise-then-cluster
the model does on its own. It is here anyway, for the reason the rejection gave
rather than in spite of it.

**What the screen actually measured was the background cut.** Across six arms,
``corr(|background error|, dye accuracy) = -0.978``. The sonnet FAIL (M2)
clustered at 0.867 — indistinguishable from the passing arms — and lost the
whole 0.23 by cutting background at 54.2% where the truth is 39.6%. It also ran
more validation than any arm in that survey (cluster-centre spectra smooth and
evenly spaced, connected-component purity 11/23 blobs >= 0.96, an RGB rendering
with sharp boundaries, a cross-check against NMF it correctly rejected) and
passed every one of its own checks while discarding a seventh of the image.

**That is the blind spot this case is an instrument for.** Every check the arms
invented is computed on *retained* pixels only, so an over-cut background is
invisible to all of them by construction. The discriminating question — *what
fraction of the field did I just call background, and does it match the
structure I can see?* — is the one no arm asked, and it is why this case scores
the background fraction as its own metric rather than folding it into accuracy.

Measured on this fixture, seeds 7/11/23, whole-field accuracy with the four dye
labels matched optimally:

  ==========================================  =========  ===========
  route                                         dye acc    bg called
  ==========================================  =========  ===========
  reference (cut in the valley, L2-normalise, k=4)  0.990      0.400
  no L2 normalisation                             0.599        0.400
  a cut outside the valley, then the reference    0.668        0.732
  TRIVIALITY -- brightest band alone              0.769        0.400
  BACK DOOR -- intensity only, at oracle strength  0.552        0.400
  chance (largest class)                          0.400           --
  truth                                           1.000        0.400
  ==========================================  =========  ===========

Spread across the three seeds is <= 0.018 on every row, so the separations are
properties of the construction rather than of one draw.

`TOLERANCE` sits in the gaps that table opens. `dye_error` is 1 - accuracy: the
reference is 0.010 and the nearest wrong route is 0.231, so 0.12 is about ten
times a clean run and about half the closest failure. `background_error` is the
absolute error in the fraction called background: the reference is 0.000 and the
over-cutting route is 0.332, so 0.08 is comfortably inside that gap. Together
the two limits put the pass band at a cut of 0.38-0.46; below that the binding
constraint is accuracy rather than the fraction, because background pixels
forced into a dye cluster cost more than they save.

**Both §11 screens were run before this case was written, and both moved it.**

*Triviality.* Taking the brightest detector band and mapping it to the nearest
dye peak scored **0.999** on the first build — the whole candidate for free.
The cause was peak spacing against band spacing: 30 nm peaks on 24.4 nm bands
give every dye its own argmax band. At 20 nm spacing adjacent dyes share a band
and the shortcut falls to 0.769 against the reference's 0.990.

*Back door.* An intensity-only classifier, given oracle strength (the true
foreground handed to it, and bin edges fitted against the answer key), scored
0.640 when every dye had the same brightness distribution *width*. Widening the
within-dye range to a decade — identical distribution per dye, so intensity
still says nothing about identity — takes it to 0.552 against a chance floor of
0.400, and to 0.469 once it has to find its own foreground. This is the failure
`skill-candidates.md` §11 records LUMoS's own first fixture having:
``BRIGHTNESS = [1500, 300, 90, 30]`` gave an intensity-only route 0.734, above
three of six arms, and six arms were spent before anyone read the fixture.

**This case does not ask a run to pick the right thresholding method, and it
must not be read as doing so.** The background here is *unambiguous*: the
brightest background pixel totals 30 and the dimmest dye pixel 119, so the
histogram carries an empty valley from 33 to 106 holding 0.1% of the field.
**Any** cut anywhere in that 3x-wide gap recovers the foreground mask exactly.
There is no method to choose and no judgement to exercise — the answer is
sitting in the histogram, and a run that plots one, or checks what fraction its
mask kept, has it.

**What the case measures is whether the run looks.** That is the prescreen's
own finding restated: M2 passed every check it invented and discarded a seventh
of the image, because all of them were computed on retained pixels. Here the
verification is cheap and available, and the fixture is built so that not doing
it is expensive.

**The cost of not looking is measured, and it is close to a coin flip.** Five
named auto-thresholds, each on linear and on log intensity, against the pass
band (`bg` in 0.32-0.48 *and* accuracy >= 0.88, which works out to a cut in
0.38-0.46):

  ==========================  =========  ===========  =========
  cut                           dye acc    bg called    verdict
  ==========================  =========  ===========  =========
  otsu on linear                  0.668        0.732       fail
  otsu on log                     0.990        0.400       pass
  li on linear                    0.836        0.562       fail
  li on log                       0.990        0.400       pass
  yen on linear                   0.612        0.788       fail
  yen on log                      0.771        0.629       fail
  triangle on linear              0.990        0.400       pass
  triangle on log                 0.641        0.191       fail
  mean on linear                  0.777        0.623       fail
  mean on log                     0.988        0.402       pass
  2-component GMM on log          0.990        0.400       pass
  ==========================  =========  ===========  =========

**Note that the pass/fail does not follow linear-versus-log.** `triangle`
inverts the pairing that `otsu`, `li` and `mean` share. So there is no rule of
the form "cut on the log histogram" to be learned here, and an earlier draft of
this docstring that claimed one was wrong. The menu is unordered: an unchecked
pick is ~5 in 11.

**A bad cut is loudly visible in the mask, and this case is weaker than the
prescreen because of it.** An earlier draft of this docstring claimed the run
"cannot tell which side it landed on", by analogy with M2. That is false here,
and measuring the mask says so at once:

  ========================  =======  =======  =========  ========  ===========
  mask                           fg    comps    largest    median   perim/area
  ========================  =======  =======  =========  ========  ===========
  truth                       0.600       10      24774       674        11.70
  otsu on linear (fails)      0.268      119       1845        29        32.59
  yen on linear (fails)       0.212      141       1486        21        35.71
  triangle on log (fails)     0.809     1165      48241         1        26.07
  otsu on log (passes)        0.600       11      24774       216        11.70
  ========================  =======  =======  =========  ========  ===========

A failing cut shatters the field into 119-1165 components against the truth's
10, with ~3x the boundary raggedness. One `ndi.label()` on its own mask catches
it, and so does looking at the mask.

**Why M2's equivalent check did not catch its over-cut**, which is the
distinction worth keeping: M2 computed connected-component *purity* -- whether
each component is spectrally homogeneous -- and a shattered mask is perfectly
pure. Purity is blind to fragmentation by construction, so the check it ran was
not a weaker version of the one that would have worked; it was a different
check that cannot fail for this reason. Count and area, not purity.

**One attempt was made to reproduce the invisible over-cut, and it failed.**
Varying brightness *between* blobs rather than within them should let an
over-cut delete whole structures and leave a smooth mask; measured, it still
fragments (191 components against 35), because Poisson noise on any blob whose
mean sits near the cut speckles its boundary whatever the brightness layout is.
That looks intrinsic to a Poisson synthetic rather than incidental, and it
raises a question for `skill-candidates.md` rather than for this file: whether
M2's plausible-looking over-cut was a property of *its* data rather than of
over-cutting in general. Under §11's stopping rule that is where this stops --
a second rebuild aimed at the same shortcut would be chosen for its effect on
the score.

**So read a failure here as the stronger indictment, not the weaker one.** The
run did not merely pick badly off an unordered menu; it declined to look at a
mask that had come apart into a hundred pieces.

**And the persona closes the menu as a route to a pass.** Without it this case
would hand ~5 in 11 of the auto-threshold menu a pass for no reason the run
could articulate, which is a lottery wearing a benchmark's clothes. The
microscopist knows two things a microscopist would know -- roughly what
fraction of the frame has cells on it, and roughly how many patches they form
-- and those are exactly the two quantities the verifier scores. A run that
asks has a target for its cut *and* a number to compare its component count
against; a run that looks has the histogram valley; a run that does neither is
back on the menu. That is the shape `skill-candidates.md` calls a
†-elicitation candidate, and it is why this case's `persona_must_know` is not
empty while `align-channels-from-landmarks`' is.

**Both halves of the safeguard are asserted in `test_verifiers.py`**, because
wording them is easier to get wrong than it looks. The pass band is asymmetric
-- under-cutting background costs accuracy fast, since a background pixel forced
into a dye cluster is wrong twice -- so an estimate that reads high fails the
run that believed it. A first draft said "call it two thirds", i.e. 0.333
background against a band starting at 0.38, and a run following the operator
exactly would have scored 0.791 and failed. The test takes the range a reader
would take from the persona's own words and requires both ends to pass.

**Two consequences for reading a result from this case.** A single failing
sample is weak evidence about a model — it may only have drawn badly from the
menu — so this case wants repetition more than most, and its verdict is about
the *rate* rather than about one run. And a passing sample is only evidence of
verification if the transcript shows verification; a run that reached for
`threshold_otsu(np.log1p(...))` first and never looked back passes for the wrong
reason. The transcript is part of this case's result in a way it is not for
`local-thickness`, where the number alone settles it.

**`sigma`, not FWHM.** The prescreen records "four dyes 30 nm apart at 50 nm
width (pairwise cosine 0.46-0.93)". Read as a FWHM that gives 0.011-0.608 and a
fixture where nothing is hard; read as a sigma it gives 0.451-0.918, which is
the recorded range. This case uses sigma, at 20 nm spacing (cosine 0.70-0.96)
for the triviality reason above.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

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

NAMESPACE = "lumos-spectral-unmixing"
CASE_ID = "four-dyes-over-a-background-nobody-checks"

#: Read off the table in the module docstring, not off taste. See there for why
#: each sits where it does.
TOLERANCE = {
    "dye_error": 0.12,
    "background_error": 0.08,
}

SHAPE = (256, 256)
#: Ten detector bands, which is the LUMoS geometry: more bands than dyes, and
#: none of them aligned to a dye.
BANDS = np.linspace(480.0, 700.0, 10)
#: 20 nm apart rather than the prescreen's 30, so adjacent dyes share an argmax
#: band and the one-line shortcut cannot match the reference (see docstring).
PEAKS = (555.0, 575.0, 595.0, 615.0)
#: nm. A sigma, not a FWHM -- that reading is what reproduces the recorded
#: pairwise-cosine range.
SIGMA = 50.0

N_DYES = len(PEAKS)
BACKGROUND_FRACTION = 0.40
#: Peak per-pixel signal before the per-pixel draw below.
PHOTONS = 400.0
#: A dim, spectrally flat haze over the whole field: autofluorescence, and the
#: thing a background cut is cutting.
BACKGROUND_PHOTONS = 6.0
#: Within-dye brightness spans 10x, log-uniform. The *width* is what makes
#: normalisation load-bearing; the *sameness across dyes* is what keeps
#: intensity from being a back door. Both were measured -- see the docstring.
BRIGHTNESS_DECADES = 1.0
#: How coarse the dye territories are, in pixels.
BLOB_SIGMA = 9.0

SEED = 7


def _spectra() -> np.ndarray:
    """``(n_dyes, n_bands)``, each row unit norm."""
    peaks = np.asarray(PEAKS, float)
    s = np.exp(-0.5 * ((BANDS[None, :] - peaks[:, None]) / SIGMA) ** 2)
    return s / np.linalg.norm(s, axis=1, keepdims=True)


@dataclass(frozen=True)
class FourDyes:
    """A lambda stack of four overlapping dyes on a haze of known extent."""

    shape: tuple[int, int] = SHAPE
    seed: int = SEED

    def _territories(self, rng) -> np.ndarray:
        """Which dye owns each pixel, and which pixels are nobody's.

        Smooth random fields, one per dye, each pixel going to the strongest --
        then the weakest `BACKGROUND_FRACTION` of those winning values is
        vacated. The fraction is *set* rather than hoped for, because it is the
        quantity the case scores.
        """
        fields = np.stack(
            [
                ndi.gaussian_filter(rng.normal(size=self.shape), sigma=BLOB_SIGMA)
                for _ in PEAKS
            ]
        )
        strength = fields.max(0)
        labels = np.zeros(self.shape, np.uint8)
        foreground = strength > np.quantile(strength, BACKGROUND_FRACTION)
        labels[foreground] = fields.argmax(0)[foreground] + 1
        return labels

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        labels = self._territories(rng)

        # Brightness varies *smoothly*, not per pixel. An independent draw per
        # pixel spans the same decade and closes the same back door, and it
        # renders as salt and pepper — which would make every look-at-it check
        # the run performs meaningless, and contradicts the one thing the
        # operator says they want out of it. The rank transform is what keeps
        # the distribution exactly identical across dyes while the *field* is
        # structured: without it, four smoothed fields have four different
        # empirical ranges and brightness starts carrying identity again.
        amplitude = np.zeros(self.shape)
        for dye in range(1, N_DYES + 1):
            here = labels == dye
            field = ndi.gaussian_filter(
                rng.normal(size=self.shape), sigma=BLOB_SIGMA / 2
            )
            values = field[here]
            uniform = np.empty(values.size)
            uniform[np.argsort(values)] = np.linspace(0.0, 1.0, values.size)
            amplitude[here] = PHOTONS * 10.0 ** (-BRIGHTNESS_DECADES * uniform)

        spectra = _spectra()
        cube = np.zeros((*self.shape, len(BANDS)))
        for dye in range(1, N_DYES + 1):
            here = labels == dye
            cube[here] = amplitude[here, None] * spectra[dye - 1][None, :]
        haze = np.ones(len(BANDS)) / np.linalg.norm(np.ones(len(BANDS)))
        cube += BACKGROUND_PHOTONS * haze[None, None, :]
        stack = rng.poisson(np.maximum(cube, 0)).astype(np.float32)
        # Bands first: a lambda stack is scrolled band by band in the viewer,
        # and (C, Y, X) is the order the rest of the tree presents a channel
        # axis in.
        stack = np.moveaxis(stack, -1, 0)

        background = float((labels == 0).mean())
        totals = stack.sum(axis=0)
        per_dye = np.array(
            [float(totals[labels == d].mean()) for d in range(1, N_DYES + 1)]
        )
        cosine = (spectra @ spectra.T)[np.triu_indices(N_DYES, 1)]

        # The properties the case rests on, checked before anyone pays for a
        # run. None of them is visible from the array alone.
        assert abs(background - BACKGROUND_FRACTION) < 0.01, (
            f"background is {background:.3f} of the field, not "
            f"{BACKGROUND_FRACTION} -- the quantity this case scores has "
            "drifted from the quantity it declares"
        )
        assert per_dye.max() / per_dye.min() < 1.15, (
            f"mean signal per dye spans {per_dye.max() / per_dye.min():.2f}x, "
            "so brightness carries dye identity and an intensity-only route is "
            "a back door (§11 -- this is LUMoS's own first-fixture failure)"
        )
        assert cosine.min() > 0.4, (
            f"the least similar dye pair is at cosine {cosine.min():.2f}; below "
            "~0.4 the dyes are separable enough that no unmixing is needed"
        )
        for dye in range(1, N_DYES + 1):
            assert (labels == dye).mean() > 0.05, (
                f"dye {dye} covers {(labels == dye).mean():.3f} of the field, "
                "too little for its share of the accuracy to mean anything"
            )

        return Fixture(
            provenance=(
                f"procedural: {self.shape[0]}x{self.shape[1]} over {len(BANDS)} "
                f"bands {BANDS[0]:.0f}-{BANDS[-1]:.0f} nm, four dyes peaking at "
                f"{'/'.join(f'{p:.0f}' for p in PEAKS)} nm with sigma {SIGMA:.0f} "
                f"nm (pairwise cosine {cosine.min():.2f}-{cosine.max():.2f}), "
                f"within-dye brightness log-uniform over "
                f"{BRIGHTNESS_DECADES:g} decade(s) and identical across dyes, "
                f"{background:.1%} background, Poisson, seed {self.seed}"
            ),
            about=(
                "Four spectrally overlapping dyes and a dim flat haze covering "
                f"{background:.1%} of the field. Normalising each pixel's "
                "spectrum before clustering is the step the candidate was "
                "nominated for and the one the model already takes; what "
                "carries the outcome is where background is cut. A cut on "
                "linear intensity looks defensible, calls 73.2% of the field "
                "background, and costs the run a third of its accuracy -- and "
                "nothing computed on the pixels it kept can see that it "
                "happened."
            ),
            data={"lambda_stack": stack},
            truth={
                "dye_map": labels,
                "background_fraction": np.array(background, float),
                "dye_spectra": spectra,
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _accuracy(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, np.ndarray]:
    """Whole-field accuracy with the four dye labels matched optimally.

    Background is **not** permutable -- the task names it 0, so it is a class
    rather than a cluster id, and calling a dye pixel background is an error
    like any other. That is the whole reason this is scored over the field
    rather than over the pixels the run chose to keep: an accuracy computed on
    retained pixels is exactly the blind spot the case exists to measure.
    """
    predicted = np.asarray(predicted).ravel()
    truth = np.asarray(truth).ravel()
    overlap = np.zeros((N_DYES, N_DYES))
    for p in range(N_DYES):
        for t in range(N_DYES):
            overlap[p, t] = np.sum((predicted == p + 1) & (truth == t + 1))
    rows, cols = linear_sum_assignment(-overlap)
    hits = overlap[rows, cols].sum() + np.sum((predicted == 0) & (truth == 0))
    return float(hits / truth.size), np.stack([rows + 1, cols + 1])


def _read_label_map(attempt: Attempt, shape: tuple[int, int]):
    """``(labels, why not)``. Strict about the convention the task states.

    A name bound to the right shape but the wrong vocabulary -- five clusters,
    or background numbered last -- has not delivered the thing that was asked
    for, and scoring it anyway would be scoring a different task.
    """
    got = attempt.arrays.get("dye_labels")
    if got is None:
        return None, "the run left no `dye_labels`"
    got = np.asarray(got)
    if got.shape != shape:
        return None, f"the run's `dye_labels` is {got.shape}, not {shape}"
    if not np.all(np.isfinite(got.astype(float))):
        return None, "`dye_labels` holds non-finite values"
    rounded = np.rint(got.astype(float))
    if not np.allclose(rounded, got.astype(float)):
        return None, "`dye_labels` is not integer-valued"
    present = set(np.unique(rounded).astype(int).tolist())
    allowed = set(range(N_DYES + 1))
    if not present <= allowed:
        return None, (
            f"`dye_labels` holds {sorted(present)}, and the task asks for "
            f"0 (background) and 1-{N_DYES}"
        )
    return rounded.astype(int), ""


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = np.asarray(fixture.truth["dye_map"])
    want_background = float(np.asarray(fixture.truth["background_fraction"]))

    labels, why = _read_label_map(attempt, truth.shape)
    if labels is None:
        return Outcome(
            fixture,
            attempt,
            [
                Metric("dye_error", None, limits["dye_error"], unavailable=why),
                Metric(
                    "background_error",
                    None,
                    limits["background_error"],
                    unavailable=why,
                ),
            ],
        )

    accuracy, assignment = _accuracy(labels, truth)
    called_background = float((labels == 0).mean())

    return Outcome(
        fixture,
        attempt,
        [
            Metric("dye_error", 1.0 - accuracy, limits["dye_error"]),
            Metric(
                "background_error",
                abs(called_background - want_background),
                limits["background_error"],
            ),
        ],
        detail={
            "dye_accuracy": round(accuracy, 4),
            "background_called": round(called_background, 4),
            "background_true": round(want_background, 4),
            "label_assignment": assignment.tolist(),
            "dye_area_called": [
                round(float((labels == d).mean()), 4) for d in range(1, N_DYES + 1)
            ],
        },
    )


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """The picture half: where the field went, not just how much of it. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    stack = fixture.data.get("lambda_stack")
    truth = fixture.truth.get("dye_map")
    if stack is None or truth is None:
        return
    stack = np.asarray(stack, float)
    truth = np.asarray(truth)

    save_png(stack.sum(axis=0), where / "band-sum.png")
    save_png(truth.astype(float), where / "dye-truth.png", vmax=float(N_DYES))

    labels = attempt.arrays.get("dye_labels")
    if labels is None:
        return
    labels = np.asarray(labels)
    if labels.shape != truth.shape:
        return
    labels = np.rint(labels.astype(float)).astype(int)
    save_png(labels.astype(float), where / "dye-called.png", vmax=float(N_DYES))
    # Where the run called background and the truth did not, and vice versa --
    # the map the run could not draw for itself.
    disagreement = np.zeros(truth.shape, float)
    disagreement[(labels == 0) & (truth != 0)] = 1.0
    disagreement[(labels != 0) & (truth == 0)] = 2.0
    save_png(disagreement, where / "background-disagreement.png", vmax=2.0)

    rows = ["class,true_fraction,called_fraction"]
    rows.append(
        f"background,{float((truth == 0).mean()):.4f},{float((labels == 0).mean()):.4f}"
    )
    for dye in range(1, N_DYES + 1):
        rows.append(
            f"dye_{dye},{float((truth == dye).mean()):.4f},"
            f"{float((labels == dye).mean()):.4f}"
        )
    (where / "class-areas.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = f"""\
A spectral image of one field is open in the napari viewer as the layer
`lambda_stack`, with axes (band, y, x) = ({len(BANDS)}, {SHAPE[0]}, {SHAPE[1]}),
float32. The {len(BANDS)} bands are evenly spaced detector windows from
{BANDS[0]:.0f} to {BANDS[-1]:.0f} nm. There is no tensor server in this session,
so that layer is the only source of pixels.

The sample was labelled with {N_DYES} dyes and imaged in one pass. Their
emission spectra overlap, and no single band belongs to any one dye. Not all of
the field is labelled.

I need to know which dye is where.

The person who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave one name bound in the kernel namespace:

  dye_labels   an ({SHAPE[0]}, {SHAPE[1]}) integer array, one entry per pixel:
               0 where the pixel is unlabelled, and 1 to {N_DYES} for the
               {N_DYES} dyes. Which dye gets which of the numbers 1 to {N_DYES}
               does not matter and is not scored -- only that pixels of one dye
               share a number and pixels of different dyes do not.

That name is how your result is collected, so bind it exactly.
"""

#: **This persona holds the safeguard**, and it is the reason the case is not a
#: lottery over the auto-threshold menu. Two facts -- roughly how much of the
#: frame has cells on it, and roughly how many patches they form -- are exactly
#: what a person who looked down a microscope knows, and between them they pin
#: both quantities the verifier scores: coverage validates the background
#: fraction, and the patch count catches the fragmentation a bad cut produces.
#:
#: Neither is precise, and neither should be: an operator's "more than half,
#: call it two thirds" is worth ~10 points of coverage, which is inside the
#: 0.32-0.48 pass band and nowhere near an answer key. What it converts is the
#: *kind* of run that passes -- from one that drew well off the menu to one that
#: asked, or looked, and had something to check against.
#:
#: It is deliberately **obtainable from the pixels too** (patches can be
#: counted, coverage measured), which the README's "what a fixture has to
#: withhold" would call a weak withholding. That is the right trade here: the
#: point is not to make the fact unobtainable, it is to make a defensible cut
#: *reachable by asking* as well as by looking, so that a failure means the run
#: did neither.
MICROSCOPIST = Persona(
    name="operator-lambda-scan",
    facts={
        "what the sample is": (
            "a fixed cell monolayer we stain with several markers at once, "
            "because there is not enough material to split across slides"
        ),
        "how it was imaged": (
            "one pass on a confocal with the detector split into ten windows, "
            "so every window sees a bit of everything"
        ),
        "why so many stains at once": (
            "we are looking at where four structures sit relative to each "
            "other in the same cell, so they have to be on the same slide"
        ),
        "whether the stains were imaged separately": (
            "no, and we cannot go back and do it -- the sample is used up. "
            "This one pass is what there is"
        ),
        "what is in the empty-looking parts": (
            "the coverslip and the medium between cells, which glow a little "
            "on their own. Nothing we stained is there"
        ),
        "how much of the frame has cells on it": (
            "getting confluent but not there yet -- a bit more than half the "
            "frame has cells on it, not much more. I did not measure it, that "
            "is just what it looks like down the eyepiece"
        ),
        "how the cells sit in the frame": (
            "they grow in patches that run into each other, so it is ten or so "
            "big irregular areas rather than hundreds of separate specks"
        ),
        "how confident I am about the stains": (
            "confident they all went on. Whether they all worked equally well "
            "on this slide I could not tell you"
        ),
        "what the result is for": (
            "a figure showing the four structures in one field, so it has to "
            "look like the cells rather than like a mosaic"
        ),
    },
    background=(
        "You stain fixed cells with several markers at once and image them in "
        "one pass on a confocal microscope. You are happy to answer questions "
        "about the sample, the stains and the microscope."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=MICROSCOPIST,
    fixture=Procedural(FourDyes()),
    layers=(Layer("lambda_stack", "lambda_stack"),),
    collect={"dye_labels": "dye_labels"},
    score=verify,
    save_artifacts=save_artifacts,
    #: The safeguard, declared so a persona edit that drops it fails a test
    #: rather than quietly turning the case back into a coin flip.
    persona_must_know=(
        "a bit more than half the frame has cells on it",
        "ten or so big irregular areas",
    ),
    persona_must_not_know=(
        "normalis",
        "normaliz",
        "k-means",
        "kmeans",
        "cluster",
        "unmix",
        "spectral angle",
        "cosine",
        "otsu",
    ),
)
