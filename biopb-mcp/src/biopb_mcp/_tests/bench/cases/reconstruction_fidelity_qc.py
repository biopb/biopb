"""Seven reconstructions of one field, no ground truth: which can be trusted?

**Scope note.** This case scores whether a run compared each reconstruction
against the data *at a matched resolution*. It does **not** ask for SQUIRREL,
or for a fitted resolution-scaling function, or for an accurate PSF: a box blur
of roughly the right width scores as well as a Gaussian, skipping the rebin
costs nothing, RSE and RSP give the same ranking, and any assumed PSF between
0.6x and 4x the true one passes. All of that is measured below. What fails is
correlating without matching resolution at all.

A deferred-tier case (`docs/skill-candidates.md`). NanoJ-SQUIRREL was
**prescreened and dropped** 2026-08-06 as tier-conditional: all six arms found
the widefield comparison unaided, so the *idea* is not the hard part. What
separated the tiers was a single step -- blur to the camera's resolution before
correlating -- worth +0.14 of Spearman, "a one-line caveat, not a procedure".
It is here anyway, because a one-line caveat that half the arms missed is
exactly what a benchmark can measure and a skill cannot usefully carry.

**The prescreen's two keepsakes are what this case is built around.** Apparent
sharpness is *anti*-correlated with fidelity (-0.643 there, -0.321 here): the
reconstructions that look most resolved are among the least faithful, so "it
looks sharper" inverts the answer. And **odd-order SOFI cumulants are
legitimately signed** -- two of six arms demoted a reconstruction as "corrupted"
for having 7.5% negative pixels, penalising a method for a property inherent to
it, and that single error was the whole of one arm's failure.

Measured on this fixture, seed 3, Spearman of each route against the true
fidelity ordering:

  ==============================================  ==========  =========
  route                                             spearman      error
  ==============================================  ==========  =========
  REFERENCE -- blur to camera resolution, rebin        0.929      0.071
  the same, without the rebin                          0.929      0.071
  a box blur of roughly the right width                0.893      0.107
  scored by RSE instead of RSP                         0.929      0.071
  TRIVIALITY -- rebin and correlate, no blur           0.714      0.286
  TRIVIALITY -- upsample the widefield, no blur        0.607      0.393
  THE TRAP -- rank by apparent sharpness              -0.321      1.321
  BACK DOOR -- consensus of the seven                  0.714      0.286
  BACK DOOR -- best per-image statistic, oracle sign   0.408      0.592
  alphabetical order                                   0.036      0.964
  ==============================================  ==========  =========

Every route in that table is recomputed against the shipped fixture in
`test_verifiers.py` and its verdict asserted there, so no row can quietly
change sides. The Spearman values themselves are recorded here rather than
asserted -- pinning them would fail on any harmless change to the fixture,
which is the wrong thing to defend.

`TOLERANCE` reads off that table. With seven items Spearman takes only the
values ``1 - k/56`` where ``k = sum d^2`` over the rank displacements, so
`ranking_error` is `k/56` for an even integer k, and one adjacent transposition
costs k=2. The reference route scores **k=4**, the un-blurred shortcut
**k=16**, and the limit sits at **k=8** -- the geometric midpoint, two adjacent
swaps above a clean run and four below the shortcut. That placement is
deliberate: the prescreen's own threshold caveat is that one adjacent swap
moves Spearman by 0.036, so a mark sitting one swap from anything is measuring
the draw rather than the work. Every one of the three disjoint adjacent swaps
that fit in seven items passes; displacing a single layer by three places does
not.

**Both §11 screens were run before this case was written, and both moved it.**

*Back door.* The first build made the residual-background `leak` monotone in
fidelity, which was tidier and completely broken: a leak is low-frequency
power, so the high-frequency power fraction -- one number per image, no
comparison with anything -- reproduced the answer key at |rho| 0.964 on two of
six seeds. Assigning leak **orthogonally** to the fidelity ordering (0.12,
0.04, 0.26, 0.06, 0.30, 0.08, 0.28 down the ranks) drops the best per-image
statistic to 0.408 while leaving its power to mislead the un-blurred route
intact. A confound has to be a confound, not a second copy of the answer.

*Triviality.* Ranking the layers in the order they are presented scores 0.036.
The letters were permuted against the fidelity order for exactly this reason.

**Three earlier fixtures failed for reasons worth recording, because each was a
plausible design.**

*Fidelity separated by resolution is unwinnable by construction.* The first
build gave the seven different nominal resolutions. But the reference route
*deliberately blurs resolution away* -- that is the whole step being measured --
so it cannot recover an ordering that resolution produced. The reference scored
0.357. All seven now sit at 47-57 nm and are separated by **artifact content**;
resolution is only the confound.

*Intensity nonlinearity separated them the wrong way.* Squaring and cubing (a
literal SOFI-2 and SOFI-3) move Pearson-against-truth and
Pearson-against-widefield in opposite directions, so the reference tracked the
truth at 0.643. The artifacts are now structural -- missing filaments, ghosted
ridges, halos, invented filaments, residual background -- and linear in
intensity.

*Adjacent ranks were tied.* Hand-tuned amplitudes left the smallest fidelity gap
at ~0.000, and a tie caps the reference route however good the route is, since
no comparison can recover an ordering that is noise. Each artifact amplitude is
now **solved** by bisection onto an evenly spaced fidelity target (0.6922 down
to 0.2722 in steps of 0.07), so the ordering is set rather than hoped for --
the same discipline `lumos-spectral-unmixing` gives its background fraction.

**A strongly signed reconstruction cannot be ranked by this route, and that
bounded the trap.** The widefield has no high frequencies, so high-passing a
reconstruction destroys the comparison: a global high-pass strong enough to
make negatives obvious dropped the reference route to 0.786. The signed
reconstruction therefore carries a *local* negative ring around structure
(0.15x a difference-of-Gaussians shell), which costs the reference route almost
nothing -- 0.968 against a clean 0.972 -- and a pedestal chosen to put the
negative fraction at exactly **8.0%**, the prescreen's own figure. Pearson is
affine invariant, so that pedestal moves the negative fraction and nothing
else: it sets the trap without touching any ranking.

**The signed-reconstruction error alone is fatal, which is the point.** Moving
that one layer from rank 2 to last -- changing nothing else -- costs
``sum d^2 = 30``, i.e. Spearman 0.464 and `ranking_error` 0.536, against a
limit of 0.143. That reproduces the prescreen's finding that one such error was
the whole of an arm's failure, and it is asserted in `test_verifiers.py` rather
than left as a claim.

**So the persona holds the safeguard, and it is three facts a person at that
microscope would have**: what the beads measure (which pins the PSF), what the
camera pixel is (which pins the grids), and that a couple of these methods
produce signed output on purpose. None of them is the procedure; all of them
are the environment, which is the §6 split. A run that asks can match resolution
and will not throw away a good reconstruction for having negative pixels. A run
that asks nothing is on the shortcut at 0.714 and fails.

**What a failure here does and does not mean.** It does not mean the run could
not implement SQUIRREL -- nothing here needs SQUIRREL. It means the run
compared a 50 nm image against a 280 nm image and read the result as fidelity,
or discarded a reconstruction for a property its method is defined by. Both are
visible in the transcript, and both should be read there rather than inferred
from the number alone.
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

NAMESPACE = "reconstruction-fidelity-qc"
CASE_ID = "seven-reconstructions-one-widefield"

#: Read off the table in the module docstring, not off taste. With seven items
#: Spearman only takes the values ``1 - k/56``; this is k=8, four adjacent
#: swaps above a clean run and eight below the un-blurred shortcut.
#:
#: There is one metric and not two. A ranking that is not a ranking -- six
#: names, a repeat, a layer that does not exist -- is reported as **unscorable**
#: rather than as a second 0/1 metric, because `Outcome.passed` already refuses
#: to read "nothing scored" as a pass and a metric whose limit is zero is a
#: silent always-fail. (`align-channels-from-landmarks` carries exactly such a
#: metric; it is never caught because that case is `OnDisk` and skips.)
TOLERANCE = {
    "ranking_error": 0.143,
}

#: The reconstruction grid, and the camera grid it was reconstructed from.
FINE = 512
BIN = 4
CAM = FINE // BIN
FINE_NM = 25.0
CAM_NM = FINE_NM * BIN

#: What beads measure on this scope. The persona holds this number; it is the
#: environment, not the procedure.
PSF_FWHM_NM = 280.0
PSF_SIGMA_FINE = PSF_FWHM_NM / 2.3548 / FINE_NM

#: The sample: filaments about this wide, which is also the finest thing in it.
#: A reconstruction showing structure below this is inventing it.
FILAMENT_FWHM_NM = 60.0
FIL_SIGMA_FINE = FILAMENT_FWHM_NM / 2.3548 / FINE_NM
N_FILAMENTS = 26

#: How diffuse the un-removed background is, in fine pixels.
LEAK_SIGMA = 9.0
#: Ghost displacement, fine pixels. Deliberately WIDER than the PSF: an
#: artifact finer than the camera's own resolution is erased by the very blur
#: the reference route applies, so it could not be what carries the ranking.
GHOST_SHIFT = 13
#: The negative fraction the signed reconstruction is pedestalled to -- the
#: prescreen's own figure for the SOFI-3 that two arms discarded.
NEG_FRACTION = 0.08
SIGNED_RING = 0.15

SEED = 3

#: ``internal name -> (recon sigma, leak, artifact kind, artifact amplitude)``.
#:
#: The amplitudes are **solved**, not chosen: each was bisected until that
#: reconstruction's Pearson correlation with the true structure landed on an
#: evenly spaced target, 0.6922 down to 0.2722 in steps of 0.07. See the
#: docstring for why an eyeballed amplitude is not good enough here.
#:
#: `leak` -- a residual diffuse background the reconstruction failed to remove
#: -- is the confound the case turns on: it makes a reconstruction look more
#: like the widefield without making it more faithful, so an un-blurred
#: comparison rewards it. It is assigned ORTHOGONALLY to the fidelity ordering.
BANK = {
    #                   sigma  leak  artifact  amplitude
    "deconv_good": (2.0, 0.12, "none", 0.0000),
    "sofi3_signed": (2.4, 0.04, "dim", 0.5796),
    "sofi2": (2.2, 0.26, "dim", 0.8868),
    "radiality_mean": (2.2, 0.06, "ghost", 0.8550),
    "deconv_halo": (2.0, 0.30, "halo", 3.5100),
    "radiality": (2.2, 0.08, "ghost", 1.5976),
    "oversharp": (2.0, 0.28, "fake", 2.8803),
}

#: Each reconstruction lands on its own arbitrary scale, because real ones do.
#: Pearson is affine invariant, which is the prescreen's finding that
#: SQUIRREL's alpha and beta are a mathematical no-op for RSP.
AFFINE = {
    "deconv_good": (2.4, 0.8),
    "sofi3_signed": (0.7, 0.0),
    "sofi2": (55.0, 40.0),
    "radiality_mean": (8.0, 5.0),
    "deconv_halo": (1.0, 0.4),
    "radiality": (140.0, 60.0),
    "oversharp": (0.35, 0.1),
}

#: The names the agent sees. Anonymised so no run can rank by which package a
#: reconstruction came from, and **permuted against the fidelity order** so
#: that presenting them in order is worth nothing: alphabetical scores 0.036.
#: The two that share the ghost mechanism (`radiality`, `radiality_mean`) get
#: non-adjacent letters for the same reason.
LAYER_NAMES = {
    "sofi2": "method_A",
    "radiality": "method_B",
    "deconv_good": "method_C",
    "oversharp": "method_D",
    "radiality_mean": "method_E",
    "sofi3_signed": "method_F",
    "deconv_halo": "method_G",
}
WIDEFIELD = "widefield"


def _filaments(rng: np.random.Generator, count: int, weight: float) -> np.ndarray:
    """Rasterise `count` smoothed random walks onto the fine grid."""
    img = np.zeros((FINE, FINE))
    for _ in range(count):
        angle = ndi.gaussian_filter1d(
            np.cumsum(rng.normal(scale=0.16, size=400)), 12.0, mode="nearest"
        )
        step = FINE / 90.0
        x = np.cumsum(np.cos(angle)) * step + rng.uniform(0, FINE)
        y = np.cumsum(np.sin(angle)) * step + rng.uniform(0, FINE)
        keep = (x >= 0) & (x < FINE) & (y >= 0) & (y < FINE)
        if keep.any():
            np.add.at(
                img,
                (
                    np.clip(np.rint(y[keep]).astype(int), 0, FINE - 1),
                    np.clip(np.rint(x[keep]).astype(int), 0, FINE - 1),
                ),
                weight,
            )
    return img


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


@dataclass(frozen=True)
class SevenReconstructions:
    """One field, its diffraction-limited average, and seven attempts at it."""

    seed: int = SEED

    def _sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """The specimen, split bright/faint.

        Split because "the dim structure did not survive" has to be a whole
        filament missing rather than an intensity curve -- a structural
        artifact, so that the un-blurred and blurred routes see it the same way.
        """
        bright = _filaments(rng, N_FILAMENTS // 2, 1.0)
        faint = _filaments(rng, N_FILAMENTS - N_FILAMENTS // 2, 0.35)
        for img, count in ((bright, 26), (faint, 14)):
            for _ in range(count):
                y, x = rng.integers(0, FINE, 2)
                img[y, x] += 4.0
        bright = ndi.gaussian_filter(bright, FIL_SIGMA_FINE)
        faint = ndi.gaussian_filter(faint, FIL_SIGMA_FINE)
        scale = np.quantile(bright + faint, 0.9995)
        return np.clip(bright / scale, 0, 1.0), np.clip(faint / scale, 0, 1.0)

    def _one(self, name, bright, faint, leak, fake, rng) -> np.ndarray:
        sigma, leak_amp, kind, amp = BANK[name]
        dim_keep = 1.0 - amp if kind == "dim" else (0.0 if kind == "fake" else 1.0)
        r = ndi.gaussian_filter(bright + dim_keep * faint, sigma)
        r = r / r.max()
        if kind == "ghost":
            shifted = np.roll(np.roll(r, GHOST_SHIFT, 0), GHOST_SHIFT - 2, 1) + np.roll(
                np.roll(r, -GHOST_SHIFT + 1, 0), GHOST_SHIFT, 1
            )
            r = r + amp * shifted / shifted.max()
        elif kind == "halo":
            r = r + amp * np.clip(ndi.gaussian_filter(r, 9.0) - 0.35 * r, 0, None)
        elif kind == "fake":
            r = r + amp * fake
        if leak_amp:
            r = r + leak_amp * leak
        r = r / r.max() + rng.normal(scale=0.002, size=r.shape)

        if name == "sofi3_signed":
            # A LOCAL negative ring, not a global high-pass. The widefield has
            # no high frequencies, so high-passing a reconstruction destroys
            # the very comparison the reference route makes -- measured, a
            # global high-pass strong enough to make negatives obvious drops
            # that route from 0.929 to 0.786.
            ring = np.clip(
                ndi.gaussian_filter(r, 5.0) - ndi.gaussian_filter(r, 2.0), 0, None
            )
            r = r - SIGNED_RING * ring / ring.max()
            # Pearson is affine invariant, so this constant moves the negative
            # FRACTION and nothing else: it sets the trap at the prescreen's
            # 8% without touching any ranking.
            low, high = -1.0, 1.0
            for _ in range(40):
                mid = 0.5 * (low + high)
                if ((r + mid) < 0).mean() > NEG_FRACTION:
                    low = mid
                else:
                    high = mid
            r = (r + 0.5 * (low + high)) / np.abs(r + 0.5 * (low + high)).max()
        else:
            r = np.clip(r, 0, None)
            r = r / r.max()

        gain, offset = AFFINE[name]
        return (gain * r + offset).astype(np.float32)

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        bright, faint = self._sample(rng)
        structure = bright + faint

        # What the camera actually recorded: the PSF, then camera pixels, then
        # shot noise. This is the time average of the raw stack -- the
        # diffraction-limited reference, which was there all along.
        blurred = ndi.gaussian_filter(structure, PSF_SIGMA_FINE)
        wf = blurred.reshape(CAM, BIN, CAM, BIN).mean((1, 3))
        wf = wf / wf.max() * 900.0 + 40.0
        widefield = (
            np.random.default_rng(self.seed + 900).poisson(wf).astype(np.float32)
        )

        leak = ndi.gaussian_filter(structure, LEAK_SIGMA)
        leak = leak / leak.max()
        fake = ndi.gaussian_filter(
            _filaments(np.random.default_rng(self.seed + 100), 14, 1.0), FIL_SIGMA_FINE
        )
        fake = fake / fake.max()

        recons, fidelity = {}, {}
        for name in BANK:
            arr = self._one(
                name,
                bright,
                faint,
                leak,
                fake,
                np.random.default_rng(self.seed + 100),
            )
            recons[LAYER_NAMES[name]] = arr
            fidelity[LAYER_NAMES[name]] = _pearson(arr, structure)

        order = sorted(fidelity, key=lambda k: -fidelity[k])
        gaps = [fidelity[order[i]] - fidelity[order[i + 1]] for i in range(6)]
        expected = [LAYER_NAMES[n] for n in BANK]

        # The properties the case rests on, checked before anyone pays for a
        # run. None of them is visible from the arrays alone.
        assert order == expected, (
            f"fidelity order is {order}, not the {expected} the solved "
            "amplitudes were bisected for -- the answer key has drifted from "
            "the construction that produced it"
        )
        assert min(gaps) > 0.04, (
            f"the closest pair of reconstructions differs by {min(gaps):.4f} in "
            "fidelity; below ~0.04 the ordering between them is noise and no "
            "route can recover it, which caps the reference route (see the "
            "docstring: an earlier build sat at 0.000 here)"
        )
        signed = recons[LAYER_NAMES["sofi3_signed"]]
        negative = float((signed < 0).mean())
        assert 0.05 < negative < 0.12, (
            f"the signed reconstruction is {negative:.1%} negative, and the "
            "trap this case carries is that ~8% reads as corruption to a run "
            "that does not ask"
        )
        for name, arr in recons.items():
            if name != LAYER_NAMES["sofi3_signed"]:
                assert arr.min() >= 0.0, (
                    f"{name} holds negative values, so 'which of these is "
                    "signed' stops being the single question it is meant to be"
                )

        return Fixture(
            provenance=(
                f"procedural: one {FINE}x{FINE} field of filaments "
                f"{FILAMENT_FWHM_NM:.0f} nm wide on a {FINE_NM:.0f} nm grid, "
                f"its {CAM}x{CAM} diffraction-limited average at "
                f"{CAM_NM:.0f} nm pixels through a {PSF_FWHM_NM:.0f} nm FWHM "
                "PSF with Poisson noise, and seven reconstructions whose "
                "artifact amplitudes were solved for evenly spaced true "
                f"fidelities {fidelity[order[0]]:.4f} down to "
                f"{fidelity[order[-1]]:.4f}; seed {self.seed}"
            ),
            about=(
                "Seven anonymised reconstructions of one field and the "
                "widefield average they were all built from. Ranking them by "
                "correlation with that average, without first matching "
                "resolution, scores 0.714 against the reference's 0.929 -- "
                "because a residual diffuse background makes a reconstruction "
                "look more like the widefield without making it more faithful. "
                "Apparent sharpness inverts the answer outright (-0.321), and "
                "one of the seven is signed by construction with 8% negative "
                "pixels, which is a property of its method and not damage."
            ),
            data={WIDEFIELD: widefield, **recons},
            truth={
                "structure": structure,
                "fidelity_order": np.array(order),
                "fidelity": np.array([fidelity[n] for n in order]),
            },
            tolerance=dict(TOLERANCE),
        )


# --- the verifier ----------------------------------------------------------


def _spearman(predicted: list[str], truth: list[str]) -> float:
    """Spearman's rho between two orderings of the same seven names.

    A ranking rather than a score is what the run is asked for, and that is a
    §8 choice: it is invariant to whatever scale the run measured fidelity on,
    so two runs that agree about the order agree here even if one reported
    correlations and the other reported errors.
    """
    n = len(truth)
    place = {name: i for i, name in enumerate(predicted)}
    d2 = sum((place[name] - i) ** 2 for i, name in enumerate(truth))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _read_ranking(attempt: Attempt, names: list[str]):
    """``(ranking, why not)``. Strict about the convention the task states.

    The harness scrapes through ``np.asarray``, so a list of seven names
    arrives as a ``<U`` array rather than a list. Anything that does not
    survive that -- a dict of scores, say -- never reaches here at all: it
    fails to save and is scraped as absent.
    """
    got = attempt.arrays.get("fidelity_ranking")
    if got is None:
        return None, "the run left no `fidelity_ranking`"
    got = np.asarray(got)
    if got.ndim != 1:
        return None, (
            f"`fidelity_ranking` has shape {got.shape}, and the task asks for a "
            "flat list of the seven layer names"
        )
    ranking = [str(x).strip() for x in got.tolist()]
    if len(ranking) != len(names):
        return None, (
            f"`fidelity_ranking` names {len(ranking)} reconstructions, and "
            f"there are {len(names)}"
        )
    if sorted(ranking) != sorted(names):
        unknown = sorted(set(ranking) - set(names))
        missing = sorted(set(names) - set(ranking))
        why = "`fidelity_ranking` is not a ranking of the seven layers"
        if unknown:
            why += f"; it names {unknown}, which are not layers"
        if missing:
            why += f"; it omits {missing}"
        if len(set(ranking)) != len(ranking):
            why += "; it repeats a name"
        return None, why
    return ranking, ""


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = [str(x) for x in np.asarray(fixture.truth["fidelity_order"]).tolist()]

    ranking, why = _read_ranking(attempt, truth)
    if ranking is None:
        return Outcome(
            fixture,
            attempt,
            [Metric("ranking_error", None, limits["ranking_error"], unavailable=why)],
        )

    rho = _spearman(ranking, truth)
    fidelity = np.asarray(fixture.truth["fidelity"], float)
    return Outcome(
        fixture,
        attempt,
        [Metric("ranking_error", 1.0 - rho, limits["ranking_error"])],
        detail={
            "spearman": round(rho, 4),
            "ranking_called": ranking,
            "ranking_true": truth,
            "true_fidelity": {
                n: round(float(f), 4) for n, f in zip(truth, fidelity, strict=False)
            },
            "displacement": {
                name: ranking.index(name) - i for i, name in enumerate(truth)
            },
        },
    )


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """What the run was looking at, and where its order went wrong. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    structure = fixture.truth.get("structure")
    widefield = fixture.data.get(WIDEFIELD)
    if structure is None or widefield is None:
        return
    save_png(np.asarray(structure, float), where / "true-structure.png")
    save_png(np.asarray(widefield, float), where / "widefield.png")
    for name in LAYER_NAMES.values():
        arr = fixture.data.get(name)
        if arr is not None:
            save_png(np.asarray(arr, float), where / f"{name.replace('_', '-')}.png")

    truth = [str(x) for x in np.asarray(fixture.truth["fidelity_order"]).tolist()]
    fidelity = np.asarray(fixture.truth["fidelity"], float)
    ranking, _ = _read_ranking(attempt, truth)
    rows = ["layer,true_rank,true_fidelity,called_rank"]
    for i, name in enumerate(truth):
        called = ranking.index(name) + 1 if ranking else ""
        rows.append(f"{name},{i + 1},{fidelity[i]:.4f},{called}")
    (where / "ranking.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = f"""\
Eight layers are open in the napari viewer.

Seven of them are `method_A`, `method_B`, `method_C`, `method_D`, `method_E`,
`method_F` and `method_G`. Each is a {FINE}x{FINE} float32 reconstruction of the
same field of view, produced by a different processing method from the same raw
acquisition. They are on a common pixel grid and are already aligned with each
other. They arrive on arbitrary and unrelated intensity scales.

The eighth, `{WIDEFIELD}`, is a {CAM}x{CAM} float32 image of that same field,
recorded on the camera's own pixel grid.

There is no tensor server in this session, so those layers are the only source
of pixels. There is no ground truth available for this field.

I need the seven ranked by how faithful each one is to the actual specimen --
most faithful first.

The person who acquired the data is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave one name bound in the kernel namespace:

  fidelity_ranking   a list of the seven layer names as strings, most faithful
                     first, least faithful last. Every one of the seven appears
                     exactly once, e.g.
                     ["method_C", "method_A", ..., "method_D"].

That name is how your result is collected, so bind it exactly.
"""

#: **This persona holds the safeguard**, in three facts that a person standing
#: at that microscope would have and the pixels do not carry:
#:
#: * what beads measure on the scope -- which pins the PSF, and so what
#:   "matched resolution" means. The prescreen found a *fitted* resolution
#:   scaling function and the *known optical PSF* rank all seven identically,
#:   so handing this over concedes nothing the case is measuring.
#: * the two pixel sizes -- which pin the relationship between the grids.
#: * that a couple of the methods emit signed output on purpose. This is the
#:   one that decides a whole pass: demoting the signed reconstruction from
#:   rank 2 to last costs `ranking_error` 0.536 against a limit of 0.143.
#:
#: None of them is the procedure. "Compare against the widefield", "match the
#: resolutions first" and "rank by correlation" are the work, and the persona
#: does not know any of it -- see `persona_must_not_know`.
#:
#: The operator also knows nothing is finer than 60 nm in this sample, which is
#: the fact that undercuts the sharpness trap without naming it: a
#: reconstruction resolving structure below that is inventing it.
MICROSCOPIST = Persona(
    name="operator-live-cell-smlm",
    facts={
        "what the sample is": (
            "a fixed cell, stained for the tubule network. It is the same "
            "field in every one of those images -- one acquisition, seven "
            "people's processing"
        ),
        "how fine the real structure is": (
            "the tubules are about 60 nm across. We had this sample under an "
            "electron microscope and there is nothing finer than that in it"
        ),
        "what the widefield layer is": (
            "the plain average of all the raw frames, straight off the camera "
            "with nothing done to it. It is just the ordinary view of the "
            "field, which is why it looks so much softer than the others"
        ),
        "what the pixel size is": (
            "the camera pixel is 100 nm at the sample. The seven are all on a "
            "grid four times finer than that, so 25 nm"
        ),
        "how good the optics are": (
            "1.35 NA silicone objective, we collect around 600 nm. When we put "
            "beads on this scope they come out about 280 nm across"
        ),
        "whether any of them produce negative values": (
            "yes, and it is not damage -- a couple of those methods produce "
            "signed output by design, so negative pixels are normal for them. "
            "I would not throw one out for that"
        ),
        "why they have no names": (
            "I stripped the names on purpose. I did not want to be swayed by "
            "which package a result came from, and neither should you"
        ),
        "whether any two came from related methods": (
            "possibly -- I did not keep track of who used what. Two of them "
            "may well share a step"
        ),
        "whether the raw frames are available": (
            "not in this session, no. The average is what I kept"
        ),
        "what the result is for": (
            "one of these goes in a figure and I have to be able to defend the "
            "choice to a referee. I would rather be told one is untrustworthy "
            "than have it quietly picked"
        ),
    },
    background=(
        "You image fixed cells on a super-resolution microscope and you have "
        "had one acquisition processed seven different ways. You are happy to "
        "answer questions about the sample, the microscope and the data."
    ),
)

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    persona=MICROSCOPIST,
    fixture=Procedural(SevenReconstructions()),
    layers=(
        Layer(WIDEFIELD, WIDEFIELD),
        *(Layer(name, name) for name in LAYER_NAMES.values()),
    ),
    collect={"fidelity_ranking": "fidelity_ranking"},
    score=verify,
    save_artifacts=save_artifacts,
    #: The safeguard, declared so a persona edit that drops it fails a test
    #: rather than quietly making the case unwinnable or turning the signed
    #: reconstruction into an unavoidable trap.
    persona_must_know=(
        "about 280 nm across",
        "the camera pixel is 100 nm",
        "about 60 nm across",
        "signed output by design",
    ),
    persona_must_not_know=(
        "correlat",
        "pearson",
        "spearman",
        "squirrel",
        "resolution scaling",
        "convolv",
        "downsampl",
        "rebin",
        "sharpness",
        "unsharp",
        "point spread",
    ),
)
