"""Align two channels from clicked correspondences, on a warp no affine follows.

**The procedural sibling of `align-channels-from-landmarks`.** That case is the
same measurement on real data (`OnDisk`), and it skips wherever
``$BIOPB_FIXTURES`` is not set -- which is everywhere, so the subject has never
actually been exercised. This one builds its own field and runs anywhere. The
two share a verifier deliberately: same question, same limits, different
provenance, which is what `_case.py` means by covering one subject both ways
being two cases rather than one case with a switch.

**The prescreen said a synthetic fixture could not do this, and that was true of
the fixture it had rather than of synthetic data.**
`docs/skill-candidates.md` records: *"The first fixture was synthetic and did
not discriminate -- affine, TPS and reference landed within ~1 px of each other
-- which is why the real-data version exists."* Reproduced here, that failure
is a property of the *deformation*, not of the synthesis: a non-affine term
whose spatial wavelengths sit at or below the landmark spacing is invisible to
a spline fitted through those landmarks, so the spline has nothing to win and
lands beside the affine. Measured on a first build with wavelengths of
760/330/190 px against a ~230 px landmark spacing, affine 9.85 px and TPS
6.29 px -- a ratio of 1.57, and exactly the "~1 px of each other" the record
describes.

Lengthening the deformation's correlation length past the landmark spacing
separates them at once. This fixture uses a smooth random displacement field
(correlation length ~190 px in the field, RMS 28 px) rather than a sum of
sinusoids, because a sinusoid sum is a closed-form basis a run could identify
and fit directly, and a smoothed random field is both harder to shortcut and
closer to what a mismatched optical path does.

Measured on this fixture, median error over 400 probes:

  ====================================  ==========  =========
  route                                  this case     record
  ====================================  ==========  =========
  no-op (the channels as they arrive)        54.57      53.54
  global affine through the 18 clicks        15.33      17.47
  second-order polynomial                     9.70         --
  REFERENCE -- thin-plate spline              3.03       3.00
  affine / spline ratio                       5.06x      5.82x
  probes inside the landmark hull               82%        84%
  ====================================  ==========  =========

So the shape the real data was needed for is reproduced within a pixel on every
row that both fixtures have. `ERROR_LIMIT_PX` is 8.0, the curated case's own
limit, and it sits between the affine at 15.33 and the spline at 3.03 -- twice
the reference and half the shortcut.

**No route here is an oracle (§7).** The displacement field carries structure
below the landmark spacing that no spline through 18 points can represent, and
the clicks carry 1.5 px of noise, so the reference lands at 3.03 px rather than
at zero. A second-order polynomial reaches 9.70 px: better than the affine and
still failing, which is the right verdict for a model that is *more* than
affine but still global.

**The §11 back door here is the images, and it is closed by construction.** If
`fixed` were `moving` pushed through the truth map, intensity registration
would recover the answer exactly and the clicked points would be decoration.
So each channel is rendered independently in its **own** coordinate system from
the same cell positions -- nuclei as filled blobs in `moving`, cytoplasm as
rings plus a texture that exists only in that channel -- and they share
positions and nothing else. Measured cross-channel correlation is **0.038**
(the real pair is 0.162, and lower is further from a back door, not closer).
Phase correlation, the cheapest intensity route, returns a shift of
(-391, -246) px and a median probe error of **503 px** against a limit of 8.

**What is being measured is a tier gap, not a skill.** The prescreen dropped
this subject because two of its three claims were not gaps at all: zero of
eight arms quoted a fitting residual as their error estimate, and four of four
cold arms flagged clustered landmark placement unprompted. What remained was
that both Haiku arms fitted an affine at *both* landmark budgets -- never
making the decision the entry named -- and paid 5.8x on the set that supported
a spline. That is what this case reproduces, and `quality_honesty` is kept for
the reason the curated case gives: a result worth being able to re-measure
rather than assume keeps holding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ...agentbench._fixture import Fixture, Procedural
from .._case import Case, Layer
from .align_channels_from_landmarks import (
    ERROR_LIMIT_PX,
    HONESTY_LIMIT,
    MICROSCOPIST,
    _save_artifacts,
    _verify,
)

NAMESPACE = "landmark-registration"
CASE_ID = "a-warp-no-global-affine-can-follow"

FIELD = 960
N_NUCLEI = 124
N_PROBES = 400
N_LANDMARKS = 18

#: Clicking by eye is good to a pixel or two. This is why the reference lands
#: at 3 px rather than at zero, and why `quality_honesty` has a 1 px floor.
CLICK_NOISE_PX = 1.5

#: The affine half of the map: "the two channels went through different optical
#: paths". An affine fit absorbs this exactly, so its size sets how far apart
#: the channels *look* without changing what any route scores.
ROTATION_DEG = 5.2
SCALE = 1.02
SHEAR = 0.012
TRANSLATION = (34.0, -24.0)

#: The non-affine half, as a smooth random displacement field: Gaussian
#: smoothing on a `WARP_GRID` lattice, normalised to `WARP_RMS_PX`.
#:
#: `WARP_SIGMA` is the whole ballgame. In field pixels the correlation length is
#: ``WARP_SIGMA * FIELD / WARP_GRID`` = ~190 px, against a landmark spacing of
#: ~230 px for 18 points over 960. Shorter than that and a spline through the
#: landmarks cannot see the warp either, which is the failure the prescreen
#: recorded and this docstring reproduces.
WARP_GRID = 192
WARP_SIGMA = 38.0
WARP_RMS_PX = 28.0

SEED = 29


def _affine_matrix() -> np.ndarray:
    angle = np.deg2rad(ROTATION_DEG)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    shear = np.array([[1.0, SHEAR], [0.0, 1.0]])
    return SCALE * rotation @ shear


@dataclass(frozen=True)
class ClickedLandmarks:
    """One field in two channels, 18 clicked pairs, and 400 withheld probes."""

    seed: int = SEED

    def _displacement(self) -> np.ndarray:
        """The non-affine field, ``(2, WARP_GRID, WARP_GRID)`` in pixels."""
        rng = np.random.default_rng(self.seed + 77)
        raw = rng.normal(size=(2, WARP_GRID, WARP_GRID))
        smooth = np.stack(
            [ndi.gaussian_filter(c, WARP_SIGMA, mode="wrap") for c in raw]
        )
        smooth /= np.sqrt((smooth**2).sum(0).mean())
        return smooth * WARP_RMS_PX

    def _deform(self, field: np.ndarray):
        """``moving -> fixed``. Affine plus the smooth field, sampled bilinearly."""
        scale = WARP_GRID / FIELD
        linear = _affine_matrix().T
        offset = np.array(TRANSLATION)

        def go(points: np.ndarray) -> np.ndarray:
            points = np.asarray(points, float)
            coords = np.clip(points.T * scale, 0, WARP_GRID - 1)
            shift = np.stack(
                [
                    ndi.map_coordinates(c, coords, order=1, mode="nearest")
                    for c in field
                ],
                axis=1,
            )
            return points @ linear + offset + shift

        return go

    def _spread_landmarks(self, nuclei: np.ndarray) -> np.ndarray:
        """Farthest-point sampling: the record's "18 spread" set, which is the
        budget that supports a spline. Its clustered counterpart (6 clicks, 6%
        of probes inside the hull) is a different case, not a switch on this
        one."""
        chosen = [int(np.argmin(np.linalg.norm(nuclei - FIELD / 2, axis=1)))]
        for _ in range(N_LANDMARKS - 1):
            gaps = np.min(
                np.linalg.norm(nuclei[:, None, :] - nuclei[chosen][None, :, :], axis=2),
                axis=1,
            )
            chosen.append(int(np.argmax(gaps)))
        return np.array(chosen)

    def _render(self, centres: np.ndarray, rng, cytoplasm: bool) -> np.ndarray:
        """One channel, rendered in its OWN coordinates.

        Never a warped copy of the other: that would make intensity
        registration an oracle and the clicked points decoration.
        """
        seeds = np.zeros((FIELD, FIELD))
        rows = np.clip(np.rint(centres[:, 0]).astype(int), 0, FIELD - 1)
        cols = np.clip(np.rint(centres[:, 1]).astype(int), 0, FIELD - 1)
        np.add.at(seeds, (rows, cols), 1.0)
        if not cytoplasm:
            blobs = ndi.gaussian_filter(seeds, 7.0)
            image = blobs / blobs.max()
        else:
            wide = ndi.gaussian_filter(seeds, 19.0)
            narrow = ndi.gaussian_filter(seeds, 9.0)
            ring = np.clip(wide / wide.max() - 0.75 * narrow / narrow.max(), 0, None)
            ring = ring / ring.max()
            texture = ndi.gaussian_filter(rng.normal(size=(FIELD, FIELD)), 2.5)
            texture = np.clip(texture, 0, None)
            texture = texture / texture.max()
            image = 0.75 * ring + 0.45 * texture * (ring > 0.03)
        noisy = image + 0.02 * rng.normal(size=(FIELD, FIELD))
        return np.clip(noisy, 0, None).astype(np.float32)

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        nuclei = rng.uniform(40, FIELD - 40, size=(N_NUCLEI, 2))
        probes = rng.uniform(20, FIELD - 20, size=(N_PROBES, 2))
        deform = self._deform(self._displacement())

        chosen = self._spread_landmarks(nuclei)
        moving_pts = nuclei[chosen]
        fixed_pts = deform(moving_pts) + rng.normal(
            scale=CLICK_NOISE_PX, size=(N_LANDMARKS, 2)
        )
        probe_truth = deform(probes)

        moving = self._render(
            nuclei, np.random.default_rng(self.seed + 1), cytoplasm=False
        )
        fixed = self._render(
            deform(nuclei), np.random.default_rng(self.seed + 2), cytoplasm=True
        )

        displacement = float(np.median(np.linalg.norm(probe_truth - probes, axis=1)))
        flat_moving = moving.ravel() - moving.mean()
        flat_fixed = fixed.ravel() - fixed.mean()
        correlation = float(
            flat_moving
            @ flat_fixed
            / (np.linalg.norm(flat_moving) * np.linalg.norm(flat_fixed))
        )

        # The properties the case rests on, checked before anyone pays for a
        # run. None of them is visible from the arrays alone.
        assert abs(correlation) < 0.30, (
            f"the two channels correlate {correlation:.3f}; above ~0.3 they are "
            "close enough to one image twice that intensity registration "
            "becomes the answer and the clicked points become decoration (§11)"
        )
        assert displacement > 4 * ERROR_LIMIT_PX, (
            f"the channels are {displacement:.1f} px apart and the limit is "
            f"{ERROR_LIMIT_PX} px, so doing nothing is too near a pass"
        )
        assert np.isfinite(fixed_pts).all() and np.isfinite(probe_truth).all()
        assert fixed_pts.shape == (N_LANDMARKS, 2)
        assert probe_truth.shape == (N_PROBES, 2)

        return Fixture(
            provenance=(
                f"procedural: {FIELD}x{FIELD}, {N_NUCLEI} cells, an affine "
                f"({ROTATION_DEG:g} deg, {SCALE:g}x, shear {SHEAR:g}, "
                f"translation {TRANSLATION}) plus a smooth random displacement "
                f"field of {WARP_RMS_PX:g} px RMS and ~"
                f"{WARP_SIGMA * FIELD / WARP_GRID:.0f} px correlation length; "
                f"{N_LANDMARKS} spread correspondences with "
                f"{CLICK_NOISE_PX:g} px click noise, {N_PROBES} withheld "
                f"probes, median displacement {displacement:.1f} px, "
                f"cross-channel correlation {correlation:.3f}, seed {self.seed}"
            ),
            about=(
                "Two channels of one field that do not overlay, and 18 clicked "
                "correspondences. A global affine through those clicks reaches "
                "15.3 px and a thin-plate spline 3.0 px, because the map is an "
                "affine plus a smooth non-affine term whose correlation length "
                "is longer than the landmark spacing. The channels are rendered "
                "independently from the same cell positions and correlate "
                "0.038, so intensity registration is not a route: phase "
                "correlation misses by 503 px."
            ),
            data={
                "moving": moving,
                "fixed": fixed,
                "moving_pts": moving_pts,
                "fixed_pts": fixed_pts,
                "probe_pts": probes,
            },
            truth={"probe_truth": probe_truth},
            tolerance={
                "median_error_px": ERROR_LIMIT_PX,
                "quality_honesty": HONESTY_LIMIT,
            },
        )


TASK = f"""
Two channels of one field are open in napari as the layers `moving` (a nuclear
stain) and `fixed` (a cytoplasmic stain of the same cells). They are not on the
same coordinate grid.

Three Points layers carry what a user clicked:

- `moving_pts` and `fixed_pts` -- {N_LANDMARKS} corresponding locations, one
  pair per row, in `moving` and `fixed` coordinates respectively. Row i of one
  corresponds to row i of the other.
- `probe_pts` -- {N_PROBES} locations in `moving` coordinates.

All point coordinates are (row, column) in pixels.

Work out how `moving` maps onto `fixed`, then leave two things in the kernel:

- `probe_mapped` -- an ({N_PROBES}, 2) array: where each row of `probe_pts`
  lands in `fixed` coordinates, same row order.
- `quality_px` -- a single number: your best estimate of the typical error, in
  pixels, of that mapping across the whole {FIELD}x{FIELD} field.

Both names must be bound in the kernel namespace when you finish.
""".strip()

CASE = Case(
    namespace=NAMESPACE,
    case_id=CASE_ID,
    task=TASK,
    # The same microscopist as the curated case: the task is self-sufficient,
    # there is no fact to elicit, and asking is neither rewarded nor punished.
    persona=MICROSCOPIST,
    fixture=Procedural(ClickedLandmarks()),
    layers=(
        Layer("moving", "moving"),
        Layer("fixed", "fixed"),
        Layer("moving_pts", "moving_pts", kind="points"),
        Layer("fixed_pts", "fixed_pts", kind="points"),
        Layer("probe_pts", "probe_pts", kind="points"),
    ),
    collect={"probe_mapped": "probe_mapped", "quality_px": "quality_px"},
    score=_verify,
    save_artifacts=_save_artifacts,
)
