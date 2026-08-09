"""`deconvolve-widefield` as benchmark data: did the restoration restore anything?

The stack is blurred by a PSF that carries a **pupil aberration** — spherical
from an index mismatch, plus astigmatism and coma. That is the point of the
construction: an aberration has no representation in NA/wavelength/RI, so no
model built from the metadata can produce it, and every aberration-free family
(Gaussian, angular spectrum, Born & Wolf, Gibson-Lanni with matched indices) is
wrong by the same physical fact rather than by a modelling detail chosen here.
The bead stack ships alongside because that is what the fact makes necessary.

The withheld fact is the one in step 2 that no pixel carries — the **voxel
spacing, and especially the 0.20 µm z-step**. It is load-bearing twice: the
PSF support is sized through it, and every axial FWHM the run reports is
`planes x dz`, so a run that assumes isotropic 1 µm voxels can restore the
stack perfectly and still report a resolution five times wrong. That is a
scale, the kind of fact §5d asks for: unlike "which of these is the bead
stack", there is no back door in the pixels through which it can be recovered.

What is *not* withheld, and is visible in the data:

  * the bead stack is a layer, so choosing to use it is a decision the run can
    make unaided — and in the 2026-08-05 prescreen both cold arms did
  * the objects are point-like, so an axial profile through one is a resolution
    measurement anybody can take

What the prescreen showed this actually separates is narrower than the entry
originally claimed, and it is the reason `residual_blur` is here: one cold arm
mistook Richardson-Lucy's rising peak for divergence, retreated to a Wiener
filter regularised ~1000x too hard, and delivered a **2%** improvement in axial
FWHM while reporting success from a visual check. Nothing in its own output
said otherwise.

The reference implementation these tolerances come from is in the pull request
that added this case, per `biopb-mcp/docs/skills.md` §11b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from ...agentbench._fixture import (
    Attempt,
    Fixture,
    Metric,
    Outcome,
    Procedural,
    read_scalar,
)
from ...agentbench._respondent import Persona
from .._case import Case, Layer

SKILL = "deconvolve-widefield"

#: The withheld triple. Anisotropic on purpose -- a 0.20 µm z-step against
#: 0.065 µm pixels is an ordinary high-NA acquisition, and the ratio is what a
#: run that assumes isotropic voxels gets wrong.
SPACING = (0.20, 0.065, 0.065)
NA = 1.4
LAMBDA_NM = 520.0
RI = 1.515

#: Set from the reference implementation over 5 seeds of this construction, not
#: from taste. The reference (PSF measured from the beads, pedestal removed,
#: then 40 RL iterations) lands at residual_blur 0.058-0.182; the failures these
#: limits have to separate it from are measured on the same seeds:
#:
#:   theoretical PSF instead of the beads ....... residual_blur 0.619-0.754
#:   untouched input (a filter regularised
#:     into a no-op -- the failure actually
#:     observed in the prescreen) ............... residual_blur 1.000
#:   `clip=True` left at its default ............ output flat, every metric
#:                                                unavailable
#:   dz assumed equal to the lateral pixel ...... fwhm_after_error 0.675
#:
#: `residual_blur` sits at 0.55 -- above the worst reference seed (0.182) and
#: below the best metadata-only one (0.619), a gap of 0.44. It is deliberately
#: on the far side of the theoretical-PSF route: the beads are *in the data*,
#: so choosing to use them is the decision this case is measuring.
#:
#: The two FWHM limits are not about the reference's accuracy. Each compares the
#: run's *reported* number against the verifier's own measurement of the very
#: same volume, so any honest implementation scores ~0 and the only ways to fail
#: are the withheld z-step (0.675) or not really measuring at all.
TOLERANCE = {
    "residual_blur": 0.55,
    "fwhm_after_error": 0.25,
    "fwhm_before_error": 0.25,
}

_ABERRATION = {"spherical": 0.38, "astigmatism": 0.09, "coma": 0.07}


# --- the optics ------------------------------------------------------------


def _zernike(rho, theta, kind):
    if kind == "spherical":  # Z(4,0)
        return np.sqrt(5.0) * (6 * rho**4 - 6 * rho**2 + 1)
    if kind == "astigmatism":  # Z(2,2)
        return np.sqrt(6.0) * rho**2 * np.cos(2 * theta)
    if kind == "coma":  # Z(3,1)
        return np.sqrt(8.0) * (3 * rho**3 - 2 * rho) * np.cos(theta)
    raise ValueError(kind)


def _psf(shape, aberration, shift=(0.0, 0.0, 0.0), grid=192, offset=None):
    """Scalar widefield PSF by angular-spectrum propagation of an aberrated pupil.

    `shift` displaces the source by (dz, dy, dx) in voxels -- laterally by a
    pupil phase ramp, axially by moving the sample planes. That is how beads get
    sub-voxel positions without the sinc ringing a shifted lattice delta would
    produce: an intensity PSF is non-negative by construction, a band-limited
    delta is not.
    """
    lam = LAMBDA_NM / 1000.0
    dz, dy, dx = SPACING
    nz, ny, nx = shape
    if offset is not None:
        shift = tuple(float(s) - float(o) for s, o in zip(shift, offset, strict=True))

    fy = np.fft.fftfreq(grid, d=dy)
    fx = np.fft.fftfreq(grid, d=dx)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    f2 = FY**2 + FX**2
    cut = NA / lam

    pupil = (f2 <= cut**2).astype(np.complex128)
    if aberration:
        rho = np.clip(np.sqrt(f2) / cut, 0, 1)
        theta = np.arctan2(FY, FX)
        phase = sum(w * _zernike(rho, theta, k) for k, w in aberration.items())
        pupil = pupil * np.exp(2j * np.pi * phase)

    sz, sy, sx = shift
    if sy or sx:
        pupil = pupil * np.exp(-2j * np.pi * (FY * sy * dy + FX * sx * dx))

    kz = 2 * np.pi * np.sqrt(np.maximum((RI / lam) ** 2 - f2, 0.0))
    out = np.empty((nz, grid, grid))
    for i, z in enumerate((np.arange(nz) - nz // 2 - sz) * dz):
        out[i] = np.abs(np.fft.fftshift(np.fft.ifft2(pupil * np.exp(1j * z * kz)))) ** 2

    c = grid // 2
    out = out[:, c - ny // 2 : c + ny // 2 + 1, c - nx // 2 : c + nx // 2 + 1]
    return (out / out.sum()).astype(np.float32)


def _peak_offset(shape, aberration):
    """Where the aberrated PSF's brightest voxel sits relative to the centre.

    Spherical aberration moves the intensity maximum away from geometric focus
    -- measured on this construction at 9 planes, 1.8 µm. Left in, it would
    translate every reconstruction by a constant, which is a choice of origin
    rather than a restoration error, and would be charged to the run. The shape
    distortion, which is the whole point, is kept.
    """
    raw = _psf(shape, aberration)
    return np.unravel_index(raw.argmax(), raw.shape) - np.array([s // 2 for s in shape])


def _stamp(vol, kernel, centre, amp=1.0):
    src, dst = [], []
    for c, k, n in zip(centre, kernel.shape, vol.shape, strict=True):
        lo, hi = c - k // 2, c - k // 2 + k
        src.append(slice(max(0, -lo), k - max(0, hi - n)))
        dst.append(slice(max(0, lo), min(n, hi)))
    vol[tuple(dst)] += amp * kernel[tuple(src)]


def _camera(blurred, rng, peak_photons, offset=100.0, read_noise=2.0):
    scaled = blurred / blurred.max() * peak_photons
    img = rng.poisson(np.clip(scaled, 0, None)).astype(np.float32)
    return (img + offset + rng.normal(0, read_noise, img.shape)).astype(np.float32)


# --- the fixture -----------------------------------------------------------


@dataclass(frozen=True)
class AberratedStack:
    """A widefield stack and a bead stack from the same mount and depth."""

    shape: tuple[int, int, int] = (40, 160, 160)
    psf_shape: tuple[int, int, int] = (31, 33, 33)
    n_points: int = 18
    n_filaments: int = 3
    peak_photons: float = 900.0
    bead_photons: float = 2200.0
    #: Object brightnesses are multiples of what the *brightest* point puts at
    #: its own voxel. They have to be, not set by eye: this PSF smears a point
    #: over ~1200 voxels so its peak is ~8e-4, and a haze picked to "look like
    #: 10%" of the object amplitude buries every object a thousand times over.
    point_amp: tuple[float, float] = (0.25, 1.5)
    haze_frac: float = 0.30
    filament_frac: float = 0.60
    aberration: dict = field(default_factory=lambda: dict(_ABERRATION))
    seed: int = 7

    def __call__(self) -> Fixture:
        rng = np.random.default_rng(self.seed)
        nz, ny, nx = self.shape
        off = _peak_offset(self.psf_shape, self.aberration)
        psf = _psf(self.psf_shape, self.aberration, offset=off)
        unit = self.point_amp[1] * float(psf.max())
        truth_vol = np.zeros(self.shape, np.float32)

        # Filaments first, so the point emitters can be placed clear of them: a
        # filament crossing a photometry window contributes several units of
        # flux and makes that point unusable as a resolution standard.
        line_gain = float(psf[psf.shape[0] // 2, psf.shape[1] // 2, :].sum())
        fil_amp = self.filament_frac * unit / line_gain
        for _ in range(self.n_filaments):
            z = int(rng.integers(10, 30))
            y0, x0 = rng.uniform(30, 130, 2)
            ang = rng.uniform(0, np.pi)
            for t in np.linspace(-45, 45, 700):
                y, x = (
                    int(round(y0 + t * np.sin(ang))),
                    int(round(x0 + t * np.cos(ang))),
                )
                if 4 <= y < ny - 4 and 4 <= x < nx - 4:
                    truth_vol[z, y, x] = max(truth_vol[z, y, x], fil_amp)

        pts, amps = [], []
        for _ in range(20000):
            if len(pts) == self.n_points:
                break
            p = (
                int(rng.integers(9, 31)),
                int(rng.integers(16, 144)),
                int(rng.integers(16, 144)),
            )
            if any(
                abs(p[0] - q[0]) <= 5 and np.hypot(p[1] - q[1], p[2] - q[2]) <= 22
                for q in pts
            ):
                continue
            win = truth_vol[
                max(0, p[0] - 7) : p[0] + 8,
                max(0, p[1] - 12) : p[1] + 13,
                max(0, p[2] - 12) : p[2] + 13,
            ]
            if win.any():  # a filament is in frame
                continue
            pts.append(p)
            amps.append(float(rng.uniform(*self.point_amp)))
        assert len(pts) == self.n_points, f"point packing failed: {len(pts)}"
        for p, a in zip(pts, amps, strict=True):
            truth_vol[p] = a

        # The out-of-focus light widefield actually suffers from. A smooth field
        # passes a normalised PSF essentially unchanged, so its amplitude is its
        # image level.
        haze = ndi.gaussian_filter(
            rng.random(self.shape).astype(np.float32), (4, 14, 14)
        )
        truth_vol = truth_vol + (
            (haze - haze.min()) / np.ptp(haze) * (self.haze_frac * unit)
        ).astype(np.float32)

        image = _camera(
            fftconvolve(truth_vol, psf, mode="same"), rng, self.peak_photons
        )

        # The bead stack: sub-resolution beads on a jittered grid, each a PSF
        # stamped at a sub-voxel offset. A grid rather than rejection sampling
        # because 12 beads at a usable separation do not fit in this field by
        # chance, and a sampler that cannot satisfy its constraint hangs.
        blur = np.zeros(self.shape, np.float32)
        for iy in range(4):
            for ix in range(3):
                centre = (
                    # centred in z so a 31-plane crop fits inside 40 planes
                    int(rng.integers(17, 23)),
                    int(22 + 38 * iy + rng.integers(-5, 6)),
                    int(30 + 50 * ix + rng.integers(-5, 6)),
                )
                jitter = tuple(rng.uniform(-0.5, 0.5, 3))
                _stamp(
                    blur,
                    _psf(self.psf_shape, self.aberration, jitter, offset=off),
                    centre,
                )
        beads = _camera(blur, rng, self.bead_photons)

        fwhm_input = _mean_axial_fwhm(image, pts, SPACING[0])
        fwhm_truth = _mean_axial_fwhm(truth_vol, pts, SPACING[0])
        assert fwhm_input > 4 * fwhm_truth, (
            f"the blur is not the problem here: input {fwhm_input:.2f} µm "
            f"against a truth of {fwhm_truth:.2f} µm"
        )

        return Fixture(
            provenance=(
                f"procedural: seed {self.seed}, {self.shape} widefield stack, "
                f"NA {NA} / {LAMBDA_NM:.0f} nm / RI {RI}, voxels "
                f"{SPACING[0]}x{SPACING[1]}x{SPACING[2]} µm, "
                f"{np.sqrt(sum(v * v for v in self.aberration.values())):.3f} "
                f"waves RMS aberration, {self.n_points} point emitters, "
                f"12 beads, ~{self.peak_photons:.0f} peak photons"
            ),
            about=(
                "The blur carries an aberration that is absent from "
                "NA/wavelength/RI, so the bead stack is the only route to the "
                "real PSF. The voxel spacing is not in the data: without the "
                "z-step the stack can be restored and the resolution still "
                "reported five times wrong."
            ),
            data={"image": image, "beads": beads},
            truth={
                "points": np.array(pts, int),
                "spacing": np.array(SPACING, float),
                "fwhm_input_um": fwhm_input,
                "fwhm_truth_um": fwhm_truth,
            },
            tolerance=dict(TOLERANCE),
        )


# --- truth-side arithmetic, shared by the builder and the verifier ----------


def _axial_fwhm(vol, zyx, dz, half_window=9):
    """FWHM along z through one point-like object, in µm.

    Measured inside a local window. Against the whole line the diffuse haze and
    the filaments set `max`, and the half-max crossing then reports the width of
    the field rather than of the object.
    """
    z, y, x = (int(v) for v in zyx)
    lo, hi = max(0, z - half_window), min(vol.shape[0], z + half_window + 1)
    p = np.asarray(vol[lo:hi, y, x], float)
    p = p - p.min()
    if not np.isfinite(p).all() or p.max() <= 0:
        return np.nan
    half, pk = p.max() / 2.0, int(np.argmax(p))
    a = b = pk
    while a > 0 and p[a] >= half:
        a -= 1
    while b < len(p) - 1 and p[b] >= half:
        b += 1
    return (b - a) * dz


def _mean_axial_fwhm(vol, points, dz):
    vals = [_axial_fwhm(vol, p, dz) for p in points]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _restored(attempt: Attempt, truth) -> tuple[np.ndarray | None, str]:
    got = attempt.arrays.get("restored")
    if got is None:
        return None, "the run left no `restored`"
    got = np.asarray(got)
    want = tuple(int(v) for v in (40, 160, 160))
    if got.shape != want:
        return None, f"the run's `restored` is {got.shape}, not {want}"
    if not np.isfinite(got).all():
        return None, "the run's `restored` holds non-finite values"
    if float(np.ptp(got.astype(float))) <= 0:
        return None, "the run's `restored` is constant -- nothing to measure"
    return got.astype(float), ""


# --- the verifier ----------------------------------------------------------


def verify(fixture: Fixture, attempt: Attempt) -> Outcome:
    """Score *attempt* against *fixture*'s truth.

    `residual_blur` is the one that matters and it is measured on the returned
    volume, never on the run's own account of it: the failure this case exists
    for delivered a 2% improvement while reporting success. 0 is the truth, 1 is
    the untouched input.

    The two FWHM metrics are scored from the run's *reported* numbers against
    what the verifier measures on the very same volume, so what they test is the
    withheld z-step -- a run that assumed isotropic voxels restores the stack
    and still reports µm that are wrong by the spacing ratio.
    """
    limits = {**TOLERANCE, **fixture.tolerance}
    truth = fixture.truth
    dz = float(truth["spacing"][0])
    pts = truth["points"]
    metrics: list[Metric] = []
    detail: dict[str, object] = {}

    vol, why = _restored(attempt, truth)
    if vol is None:
        metrics.append(
            Metric("residual_blur", None, limits["residual_blur"], unavailable=why)
        )
    else:
        after = _mean_axial_fwhm(vol, pts, dz)
        floor, ceil = truth["fwhm_truth_um"], truth["fwhm_input_um"]
        if not np.isfinite(after):
            metrics.append(
                Metric(
                    "residual_blur",
                    None,
                    limits["residual_blur"],
                    unavailable="no axial profile through the truth's point emitters "
                    "could be measured on the run's `restored`",
                )
            )
        else:
            metrics.append(
                Metric(
                    "residual_blur",
                    max(0.0, (after - floor) / (ceil - floor)),
                    limits["residual_blur"],
                    unit=" of the input blur left (0 = truth, 1 = untouched)",
                )
            )
            detail |= {
                "axial_fwhm_um_measured": after,
                "axial_fwhm_um_input": float(ceil),
                "axial_fwhm_um_truth": float(floor),
            }

    for name, key, on in (
        ("fwhm_before_error", "axial_fwhm_before_um", "input"),
        ("fwhm_after_error", "axial_fwhm_after_um", "restored"),
    ):
        said, why = read_scalar(attempt, key)
        if on == "input":
            want = truth["fwhm_input_um"]
        elif vol is None or not np.isfinite(_mean_axial_fwhm(vol, pts, dz)):
            metrics.append(
                Metric(
                    name,
                    None,
                    limits[name],
                    unavailable="there is no usable `restored` to compare against",
                )
            )
            continue
        else:
            want = _mean_axial_fwhm(vol, pts, dz)
        if said is None:
            metrics.append(Metric(name, None, limits[name], unavailable=why))
        else:
            metrics.append(Metric(name, abs(said - want) / want, limits[name]))
            detail |= {key: float(said), f"{key}_measured": float(want)}

    return Outcome(fixture=fixture, attempt=attempt, metrics=metrics, detail=detail)


def save_artifacts(outcome: Outcome, where: Path) -> None:
    """Per-object axial FWHM before and after, so a human can see whether the
    number came from a real restoration or from one lucky point. Never raises."""
    fixture, attempt = outcome.fixture, outcome.attempt
    truth = fixture.truth
    vol, _ = _restored(attempt, truth)
    if vol is None:
        return
    dz = float(truth["spacing"][0])
    lines = ["z,y,x,fwhm_input_um,fwhm_restored_um"]
    for p in truth["points"]:
        lines.append(
            f"{p[0]},{p[1]},{p[2]},"
            f"{_axial_fwhm(fixture.data['image'], p, dz):.3f},"
            f"{_axial_fwhm(vol, p, dz):.3f}"
        )
    (where / "axial_fwhm.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- the conversation ------------------------------------------------------

TASK = """\
Two 3D widefield fluorescence stacks are open in the napari viewer as Image
layers. Both are float32 with axes (Z, Y, X) = (40, 160, 160).

  image   the acquisition I care about
  beads   a stack of sub-resolution fluorescent beads

There is no tensor server in this session, so those layers are the only source
of pixels.

The stack is badly smeared along z and I want it restored so I can look at the
structures in it. I would also like to know how much the axial resolution
actually improved.

The person who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave three names bound in the kernel namespace:

  restored                a float array with the same (40, 160, 160) shape as
                          `image`, holding the restored stack
  axial_fwhm_before_um    a float. The mean full width at half maximum, along
                          z, through the point-like objects in `image`, in
                          microns.
  axial_fwhm_after_um     a float. The same measurement made on `restored`.

Those three names are how your result is collected, so bind them exactly.
"""

#: What is *not* here: no iteration count, nothing about Richardson-Lucy or
#: clipping, no hint that the beads are the route to the PSF, and nothing about
#: how to tell a working restoration from a no-op. This person knows their
#: sample and their microscope.
OPERATOR = Persona(
    name="operator-widefield-deconvolution",
    facts={
        "what the voxel spacing is": (
            "0.065 microns in x and y, and the z-step is 0.20 microns -- I set "
            "the step on the piezo myself"
        ),
        "what objective was used": (
            "a 1.4 NA oil immersion lens, the immersion oil is 1.515"
        ),
        "what the emission wavelength is": "520 nanometres, it is a green dye",
        "what the beads are": (
            "sub-resolution fluorescent beads, well under the resolution limit. "
            "I mounted them in the same medium and imaged them at the same "
            "depth on the same day, right after the sample"
        ),
        "what the sample is": (
            "fixed cells with small punctate structures in them, and some "
            "filamentous material"
        ),
        "how deep into the sample this was imaged": (
            "about 12 microns above the coverslip, in an aqueous mounting medium"
        ),
        "what the camera does": (
            "it is an sCMOS, there is an offset pedestal of around 100 counts "
            "and a couple of counts of read noise"
        ),
        "what the result is for": (
            "looking at it, and measuring the shapes of the puncta. I am not "
            "trying to measure how bright anything is"
        ),
    },
    background=(
        "A 3D widefield fluorescence stack of fixed cells, acquired alongside a "
        "bead stack. You are happy to answer questions about the sample, the "
        "microscope and the acquisition."
    ),
)

CASE = Case(
    skill=SKILL,
    case_id="aberrated-no-spacing-in-the-pixels",
    task=TASK,
    persona=OPERATOR,
    fixture=Procedural(AberratedStack()),
    layers=(Layer("image", "image"), Layer("beads", "beads")),
    collect={
        "restored": "restored",
        "axial_fwhm_before_um": "axial_fwhm_before_um",
        "axial_fwhm_after_um": "axial_fwhm_after_um",
    },
    score=verify,
    save_artifacts=save_artifacts,
    catalog_query="deconvolution",
    # It must be able to answer: the fixture strips the spacing, and this person
    # knows it, along with the optics and what the beads are.
    persona_must_know=("0.065", "0.20", "1.4", "520", "beads"),
    # And it must not know the procedure.
    persona_must_not_know=(
        "richardson",
        "lucy",
        "num_iter",
        "clip=",
        "point spread",
        "peak_local_max",
        "iterations",
    ),
)
