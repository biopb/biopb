"""Image resolution in physical units: Fourier ring correlation and decorrelation.

Answers "how good is this image, in nanometres" — which is **not** the question a
focus metric or a localization precision answers. Localization precision is
per-molecule and optical; resolution is additionally capped by label density,
residual drift and linking errors, so an SMLM run that reports its median CRLB as
"20 nm resolution" is overclaiming by a factor nothing in its own output reveals.
This module measures the image.

Two estimators, and the data picks which one:

- ``frc(a, b)`` — **Fourier ring correlation** (Nieuwenhuizen et al. 2013), when two
  statistically independent images of the same field exist. For SMLM they always
  do: ``frc_from_localizations`` splits one localization list into halves and
  renders both.
- ``decorrelation_resolution(image)`` — **decorrelation analysis** (Descloux,
  Grussmayer & Radenovic 2019), when only one image exists: a deconvolved
  widefield, a SIM or eSRRF reconstruction, a single micrograph.

Delivered as a kernel plugin rather than as a snippet in a skill body for the same
reason ``segmentation_qc`` is: the arithmetic is short but wrong in ways that are
invisible in the output. Every one of these changes the reported number by tens of
percent and none of them changes how the answer *looks*:

- **Apodization is not cosmetic.** An unwindowed FFT sees the wrap-around edge
  discontinuity as a cross of power along the axes, and that cross is *identical*
  in both halves — so it correlates at every frequency and lifts the whole FRC
  curve. Both estimators taper the edges before transforming.
- **The threshold is part of the answer.** 1/7, the ½-bit curve and 3σ do not
  agree, so a resolution quoted without its criterion is not comparable to
  anything. ``FRCResult`` carries ``threshold_name``; ``summary()`` prints it.
- **The sampling is a floor.** No method reports below twice the pixel (or render)
  size. A localization list rendered at 20 nm/px cannot resolve better than 40 nm
  no matter how precise the localizations are, and a run that renders coarsely
  measures its own renderer. Flagged as ``nyquist_limited``.
- **Label density is a second floor** and it is the one people forget. Structure
  sampled at a mean label spacing *d* cannot be resolved below ~2*d* regardless of
  precision (Shroff et al. 2008). ``frc_from_localizations`` computes it from the
  localizations it was handed and warns when the FRC number is below it.
- **One number describes the whole field.** Both estimators average over
  orientation and over the entire image, so on a field with heterogeneous label
  density or anisotropic structure the number describes nowhere in particular.
  Tile the field and call this per tile if that is a risk.

Four public callables plus their result records, reached through the module the
agent gets bound (``image_resolution``): ``frc``, ``frc_from_localizations``,
``split_localizations``, ``decorrelation_resolution``, and the ``FRCResult`` /
``DecorrelationResult`` records they return.
"""

# Private aliases keep the module's own surface to its public API, so
# `inspect_object("image_resolution")` shows the agent the callables rather than
# every scipy handle this file imported. Style, not protection: as a kernel plugin
# this module is bound under one name.
from dataclasses import dataclass as _dataclass, field as _dc_field

import numpy as np
import scipy.ndimage as _ndi

__all__ = [
    "frc",
    "frc_from_localizations",
    "split_localizations",
    "decorrelation_resolution",
    "FRCResult",
    "DecorrelationResult",
    "THRESHOLDS",
]

# The three criteria in use. "1/7" is the SMLM convention (Nieuwenhuizen 2013) and
# the default here so numbers are comparable to the field's; the other two are
# ring-count dependent (van Heel & Schatz 2005) and stricter where rings are sparse.
THRESHOLDS = ("1/7", "half-bit", "3sigma")

# Fewer rings than this and a ring average is not an average of anything.
_MIN_RINGS = 8

# The count-dependent criteria are functions of how many Fourier samples a ring
# holds, and the innermost rings hold almost none (1, 8, 12, 16, 32 ... on a 512
# grid). There the threshold is reporting the ring's emptiness rather than any
# property of the images: 3sigma asks for a correlation above 1 in the first four
# rings, and is still at 0.75 by ring 4, which a perfectly good low-SNR-per-ring
# curve dips under before recovering for another 175 rings. Below this many
# samples the criterion is not applied. 3sigma needs 18 samples to be attainable
# at all and 72 to ask for less than 0.5; 64 is the round number in between.
_MIN_RING_SAMPLES = 64

# im - gaussian_blur(im, g) as a Fourier multiplier: a real-space Gaussian of
# sigma g (pixels) is exp(-2 pi^2 g^2 f^2) at f cycles/pixel, and the normalized
# frequency used here is k = 2f, so the high-pass is 1 - exp(-_HP_A g^2 k^2).
_HP_A = np.pi**2 / 2.0


@_dataclass
class FRCResult:
    """One FRC measurement. Curves are indexed by ring, ring 0 being DC."""

    resolution: float
    threshold_name: str
    crossing_frequency: float  # cycles per unit length (1 / pixel_size units)
    pixel_size: float
    nyquist_resolution: float
    nyquist_limited: bool
    n_crossings: int
    frequency: np.ndarray = _dc_field(repr=False)  # cycles per unit length
    curve: np.ndarray = _dc_field(repr=False)  # raw FRC per ring
    curve_smoothed: np.ndarray = _dc_field(repr=False)
    threshold_curve: np.ndarray = _dc_field(repr=False)
    ring_counts: np.ndarray = _dc_field(repr=False)
    shape: tuple = ()
    split: str = ""
    label_nyquist: float = float("nan")
    warnings: list = _dc_field(default_factory=list)

    def to_dict(self) -> dict:
        """Flat dict of the scalar fields, for logging or a results table."""
        curves = {
            "frequency",
            "curve",
            "curve_smoothed",
            "threshold_curve",
            "ring_counts",
        }
        return {k: v for k, v in self.__dict__.items() if k not in curves}

    def summary(self) -> str:
        """One-line report carrying the criterion, because the number alone is not
        comparable to a published figure without it, plus any warnings."""
        head = f"FRC resolution {self.resolution:.4g} (threshold {self.threshold_name}"
        if self.split:
            head += f", split={self.split}"
        head += ")"
        return "\n".join([head] + [f"  ! {w}" for w in self.warnings])


@_dataclass
class DecorrelationResult:
    """One decorrelation measurement (Descloux et al. 2019)."""

    resolution: float
    kc: float  # normalized frequency of the winning peak; 1.0 == Nyquist
    amplitude: float
    pixel_size: float
    nyquist_resolution: float
    nyquist_limited: bool
    radii: np.ndarray = _dc_field(repr=False)  # normalized frequency grid
    curves: np.ndarray = _dc_field(repr=False)  # (n_filters + 1, n_r)
    filter_sigmas: np.ndarray = _dc_field(repr=False)  # real-space pixels; 0 == none
    kc_per_filter: np.ndarray = _dc_field(repr=False)
    amplitude_per_filter: np.ndarray = _dc_field(repr=False)
    shape: tuple = ()
    warnings: list = _dc_field(default_factory=list)

    def to_dict(self) -> dict:
        """Flat dict of the scalar fields, for logging or a results table."""
        curves = {
            "radii",
            "curves",
            "filter_sigmas",
            "kc_per_filter",
            "amplitude_per_filter",
        }
        return {k: v for k, v in self.__dict__.items() if k not in curves}

    def summary(self) -> str:
        head = (
            f"decorrelation resolution {self.resolution:.4g} "
            f"(peak at {self.kc:.3f} of Nyquist, height {self.amplitude:.3f})"
        )
        return "\n".join([head] + [f"  ! {w}" for w in self.warnings])


# --------------------------------------------------------------------------- #
# shared preparation
# --------------------------------------------------------------------------- #


def _as_plane(image, name: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D plane; got shape {arr.shape}. Slice a stack "
            "first -- resolution is a property of one plane, and the mean of the "
            "measurement over Z is not the measurement of the mean."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=True)


def _center_crop_square(arr: np.ndarray) -> np.ndarray:
    """Largest centred even-sided square. A ring average is only isotropic on one."""
    n = min(arr.shape)
    n -= n % 2
    y0 = (arr.shape[0] - n) // 2
    x0 = (arr.shape[1] - n) // 2
    return arr[y0 : y0 + n, x0 : x0 + n]


def _tukey(n: int, alpha: float) -> np.ndarray:
    """1-D Tukey (tapered-cosine) window; ``alpha`` is the fraction spent tapering."""
    if alpha <= 0 or n < 3:
        return np.ones(n)
    alpha = min(alpha, 1.0)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return w
    k = np.arange(edge + 1)
    ramp = 0.5 * (1.0 + np.cos(np.pi * (2.0 * k / (alpha * (n - 1)) - 1.0)))
    w[: edge + 1] = ramp
    w[n - edge - 1 :] = ramp[::-1]
    return w


def _apodize(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Mean-subtract, then taper the edges.

    Mean first, so the taper does not leave a DC pedestal shaped like the window;
    DC carries no resolution information and would otherwise dominate ring 0.
    """
    arr = arr - arr.mean()
    if alpha <= 0:
        return arr
    return arr * np.outer(_tukey(arr.shape[0], alpha), _tukey(arr.shape[1], alpha))


def _radial_frequency(n: int) -> np.ndarray:
    """|k| in cycles per pixel over an ``n x n`` un-shifted FFT grid."""
    f = np.fft.fftfreq(n)
    return np.hypot(f[:, None], f[None, :])


# --------------------------------------------------------------------------- #
# Fourier ring correlation
# --------------------------------------------------------------------------- #


def _threshold_curve(name: str, counts: np.ndarray) -> np.ndarray:
    """The chosen criterion evaluated per ring.

    ``1/7`` is a constant; the other two tighten where a ring holds few samples,
    which is the low-frequency end where a constant threshold is most permissive.
    """
    if name == "1/7":
        # A constant does not depend on the ring count, so it needs no sample-count
        # gate: a crossing in the first ring under 1/7 really does mean the two
        # images have nothing in common, and is reported rather than suppressed.
        return np.full(counts.shape, 1.0 / 7.0)
    n = np.maximum(counts.astype(np.float64), 1.0)
    if name == "half-bit":
        thr = (0.2071 + 1.9102 / np.sqrt(n)) / (1.2071 + 0.9102 / np.sqrt(n))
    elif name == "3sigma":
        thr = 3.0 / np.sqrt(n / 2.0)
    else:
        raise ValueError(f"unknown threshold {name!r}; expected one of {THRESHOLDS}")
    # inf, not a clip to 1: an unattainable threshold must read as "not applicable
    # here" so the crossing search skips the ring. Clipped to 1 it would instead
    # read as an instant crossing and report a resolution of about one ring.
    return np.where(counts < _MIN_RING_SAMPLES, np.inf, thr)


def _first_crossing(curve: np.ndarray, thr: np.ndarray) -> tuple[float, int]:
    """Fractional ring where ``curve`` first falls to ``thr``, and how many times
    it crosses in total.

    The first crossing is the resolution; the count is the honesty check. A curve
    that crosses once is a measurement; a curve that crosses five times is noise,
    and the first crossing is then an accident of where the noise landed.

    Rings whose threshold is >= 1 are skipped: no correlation can reach them, so
    they are not a crossing but an unanswerable question. Skipped means *dropped
    from the search*, not "start after them" -- ring sample counts are not
    monotonic in radius (they wobble with how the grid quantises each circle, and
    fall again at the outermost rings once the corners are excluded), so gated
    rings turn up well past the first usable one. Left in the array they read as
    an infinitely negative margin, which is to say as a crossing, and the reported
    resolution snaps to the gate for every input.

    Returns ``(ring, n_crossings, at_floor)``. ``at_floor`` means the curve was
    already below the threshold on the first ring the criterion applied to, so the
    answer is the gate rather than a measurement.
    """
    rings = np.arange(1.0, curve.size)  # ring 0 is DC and carries no information
    usable = thr[1:] < 1.0
    if not usable.any():
        return float("nan"), 0, True
    r_u = rings[usable]
    g_u = (curve[1:] - thr[1:])[usable]
    n_cross = int(np.count_nonzero(np.diff(np.signbit(g_u))))
    below = np.flatnonzero(g_u <= 0)
    if below.size == 0:
        return float("nan"), n_cross, False
    j = int(below[0])
    if j == 0:
        # Below on the very first ring the criterion could speak about. That is a
        # real "nothing in common" answer if it is also the first ring overall,
        # and the gate otherwise.
        return float(r_u[0]), n_cross, bool(r_u[0] > 1)
    a, b = g_u[j - 1], g_u[j]
    t = a / (a - b) if a != b else 0.0
    return float(r_u[j - 1] + t * (r_u[j] - r_u[j - 1])), n_cross, False


def frc(
    image_a,
    image_b,
    *,
    pixel_size: float = 1.0,
    threshold: str = "1/7",
    smooth: int | None = None,
    apodize: float = 0.25,
    split: str = "",
    label_nyquist: float = float("nan"),
) -> FRCResult:
    """Fourier ring correlation between two independent images of the same field.

    The two inputs must be **statistically independent realizations** -- different
    noise, same structure. Two consecutive frames, two halves of a localization
    list, two sequential acquisitions. Handing this the same image twice, or an
    image and a filtered copy of it, returns a correlation near 1 at every
    frequency and a resolution of exactly Nyquist, which measures nothing.

    Args:
        image_a, image_b: 2-D planes of identical shape. Non-square input is
            centre-cropped to the largest even square, because a ring average is
            only isotropic on a square grid.
        pixel_size: Physical size of one pixel. The returned resolution is in these
            units; leave it at 1.0 to get an answer in pixels.
        threshold: One of ``THRESHOLDS``. ``"1/7"`` (default) is the SMLM
            convention; ``"half-bit"`` and ``"3sigma"`` tighten where rings are
            sparse. **Report which one you used** -- they do not agree.
        smooth: Boxcar width in rings applied before the crossing search (forced
            odd; default ``max(3, n_rings // 20)``). The raw curve is returned too.
        apodize: Tukey taper fraction (default 0.25). Set 0 only if the input is
            already windowed; an unwindowed FFT lifts the whole curve.
        split: Free-text label recorded in the result, used by
            ``frc_from_localizations`` to record how the halves were made.
        label_nyquist: Optional density-imposed resolution floor to check against.

    Returns:
        An :class:`FRCResult`. Read ``.summary()`` before ``.resolution`` -- the
        warnings are where a plausible-looking number turns out not to be one.
    """
    a = _as_plane(image_a, "image_a")
    b = _as_plane(image_b, "image_b")
    if a.shape != b.shape:
        raise ValueError(
            f"the two images must match in shape; got {a.shape} and {b.shape}"
        )
    notes: list[str] = []
    if a.shape[0] != a.shape[1]:
        side = min(a.shape) - min(a.shape) % 2
        notes.append(
            f"input {a.shape} is not square; centre-cropped to {side}x{side} so "
            "the ring average is isotropic"
        )
    a = _center_crop_square(a)
    b = _center_crop_square(b)
    n = a.shape[0]
    if n < 2 * _MIN_RINGS:
        raise ValueError(
            f"image is {n}x{n} after cropping, too small for a ring average; "
            f"need at least {2 * _MIN_RINGS} on a side"
        )

    fa = np.fft.fft2(_apodize(a, apodize))
    fb = np.fft.fft2(_apodize(b, apodize))

    ring = np.rint(_radial_frequency(n) * n).astype(np.intp)
    n_rings = n // 2 + 1
    keep = ring < n_rings  # drop the corners; those bins see only two orientations
    idx = ring[keep]

    num = np.bincount(idx, np.real(fa * np.conj(fb))[keep], n_rings)
    den_a = np.bincount(idx, (np.abs(fa) ** 2)[keep], n_rings)
    den_b = np.bincount(idx, (np.abs(fb) ** 2)[keep], n_rings)
    counts = np.bincount(idx, None, n_rings)

    with np.errstate(invalid="ignore", divide="ignore"):
        curve = num / np.sqrt(den_a * den_b)
    curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
    curve[0] = 1.0  # DC is zero by construction after mean subtraction

    if smooth is None:
        # Measured on a hard band limit: widths up to 5 move the crossing by under
        # 1%, 13 costs 3-9% and 21 costs 4-40%, worst where the crossing is at a
        # low ring and the boxcar spans a large fraction of the way to it. Cheap
        # noise robustness is worth a percent; a twentieth of the curve is not.
        smooth = max(3, min(9, n_rings // 40))
    smooth = int(smooth)
    if smooth > 1:
        smooth += 1 - smooth % 2  # odd, so the boxcar stays centred
        # Ring 0 is a display convention (see below), not a measurement -- smooth
        # from ring 1 on, or that forced 1.0 leaks into the rings that decide the
        # crossing and holds the curve up over exactly the range it matters most.
        smoothed = curve.copy()
        smoothed[1:] = _ndi.uniform_filter1d(curve[1:], smooth, mode="nearest")
    else:
        smoothed = curve.copy()

    thr = _threshold_curve(threshold, counts)
    ring_c, n_cross, at_floor = _first_crossing(smoothed, thr)

    nyquist = 2.0 * pixel_size
    if np.isnan(ring_c):
        resolution = nyquist
        freq_c = 0.5 / pixel_size
        nyq_limited = True
        notes.append(
            "the FRC curve never falls to the threshold: the two images stay "
            f"correlated out to Nyquist, so {nyquist:.4g} is a bound and not a "
            "measurement. Either the inputs are not independent, or the image is "
            "undersampled -- render or acquire finer and repeat"
        )
    else:
        freq_c = (ring_c / n) / pixel_size  # cycles per unit length
        resolution = 1.0 / freq_c
        nyq_limited = bool(resolution <= nyquist * 1.1)
        if ring_c <= 1.0:
            notes.append(
                "the FRC curve is below the threshold at the first ring: the two "
                "images share no common structure at any frequency. Check they "
                "are the same field, and that drift between them was corrected "
                "before the split"
            )
        elif at_floor:
            notes.append(
                f"the curve was already below the {threshold} threshold on the "
                "first ring that criterion applies to, so this is the gate and "
                "not a measurement. The count-dependent criteria are not applied "
                f"to rings holding fewer than {_MIN_RING_SAMPLES} Fourier "
                "samples; re-run with threshold='1/7', which has no such gate"
            )
        elif ring_c < _MIN_RINGS:
            notes.append(
                f"the crossing is at ring {ring_c:.1f}, so fewer than "
                f"{_MIN_RINGS} rings carry the whole measurement and the "
                "innermost of those hold only a handful of Fourier samples each. "
                "The number is a coarse bound at best -- the two images have very "
                "little in common, or the field is too small for this structure"
            )
        if nyq_limited:
            notes.append(
                f"resolution {resolution:.4g} is within 10% of the {nyquist:.4g} "
                "sampling floor -- this measures the pixel size, not the data"
            )
    if n_cross > 1:
        notes.append(
            f"the curve crosses the threshold {n_cross} times; the first crossing "
            "was used. More than one crossing means the tail is noise -- raise "
            "`smooth`, or measure a larger field, before quoting this"
        )
    if not np.isnan(label_nyquist) and resolution < label_nyquist:
        notes.append(
            f"reported resolution {resolution:.4g} is finer than the "
            f"{label_nyquist:.4g} set by emitter density averaged over the whole "
            "field. If the emitters cover the field, structure is not resolved at "
            "this scale whatever the FRC says, and the density figure is the one "
            "to quote. If they sit on sparse structure with empty field between, "
            "this floor is pessimistic -- compare against the emitter spacing "
            "along the structure instead"
        )

    return FRCResult(
        resolution=float(resolution),
        threshold_name=threshold,
        crossing_frequency=float(freq_c),
        pixel_size=float(pixel_size),
        nyquist_resolution=float(nyquist),
        nyquist_limited=nyq_limited,
        n_crossings=n_cross,
        frequency=np.arange(n_rings) / (n * pixel_size),
        curve=curve,
        curve_smoothed=smoothed,
        threshold_curve=thr,
        ring_counts=counts,
        shape=(n, n),
        split=split,
        label_nyquist=float(label_nyquist),
        warnings=notes,
    )


# --------------------------------------------------------------------------- #
# the SMLM path: splitting a localization list
# --------------------------------------------------------------------------- #


def split_localizations(
    n_loc: int,
    frames=None,
    *,
    split: str = "blocks",
    block_frames: int = 500,
    seed: int = 0,
):
    """Boolean mask selecting the first of two halves of a localization list.

    **The split is the measurement.** A localization list is not a bag of
    independent samples: one fluorophore blinks many times and is localized once
    per blink, so any split that can put two blinks of the *same* molecule into
    opposite halves leaves those halves correlated by something that is not
    structure. FRC then reports a resolution better than the truth, silently.

    - ``"blocks"`` (default) -- consecutive runs of ``block_frames`` frames go
      alternately to A and B. A molecule's blinks cluster in time, so nearly all of
      them land in one block. Use this unless you have a reason not to. Its cost is
      that slow drift now differs between the halves and is charged against the
      resolution, which is the honest direction to err.
    - ``"halves"`` -- first half of the acquisition against the second. Fully
      decorrelates blinking, but confounds it with drift and bleaching, which make
      the two halves genuinely different images.
    - ``"random"`` -- every localization independently to A or B. Best matched for
      drift and density, and **wrong for resolution**: repeated blinks of one
      molecule are split across the halves and correlate them. Offered for
      comparison, not for reporting; it warns.

    Args:
        n_loc: Number of localizations.
        frames: Per-localization frame index. Required for ``"blocks"``/``"halves"``.
        split: One of ``"blocks"``, ``"halves"``, ``"random"``.
        block_frames: Frames per block for ``"blocks"``. Set it to a few times the
            typical molecular on-time, not to 1.
        seed: RNG seed for ``"random"``.

    Returns:
        ``(mask, notes)`` -- a boolean array selecting half A, and a list of
        warnings to carry into the result.
    """
    notes: list[str] = []
    if split == "random":
        rng = np.random.default_rng(seed)
        notes.append(
            "split='random' puts repeated blinks of the same molecule into both "
            "halves, correlating them and reporting a resolution better than the "
            "truth. Use split='blocks' for a number you intend to quote"
        )
        return rng.random(n_loc) < 0.5, notes
    if frames is None:
        raise ValueError(
            f"split={split!r} needs a per-localization frame index; pass `frames`, "
            "or use split='random' (not safe to quote -- see the docstring)"
        )
    fr = np.asarray(frames).ravel()
    if fr.shape != (n_loc,):
        raise ValueError(
            f"frames must have one entry per localization; got {fr.shape} for "
            f"{n_loc} localizations"
        )
    if split == "halves":
        mid = 0.5 * (float(fr.min()) + float(fr.max()))
        notes.append(
            "split='halves' separates the acquisition in time, so drift and "
            "bleaching between the halves are charged against the resolution"
        )
        return fr <= mid, notes
    if split == "blocks":
        if block_frames < 1:
            raise ValueError("block_frames must be >= 1")
        span = float(fr.max()) - float(fr.min())
        if span < 2 * block_frames:
            notes.append(
                f"the acquisition spans {span:.0f} frames but block_frames is "
                f"{block_frames}, giving fewer than two blocks per half -- this is "
                "effectively split='halves' and carries its drift caveat"
            )
        return ((fr - fr.min()) // block_frames) % 2 == 0, notes
    raise ValueError(
        f"unknown split {split!r}; expected 'blocks', 'halves' or 'random'"
    )


def frc_from_localizations(
    x,
    y,
    frames=None,
    *,
    render_pixel_size: float,
    split: str = "blocks",
    block_frames: int = 500,
    seed: int = 0,
    n_emitters: int | None = None,
    extent=None,
    **frc_kwargs,
) -> FRCResult:
    """Resolution of an SMLM reconstruction, from the localization list.

    Splits the list into two independent halves (see :func:`split_localizations`),
    renders each as a 2-D histogram on shared bin edges, and runs :func:`frc` on the
    pair. Renders **plain counts, not precision-weighted**: a weighted render
    convolves every localization with its own uncertainty, which is a filter applied
    equally to both halves and therefore invisible to FRC -- the number would then
    describe the renderer rather than the data.

    ``x`` and ``y`` are in whatever units you like; ``render_pixel_size`` must be in
    the same ones, and the reported resolution comes back in them. That is the whole
    unit contract -- there is no second pixel size here to get out of step with.

    Also computes the **density floor**: with *N* independent emitters spread over
    an area *A* the mean label spacing is ``sqrt(A/N)``, and structure cannot be
    resolved below about twice that (Shroff et al. 2008) whatever the localization
    precision. If FRC comes back finer than the floor, the result warns.

    The floor assumes the emitters **cover the rendered field**, which is the case
    it is a floor for. On a sample where they do not -- filaments, a membrane, any
    sparse structure in a mostly empty field -- it is pessimistic, because the
    spacing that limits a filament is the spacing *along* the filament and not the
    areal average over the empty space between them. Read the warning as "check
    this", not as a verdict: the number to compare against there is the emitter
    spacing along the structure.

    *N* is the number of **molecules**, not of localizations. One fluorophore
    localized twelve times samples the structure once, so passing an un-merged list
    without ``n_emitters`` understates the floor by the square root of the mean
    blink count -- 3.5x at twelve blinks, which is the difference between a floor
    that catches an overclaim and one that waves it through. Pass ``n_emitters`` if
    you have merged or linked the list; otherwise the result says the figure is
    optimistic.

    Args:
        x, y: Localization coordinates, in physical units.
        frames: Per-localization frame index. Required unless ``split="random"``.
        render_pixel_size: Render bin size, in the units of ``x``/``y``. It sets a
            hard floor of twice itself on the answer -- pick it well below the
            resolution you expect, typically 5-10x finer.
        split, block_frames, seed: Passed to :func:`split_localizations`.
        n_emitters: Number of distinct molecules, for the density floor. Defaults
            to the localization count, which is optimistic for blinking data.
        extent: ``(ymin, ymax, xmin, xmax)`` to render; default is a square
            bounding box of the localizations. Squared by expanding the shorter
            axis, so ``frc`` never has to centre-crop and discard data.
        **frc_kwargs: Forwarded to :func:`frc` (``threshold``, ``smooth``,
            ``apodize``).

    Returns:
        An :class:`FRCResult` with ``split`` and ``label_nyquist`` filled in.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must match in length; got {x.size} and {y.size}")
    if x.size < 2:
        raise ValueError("need at least two localizations")
    if render_pixel_size <= 0:
        raise ValueError("render_pixel_size must be positive")

    mask, notes = split_localizations(
        x.size, frames, split=split, block_frames=block_frames, seed=seed
    )
    n_a, n_b = int(mask.sum()), int((~mask).sum())
    if min(n_a, n_b) < 2:
        raise ValueError(
            f"the {split!r} split left {n_a} and {n_b} localizations; it cannot "
            "make two images. Check `frames`, or lower `block_frames`"
        )
    if max(n_a, n_b) > 3 * min(n_a, n_b):
        notes.append(
            f"the halves are unbalanced ({n_a} against {n_b} localizations), so "
            "their noise levels differ and the FRC curve is pulled down by the "
            "sparser one. Adjust `block_frames` to even them out"
        )

    if extent is None:
        ymin, ymax = float(y.min()), float(y.max())
        xmin, xmax = float(x.min()), float(x.max())
        # Square it up here by growing the shorter axis, rather than letting `frc`
        # centre-crop: the localizations are all real data and cropping throws
        # some away, whereas the empty margin an expansion adds costs nothing.
        side = max(ymax - ymin, xmax - xmin)
        ycen, xcen = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
        ymin, ymax = ycen - side / 2, ycen + side / 2
        xmin, xmax = xcen - side / 2, xcen + side / 2
    else:
        ymin, ymax, xmin, xmax = (float(v) for v in extent)
    ny = max(int(np.ceil((ymax - ymin) / render_pixel_size)), 1)
    nx = max(int(np.ceil((xmax - xmin) / render_pixel_size)), 1)
    yedges = ymin + render_pixel_size * np.arange(ny + 1)
    xedges = xmin + render_pixel_size * np.arange(nx + 1)

    img_a, _, _ = np.histogram2d(y[mask], x[mask], bins=(yedges, xedges))
    img_b, _, _ = np.histogram2d(y[~mask], x[~mask], bins=(yedges, xedges))

    area = (ymax - ymin) * (xmax - xmin)
    n_independent = x.size if n_emitters is None else int(n_emitters)
    label_nyquist = (
        2.0 * np.sqrt(area / n_independent)
        if area > 0 and n_independent > 0
        else float("nan")
    )
    if n_emitters is None:
        notes.append(
            f"the {label_nyquist:.4g} density floor counts localizations, not "
            "molecules. If this list is un-merged blinking data the true floor is "
            "coarser by the square root of the mean blink count -- pass "
            "`n_emitters` after linking to get a floor that can actually catch an "
            "overclaim"
        )

    result = frc(
        img_a,
        img_b,
        pixel_size=render_pixel_size,
        split=split,
        label_nyquist=label_nyquist,
        **frc_kwargs,
    )
    result.warnings = notes + result.warnings
    return result


# --------------------------------------------------------------------------- #
# decorrelation analysis (single image)
# --------------------------------------------------------------------------- #


def _decorrelation_curve(
    spec: np.ndarray, bins: np.ndarray, n_r: int, norm: float
) -> np.ndarray:
    """d(r) over the whole radius grid, in one pass over the image.

    d(r) is the correlation coefficient between the spectrum and its own phase-only
    version masked to ``|k| <= r``. The phase-only version has unit modulus, so the
    numerator ``sum Re(I conj(I/|I|))`` collapses to ``sum |I|`` and its norm to
    ``sqrt(count)``. Cumulative sums over radius bins then give every radius at once
    rather than one masked pass per radius, which is what makes the filter sweep
    affordable on a large field.

    ``norm`` is ``||I||`` over the full analysed disk and is constant in r: the
    coefficient is between the *whole* spectrum and the *masked* phase-only one.
    """
    mag = np.abs(spec)
    num = np.cumsum(np.bincount(bins, mag, n_r))
    counts = np.cumsum(np.bincount(bins, None, n_r)).astype(np.float64)
    if norm <= 0:
        return np.zeros(n_r)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = num / (norm * np.sqrt(counts))
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _local_max(d: np.ndarray) -> tuple[int, float]:
    """Position and height of the last meaningful local maximum of ``d``.

    A trailing rise is not a peak: a curve still climbing at the last radius has not
    told us where the signal stops. Strip the tail until the maximum is interior.
    """
    end = d.size
    while end > 1:
        i = int(np.argmax(d[:end]))
        if i < end - 1:
            return i, float(d[i])
        end -= 1
    return 0, float(d[0])


def decorrelation_resolution(
    image,
    *,
    pixel_size: float = 1.0,
    n_r: int = 50,
    n_filters: int = 10,
    apodize: float = 0.25,
    min_amplitude: float = 0.05,
) -> DecorrelationResult:
    """Single-image resolution by decorrelation analysis (Descloux et al. 2019).

    Use when no second independent realization exists: a deconvolved widefield, a
    SIM or eSRRF reconstruction, one micrograph. Where FRC asks "out to what
    frequency do two noisy copies agree", this asks "out to what frequency does the
    image agree with a whitened copy of itself" -- a disk of radius *r* in Fourier
    space is correlated against the phase-only spectrum, and the radius at which
    that correlation peaks marks where structure gives way to noise. Sweeping a
    Gaussian high-pass first exposes peaks the low-frequency bulk would otherwise
    bury; the highest-frequency surviving peak across the sweep wins.

    This is a reimplementation from the published description, not a port of the
    authors' MATLAB. It recovers a known band limit to within a few percent on
    synthetic fields (``_tests/test_image_resolution.py``) but has **not** been
    checked number-for-number against the reference implementation, so treat a
    few-percent disagreement with a published figure as expected.

    Args:
        image: 2-D plane. Non-square input is centre-cropped to a square.
        pixel_size: Physical size of one pixel; the result is in these units.
        n_r: Radii sampled between DC and Nyquist (default 50).
        n_filters: High-pass widths swept (default 10), geometrically spaced from a
            strong high-pass to one that removes little more than DC.
        apodize: Tukey taper fraction (default 0.25).
        min_amplitude: Peaks below this are discarded as noise. A heavily
            high-passed copy of a noisy image peaks near Nyquist on nothing at all,
            and without this floor that peak would set the answer.

    Returns:
        A :class:`DecorrelationResult`.
    """
    arr = _center_crop_square(_as_plane(image, "image"))
    n = arr.shape[0]
    if n < 2 * _MIN_RINGS:
        raise ValueError(f"image is {n}x{n} after cropping; too small to analyse")
    notes: list[str] = []

    spec = np.fft.fft2(_apodize(arr, apodize))
    # Normalized frequency, 1.0 at Nyquist: the corners run out to sqrt(2) and are
    # excluded, so no bin is fed by only two orientations.
    kn = 2.0 * _radial_frequency(n)
    inside = kn <= 1.0

    radii = np.linspace(0.0, 1.0, n_r)
    # Bin j collects |k| in (r_{j-1}, r_j], so a cumulative sum over bins is exactly
    # the sum over the disk |k| <= r_j.
    bins = np.clip(np.ceil(kn * (n_r - 1)).astype(np.intp), 0, n_r - 1)[inside]
    kn_in = kn[inside]
    spec_in = spec[inside]

    # Small sigma is a strong high-pass (im - blur(im, sigma) keeps only what the
    # blur removed); large sigma removes little beyond DC. Both ends are wanted:
    # the strong end finds peaks buried under the low-frequency bulk.
    sigmas = np.concatenate([[0.0], np.geomspace(0.2, max(n / 4.0, 1.0), n_filters)])
    curves = np.empty((sigmas.size, n_r))
    kcs = np.zeros(sigmas.size)
    amps = np.zeros(sigmas.size)
    for i, sigma in enumerate(sigmas):
        if sigma == 0.0:
            filtered = spec_in
        else:
            filtered = spec_in * (1.0 - np.exp(-_HP_A * (sigma**2) * (kn_in**2)))
        norm = float(np.sqrt(np.sum(np.abs(filtered) ** 2)))
        d = _decorrelation_curve(filtered, bins, n_r, norm)
        curves[i] = d
        j, a = _local_max(d)
        kcs[i] = radii[j]
        amps[i] = a

    ok = amps >= min_amplitude
    if not np.any(ok):
        notes.append(
            f"no decorrelation peak reached amplitude {min_amplitude}; there is no "
            "frequency band where structure dominates noise here, so the figure "
            "below is the sampling limit rather than a measurement"
        )
        kc, amp = 1.0, float(amps.max())
    else:
        best = int(np.argmax(np.where(ok, kcs, -1.0)))
        kc, amp = float(kcs[best]), float(amps[best])

    nyquist = 2.0 * pixel_size
    if kc <= 0:
        resolution = float("inf")
        nyq_limited = False
        notes.append(
            "the decorrelation curve peaked at DC: no resolvable structure in "
            "this field"
        )
    else:
        resolution = 2.0 * pixel_size / kc
        nyq_limited = bool(kc >= radii[-2])
        if nyq_limited:
            notes.append(
                f"the peak sits at the Nyquist end of the sweep, so {nyquist:.4g} "
                "is a bound and not a measurement -- the image is undersampled "
                "for whatever resolution it actually has"
            )

    return DecorrelationResult(
        resolution=float(resolution),
        kc=float(kc),
        amplitude=float(amp),
        pixel_size=float(pixel_size),
        nyquist_resolution=float(nyquist),
        nyquist_limited=nyq_limited,
        radii=radii,
        curves=curves,
        filter_sigmas=sigmas,
        kc_per_filter=kcs,
        amplitude_per_filter=amps,
        shape=(n, n),
        warnings=notes,
    )
