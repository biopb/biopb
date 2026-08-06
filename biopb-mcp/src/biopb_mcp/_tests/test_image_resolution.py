"""Unit tests for the resolution plugin (biopb_mcp.plugins.image_resolution).

The point of this plugin is that its output is a single plausible-looking number,
so these tests are mostly about **whether the number is right** rather than about
whether the code runs. The fixture is a field with a hard band limit at a known
cutoff: past that cutoff two noisy copies share nothing, so the FRC curve must
fall through its threshold there, and the decorrelation peak must sit there too.
That truth is not reachable from anything the estimators are handed.

Both estimators come back slightly *optimistic* on this fixture, by a roughly
constant number of rings rather than a constant fraction. That is the edge taper
convolving the spectrum and smearing a hard cutoff -- an artifact of a fixture
with a discontinuity in Fourier space, which no real image has. Tolerances are
therefore one-sided and stated in rings, not percent.

Also pins the two regressions that cost the most to rediscover: the count-
dependent thresholds snapping to a fixed low ring for every input, and ring 0's
forced 1.0 leaking through the smoother into the rings that decide the crossing.
And the delivery path, as the other plugin suites do. No kernel or display needed.
"""

import numpy as np
import pytest

from biopb_mcp.plugins import image_resolution as ir

N = 256
FIELD, N_SEG, N_FRAMES = 4000.0, 25, 4000


def bandlimited(n=N, kc=0.2, seed=0):
    """White noise with every frequency above ``kc`` (cycles/pixel) removed."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    spec = np.fft.fft2(rng.standard_normal((n, n)))
    spec[np.hypot(f[:, None], f[None, :]) > kc] = 0
    img = np.real(np.fft.ifft2(spec))
    return img / img.std()


def noisy_pair(kc=0.2, n=N, sd=1.0, seed=99):
    """Two independent noisy realizations of one band-limited field."""
    truth = bandlimited(n, kc)
    rng = np.random.default_rng(seed)
    return (
        truth + sd * rng.standard_normal((n, n)),
        truth + sd * rng.standard_normal((n, n)),
    )


def smlm(per_seg=160, mean_blinks=12, precision=15.0, seed=0):
    """Molecules decorating random filaments, each blinking in a burst.

    Structure the two halves can *both* see, which is what FRC needs: a field of
    isolated molecules has no structure beyond the molecules themselves, so a
    split that separates them correctly reports nothing in common.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N_SEG):
        p0 = rng.uniform(0, FIELD, 2)
        ang = rng.uniform(0, np.pi)
        t = rng.uniform(0, 1, per_seg)[:, None]
        pts.append(p0 + t * 1600.0 * np.array([np.cos(ang), np.sin(ang)]))
    mols = np.clip(np.vstack(pts), 0, FIELD)
    n_mol = mols.shape[0]
    blinks = rng.poisson(mean_blinks, n_mol) + 1
    mol = np.repeat(np.arange(n_mol), blinks)
    onset = rng.uniform(0, N_FRAMES, n_mol)
    frames = np.clip(
        np.repeat(onset, blinks) + rng.normal(0, 30, mol.size), 0, N_FRAMES - 1
    )
    x = mols[mol, 0] + rng.normal(0, precision, mol.size)
    y = mols[mol, 1] + rng.normal(0, precision, mol.size)
    return x, y, frames, n_mol


class TestRecoversAKnownBandLimit:
    """The measurement itself: does the number match a truth it cannot see?"""

    @pytest.mark.parametrize("kc", [0.15, 0.25, 0.35])
    @pytest.mark.parametrize("threshold", ir.THRESHOLDS)
    def test_crossing_lands_on_the_cutoff(self, kc, threshold):
        r = ir.frc(*noisy_pair(kc), threshold=threshold)
        rings = r.crossing_frequency * N
        # One-sided: the taper smears the hard cutoff outward by a few rings and
        # never inward. 8 rings is 3% at the coarsest cutoff tested.
        assert 0 <= rings - kc * N <= 8, (
            f"{threshold} crossed at ring {rings:.1f}, truth {kc * N:.1f}"
        )
        assert r.n_crossings == 1

    def test_resolution_is_the_reciprocal_in_physical_units(self):
        r = ir.frc(*noisy_pair(0.25), pixel_size=0.1)
        assert r.resolution == pytest.approx(1.0 / r.crossing_frequency)
        # 0.25 cyc/px at 0.1 um/px is 4 px, i.e. 0.4 um.
        assert r.resolution == pytest.approx(0.4, rel=0.1)
        assert r.nyquist_resolution == pytest.approx(0.2)

    def _crossing_at_noise(self, sd, kc=0.25):
        truth = bandlimited(N, kc)
        rng = np.random.default_rng(7)
        return ir.frc(
            truth + sd * rng.standard_normal((N, N)),
            truth + sd * rng.standard_normal((N, N)),
        )

    def test_the_answer_holds_over_a_wide_noise_range(self):
        # A band limit is a band limit: over 20x in noise the crossing moves by
        # 5%, because SNR sets how far above the threshold the plateau sits and
        # not where the signal stops.
        got = [self._crossing_at_noise(sd).crossing_frequency for sd in (0.1, 1.0, 2.0)]
        assert max(got) - min(got) < 0.02
        assert all(g == pytest.approx(0.25, rel=0.1) for g in got)

    def test_but_it_does_collapse_below_usable_snr_and_says_so(self):
        """Resolution is a property of the image, not of the optics, so a
        band-limited field at SNR 0.3 really is worse than its band limit. What
        matters is that it collapses loudly rather than degrading quietly."""
        r = self._crossing_at_noise(4.0)
        assert r.crossing_frequency < 0.05
        assert any("no common structure" in w for w in r.warnings)

    @pytest.mark.parametrize("kc", [0.1, 0.2, 0.35])
    def test_decorrelation_finds_the_same_cutoff(self, kc):
        truth = bandlimited(N, kc)
        rng = np.random.default_rng(5)
        d = ir.decorrelation_resolution(
            truth + 0.5 * rng.standard_normal((N, N)), pixel_size=1.0
        )
        # kc is reported against Nyquist (0.5 cyc/px), hence the factor 2. The
        # tolerance is set by the radius grid, not by the estimator: 50 samples
        # over [0, 1] quantise kc to 1/49, which is 5% of a mid-range answer.
        assert d.kc == pytest.approx(2 * kc, abs=0.04)
        assert d.resolution == pytest.approx(1.0 / kc, rel=0.10)

    def test_a_finer_radius_grid_tightens_the_decorrelation_answer(self):
        truth = bandlimited(N, 0.2)
        rng = np.random.default_rng(5)
        img = truth + 0.5 * rng.standard_normal((N, N))
        coarse = ir.decorrelation_resolution(img, n_r=20)
        fine = ir.decorrelation_resolution(img, n_r=200)
        assert abs(fine.kc - 0.4) < abs(coarse.kc - 0.4)
        # What is left once the grid is fine is the taper leakage, and it runs the
        # same way as FRC's: optimistic, never pessimistic, by a few percent.
        assert 0.94 * 5.0 <= fine.resolution <= 5.0

    def test_the_two_estimators_agree_on_one_field(self):
        a, b = noisy_pair(0.2)
        f = ir.frc(a, b)
        d = ir.decorrelation_resolution(a)
        assert f.resolution == pytest.approx(d.resolution, rel=0.2)


class TestThresholds:
    def test_count_dependent_criteria_do_not_snap_to_a_fixed_ring(self):
        """Regression: ring sample counts are not monotonic in radius, so gated
        rings appear well past the first usable one. Left in the search they read
        as an infinitely negative margin -- a crossing -- and every input reports
        the same low ring whatever its real cutoff."""
        for name in ("half-bit", "3sigma"):
            got = [
                ir.frc(*noisy_pair(kc), threshold=name).crossing_frequency
                for kc in (0.1, 0.2, 0.35)
            ]
            assert got[0] < got[1] < got[2], f"{name} reported {got}"

    def test_the_gate_is_not_applied_to_the_constant_criterion(self):
        # 1/7 does not depend on the ring count, so it has nothing to gate and a
        # first-ring crossing under it is a real answer rather than an artifact.
        counts = np.arange(200)
        assert np.all(np.isfinite(ir._threshold_curve("1/7", counts)))
        gated = ir._threshold_curve("3sigma", counts)
        assert np.isinf(gated[:8]).all()
        assert np.isfinite(gated[ir._MIN_RING_SAMPLES :]).all()

    def test_unknown_threshold_is_refused(self):
        with pytest.raises(ValueError, match="unknown threshold"):
            ir.frc(*noisy_pair(), threshold="1/2-bit")

    def test_the_criterion_is_carried_in_the_result(self):
        r = ir.frc(*noisy_pair(), threshold="half-bit")
        assert r.threshold_name == "half-bit"
        assert "half-bit" in r.summary()
        assert r.to_dict()["threshold_name"] == "half-bit"


class TestSmoothingAndApodization:
    def test_ring_zero_does_not_leak_into_the_crossing(self):
        """Regression: ring 0 is set to 1.0 as a display convention. Smoothed in
        with the rest it holds up exactly the low rings that decide a crossing,
        and an image with nothing in common reports a resolution instead."""
        rng = np.random.default_rng(3)
        a, b = rng.standard_normal((N, N)), rng.standard_normal((N, N))
        r = ir.frc(a, b, smooth=9)
        assert r.curve[0] == 1.0
        assert r.curve_smoothed[0] == 1.0
        assert abs(r.curve_smoothed[1]) < 0.3, "ring 0 bled into ring 1"
        assert any("no common structure" in w for w in r.warnings)

    def test_heavy_smoothing_biases_the_crossing_outward(self):
        a, b = noisy_pair(0.1)
        light = ir.frc(a, b, smooth=3).crossing_frequency
        heavy = ir.frc(a, b, smooth=41).crossing_frequency
        assert heavy > light * 1.05
        # ... which is why the default stays small rather than tracking the curve
        # length: a twentieth of the curve is a several-percent bias.
        assert ir.frc(a, b).crossing_frequency == pytest.approx(light, rel=0.02)

    def test_apodization_pulls_a_non_periodic_field_back_toward_truth(self):
        # Generated large and cropped, so the field does not wrap: the FFT sees a
        # step at the edge and puts correlated power along the axes in both halves.
        big = bandlimited(2 * N, 0.15)
        rng = np.random.default_rng(11)
        a = big[:N, :N] + rng.standard_normal((N, N))
        b = big[:N, :N] + rng.standard_normal((N, N))
        bare = ir.frc(a, b, apodize=0.0).crossing_frequency
        taper = ir.frc(a, b, apodize=0.25).crossing_frequency
        assert taper < bare, "the taper should reduce the leakage, not add to it"
        assert abs(taper - 0.15) < abs(bare - 0.15)


class TestDegenerateInput:
    def test_identical_images_report_nyquist_and_say_so(self):
        t = bandlimited(N, 0.2)
        r = ir.frc(t, t.copy(), pixel_size=0.1)
        assert r.resolution == pytest.approx(0.2)
        assert r.nyquist_limited
        assert any("never falls to the threshold" in w for w in r.warnings)

    def test_an_uncorrelated_pair_is_flagged_not_scored(self):
        rng = np.random.default_rng(3)
        r = ir.frc(rng.standard_normal((N, N)), rng.standard_normal((N, N)))
        assert r.warnings, "a meaningless answer came back with nothing said"

    def test_shape_and_dimensionality_are_refused_not_coerced(self):
        with pytest.raises(ValueError, match="match in shape"):
            ir.frc(np.zeros((64, 64)), np.zeros((64, 32)))
        with pytest.raises(ValueError, match="2-D plane"):
            ir.frc(np.zeros((4, 64, 64)), np.zeros((4, 64, 64)))
        with pytest.raises(ValueError, match="too small"):
            ir.frc(np.zeros((8, 8)), np.zeros((8, 8)))
        with pytest.raises(ValueError, match="too small to analyse"):
            ir.decorrelation_resolution(np.zeros((8, 8)))

    def test_non_square_is_cropped_and_announced(self):
        a, b = noisy_pair(0.2)
        r = ir.frc(a[:, :200], b[:, :200])
        assert r.shape == (200, 200)
        assert any("not square" in w for w in r.warnings)


class TestSplittingALocalizationList:
    def test_blocks_keeps_a_molecules_blinks_on_one_side(self):
        # Two molecules, each blinking inside its own 100-frame window; the
        # windows fall in adjacent blocks, which alternate between the halves.
        frames = np.array([10, 20, 30, 110, 120, 130])
        mask, _ = ir.split_localizations(6, frames, split="blocks", block_frames=100)
        assert mask[:3].all() and not mask[3:].any()
        # Blocks alternate, so two molecules four blocks apart land on the same
        # side -- the guarantee is that one molecule is not split, not that two
        # different ones are separated.
        far = np.array([10, 20, 30, 410, 420, 430])
        mask, _ = ir.split_localizations(6, far, split="blocks", block_frames=100)
        assert mask.all()

    def test_random_split_warns_that_it_is_not_quotable(self):
        _, notes = ir.split_localizations(100, split="random")
        assert any("better than the truth" in n for n in notes)

    def test_halves_warns_about_drift(self):
        _, notes = ir.split_localizations(4, np.arange(4), split="halves")
        assert any("drift" in n for n in notes)

    def test_too_few_blocks_is_called_out(self):
        _, notes = ir.split_localizations(
            4, np.arange(4), split="blocks", block_frames=500
        )
        assert any("effectively split='halves'" in n for n in notes)

    def test_a_time_split_without_frames_is_refused(self):
        with pytest.raises(ValueError, match="needs a per-localization frame index"):
            ir.split_localizations(10, None, split="blocks")
        with pytest.raises(ValueError, match="one entry per localization"):
            ir.split_localizations(10, np.arange(4), split="blocks")

    def test_unknown_split_is_refused(self):
        with pytest.raises(ValueError, match="unknown split"):
            ir.split_localizations(10, np.arange(10), split="odd-even")


class TestFromLocalizations:
    def test_a_random_split_reports_a_better_number_than_it_should(self):
        """The claim the module is built around, on the sparse labelling where it
        bites: the same molecule lands in both halves and correlates them, so the
        answer improves without the data improving."""
        x, y, frames, n_mol = smlm(per_seg=12)
        kw = {"render_pixel_size": 6.0, "block_frames": 200, "n_emitters": n_mol}
        blocks = ir.frc_from_localizations(x, y, frames, split="blocks", **kw)
        random = ir.frc_from_localizations(x, y, frames, split="random", **kw)
        assert random.resolution < blocks.resolution * 0.8
        assert random.split == "random"
        assert any("better than the truth" in w for w in random.warnings)

    def test_dense_labelling_is_where_the_split_stops_mattering(self):
        # Same trap, structure sampled finely enough that the molecules are not
        # what limits it -- the two splits then agree, which is why the warning is
        # a warning and not a correction.
        x, y, frames, n_mol = smlm(per_seg=400)
        kw = {"render_pixel_size": 6.0, "block_frames": 200, "n_emitters": n_mol}
        blocks = ir.frc_from_localizations(x, y, frames, split="blocks", **kw)
        random = ir.frc_from_localizations(x, y, frames, split="random", **kw)
        assert random.resolution == pytest.approx(blocks.resolution, rel=0.1)

    def test_the_render_is_squared_by_expansion_so_nothing_is_cropped(self):
        x, y, frames, n_mol = smlm(per_seg=60)
        y = y * 0.5  # a decidedly non-square field
        r = ir.frc_from_localizations(
            x, y, frames, render_pixel_size=8.0, n_emitters=n_mol
        )
        assert r.shape[0] == r.shape[1]
        assert not any("not square" in w for w in r.warnings)

    def test_render_pixel_size_is_a_hard_floor(self):
        x, y, frames, n_mol = smlm(per_seg=160)
        kw = {"block_frames": 200, "n_emitters": n_mol}
        fine = ir.frc_from_localizations(x, y, frames, render_pixel_size=6.0, **kw)
        coarse = ir.frc_from_localizations(x, y, frames, render_pixel_size=60.0, **kw)
        assert coarse.resolution >= 2 * 60.0
        assert coarse.nyquist_limited and not fine.nyquist_limited
        # Two ways to hit the floor and both must speak: the curve either never
        # reaches the threshold (a bound), or crosses within 10% of it (a floor).
        assert any(
            "bound and not a measurement" in w or "sampling floor" in w
            for w in coarse.warnings
        )

    def test_the_density_floor_counts_molecules_not_blinks(self):
        x, y, frames, n_mol = smlm(per_seg=60, mean_blinks=12)
        kw = {"render_pixel_size": 8.0, "block_frames": 200}
        merged = ir.frc_from_localizations(x, y, frames, n_emitters=n_mol, **kw)
        raw = ir.frc_from_localizations(x, y, frames, **kw)
        # sqrt of the mean blink count, which is the whole point of the argument.
        assert merged.label_nyquist > raw.label_nyquist * 2
        assert any("counts localizations, not" in w for w in raw.warnings)
        assert not any("counts localizations, not" in w for w in merged.warnings)

    def test_units_are_whatever_the_coordinates_are(self):
        x, y, frames, n_mol = smlm(per_seg=160)
        kw = {"split": "blocks", "block_frames": 200, "n_emitters": n_mol}
        nm = ir.frc_from_localizations(x, y, frames, render_pixel_size=8.0, **kw)
        um = ir.frc_from_localizations(
            x / 1000, y / 1000, frames, render_pixel_size=0.008, **kw
        )
        assert um.resolution * 1000 == pytest.approx(nm.resolution, rel=0.02)

    def test_a_split_that_empties_a_half_is_refused(self):
        with pytest.raises(ValueError, match="cannot\n?\\s*make two images"):
            ir.frc_from_localizations(
                np.arange(10.0),
                np.arange(10.0),
                np.zeros(10),
                render_pixel_size=1.0,
                block_frames=500,
            )


class TestDecorrelation:
    def test_pure_noise_does_not_produce_a_confident_number(self):
        rng = np.random.default_rng(4)
        d = ir.decorrelation_resolution(rng.standard_normal((N, N)))
        assert d.warnings

    def test_the_filter_sweep_is_reported(self):
        d = ir.decorrelation_resolution(bandlimited(N, 0.2), n_filters=6)
        assert d.curves.shape == (7, 50)  # the unfiltered curve plus the sweep
        assert d.filter_sigmas[0] == 0.0
        assert d.kc in set(d.kc_per_filter)
        assert "decorrelation resolution" in d.summary()

    def test_it_inverts_on_a_point_cloud_which_is_why_the_docstring_forbids_it(self):
        """Pins the reason ``frc`` is the only estimator offered for localizations.

        Two samplings of one structure: many points, and few. The sparse render is
        genuinely sharp and genuinely unfaithful; the dense one is shot-noise
        limited and faithful. With a single image there is nothing to tell those
        apart, so decorrelation calls the sparse one finer -- backwards, and by a
        larger factor than the FRC splitting mistake it would be standing in for.
        """
        side = 512
        f = np.fft.fftfreq(side)
        q = np.hypot(f[:, None], f[None, :])
        spec = np.fft.fft2(np.random.default_rng(0).standard_normal((side, side)))
        with np.errstate(divide="ignore"):
            spec = spec * np.where(q > 0, q**-1.5, 0.0)  # low-frequency-heavy, as
        spec[q > 0.25] = 0  # biological structure is
        m = np.real(np.fft.ifft2(spec))
        m -= m.min()
        cdf = np.cumsum((m / m.sum()).ravel())

        got = {}
        # Same total counts in both, so the two renders carry the same number of
        # dots; only the number of distinct sites those dots came from differs.
        for label, n_sites, repeats in (("dense", 200_000, 1), ("sparse", 4_000, 50)):
            rng = np.random.default_rng(0)
            flat = np.minimum(np.searchsorted(cdf, rng.random(n_sites)), m.size - 1)
            img = (
                np.bincount(np.repeat(flat, repeats), minlength=m.size)
                .reshape(side, side)
                .astype(float)
            )
            got[label] = ir.decorrelation_resolution(img)

        assert got["sparse"].resolution < got["dense"].resolution, (
            "decorrelation is expected to invert on point clouds; if it now orders "
            "them correctly, the module docstring's prohibition is stale"
        )
        # Sharper than the inversion: the sparse render, which is the *worse*
        # reconstruction, comes back pinned at the sampling floor -- reported as
        # perfect. There is no threshold at which a caller could catch that.
        assert got["sparse"].nyquist_limited

    def test_to_dict_drops_the_curves(self):
        d = ir.decorrelation_resolution(bandlimited(N, 0.2))
        flat = d.to_dict()
        assert "curves" not in flat and "radii" not in flat
        assert flat["resolution"] == d.resolution


class TestSeeding:
    """The delivery path: the installer seeds the plugin into the kernel dir."""

    def test_seed_includes_the_resolution_plugin(self, tmp_path):
        from biopb_mcp.plugins._seed import SEED_FILES, seed_kernel_plugins

        assert "image_resolution.py" in SEED_FILES
        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)
        assert (dest / "image_resolution.py").exists()

    def test_seeded_file_loads_with_a_clean_namespace_surface(self, tmp_path):
        from biopb_mcp.mcp import _bootstrap
        from biopb_mcp.plugins._seed import seed_kernel_plugins

        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)
        # Other seeded plugins have their own surface tests; drop them so this
        # assertion stays an exact set for *this* file rather than a superset.
        for other in dest.glob("*.py"):
            if other.name not in ("__init__.py", "image_resolution.py"):
                other.unlink()

        class IP:
            def __init__(self):
                self.user_ns = {"viewer": 1, "client": 1, "np": np, "da": 1, "ops": {}}

        ip = IP()
        _bootstrap._load_plugin_files(ip, dest)
        builtins_ = {"viewer", "client", "np", "da", "ops"}
        contributed = {
            n for n in ip.user_ns if not n.startswith("_") and n not in builtins_
        }
        assert contributed == {"image_resolution"}
        plug = ip.user_ns["image_resolution"]
        assert set(ir.__all__) <= set(dir(plug))
        assert ip.user_ns["np"] is np  # reserved handle untouched

    def test_seeded_plugin_is_callable_from_the_namespace(self, tmp_path):
        from biopb_mcp.mcp import _bootstrap
        from biopb_mcp.plugins._seed import seed_kernel_plugins

        dest = tmp_path / "kernel"
        seed_kernel_plugins(dest)

        class IP:
            def __init__(self):
                self.user_ns = {"viewer": 1, "client": 1, "np": np, "da": 1, "ops": {}}

        ip = IP()
        _bootstrap._load_plugin_files(ip, dest)
        a, b = noisy_pair(0.25)
        got = ip.user_ns["image_resolution"].frc(a, b)
        assert got.resolution == pytest.approx(4.0, rel=0.15)
