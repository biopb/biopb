"""A streamed reduction must equal the whole-extent one, byte for byte.

Streaming is a residency and read-shape decision; it must not also be a decision
about what the data is. So the core of this file is one differential test: fold
an extent unit by unit, reduce the same extent whole, demand identical bytes.

Two axes are swept deliberately, because each has a history of being the one
that breaks:

- **Geometry.** An extent that ends mid-block carries ``downsample_block``'s
  edge-replicate pad and its full-block divisor, and it is the terminal unit
  that has to reproduce them. Offsets and per-axis scales are swept with it.
- **Block size across the kernel gate.** ``downsample_block`` switches
  implementation at ``_STRIDED_ADD_MAX_BLOCK``. Cases on both sides are
  required: a suite that only exercises small blocks leaves the branch every
  scale-32 read actually takes untested.

The one thing not swept is a unit that splits a reduction block, because
:func:`stream_reduce` refuses it rather than folding it -- see
``test_a_unit_that_splits_a_block_is_refused``.
"""

from __future__ import annotations

import numpy as np
import pytest
from biopb_tensor_server.core.downsample import (
    _STRIDED_ADD_MAX_BLOCK,
    downsample_block,
)
from biopb_tensor_server.core.stream_reduce import (
    _CONTIGUOUS_BAND_BYTES,
    covering_units,
    stream_reduce,
    streaming_unit,
)

# name -> (shape, start, stop, unit, scale)
_CASES = {
    # The easy case: the unit divides the extent and the scale divides the unit.
    "divides": (
        (1, 4, 1, 64, 64),
        (0, 0, 0, 0, 0),
        None,
        (1, 4, 1, 8, 8),
        (1, 1, 1, 4, 4),
    ),
    # The unit does not divide the extent, so the last one is short.
    "short-last-unit": (
        (1, 3, 1, 60, 60),
        (0, 0, 0, 0, 0),
        None,
        (1, 3, 1, 8, 8),
        (1, 1, 1, 4, 4),
    ),
    # The extent ends mid-block, so the edge pad and full-block divisor apply.
    "ragged": (
        (1, 2, 1, 30, 30),
        (0, 0, 0, 0, 0),
        None,
        (1, 2, 1, 8, 8),
        (1, 1, 1, 4, 4),
    ),
    # Ragged on both spatial axes at once, so the pad has a corner.
    "corner": (
        (1, 1, 1, 30, 26),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 8, 8),
        (1, 1, 1, 4, 4),
    ),
    # An extent that does not start at the origin: units lay from the extent.
    "offset": (
        (1, 2, 1, 64, 64),
        (0, 0, 0, 16, 16),
        (1, 2, 1, 48, 48),
        (1, 2, 1, 8, 8),
        (1, 1, 1, 2, 2),
    ),
    # A different scale per axis, including a reduced C.
    "aniso": (
        (1, 4, 1, 32, 48),
        (0, 0, 0, 0, 0),
        None,
        (1, 4, 1, 8, 8),
        (1, 2, 1, 4, 8),
    ),
    # Block 256: the last size the strided kernel takes.
    "block256": (
        (1, 1, 1, 128, 128),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 32, 32),
        (1, 1, 1, 16, 16),
    ),
    # Block 1024: what a scale-32 read asks for, on the far side of the gate.
    "block1024": (
        (1, 1, 1, 256, 256),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 64, 64),
        (1, 1, 1, 32, 32),
    ),
    # Deep scale AND a ragged end AND a short last unit, all at once.
    "block1024-ragged": (
        (1, 1, 1, 200, 200),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 64, 64),
        (1, 1, 1, 32, 32),
    ),
    # Float area and a non-dyadic scale: the cases that used to have to decline.
    "float-area": (
        (1, 1, 1, 96, 96),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 24, 24),
        (1, 1, 1, 8, 8),
    ),
    "non-dyadic": (
        (1, 1, 1, 90, 90),
        (0, 0, 0, 0, 0),
        None,
        (1, 1, 1, 15, 15),
        (1, 1, 1, 5, 5),
    ),
}


def _case(name):
    shape, start, stop, unit, scale = _CASES[name]
    return shape, start, (stop or shape), unit, scale


def _random(shape, dtype, seed=0):
    info = np.iinfo(np.dtype(dtype))
    rng = np.random.default_rng(seed)
    return rng.integers(info.min, info.max + 1, size=shape).astype(dtype)


def _reader(data):
    def fetch(unit_start, unit_stop):
        return data[
            tuple(slice(a, b) for a, b in zip(unit_start, unit_stop, strict=True))
        ].copy()

    return fetch


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("dtype", ["<u2", "|u1", "<i2", "<u4"])
@pytest.mark.parametrize("method", ["area", "nearest"])
def test_streamed_reduction_is_byte_identical(case, dtype, method):
    shape, start, stop, unit, scale = _case(case)
    data = _random(shape, dtype)

    streamed = stream_reduce(
        _reader(data), start, stop, shape, unit, scale, method, dtype
    )
    direct = downsample_block(
        data[tuple(slice(a, b) for a, b in zip(start, stop, strict=True))],
        scale,
        method,
    )

    assert streamed is not None
    assert streamed.dtype == direct.dtype
    assert streamed.shape == direct.shape
    assert np.array_equal(streamed, direct)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_both_sides_of_the_kernel_gate_are_covered(case):
    """The suite is worthless if every case lands on one side of the gate."""
    blocks = [int(np.prod(_case(name)[4])) for name in _CASES]
    assert any(block <= _STRIDED_ADD_MAX_BLOCK for block in blocks)
    assert any(block > _STRIDED_ADD_MAX_BLOCK for block in blocks)


@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
@pytest.mark.parametrize("case", ["float-area", "non-dyadic", "divides"])
def test_float_area_streams_rather_than_declining(case, dtype):
    """The staged float means are reproduced, not approximated and not refused.

    They neither commute nor associate, which is why a *straddling* fold cannot
    do this. Whole blocks inside a unit can: every output element is computed
    from its own block, by the same code, in the same order.
    """
    shape, start, stop, unit, scale = _case(case)
    data = (np.random.default_rng(0).random(shape) * 1000).astype(dtype)

    streamed = stream_reduce(
        _reader(data), start, stop, shape, unit, scale, "area", dtype
    )
    direct = downsample_block(data, scale, "area")

    assert np.array_equal(streamed, direct)


def test_a_unit_that_splits_a_block_is_refused():
    """Folding the halves of a split block separately would change the pixel.

    A programming error rather than a runtime condition -- ``streaming_unit``
    cannot produce one -- so it raises instead of silently reading whole.
    """
    shape, start, stop, _, scale = _case("divides")
    data = _random(shape, "<u2")
    with pytest.raises(ValueError, match="splits a reduction block"):
        stream_reduce(
            _reader(data), start, stop, shape, (1, 4, 1, 6, 6), scale, "area", "<u2"
        )


class TestCoveringUnits:
    def test_units_tile_the_extent_exactly(self):
        start, stop, unit = (0, 16), (48, 80), (16, 32)
        seen = np.zeros((48, 80), dtype=int)
        for a, b in covering_units(start, stop, unit, (48, 80)):
            seen[a[0] : b[0], a[1] : b[1]] += 1
        covered = seen[start[0] : stop[0], start[1] : stop[1]]
        assert (covered == 1).all(), "units must tile without gap or overlap"
        assert seen.sum() == covered.size, "no unit may reach outside the extent"

    def test_units_lay_from_the_extent_not_the_origin(self):
        """An extent starting off-grid still tiles; the first unit is short."""
        first = next(iter(covering_units((5,), (37,), (16,), (64,))))
        assert first == ((5,), (21,))


class TestStreamingUnit:
    """A unit is the largest sequential run the backend can deliver.

    Which shape that is depends on the layout, and the two disasters this rule
    exists to avoid are symmetric: bands on a chunked store re-read stored chunks
    (7.1x), square tiles on a contiguous one read strided (1.3x on a cold plane).
    """

    def test_chunked_store_takes_the_transfer_grid(self):
        unit = streaming_unit((8192, 8192), (2048, 2048), (512, 512), (8, 8), 2)
        assert unit == (2048, 2048)

    def test_chunked_store_floors_at_a_block_larger_than_the_grid(self):
        """The 3.2x case: a stored chunk bigger than the transfer target."""
        unit = streaming_unit((8192, 8192), (2048, 2048), (4096, 4096), (8, 8), 2)
        assert unit == (4096, 4096)

    def test_floor_rounds_up_to_a_whole_multiple_of_the_grid(self):
        """A unit boundary must not land inside a stored block."""
        unit = streaming_unit((16384, 16384), (2048, 2048), (3000, 3000), (8, 8), 2)
        assert unit == (4096, 4096)
        assert unit[0] % 2048 == 0

    def test_chunked_unit_never_exceeds_the_extent(self):
        assert streaming_unit((1000, 1000), (2048, 2048), (4096, 4096), (8, 8), 2) == (
            1000,
            1000,
        )

    def test_contiguous_store_takes_a_full_width_band(self):
        """Full width: a narrower unit reads strided, at 1.3x on a cold plane."""
        unit = streaming_unit((14234, 14234), (1182, 1182), None, (8, 8), 2)
        assert unit[1] == 14234, "band must span the full width"
        assert unit[0] < 14234, "band must not be the whole extent"

    def test_band_depth_fills_the_budget(self):
        unit = streaming_unit((14234, 14234), (1182, 1182), None, (8, 8), 2)
        assert unit[0] * 14234 * 2 <= _CONTIGUOUS_BAND_BYTES
        assert (unit[0] + 8) * 14234 * 2 > _CONTIGUOUS_BAND_BYTES

    def test_band_depth_is_a_whole_number_of_blocks(self):
        """So the fold can use the shared kernels instead of reduceat."""
        for scale in (2, 4, 8, 32):
            unit = streaming_unit((14234, 14234), (1182, 1182), None, (scale, scale), 2)
            assert unit[0] % scale == 0

    def test_band_spans_every_axis_but_rows(self):
        """An interleaved ND2 keeps C whole: only the row axis is divided."""
        extent = (1, 3, 1, 14234, 14234)
        unit = streaming_unit(extent, (1, 3, 1, 1182, 1182), None, (1, 1, 1, 8, 8), 2)
        assert unit[:3] == (1, 3, 1)
        assert unit[4] == 14234
        assert unit[3] < 14234

    def test_band_never_exceeds_the_extent(self):
        assert streaming_unit((64, 64), (32, 32), None, (4, 4), 2) == (64, 64)
