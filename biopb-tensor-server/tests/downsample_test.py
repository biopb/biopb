"""Equivalence and safety tests for the area downsample (biopb/biopb#639).

The area reduction was rewritten to sum at integer width and divide once on the
reduced array, instead of promoting the full-resolution block to float64 and
taking a mean per axis. The rewrite is only worth having if it is *bit*
identical, so the oracle here is the pre-#639 implementation itself, kept
verbatim below and compared against exhaustively rather than sampled.
"""

import zlib

import numpy as np
import pytest
from biopb_tensor_server.core import downsample as _ds


def legacy_downsample_block(data, scale_hint, reduction_method="area"):
    """The pre-#639 implementation, verbatim. The bit-identity oracle.

    Do not "clean up" or refactor this: its value is being an untouched copy of
    the behavior callers already depend on.
    """
    reduction_method = _ds.normalize_reduction_method(reduction_method)
    original_dtype = data.dtype

    if reduction_method == "nearest":
        return data[tuple(slice(0, None, scale) for scale in scale_hint)]

    padded_shape = _ds._pad_shape_to_scale_multiple(
        tuple(int(dim) for dim in data.shape), scale_hint
    )
    padded = _ds._pad_array_edge(data, padded_shape)

    reduced = np.asarray(padded, dtype=np.float64)
    for axis in reversed(range(reduced.ndim)):
        scale = scale_hint[axis]
        new_shape = (
            reduced.shape[:axis]
            + (reduced.shape[axis] // scale, scale)
            + reduced.shape[axis + 1 :]
        )
        reduced = reduced.reshape(new_shape).mean(axis=axis + 1)

    result = reduced
    if original_dtype != result.dtype:
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            result = np.clip(np.round(result), info.min, info.max)
        result = result.astype(original_dtype)
    return result


def _sample(dtype, shape, seed):
    """Random data spanning the dtype's full range, with the extremes present.

    Rounding disagreements hide at the boundaries, so min/max are planted
    explicitly rather than left to chance.
    """
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        data = rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=dtype)
        flat = data.reshape(-1)
        flat[0] = info.min
        flat[-1] = info.max
        return data
    data = rng.standard_normal(shape).astype(dtype) * 1000.0
    return data


# Shapes deliberately mix multiples of the scale with extents that need edge
# padding, since padding runs before the reduction and changes the values fed
# into it.
_SHAPES = [
    (1, 1, 1, 64, 64),
    (1, 1, 1, 63, 65),
    (1, 2, 3, 33, 17),
    (2, 8, 8),
    (16,),
]
_SCALES = {
    # (1, 1, 1, 64, 64) etc: the realistic case -- unit axes plus a 4x4 XY block.
    5: [(1, 1, 1, 4, 4), (1, 1, 1, 1, 1), (1, 1, 2, 8, 16), (1, 1, 1, 64, 2)],
    3: [(1, 2, 2), (2, 8, 8), (1, 1, 4)],
    1: [(4,), (1,), (16,)],
}
_DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "bool",
]


def _cases():
    for shape in _SHAPES:
        for scale in _SCALES[len(shape)]:
            for dtype in _DTYPES:
                yield shape, scale, dtype


@pytest.mark.parametrize(
    "shape,scale_hint,dtype",
    list(_cases()),
    ids=lambda v: str(v).replace(" ", ""),
)
def test_area_is_bit_identical_to_legacy(shape, scale_hint, dtype):
    # crc32, not hash(): PYTHONHASHSEED would make the sample -- and so any
    # failure -- irreproducible from run to run.
    seed = zlib.crc32(repr((shape, scale_hint, dtype)).encode())
    data = _sample(dtype, shape, seed=seed)

    expected = legacy_downsample_block(data, scale_hint, "area")
    actual = _ds.downsample_block(data, scale_hint, "area")

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), (
        f"area reduction diverged for {dtype} {shape} scale {scale_hint}: "
        f"max |diff| = "
        f"{np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max()}"
    )


@pytest.mark.parametrize(
    "scale_hint", [(1, 1, 1, 3, 3), (1, 1, 1, 6, 4), (1, 1, 1, 5, 1)]
)
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16"])
def test_non_power_of_two_scale_matches_legacy(dtype, scale_hint):
    """Non-dyadic scales take the float64 path, and must still agree.

    Dividing once by 9 is *not* the same as two float64 means by 3 -- each
    stage rounds -- so these fall back rather than fold. The guard is what this
    checks: the fallback is silent, so a bad gate would only show up as drifted
    pixels.
    """
    data = _sample(dtype, (1, 1, 1, 30, 30), seed=7)

    expected = legacy_downsample_block(data, scale_hint, "area")
    actual = _ds.downsample_block(data, scale_hint, "area")

    assert np.array_equal(actual, expected)
    assert _ds._plan_integer_area(np.dtype(dtype), scale_hint)[0] is None


class TestIntegerAccumulatorSizing:
    """The accumulator is sized from block_size * dtype_max, not hardcoded."""

    def test_realistic_case_uses_uint32(self):
        # uint16 at a 4x4 XY block: 16 * 65535 fits uint32 with room to spare.
        accumulator, block_size = _ds._plan_integer_area(
            np.dtype("uint16"), (1, 1, 1, 4, 4)
        )
        assert accumulator == np.dtype(np.uint32)
        assert block_size == 16

    def test_wide_input_widens_the_accumulator(self):
        # 4 * (2**32 - 1) overflows uint32, so uint64 is required.
        accumulator, _ = _ds._plan_integer_area(np.dtype("uint32"), (2, 2))
        assert accumulator == np.dtype(np.uint64)

    def test_signed_input_uses_a_signed_accumulator(self):
        accumulator, _ = _ds._plan_integer_area(np.dtype("int16"), (1, 4, 4))
        assert accumulator == np.dtype(np.int32)

    @pytest.mark.parametrize("dtype", ["uint64", "int64"])
    def test_64_bit_input_falls_back(self, dtype):
        """A 64-bit block sum can exceed float64's exact range, so the closing
        divide would no longer be exact -- refuse the fast path instead.

        Note the equivalence tests above emit "invalid value encountered in
        cast" for these dtypes, symmetrically from both implementations: the
        float64 path saturates at 2**63 / 2**64, which no 64-bit integer holds.
        That predates #639 and is untouched by it, since these inputs never take
        the new path.
        """
        accumulator, _ = _ds._plan_integer_area(np.dtype(dtype), (1, 4, 4))
        assert accumulator is None

    def test_extreme_block_size_falls_back(self):
        accumulator, _ = _ds._plan_integer_area(np.dtype("uint32"), (2**32, 1))
        assert accumulator is None

    @pytest.mark.parametrize("dtype", ["float32", "float64", "bool"])
    def test_non_integer_input_falls_back(self, dtype):
        accumulator, _ = _ds._plan_integer_area(np.dtype(dtype), (1, 4, 4))
        assert accumulator is None


class TestNoAccumulatorOverflow:
    """Saturated inputs: a too-narrow accumulator would wrap, not just drift."""

    def test_uint32_saturated(self):
        data = np.full((8, 8), np.iinfo(np.uint32).max, dtype=np.uint32)
        out = _ds.downsample_block(data, (2, 2), "area")
        assert out.dtype == np.uint32
        assert np.all(out == np.iinfo(np.uint32).max)

    def test_uint16_saturated_deep_reduction(self):
        data = np.full((256, 256), np.iinfo(np.uint16).max, dtype=np.uint16)
        out = _ds.downsample_block(data, (256, 256), "area")
        assert out.shape == (1, 1)
        assert out[0, 0] == np.iinfo(np.uint16).max

    def test_int16_saturated_negative(self):
        data = np.full((8, 8), np.iinfo(np.int16).min, dtype=np.int16)
        out = _ds.downsample_block(data, (4, 4), "area")
        assert np.all(out == np.iinfo(np.int16).min)


class TestUnitAxesAreSkipped:
    def test_area_reduce_skips_scale_one(self):
        """A scale-1 axis is left alone instead of reshaped-and-reduced.

        Identity is the point: reducing blocks of one cannot change a value, but
        it does copy the array -- three wasted full-resolution passes for the
        common [1,1,1,s,s] hint.
        """
        arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert _ds._area_reduce(arr, (1, 1, 1)) is arr

    def test_area_reduce_still_reduces_non_unit_axes(self):
        arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        reduced = _ds._area_reduce(arr, (1, 1, 2))
        assert reduced.shape == (2, 3, 2)
        assert np.array_equal(reduced, arr.reshape(2, 3, 2, 2).mean(axis=3))


class TestDegenerateInputs:
    def test_empty_array(self):
        data = np.zeros((0, 8), dtype=np.uint16)
        out = _ds.downsample_block(data, (2, 2), "area")
        assert out.shape == (0, 4)
        assert out.dtype == np.uint16

    def test_nearest_is_unchanged(self):
        data = _sample("uint16", (1, 1, 1, 63, 65), seed=11)
        expected = legacy_downsample_block(data, (1, 1, 1, 4, 4), "nearest")
        actual = _ds.downsample_block(data, (1, 1, 1, 4, 4), "nearest")
        assert np.array_equal(actual, expected)

    def test_output_never_aliases_the_input(self):
        """resolve_chunk_data caches what this returns, so a view over adapter
        memory would be a lifetime hazard (see the handle reaper)."""
        data = _sample("uint16", (8, 8), seed=13)
        out = _ds.downsample_block(data, (1, 1), "area")
        assert np.array_equal(out, data)
        assert not np.shares_memory(out, data)
