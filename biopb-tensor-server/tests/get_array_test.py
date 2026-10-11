"""``get_array``: an eager read that agrees with ``get_tensor(...).compute()``.

A one-chunk plan skips dask; anything else is computed through it. Either way
the result must be what the lazy form gives, so the tests compare the two.
"""

import numpy as np
import pytest
from biopb.tensor import _session


def _zarr_source(writable_server, tmp_path, *, shape, chunks, name="a"):
    import zarr
    from biopb_tensor_server.adapters.zarr import ZarrAdapter

    data = np.arange(int(np.prod(shape)), dtype="uint16").reshape(shape)
    path = tmp_path / f"{name}.zarr"
    z = zarr.open_array(str(path), mode="w", shape=shape, chunks=chunks, dtype="uint16")
    z[:] = data
    writable_server.register_source(
        name,
        ZarrAdapter(
            zarr.open_array(str(path), mode="r"), name, ["y", "x"][: len(shape)]
        ),
    )
    return name, data


@pytest.fixture
def no_dask(monkeypatch):
    """Any attempt to build a dask array fails the test."""

    def refuse(*_a, **_k):
        raise AssertionError("dask was used")

    monkeypatch.setattr(_session, "_dask_from_flight_info", refuse)


@pytest.fixture
def dask_calls(monkeypatch):
    calls = []
    real = _session._dask_from_flight_info

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(_session, "_dask_from_flight_info", spy)
    return calls


class TestOneChunk:
    def test_whole_tensor_without_dask(
        self, client, writable_server, tmp_path, no_dask
    ):
        sid, data = _zarr_source(writable_server, tmp_path, shape=(8, 8), chunks=(8, 8))
        np.testing.assert_array_equal(client.get_array(sid), data)

    def test_a_region_inside_the_chunk_is_cropped(
        self, client, writable_server, tmp_path, no_dask
    ):
        """The server snaps the slice out to the chunk; the crop maps it back."""
        sid, data = _zarr_source(writable_server, tmp_path, shape=(8, 8), chunks=(8, 8))
        hint = (slice(2, 5), slice(1, 7))
        got = client.get_array(sid, slice_hint=hint)
        np.testing.assert_array_equal(got, data[2:5, 1:7])

    def test_a_small_crop_does_not_keep_its_chunk_alive(
        self, client, writable_server, tmp_path
    ):
        """A view of the whole chunk would pin it (an mmap'd segment or a decoded
        transfer buffer) for as long as the caller holds the crop. The lazy form
        copies such a crop (dask's getitem), so this one must too."""
        sid, data = _zarr_source(
            writable_server, tmp_path, shape=(64, 64), chunks=(64, 64)
        )
        hint = (slice(0, 4), slice(0, 4))
        got = client.get_array(sid, slice_hint=hint)
        np.testing.assert_array_equal(got, data[:4, :4])
        assert got.flags.owndata
        assert client.get_tensor(sid, slice_hint=hint).compute().flags.owndata

    def test_matches_the_lazy_form(self, client, writable_server, tmp_path):
        sid, _ = _zarr_source(writable_server, tmp_path, shape=(8, 8), chunks=(8, 8))
        hint = (slice(1, 6), slice(3, 8))
        np.testing.assert_array_equal(
            client.get_array(sid, slice_hint=hint),
            client.get_tensor(sid, slice_hint=hint).compute(),
        )

    def test_a_scaled_read_matches_the_lazy_form(
        self, client, writable_server, tmp_path
    ):
        sid, _ = _zarr_source(writable_server, tmp_path, shape=(8, 8), chunks=(8, 8))
        np.testing.assert_array_equal(
            client.get_array(sid, scale_hint=[2, 2]),
            client.get_tensor(sid, scale_hint=[2, 2]).compute(),
        )


class TestSeveralChunks:
    """At the default transfer target a small fixture is one chunk, so these
    lower it until the plan really has several endpoints."""

    def test_goes_through_dask_and_agrees(
        self, client, writable_server, tmp_path, dask_calls, transfer_target
    ):
        transfer_target(8)  # one 2x2 uint16 chunk per endpoint
        sid, data = _zarr_source(writable_server, tmp_path, shape=(4, 4), chunks=(2, 2))
        np.testing.assert_array_equal(client.get_array(sid), data)
        assert dask_calls, "a four-endpoint plan should have used the dask form"

    def test_a_region_spanning_chunks_matches_the_lazy_form(
        self, client, writable_server, tmp_path, dask_calls, transfer_target
    ):
        transfer_target(32)  # one 4x4 uint16 chunk per endpoint
        sid, data = _zarr_source(writable_server, tmp_path, shape=(8, 8), chunks=(4, 4))
        hint = (slice(2, 7), slice(1, 6))
        np.testing.assert_array_equal(
            client.get_array(sid, slice_hint=hint), data[2:7, 1:6]
        )
        assert dask_calls


def test_a_missing_tensor_raises_as_get_tensor_does(client):
    with pytest.raises(ValueError):
        client.get_array("nope/none")
