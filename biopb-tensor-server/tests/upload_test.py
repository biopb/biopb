"""Unit tests for data upload functionality.

Tests for CachedSourceAdapter, server-side do_put handler,
and Python client upload methods.
"""

import itertools
import pickle
import tempfile
import threading
import time
from itertools import product
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from biopb.tensor import TensorFlightClient, UploadRefused, _upload
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload
from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.adapters.ome_zarr import minimal_ome_metadata
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.core.chunk import (
    content_version_of,
    encode_chunk_id,
    get_bounds_from_chunk_id,
    wrap_content_version,
)
from biopb_tensor_server.core.config import CacheConfig
from google.protobuf.field_mask_pb2 import FieldMask

from tests import catalog_server


def _read_cached_batch(cache_manager: CacheManager, chunk_id: bytes) -> pa.RecordBatch:
    """The batch stored under ``chunk_id``, or fail if it is not cached.

    Reads it the way production does (``CachedSourceAdapter.resolve_chunk_data``):
    ``get_or_acquire`` with a compute_fn that must never fire, then release. Do
    not peek with ``backend.start_compute`` -- that hands back a *referenced*
    entry, and forgetting to release it is the leak this pattern replaces
    (biopb/biopb#545).
    """

    def _must_not_compute():
        raise AssertionError(f"chunk {chunk_id!r} was not in the cache")

    entry = cache_manager.get_or_acquire(chunk_id, _must_not_compute)
    try:
        return entry.data
    finally:
        cache_manager.release(chunk_id)


@pytest.fixture
def writable_server(tmp_path):
    """A live, writable server on an OS-assigned port.

    No wait after `serve()`: `FlightServerBase` binds and starts serving in
    `__init__`, so the port is live before this returns -- the thread only parks
    on it.
    """

    CacheManager.reset()
    CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
    server = catalog_server(
        location="grpc://localhost:0", writable=True, write_dir=Path(tmp_path)
    )
    server.mark_ready()
    threading.Thread(target=server.serve, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        CacheManager.reset()


@pytest.fixture
def client(writable_server):
    """A client connected to `writable_server`."""
    client = TensorFlightClient(f"grpc://localhost:{writable_server.port}")
    yield client
    client.close()


def _catalog_ids(db):
    return {
        r["source_id"] for r in db.query("SELECT source_id FROM sources").to_pylist()
    }


class TestCachedSourceAdapter:
    """Tests for CachedSourceAdapter."""

    def test_init(self):
        """Adapter initialization."""
        adapter = CachedSourceAdapter(
            source_id="cache_test123",
            shape=[100, 100, 50],
            dtype="uint16",
            chunk_shape=[32, 32, 32],
            dim_labels=["z", "y", "x"],
        )

        assert adapter.source_id == "cache_test123"
        assert adapter._shape == (100, 100, 50)
        assert adapter._dtype == "uint16"
        assert adapter._chunk_shape == (32, 32, 32)
        assert adapter._dim_labels == ["z", "y", "x"]

    def test_get_tensor_descriptor(self):
        """TensorDescriptor generation."""
        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 200],
            dtype="float32",
            chunk_shape=[50, 100],
            dim_labels=["y", "x"],
        )

        desc = adapter.get_tensor_descriptor()

        assert desc.array_id == "test"
        assert list(desc.shape) == [100, 200]
        assert list(desc.chunk_shape) == [50, 100]
        assert desc.dtype == "float32"
        assert list(desc.dim_labels) == ["y", "x"]

    def test_physical_scale_echoed_verbatim(self):
        """An uploaded calibration is surfaced 1:1 by _physical_scale (issue #272)."""
        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 200],
            dtype="uint16",
            chunk_shape=[50, 100],
            dim_labels=["y", "x"],
            physical_scale=[0.65, 0.65],
            physical_unit=["µm", "µm"],
        )

        scale, unit = adapter._physical_scale()
        assert scale == [0.65, 0.65]
        assert unit == ["µm", "µm"]

    def test_physical_scale_none_without_calibration(self):
        """No uploaded calibration -> _physical_scale is None (base clears fields)."""
        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 200],
            dtype="uint16",
            chunk_shape=[50, 100],
            dim_labels=["y", "x"],
        )
        assert adapter._physical_scale() is None

    def test_list_tensor_descriptors_single(self):
        """Cache sources are single-tensor."""
        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )

        descriptors = adapter.list_tensor_descriptors()
        assert len(descriptors) == 1
        assert descriptors[0].array_id == "test"

    def test_get_metadata_with_ome(self):
        """OME metadata handling."""
        ome_metadata = {
            "multiscales": [
                {
                    "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                }
            ]
        }

        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 100, 100],
            dtype="uint8",
            chunk_shape=[50, 50, 50],
            ome_metadata=ome_metadata,
        )

        metadata = adapter.get_metadata()
        assert "multiscales" in metadata
        assert len(metadata["multiscales"]) == 1

    def test_write_chunk(self, tmp_path):
        """write_chunk stores data in cache."""
        CacheManager.reset()
        config = CacheConfig(file_cache_dir=tmp_path / "cache")
        CacheManager.initialize(config)

        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )

        # Write a chunk
        data = np.ones((50, 50), dtype=np.uint8)
        bounds = ChunkBounds(start=[0, 0], stop=[50, 50])

        adapter.write_chunk(bounds, data)

        # Verify chunk is in cache
        chunk_id = encode_chunk_id("test", bounds)

        batch = _read_cached_batch(CacheManager.get_instance(), chunk_id)

        # Verify batch schema: [data: list<dtype>, shape: list<int64>, dtype: string]
        assert batch.schema.names == ["data", "shape", "dtype"]

        # Verify data shape via the list array
        flat_data = batch.column("data").to_pylist()[0]
        assert len(flat_data) == 50 * 50

        CacheManager.reset()

    def test_write_chunk_leaves_no_cache_reference(self, tmp_path):
        """An uploaded chunk is not pinned in memory (biopb/biopb#545).

        write_chunk_arrow used to drive start_compute/complete_entry itself and
        keep the reservation's reference, so every chunk ever uploaded stayed
        mirrored in RAM for the life of the process -- unevictable, and not
        bounded by the *disk* cache budget.
        """
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))

        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )
        bounds = ChunkBounds(start=[0, 0], stop=[50, 50])
        adapter.write_chunk(bounds, np.ones((50, 50), dtype=np.uint8))

        chunk_id = encode_chunk_id("test", bounds)
        entry = CacheManager.get_instance().backend._entries[chunk_id]
        assert entry.ref_count == 0
        assert entry.is_evictable()
        assert CacheManager.get_instance().backend.remove(chunk_id) is True

        CacheManager.reset()

    def test_resolve_chunk_data_for_cache_backed_adapter(self, tmp_path):
        """Regression test: resolve_chunk_data must work for CachedSourceAdapter.

        CachedSourceAdapter has no backend data source - all data is in cache.
        Without override, retrieval would call get_data() which raises error.

        This tests the resolve_chunk_data override in CachedSourceAdapter.
        """
        CacheManager.reset()
        config = CacheConfig(file_cache_dir=tmp_path / "cache")
        CacheManager.initialize(config)

        adapter = CachedSourceAdapter(
            source_id="test_resolve",
            shape=[100, 100],
            dtype="uint16",
            chunk_shape=[50, 50],
        )

        # Write test data
        test_data = np.arange(50 * 50, dtype=np.uint16).reshape(50, 50)
        bounds = ChunkBounds(start=[0, 0], stop=[50, 50])
        adapter.write_chunk(bounds, test_data)

        # Retrieve via resolve_chunk_data (not direct cache access)
        chunk_id = encode_chunk_id("test_resolve", bounds)
        batch = adapter.resolve_chunk_data(chunk_id, CacheManager.get_instance())

        # Verify data matches - the chunk is the unified binary schema now.
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        arr = unpack_chunk_array(batch)
        np.testing.assert_array_equal(arr, test_data)

        CacheManager.reset()

    def test_read_plan_uses_the_uploaded_write_grid(self):
        """A cache-backed read must plan on the grid the upload was written on.

        This source has no backend: ``get_data`` raises and
        ``resolve_chunk_data`` serves only chunk_ids that were actually written.
        While the server re-sized every adapter's grid, a 2 MiB-block upload was
        planned at four blocks per chunk and *none* of the planned chunk_ids
        existed -- every read raised "Chunk not found" (biopb/biopb#809).

        The array is deliberately big enough that coalescing had room to move.
        The older tests here use 100x100, which the sizing policy of the day
        happened to leave on its native grid -- so they passed either way and
        hid the bug.
        """
        shape = [1, 4, 64, 1024, 1024]
        grid = [1, 1, 1, 1024, 1024]
        adapter = CachedSourceAdapter(
            source_id="grid",
            shape=shape,
            dtype="<u2",
            chunk_shape=grid,
            dim_labels=["t", "c", "z", "y", "x"],
        )

        assert adapter.get_transfer_chunk_size() == tuple(grid)

        plan = adapter.get_read_plan(adapter.get_tensor_descriptor())
        written = {
            encode_chunk_id(
                "grid",
                ChunkBounds(
                    start=list(index),
                    stop=[
                        min(start + step, dim)
                        for start, step, dim in zip(index, grid, shape, strict=True)
                    ],
                ),
            )
            for index in product(
                *(range(0, d, c) for d, c in zip(shape, grid, strict=True))
            )
        }
        planned = [bytes(ce.chunk_id) for ce in plan.chunk_endpoints]
        assert planned, "a read plan with no endpoints would pass vacuously"
        assert all(chunk_id in written for chunk_id in planned)

    def test_write_chunk_arbitrary_bounds(self, tmp_path):
        """Cache sources accept arbitrary chunk bounds."""
        CacheManager.reset()
        config = CacheConfig(file_cache_dir=tmp_path / "cache")
        CacheManager.initialize(config)

        adapter = CachedSourceAdapter(
            source_id="test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )

        # Write a chunk with arbitrary bounds (not aligned to chunk_shape)
        data = np.ones((30, 40), dtype=np.uint8)
        bounds = ChunkBounds(start=[10, 20], stop=[40, 60])

        adapter.write_chunk(bounds, data)

        # Verify chunk is stored
        chunk_id = encode_chunk_id("test", bounds)

        _read_cached_batch(CacheManager.get_instance(), chunk_id)
        CacheManager.reset()

    def test_write_chunk_arrow_rejects_list_wrapper(self, tmp_path):
        # The binary chunk schema stores the flat value buffer, and the upload
        # dtype is derived from the primitive Arrow field type; a list<T> wrapper
        # would corrupt both (buffer[1] is offsets, to_pandas_dtype() is wrong).
        # write_chunk_arrow must refuse it rather than silently mis-store.
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        adapter = CachedSourceAdapter(
            source_id="t", shape=[4], dtype="int16", chunk_shape=[4]
        )
        list_wrapper = pa.array([[1, 2, 3, 4]])  # list<int64>, not flat values
        with pytest.raises(TypeError, match="not a list"):
            adapter.write_chunk_arrow(
                ChunkBounds(start=[0], stop=[4]), list_wrapper, [4], np.int16
            )
        CacheManager.reset()

    def test_write_chunk_with_file_backend_schema(self):
        """Regression test: write_chunk must emit the unified binary chunk schema.

        A cache-backed source stores the batch it will later serve verbatim
        (resolve_chunk_data returns entry.data), so write_chunk must produce the
        same [data: binary, shape, dtype] wire schema every read path expects
        (biopb/biopb#293).
        """
        import shutil
        import tempfile

        CacheManager.reset()

        # Use file backend - this calls _cast_to_unified_schema
        cache_dir = tempfile.mkdtemp(prefix="test-cached-source-")
        try:
            config = CacheConfig(
                file_cache_dir=Path(cache_dir),
                file_max_total_bytes=100 * 1024 * 1024,  # 100MB
            )
            CacheManager.initialize(config)

            adapter = CachedSourceAdapter(
                source_id="test_file_schema",
                shape=[4096, 1025],
                dtype="float32",
                chunk_shape=[4096, 1025],
            )

            # Write a chunk - this must create batch with correct schema
            data = np.random.rand(4096, 1025).astype(np.float32)
            bounds = ChunkBounds(start=[0, 0], stop=[4096, 1025])

            adapter.write_chunk(bounds, data)

            # Verify chunk is stored and readable
            chunk_id = encode_chunk_id("test_file_schema", bounds)
            cache_manager = CacheManager.get_instance()

            # Read back via cache backend (served as the unified binary schema)
            entry = cache_manager.get_or_acquire(chunk_id, lambda: (None, 0))
            assert entry.state.name == "READY"

            # Verify batch schema: [data: binary, shape: list<int64>, dtype: string]
            batch = entry.data
            assert batch.schema.names == ["data", "shape", "dtype"]
            assert pa.types.is_binary(batch.column("data").type)

            # Verify data can be reconstructed
            from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

            reconstructed = unpack_chunk_array(batch)
            assert reconstructed.shape == data.shape
            assert reconstructed.dtype == data.dtype

            cache_manager.release(chunk_id)
        finally:
            CacheManager.reset()
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_the_write_grid_is_not_resplit_at_the_wire_bound(self):
        """A chunk over MAX_ARROW_BATCH_BYTES is served whole, because the
        pieces the base planner would fetch instead were never written.

        A consumer that replans a handle -- every fast-return consumer, whose
        handle carries no endpoints -- meets this; the producer's own embedded
        endpoints used to hide it.
        """
        from biopb_tensor_server.cache import MAX_ARROW_BATCH_BYTES

        side = int((MAX_ARROW_BATCH_BYTES * 2) ** 0.5) + 1  # > 64 MiB of uint8
        adapter = CachedSourceAdapter(
            source_id="whole", shape=[side, side], dtype="|u1", chunk_shape=[side, side]
        )

        assert adapter.get_transfer_chunk_size() == (side, side)
        plan = adapter.get_read_plan(adapter.get_tensor_descriptor())
        assert len(plan.chunk_endpoints) == 1


class TestScaledReads:
    """A cache source serves the pyramid it advertises (biopb/biopb#265).

    The server advertises a computed ladder for every tensor, uploads included,
    but this adapter used to reject every scaled chunk_id -- so an uploaded
    result was addressable, opened, and then 404'd on the first coarse tile.
    That is every upload of 2048^2 or more, which is where the ladder gains its
    second rung, and it is the shape the documented agent workflow produces:
    upload a result as ``cache:``, then display it.
    """

    SHAPE = [1024, 1024]
    GRID = [256, 256]

    @pytest.fixture
    def cache(self, tmp_path):
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        yield CacheManager.get_instance()
        CacheManager.reset()

    def _upload(self, cache, *, skip=()):
        """An adapter with every chunk written but those in ``skip``."""
        adapter = CachedSourceAdapter(
            source_id="up",
            shape=self.SHAPE,
            dtype="<u2",
            chunk_shape=self.GRID,
            dim_labels=["y", "x"],
        )
        source = np.zeros(self.SHAPE, dtype="<u2")
        step = self.GRID[0]
        for y in range(0, self.SHAPE[0], step):
            for x in range(0, self.SHAPE[1], step):
                value = (y // step) * 4 + (x // step) + 1
                source[y : y + step, x : x + step] = value
                if (y, x) in skip:
                    continue
                adapter.write_chunk(
                    ChunkBounds(start=[y, x], stop=[y + step, x + step]),
                    np.full((step, step), value, dtype="<u2"),
                )
        return adapter, source

    @staticmethod
    def _scaled_id(bounds, scale, method="area"):
        from biopb_tensor_server.core.chunk import encode_chunk_id_with_scale

        return encode_chunk_id_with_scale("up", bounds, scale, method)

    def test_a_scaled_read_is_assembled_from_the_uploaded_chunks(self, cache):
        """The reduction comes out of the cache, which is where an upload lives
        -- so it needs no backend, and ``get_data`` is never reached."""
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        adapter, source = self._upload(cache)
        adapter.get_data = lambda bounds: pytest.fail("reached get_data")

        bounds = ChunkBounds(start=[0, 0], stop=[512, 512])
        batch = adapter.resolve_chunk_data(self._scaled_id(bounds, (2, 2)), cache)

        expected = (
            source[0:512, 0:512].reshape(256, 2, 256, 2).mean(axis=(1, 3)).astype("<u2")
        )
        np.testing.assert_array_equal(unpack_chunk_array(batch), expected)

    def test_nearest_is_served_too(self, cache):
        """The advertised ladder asks for ``nearest``, so the method the viewer
        actually requests has to round-trip, not just the default."""
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        adapter, source = self._upload(cache)
        bounds = ChunkBounds(start=[0, 0], stop=[512, 512])

        batch = adapter.resolve_chunk_data(
            self._scaled_id(bounds, (2, 2), "nearest"), cache
        )

        np.testing.assert_array_equal(
            unpack_chunk_array(batch), source[0:512:2, 0:512:2]
        )

    def test_a_scaled_read_over_a_hole_raises_and_names_it(self, cache):
        """Missing data always raises -- there is nothing to reconstruct it
        from. The message has to name the bounds, because by then the caller
        asked for a coarse tile and the hole is two layers down."""
        adapter, _ = self._upload(cache, skip={(0, 256)})
        bounds = ChunkBounds(start=[0, 0], stop=[512, 512])

        with pytest.raises(
            flight.FlightServerError,
            match=r"scaled read of \[0, 0\]\.\.\[512, 512\]: 65536 of 262144",
        ):
            adapter.resolve_chunk_data(self._scaled_id(bounds, (2, 2)), cache)

    def test_an_unwritten_full_resolution_chunk_raises_and_names_it(self, cache):
        adapter, _ = self._upload(cache, skip={(0, 0)})

        with pytest.raises(flight.FlightServerError, match=r"no chunk at \[0, 0\]"):
            adapter.resolve_chunk_data(
                encode_chunk_id("up", ChunkBounds(start=[0, 0], stop=[256, 256])),
                cache,
            )

    def test_every_advertised_rung_serves(self, cache):
        """The regression itself, through the seam the viewers use: whatever the
        server advertises for this tensor, it must also serve."""
        from biopb.tensor.descriptor_pb2 import TensorReadOption
        from biopb_tensor_server.core.config import PyramidConfig

        # Big enough that the computed ladder has a second rung at all.
        self.SHAPE = [2048, 2048]
        adapter, _ = self._upload(cache)

        plan = adapter.plan_flight_info(
            TensorReadOption(fields=FieldMask(paths=["pyramid"])), PyramidConfig()
        )
        rungs = [tuple(level.scale_hint) for level in plan.descriptor.pyramid]
        assert len(rungs) > 1, "fixture too small to cover the regression"

        extent = ChunkBounds(start=[0, 0], stop=[512, 512])
        for scale in rungs:
            if all(factor == 1 for factor in scale):
                continue
            adapter.resolve_chunk_data(self._scaled_id(extent, scale), cache)


class TestChunkEncoding:
    """Tests for chunk ID encoding with cache sources."""

    def test_encode_chunk_id_cache(self):
        """Chunk ID encoding for cache sources."""
        bounds = ChunkBounds(start=[0, 0, 0], stop=[32, 32, 32])
        chunk_id = encode_chunk_id("cache_abc123", bounds)

        # Should contain source_id
        assert chunk_id is not None
        assert len(chunk_id) > 0

    def test_bounds_in_chunk_id(self):
        """Bounds are encoded directly in chunk_id."""
        bounds = ChunkBounds(start=[10, 20, 30], stop=[40, 50, 60])
        chunk_id = encode_chunk_id("test", bounds)

        # Decode and verify
        decoded_bounds = get_bounds_from_chunk_id(chunk_id)
        assert list(decoded_bounds.start) == [10, 20, 30]
        assert list(decoded_bounds.stop) == [40, 50, 60]


class TestServerDoPutHandler:
    """Tests for server-side do_put handler."""

    def test_create_source_cache_backed(self):
        """Create cache-backed source."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",  # Use random port
            writable=True,
        )

        # Create source request
        req_desc = TensorDescriptor(
            array_id="cache:my-test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
            dim_labels=["y", "x"],
        )

        # Process creation
        response_desc = server.uploads.create_tensor(req_desc)

        # Check response
        assert response_desc is not None
        assert response_desc.array_id.startswith("cache_")

        # Check adapter was registered
        adapter = server.sources.get(response_desc.array_id)
        assert adapter is not None
        assert isinstance(adapter, CachedSourceAdapter)

    def test_create_source_threads_physical_scale(self):
        """A calibrated upload survives the create_source round-trip (issue #272).

        The DoPut request descriptor already carries physical_scale / physical_unit;
        create_source must thread them onto the CachedSourceAdapter (so the read
        hot path advertises them) and echo them on the response descriptor.
        """
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
        )

        req_desc = TensorDescriptor(
            array_id="cache:calibrated",
            shape=[10, 100, 100],
            dtype="uint16",
            chunk_shape=[1, 50, 50],
            dim_labels=["z", "y", "x"],
            physical_scale=[2.0, 0.325, 0.325],
            physical_unit=["µm", "µm", "µm"],
        )

        response_desc = server.uploads.create_tensor(req_desc)

        # The response echoes the uploader's calibration.
        assert list(response_desc.physical_scale) == [2.0, 0.325, 0.325]
        assert list(response_desc.physical_unit) == ["µm", "µm", "µm"]

        # The registered adapter surfaces it on the read hot path.
        adapter = server.sources.get(response_desc.array_id)
        scale, unit = adapter._physical_scale()
        assert scale == [2.0, 0.325, 0.325]
        assert unit == ["µm", "µm", "µm"]

    def test_create_source_ome_zarr_does_not_echo_physical_scale(self):
        """The physical-scale echo is scoped to cache sources (issue #272).

        Only CachedSourceAdapter stores AND re-serves the uploaded calibration,
        so only cache responses echo it. The ome_zarr branch persists scale via
        the .zattrs (the minimal-metadata path drops it), so echoing req_desc
        here would advertise a vector a later read won't reproduce -- it must not.
        """
        from biopb_tensor_server.serving.server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                writable=True,
                write_dir=Path(tmpdir),
            )

            req_desc = TensorDescriptor(
                array_id="ome_zarr:calibrated",
                shape=[100, 100],
                dtype="uint8",
                chunk_shape=[50, 50],
                dim_labels=["y", "x"],
                physical_scale=[0.5, 0.5],
                physical_unit=["µm", "µm"],
            )

            response_desc = server.uploads.create_tensor(req_desc)

            # No unpersisted calibration is advertised on the zarr response.
            assert list(response_desc.physical_scale) == []
            assert list(response_desc.physical_unit) == []

    def test_create_source_cache_server_generated_name(self):
        """Create cache-backed source with server-generated name."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
        )

        req_desc = TensorDescriptor(
            array_id="cache:",  # No name - server generates
            shape=[100, 100],
            dtype="float32",
            chunk_shape=[50, 50],
        )

        response_desc = server.uploads.create_tensor(req_desc)
        assert response_desc.array_id.startswith("cache_")

    def test_create_source_not_writable_raises(self):
        """Non-writable server rejects source creation."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=False,
        )

        req_desc = TensorDescriptor(
            array_id="cache:test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )

        # The writable check is in do_put, not the UploadManager. This test
        # verifies create_source itself works on a non-writable server (the check
        # happens at the do_put boundary).
        response_desc = server.uploads.create_tensor(req_desc)
        # Source should be created (do_put would reject before reaching this).
        assert response_desc is not None

    def test_create_source_ome_zarr_requires_write_dir(self):
        """Zarr-backed source requires write_dir."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
            write_dir=None,  # No write_dir
        )

        req_desc = TensorDescriptor(
            array_id="ome_zarr:test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )

        with pytest.raises(flight.FlightServerError, match="write_dir not configured"):
            server.uploads.create_tensor(req_desc)

    def test_create_source_ome_zarr_with_write_dir(self):
        """Create zarr-backed source with write_dir."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                writable=True,
                write_dir=Path(tmpdir),
            )

            req_desc = TensorDescriptor(
                array_id="ome_zarr:my-zarr",
                shape=[100, 100],
                dtype="uint8",
                chunk_shape=[50, 50],
                dim_labels=["y", "x"],
            )

            response_desc = server.uploads.create_tensor(req_desc)
            assert response_desc.array_id.startswith("ome_zarr_")

            # Check zarr directory was created
            zarr_dirs = list(Path(tmpdir).glob("*.zarr"))
            assert len(zarr_dirs) == 1

    def test_ome_zarr_upload_synced_to_catalog(self):
        """File-backed (durable) uploads are added to the catalog so they are
        discoverable via list_sources/query_sources (biopb/biopb#265)."""
        from biopb_tensor_server.serving.metadata_db import MetadataDatabase
        from biopb_tensor_server.serving.server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            db = MetadataDatabase()
            server = TensorFlightServer(
                location="grpc://localhost:0",
                writable=True,
                write_dir=Path(tmpdir),
                metadata_db=db,
            )
            req_desc = TensorDescriptor(
                array_id="ome_zarr:persist",
                shape=[64, 64],
                dtype="uint8",
                chunk_shape=[32, 32],
                dim_labels=["y", "x"],
            )
            response_desc = server.uploads.create_tensor(req_desc)

            # The durable upload appears in the catalog.
            assert response_desc.array_id in _catalog_ids(db)

    def test_cache_upload_not_synced_but_readable_by_id(self):
        """Ephemeral cache-backed uploads are NOT catalogued (no removal hook ->
        the row would dangle), but stay readable by their returned id."""
        from biopb_tensor_server.serving.metadata_db import MetadataDatabase
        from biopb_tensor_server.serving.server import TensorFlightServer

        db = MetadataDatabase()
        server = TensorFlightServer(
            location="grpc://localhost:0", writable=True, metadata_db=db
        )
        req_desc = TensorDescriptor(
            array_id="cache:ephemeral",
            shape=[64, 64],
            dtype="uint8",
            chunk_shape=[32, 32],
        )
        response_desc = server.uploads.create_tensor(req_desc)

        # Not enumerable via the catalog...
        assert response_desc.array_id not in _catalog_ids(db)
        # ...but still registered and readable by its returned id.
        assert isinstance(
            server.sources.get(response_desc.array_id), CachedSourceAdapter
        )

    def test_create_source_invalid_prefix(self):
        """Invalid array_id prefix raises error."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
        )

        req_desc = TensorDescriptor(
            array_id="invalid:test",  # Invalid prefix
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )

        with pytest.raises(flight.FlightServerError, match="Invalid array_id format"):
            server.uploads.create_tensor(req_desc)

    def test_create_tensor_action_round_trip(self, tmp_path):
        """Live client create_source should return the server-assigned source_id."""
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server.serving.server import TensorFlightServer

        CacheManager.reset()
        config = CacheConfig(file_cache_dir=tmp_path / "cache")
        CacheManager.initialize(config)

        server = TensorFlightServer(
            location="grpc://127.0.0.1:0",
            writable=True,
        )
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()

        client = TensorFlightClient(f"grpc://127.0.0.1:{server.port}")
        try:
            source_id = client.create_tensor(
                "cache:test-action", np.empty((10, 10), np.uint8), chunk_shape=(5, 5)
            ).array_id
            assert source_id.startswith("cache_")
            assert source_id in server.sources
        finally:
            client.close()
            server.shutdown()
            CacheManager.reset()


class TestDoPutErrorTranslation:
    """A create-source failure must surface as itself (biopb/biopb#354).

    A source is created by the ``create_tensor`` action only; DoPut takes a
    ``PutCommand`` and refuses anything else up front, so the historical
    parse-and-guess discrimination (which mis-reported a genuine create error
    as "Invalid upload command") has no path left. ``_do_put`` here is the
    action, kept under its old name so each error case is still checked on
    both entry points.
    """

    class _MockWriter:
        """Minimal FlightMetadataWriter: capture the response bytes."""

        def __init__(self):
            self.written = []

        def write(self, buf):
            self.written.append(buf)

    @staticmethod
    def _server(**kwargs):
        from biopb_tensor_server.serving.server import TensorFlightServer

        return TensorFlightServer(
            location="grpc://localhost:0", writable=True, **kwargs
        )

    def _do_put(self, server, req_desc):
        action = flight.Action("create_tensor", req_desc.SerializeToString())
        return list(server.do_action(None, action))

    def test_do_put_refuses_a_bare_descriptor(self):
        """The old create-over-DoPut wire shape is refused, not guessed at."""
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="cache:ok", shape=[10, 10], dtype="uint8", chunk_shape=[5, 5]
        )
        descriptor = flight.FlightDescriptor.for_command(req_desc.SerializeToString())
        with pytest.raises(flight.FlightServerError, match="PutCommand"):
            server.do_put(None, descriptor, None, self._MockWriter())

    def _create_tensor_action(self, server, req_desc):
        action = flight.Action("create_tensor", req_desc.SerializeToString())
        list(server.do_action(None, action))

    # -- invalid array_id prefix ------------------------------------------------

    def test_do_put_invalid_prefix_surfaces_real_error(self):
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="bogus:test", shape=[10, 10], dtype="uint8", chunk_shape=[5, 5]
        )
        with pytest.raises(flight.FlightServerError) as exc:
            self._do_put(server, req_desc)
        assert "Invalid array_id format" in str(exc.value)
        assert "Invalid upload command" not in str(exc.value)

    def test_action_invalid_prefix_surfaces_real_error(self):
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="bogus:test", shape=[10, 10], dtype="uint8", chunk_shape=[5, 5]
        )
        with pytest.raises(flight.FlightServerError, match="Invalid array_id format"):
            self._create_tensor_action(server, req_desc)

    # -- missing write_dir ------------------------------------------------------

    def test_do_put_missing_write_dir_surfaces_real_error(self):
        server = self._server(write_dir=None)
        req_desc = TensorDescriptor(
            array_id="ome_zarr:test", shape=[10, 10], dtype="uint8", chunk_shape=[5, 5]
        )
        with pytest.raises(flight.FlightServerError) as exc:
            self._do_put(server, req_desc)
        assert "write_dir not configured" in str(exc.value)
        assert "Invalid upload command" not in str(exc.value)

    def test_action_missing_write_dir_surfaces_real_error(self):
        server = self._server(write_dir=None)
        req_desc = TensorDescriptor(
            array_id="ome_zarr:test", shape=[10, 10], dtype="uint8", chunk_shape=[5, 5]
        )
        with pytest.raises(flight.FlightServerError, match="write_dir not configured"):
            self._create_tensor_action(server, req_desc)

    # -- malformed metadata_json ------------------------------------------------

    def test_do_put_malformed_metadata_json_surfaces_real_error(self):
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="cache:test",
            shape=[10, 10],
            dtype="uint8",
            chunk_shape=[5, 5],
            metadata_json="{not valid json",
        )
        with pytest.raises(flight.FlightServerError) as exc:
            self._do_put(server, req_desc)
        assert "invalid metadata_json" in str(exc.value)
        assert "Invalid upload command" not in str(exc.value)

    def test_action_malformed_metadata_json_surfaces_real_error(self):
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="cache:test",
            shape=[10, 10],
            dtype="uint8",
            chunk_shape=[5, 5],
            metadata_json="{not valid json",
        )
        with pytest.raises(flight.FlightServerError, match="invalid metadata_json"):
            self._create_tensor_action(server, req_desc)

    def test_do_put_malformed_metadata_json_ome_zarr(self):
        """The ome_zarr branch's metadata_json is validated too, not just cache --
        and metadata is parsed *before* any disk write, so a rejected request
        leaves no orphaned .zarr behind (which would block a corrected retry
        under the same name; biopb/biopb#354)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = self._server(write_dir=Path(tmpdir))
            req_desc = TensorDescriptor(
                array_id="ome_zarr:test",
                shape=[10, 10],
                dtype="uint8",
                chunk_shape=[5, 5],
                metadata_json="{not valid json",
            )
            with pytest.raises(flight.FlightServerError, match="invalid metadata_json"):
                self._do_put(server, req_desc)
            # Nothing was written to disk -- no partial store to block a retry.
            assert list(Path(tmpdir).iterdir()) == []

    def test_do_put_ome_zarr_retry_after_bad_metadata_succeeds(self):
        """A malformed-metadata create must not poison a corrected retry under
        the same name: parse-first leaves no store for zarr.create to collide
        with (biopb/biopb#354)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = self._server(write_dir=Path(tmpdir))
            bad = TensorDescriptor(
                array_id="ome_zarr:retry",
                shape=[10, 10],
                dtype="uint8",
                chunk_shape=[5, 5],
                metadata_json="{not valid json",
            )
            with pytest.raises(flight.FlightServerError, match="invalid metadata_json"):
                self._do_put(server, bad)

            # Same name, now well-formed: previously the orphaned store made
            # zarr.create raise instead of creating the source.
            good = TensorDescriptor(
                array_id="ome_zarr:retry",
                shape=[10, 10],
                dtype="uint8",
                chunk_shape=[5, 5],
            )
            (written,) = self._do_put(server, good)
            response = TensorDescriptor.FromString(bytes(written))
            assert response.array_id.startswith("ome_zarr_")

    def test_do_put_non_object_metadata_json_surfaces_real_error(self):
        """Well-formed JSON that is not an object is rejected: callers spread the
        result into a mapping, so a scalar/list must fail at the create boundary
        rather than downstream (biopb/biopb#354)."""
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="cache:test",
            shape=[10, 10],
            dtype="uint8",
            chunk_shape=[5, 5],
            metadata_json="[1, 2, 3]",
        )
        with pytest.raises(flight.FlightServerError) as exc:
            self._do_put(server, req_desc)
        assert "expected a JSON object" in str(exc.value)
        assert "Invalid upload command" not in str(exc.value)

    # -- the good path still writes the resolved descriptor ---------------------

    def test_do_put_valid_create_writes_response(self):
        """A well-formed create still round-trips its resolved descriptor."""
        server = self._server()
        req_desc = TensorDescriptor(
            array_id="cache:ok", shape=[10, 10], dtype="uint8", chunk_shape=[5, 5]
        )
        (written,) = self._do_put(server, req_desc)

        response = TensorDescriptor.FromString(bytes(written))
        assert response.array_id.startswith("cache_")


class TestChunkUpload:
    """Tests for chunk upload handling."""

    def test_upload_chunk_cache_backed(self, tmp_path):
        """Upload chunk to cache-backed source."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        CacheManager.reset()
        config = CacheConfig(file_cache_dir=tmp_path / "cache")
        CacheManager.initialize(config)

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
        )

        # Create source first
        req_desc = TensorDescriptor(
            array_id="cache:test",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )
        response_desc = server.uploads.create_tensor(req_desc)
        source_id = response_desc.array_id

        # Upload chunk
        upload = ChunkUpload(
            source_id=source_id,
            bounds=ChunkBounds(start=[0, 0], stop=[50, 50]),
        )

        # Create mock data
        data = np.ones((50, 50), dtype=np.uint8)
        batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])

        # Mock reader
        class MockReader:
            def read_all(self):
                return pa.Table.from_batches([batch])

        # Process upload (returns None now)
        server.uploads.write_chunk(upload, MockReader())

        # Check chunk was stored
        adapter = server.sources.get(source_id)
        assert adapter is not None

        CacheManager.reset()

    def test_upload_chunk_cache_backed_preserves_logical_shape(self, tmp_path):
        """Cache-backed uploads must store shape metadata for reconstruction."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        CacheManager.reset()
        config = CacheConfig(file_cache_dir=tmp_path / "cache")
        CacheManager.initialize(config)

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
        )

        req_desc = TensorDescriptor(
            array_id="cache:test-shape",
            shape=[100, 100],
            dtype="uint8",
            chunk_shape=[50, 50],
        )
        response_desc = server.uploads.create_tensor(req_desc)
        source_id = response_desc.array_id

        bounds = ChunkBounds(start=[10, 20], stop=[40, 60])
        upload = ChunkUpload(source_id=source_id, bounds=bounds)

        data = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
        batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])

        class MockReader:
            def read_all(self):
                return pa.Table.from_batches([batch])

        server.uploads.write_chunk(upload, MockReader())

        # create_source assigns a per-upload content_version (#178), so the chunk
        # is stored under the version-wrapped id the read plan also mints.
        adapter = server.sources.get(source_id)
        chunk_id = encode_chunk_id(source_id, bounds)
        if adapter.content_version is not None:
            chunk_id = wrap_content_version(chunk_id, adapter.content_version)
        cache_manager = CacheManager.get_instance()
        stored_batch = _read_cached_batch(cache_manager, chunk_id)
        assert stored_batch.column("shape").to_pylist()[0] == [30, 40]
        assert stored_batch.column("dtype").to_pylist()[0] == np.dtype(np.uint8).str

        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        reconstructed = unpack_chunk_array(stored_batch)
        assert reconstructed.shape == data.shape
        assert np.array_equal(reconstructed, data)

        CacheManager.reset()

    def test_upload_chunk_missing_source(self):
        """Upload to missing source raises error."""
        from biopb_tensor_server.serving.server import TensorFlightServer

        server = TensorFlightServer(
            location="grpc://localhost:0",
            writable=True,
        )

        upload = ChunkUpload(
            source_id="nonexistent",
            bounds=ChunkBounds(start=[0, 0], stop=[50, 50]),
        )

        class MockReader:
            def read_all(self):
                return pa.Table.from_arrays([pa.array([1, 2, 3])], ["data"])

        with pytest.raises(flight.FlightServerError, match="Source not found"):
            server.uploads.write_chunk(upload, MockReader())


class TestOmeZarrChunkAlignment:
    """Tests for chunk alignment validation for zarr-backed sources."""

    def test_aligned_chunk_accepted(self):
        """Aligned chunk upload is accepted."""
        pytest.importorskip("zarr")

        from biopb_tensor_server.serving.server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            CacheManager.reset()
            config = CacheConfig(file_cache_dir=Path(tmpdir) / "cache")
            CacheManager.initialize(config)

            server = TensorFlightServer(
                location="grpc://localhost:0",
                writable=True,
                write_dir=Path(tmpdir),
            )

            # Create zarr-backed source
            req_desc = TensorDescriptor(
                array_id="ome_zarr:test",
                shape=[100, 100],
                dtype="uint8",
                chunk_shape=[50, 50],
            )
            response_desc = server.uploads.create_tensor(req_desc)
            source_id = response_desc.array_id

            # Upload aligned chunk
            upload = ChunkUpload(
                source_id=source_id,
                bounds=ChunkBounds(start=[0, 0], stop=[50, 50]),  # Aligned
            )

            data = np.ones((50, 50), dtype=np.uint8)
            batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])

            class MockReader:
                def read_all(self):
                    return pa.Table.from_batches([batch])

            # Should succeed
            server.uploads.write_chunk(upload, MockReader())

            CacheManager.reset()

    def test_unaligned_chunk_rejected(self):
        """Unaligned chunk upload is rejected."""
        pytest.importorskip("zarr")

        from biopb_tensor_server.serving.server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = TensorFlightServer(
                location="grpc://localhost:0",
                writable=True,
                write_dir=Path(tmpdir),
            )

            # Create zarr-backed source with chunk_shape [50, 50]
            req_desc = TensorDescriptor(
                array_id="ome_zarr:test",
                shape=[100, 100],
                dtype="uint8",
                chunk_shape=[50, 50],
            )
            response_desc = server.uploads.create_tensor(req_desc)
            source_id = response_desc.array_id

            # Upload unaligned chunk (start not on grid)
            upload = ChunkUpload(
                source_id=source_id,
                bounds=ChunkBounds(start=[10, 20], stop=[60, 70]),  # Not aligned to 50
            )

            data = np.ones((50, 50), dtype=np.uint8)
            batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])

            class MockReader:
                def read_all(self):
                    return pa.Table.from_batches([batch])

            with pytest.raises(flight.FlightServerError, match="not aligned"):
                server.uploads.write_chunk(upload, MockReader())


class TestBuildMinimalOmeMetadata:
    """Tests for OME metadata generation."""

    def test_minimal_metadata_structure(self):
        """Generated metadata has required fields."""
        desc = TensorDescriptor(
            array_id="test",
            shape=[100, 100, 100],
            dtype="uint8",
            chunk_shape=[50, 50, 50],
            dim_labels=["z", "y", "x"],
        )

        metadata = minimal_ome_metadata(desc)

        assert "multiscales" in metadata
        assert len(metadata["multiscales"]) == 1

        multiscale = metadata["multiscales"][0]
        assert "axes" in multiscale
        assert len(multiscale["axes"]) == 3
        assert "datasets" in multiscale
        assert len(multiscale["datasets"]) == 1

    def test_axis_types_detected(self):
        """Axis types are detected from labels."""
        desc = TensorDescriptor(
            array_id="test",
            shape=[1, 100, 100, 100],
            dtype="uint8",
            dim_labels=["c", "z", "y", "x"],
        )

        metadata = minimal_ome_metadata(desc)
        axes = metadata["multiscales"][0]["axes"]

        assert axes[0]["type"] == "channel"  # 'c' detected as channel
        assert axes[1]["type"] == "space"  # 'z' detected as space
        assert axes[2]["type"] == "space"  # 'y' detected as space
        assert axes[3]["type"] == "space"  # 'x' detected as space


class TestCachedSourceContentVersion:
    """Per-upload generation token folded into cache keys (biopb/biopb#178).

    cache: sources have deterministic ids, so a re-upload reuses the id. The
    generation token gives each upload a fresh cache namespace; without it
    ``CacheManager.put`` would decline to overwrite the prior upload's
    chunk and serve stale data.
    """

    def _adapter(self, cv):
        return CachedSourceAdapter(
            source_id="cache_v",
            shape=[4, 4],
            dtype="uint8",
            chunk_shape=[4, 4],
            content_version=cv,
        )

    def test_versioned_write_read_roundtrip(self, tmp_path):
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        try:
            cv = b"gen:1"
            adapter = self._adapter(cv)
            data = np.arange(16, dtype=np.uint8).reshape(4, 4)
            bounds = ChunkBounds(start=[0, 0], stop=[4, 4])
            adapter.write_chunk(bounds, data)

            # The base read plan mints version-wrapped chunk_ids; the client echoes
            # one back on read -> it must resolve to the written data.
            wrapped = wrap_content_version(encode_chunk_id("cache_v", bounds), cv)
            assert content_version_of(wrapped) == cv
            batch = adapter.resolve_chunk_data(wrapped, CacheManager.get_instance())
            np.testing.assert_array_equal(unpack_chunk_array(batch), data)

            # A legacy unwrapped id for the same bounds is a different namespace and
            # must NOT resolve on a versioned source.
            with pytest.raises(flight.FlightServerError):
                adapter.resolve_chunk_data(
                    encode_chunk_id("cache_v", bounds), CacheManager.get_instance()
                )
        finally:
            CacheManager.reset()

    def test_reupload_new_generation_serves_fresh_data(self, tmp_path):
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        try:
            bounds = ChunkBounds(start=[0, 0], stop=[4, 4])
            old = np.zeros((4, 4), dtype=np.uint8)
            new = np.full((4, 4), 7, dtype=np.uint8)

            a1 = self._adapter(b"gen:1")
            a1.write_chunk(bounds, old)

            # Re-upload: same deterministic source_id, new generation, new bytes.
            a2 = self._adapter(b"gen:2")
            a2.write_chunk(bounds, new)

            # Without the version namespace, a2's start_compute would find gen:1's
            # entry and keep the STALE bytes. The fresh namespace serves the new data.
            id2 = wrap_content_version(encode_chunk_id("cache_v", bounds), b"gen:2")
            batch2 = a2.resolve_chunk_data(id2, CacheManager.get_instance())
            np.testing.assert_array_equal(unpack_chunk_array(batch2), new)

            # The prior generation's chunk is a distinct entry, still intact.
            id1 = wrap_content_version(encode_chunk_id("cache_v", bounds), b"gen:1")
            batch1 = a1.resolve_chunk_data(id1, CacheManager.get_instance())
            np.testing.assert_array_equal(unpack_chunk_array(batch1), old)
        finally:
            CacheManager.reset()

    def test_unversioned_adapter_is_legacy(self, tmp_path):
        from biopb_tensor_server.core.chunk_batch import unpack_chunk_array

        CacheManager.reset()
        CacheManager.initialize(CacheConfig(file_cache_dir=tmp_path / "cache"))
        try:
            adapter = CachedSourceAdapter(
                source_id="cache_legacy",
                shape=[4, 4],
                dtype="uint8",
                chunk_shape=[4, 4],
            )
            assert adapter.content_version is None
            data = np.ones((4, 4), dtype=np.uint8)
            bounds = ChunkBounds(start=[0, 0], stop=[4, 4])
            adapter.write_chunk(bounds, data)
            # Unversioned -> legacy unwrapped id resolves, byte-identical to pre-#178.
            batch = adapter.resolve_chunk_data(
                encode_chunk_id("cache_legacy", bounds), CacheManager.get_instance()
            )
            np.testing.assert_array_equal(unpack_chunk_array(batch), data)
        finally:
            CacheManager.reset()

    def test_generation_monotonic_and_distinct(self):
        tokens = [CachedSourceAdapter.next_content_version() for _ in range(5)]
        assert all(t.startswith(b"gen:") for t in tokens)
        assert len(set(tokens)) == 5  # all distinct
        gens = [int(t.split(b":")[1]) for t in tokens]
        assert gens == sorted(gens)  # strictly increasing even under a rapid loop

    def test_each_upload_of_a_name_gets_its_own_generation(self):
        """Two adapters built for one name share the id and differ in gen.

        Built directly rather than through ``create_source``, which refuses the
        second: the case this guards is an upload after a *restart*, where the
        registry is empty but the persisted file cache may still hold the
        prior upload's chunks under the same deterministic id.
        """
        req = TensorDescriptor(
            array_id="cache:reupload",
            shape=[8, 8],
            dtype="uint8",
            chunk_shape=[8, 8],
            dim_labels=["y", "x"],
        )
        a1 = CachedSourceAdapter.create_upload(
            "reupload", req, metadata=None, write_dir=None
        )
        a2 = CachedSourceAdapter.create_upload(
            "reupload", req, metadata=None, write_dir=None
        )

        assert a1.source_id == a2.source_id  # deterministic id -> same source
        assert a1.content_version is not None and a2.content_version is not None
        assert a1.content_version.startswith(b"gen:")
        assert a1.content_version != a2.content_version  # a fresh namespace each


def _upload_whole(client, arr, name):
    """Declare from the array, fill it, seal it; the id for reading back."""
    desc = client.create_tensor(name, arr)
    client.upload_array(desc, arr)
    return desc.array_id


class TestConcurrentChunkUpload:
    """`upload_array` ships several chunks at once, via `da.store` (#590).

    The serial loop this replaced paid a full `do_put` round trip
    (open/write/done/close/read) per chunk with nothing overlapping it, so the
    cost was latency multiplied by chunk count.
    """

    @staticmethod
    def _counting_put(monkeypatch, hold=0.0):
        """Swap in a `_put_chunk` that records how many run at once.

        `_put_chunk` is the seam because that is what the store target calls --
        it holds connection parameters rather than a session, precisely so it
        can be pickled to a worker, so there is no session method left to wrap.
        """
        real = _upload._put_chunk
        lock = threading.Lock()
        state = {"live": 0, "peak": 0}

        def counting(*args, **kwargs):
            with lock:
                state["live"] += 1
                state["peak"] = max(state["peak"], state["live"])
            try:
                if hold:
                    time.sleep(hold)
                return real(*args, **kwargs)
            finally:
                with lock:
                    state["live"] -= 1

        monkeypatch.setattr(_upload, "_put_chunk", counting)
        return state

    def test_a_fanned_out_upload_round_trips_every_chunk(self, client):
        """The whole point: concurrent chunks, byte-identical on the way back.

        Values differ per chunk, which is what makes a swapped or dropped chunk
        visible -- a uniform array would round-trip through almost any bug.
        """
        source = np.arange(12 * 40 * 40, dtype=np.uint16).reshape(12, 40, 40)
        source_id = _upload_whole(
            client, da.from_array(source, chunks=(1, 40, 40)), "cache:fanout"
        )

        assert client.get_upload_status(source_id)["state"] == "READY"
        np.testing.assert_array_equal(client.get_tensor(source_id).compute(), source)

    def test_the_chunks_really_do_overlap(self, client, monkeypatch):
        """Concurrency, observed rather than assumed.

        Without this the suite would pass just as happily on something that
        still uploads one at a time, and #590 would be "fixed" by code that
        does not overlap anything. The hold is load-bearing: over loopback a
        round trip can finish before the next task is even scheduled.
        """
        state = self._counting_put(monkeypatch, hold=0.05)
        arr = da.from_array(np.zeros((16, 20, 20), dtype=np.uint8), chunks=(1, 20, 20))

        with dask.config.set(scheduler="threads", num_workers=4):
            _upload_whole(client, arr, "cache:overlap")

        assert state["peak"] > 1, "uploads ran one at a time"
        assert state["peak"] <= 4, f"asked for 4 at once, ran {state['peak']}"

    def test_the_caller_can_still_ask_for_a_serial_upload(self, client, monkeypatch):
        """Throttling is dask's own dial, not an argument this SDK invents.

        `upload_array` names no scheduler, so a link or a backend that wants
        exactly one write in flight says so the way it would for any other dask
        call -- and that has to actually reach `da.store`.
        """
        state = self._counting_put(monkeypatch)
        source = np.arange(6 * 20 * 20, dtype=np.uint16).reshape(6, 20, 20)
        arr = da.from_array(source, chunks=(1, 20, 20))

        with dask.config.set(scheduler="threads", num_workers=1):
            source_id = _upload_whole(client, arr, "cache:serial")

        assert state["peak"] == 1
        assert client.get_upload_status(source_id)["state"] == "READY"
        np.testing.assert_array_equal(client.get_tensor(source_id).compute(), source)

    def test_a_failed_chunk_surfaces_as_itself(self, client, monkeypatch):
        """A chunk that raises must reach the caller, with its own type.

        Callers catch on type (`flight.FlightError` and friends), so a write
        failing inside the graph has to come back out as itself rather than
        wrapped in something dask-shaped -- and must not hang on the siblings.
        """
        real = _upload._put_chunk
        # Not the first, so some have landed and others are still to come --
        # the state the unwind actually has to handle. `count.__next__` is
        # atomic, so the workers need no lock of their own.
        calls = itertools.count()

        class ChunkRefused(RuntimeError):
            pass

        def failing(*args, **kwargs):
            if next(calls) == 3:
                raise ChunkRefused("no")
            return real(*args, **kwargs)

        monkeypatch.setattr(_upload, "_put_chunk", failing)
        arr = da.from_array(np.zeros((10, 20, 20), dtype=np.uint8), chunks=(1, 20, 20))

        with pytest.raises(ChunkRefused):
            _upload_whole(client, arr, "cache:doomed")

    def test_out_of_order_arrival_still_completes_the_upload(self, client):
        """Readiness counts chunks into a set, so order is not observable.

        Fanning out makes arrival order arbitrary; this pins the server-side
        property that makes that safe, by uploading the chunks deliberately
        backwards and asking for READY.
        """
        session = client._upload
        source = np.arange(5 * 20 * 20, dtype=np.uint16).reshape(5, 20, 20)
        desc = session.create_tensor(
            "cache:backwards",
            source,
            chunk_shape=(1, 20, 20),
            dim_labels=["z", "y", "x"],
        )
        for z in reversed(range(5)):
            session.upload_chunk(
                desc,
                ChunkBounds(start=[z, 0, 0], stop=[z + 1, 20, 20]),
                source[z : z + 1],
            )
        status = session.finish_upload(desc)
        assert status["state"] == "READY"
        assert status["uploaded_chunks"] == 5

    def test_a_shared_upstream_block_is_computed_once(self, client):
        """Why `da.store` and not a compute-and-ship loop per chunk.

        `upload_array` rechunks onto the upload grid whenever the caller's
        array is not already on it, so an output chunk drawing on a coarser
        source block is the ordinary case rather than a corner. A per-chunk
        loop would slice that rechunked array and `.compute()` once per output
        chunk, and separate `.compute()` calls share no cache -- measured at 34
        runs of the source block against 10 on this very graph.

        Counted rather than timed: duplication shows up in invocations however
        cheap the upstream task is, where a stopwatch would drown it in
        per-`.compute()` fixed cost.
        """
        # A list, not an `itertools.count`: this function rides into the dask
        # graph and gets pickled, which itertools objects will stop supporting
        # in 3.14. `append` is atomic and a total is all this needs.
        runs = []

        def count(block):
            runs.append(1)
            return block

        # Source on a 4-thick grid, uploaded on a 1-thick one: eight source
        # blocks feeding 32 output chunks.
        coarse = da.from_array(
            np.zeros((32, 64, 64), dtype=np.uint16), chunks=(4, 64, 64)
        )
        _upload_whole(
            client, coarse.map_blocks(count).rechunk((1, 64, 64)), "cache:shared"
        )

        # Eight source blocks; dask's rechunk splits two, hence ten. What
        # matters is the order of magnitude against a loop's 34.
        assert len(runs) <= 10

    def test_the_store_target_survives_a_trip_to_a_worker(self, client):
        """`da.store` puts the target in the graph, so it must pickle -- and work.

        A target holding an `UploadSession` would drag a `FlightClient` along,
        which does not pickle at all, failing every upload the moment an agent
        called `_dask_ctl.attach()`. Surviving the round trip is the easy half:
        this asserts the revived target, holding no session, dials for itself.
        """
        session = client._upload
        source = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
        desc = session.create_tensor(
            "cache:revived", source, chunk_shape=(1, 8, 8), dim_labels=["z", "y", "x"]
        )
        target = _upload._UploadTarget(
            session._state.location,
            session._state.token,
            session._state.tls_trust,
            desc.array_id,
            source.shape,
            source.dtype,
        )

        revived = pickle.loads(pickle.dumps(target))

        for z in range(4):
            revived[(slice(z, z + 1), slice(0, 8), slice(0, 8))] = source[z : z + 1]

        assert session.finish_upload(desc)["state"] == "READY"
        np.testing.assert_array_equal(
            client.get_tensor(desc.array_id).compute(), source
        )


class TestDiscard:
    """Giving up on an upload: the adapter stays, in a terminal state (#1).

    Disposal rather than cancellation -- stopping whatever was producing the
    data belongs to whoever runs it. What is tested here is only the fate of the
    source, and that a writer still in flight learns *why* it can no longer
    write rather than that its source is missing.
    """

    @staticmethod
    def _tombstone(source_id="cache_tombstone", shape=(4,), chunk=(2,)):
        """A discarded cache adapter, off the wire, for the guards it keeps."""
        adapter = CachedSourceAdapter(
            source_id=source_id, shape=list(shape), dtype="<u2", chunk_shape=list(chunk)
        )
        adapter.discard("gone")
        return adapter

    @staticmethod
    def _make_source(client, name="cache:discard-me", shape=(4, 4), chunk=(2, 2)):
        return client.create_tensor(
            name, np.empty(shape, dtype=np.uint16), chunk_shape=chunk
        )

    @staticmethod
    def _put(client, desc, start, stop, fill=7):
        data = np.full(
            [b - a for a, b in zip(start, stop, strict=True)], fill, dtype=np.uint16
        )
        client.upload_chunk(desc, ChunkBounds(start=list(start), stop=list(stop)), data)

    def test_discard_leaves_a_tombstone_that_refuses_reads_too(
        self, writable_server, client
    ):
        """The adapter stays registered, terminal, so both a writer and a
        reader still unwinding learn the reason instead of "not found"."""
        desc = self._make_source(client, shape=(2, 2), chunk=(2, 2))
        self._put(client, desc, (0, 0), (2, 2))
        adapter = writable_server.sources.get(desc.array_id)

        status = writable_server.uploads.discard(desc.array_id, "client went away")

        assert status["state"] == "DISCARDED"
        assert status["reason"] == "client went away"
        assert writable_server.sources.get(desc.array_id) is adapter
        assert client.get_upload_status(desc.array_id)["state"] == "DISCARDED"
        # It is a tombstone, not a source: not listed, and its bytes are gone.
        assert desc.array_id not in client.list_sources()
        chunk_id = encode_chunk_id(
            desc.array_id, ChunkBounds(start=[0, 0], stop=[2, 2])
        )
        with pytest.raises(flight.FlightServerError, match="client went away"):
            adapter.resolve_chunk_data(chunk_id, CacheManager.get_instance())

    def test_a_write_after_discard_says_discarded_not_missing(
        self, writable_server, client
    ):
        """The reason the tombstone exists.

        Were the adapter dropped at discard, the lookup would fail first and
        tell a job unwinding after its own discard that the source never
        existed.
        """
        desc = self._make_source(client)
        writable_server.uploads.discard(desc.array_id, "superseded")

        with pytest.raises(UploadRefused) as exc:
            self._put(client, desc, (0, 0), (2, 2))

        # Type discriminates, and the state and reason are fields, so a client
        # never has to match on the message.
        assert exc.value.state == "DISCARDED"
        assert exc.value.reason == "superseded"
        assert exc.value.source_id == desc.array_id

    def test_a_forgotten_source_is_missing_not_discarded(self, writable_server, client):
        """An unregistered source still means "not found" -- the states are distinct."""
        desc = self._make_source(client)
        writable_server.unregister_source(desc.array_id)

        with pytest.raises(flight.FlightError) as exc:
            self._put(client, desc, (0, 0), (2, 2))

        assert not isinstance(exc.value, flight.FlightCancelledError)
        assert "not found" in str(exc.value).lower()

    def test_an_in_flight_upload_stops_at_a_chunk_boundary(
        self, writable_server, client, monkeypatch
    ):
        """Discard mid-upload surfaces through `da.store` as the cancelled error.

        One worker, so the order is deterministic: the first chunk lands, the
        discard happens, the second is refused. Disposal stops the upload where
        it stands rather than rewinding it -- the chunk already written stays
        counted on the tombstone.
        """
        desc = self._make_source(client, shape=(8, 2), chunk=(2, 2))
        real = _upload._put_chunk
        written = []

        def put_then_discard(*args, **kwargs):
            real(*args, **kwargs)
            written.append(1)
            if len(written) == 1:
                writable_server.uploads.discard(desc.array_id, "stopped early")

        monkeypatch.setattr(_upload, "_put_chunk", put_then_discard)

        arr = da.from_array(np.arange(16, dtype=np.uint16).reshape(8, 2), chunks=(2, 2))
        with pytest.raises(UploadRefused, match="stopped early"):
            with dask.config.set(scheduler="threads", num_workers=1):
                client._upload._store_chunks(desc.array_id, arr)

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert status["uploaded_chunks"] == 1
        assert len(written) == 1
        # How far it got is reported, but the chunk ids behind that number are
        # not kept: a tombstone outlives its upload and must not scale with it.
        assert (
            writable_server.sources.get(desc.array_id).upload.uploaded_chunk_ids
            == set()
        )

    def test_a_straggler_does_not_walk_the_tombstone_back(self):
        """A write that passed the refusal check before the discard landed still
        stores its chunk, but must not count it: the mark guards itself."""
        adapter = self._tombstone()

        adapter._mark_chunk(ChunkBounds(start=[0], stop=[2]))

        status = adapter.upload_status()
        assert status["state"] == "DISCARDED"
        assert status["uploaded_chunks"] == 0

    def test_discard_is_idempotent_and_keeps_the_first_reason(
        self, writable_server, client
    ):
        desc = self._make_source(client)
        writable_server.uploads.discard(desc.array_id, "first")

        again = writable_server.uploads.discard(desc.array_id, "second")

        assert again["state"] == "DISCARDED"
        assert again["reason"] == "first"

    def test_discarding_an_untracked_source_reports_unknown(self, writable_server):
        """Total, so a retry after the tombstone is reaped is not an error."""
        assert writable_server.uploads.discard("cache_never", "x")["state"] == "UNKNOWN"

    def test_discarding_a_source_that_is_not_an_upload_reports_unknown(
        self, writable_server
    ):
        """A catalogued store accepts writes untracked; there is nothing to discard."""
        from biopb_tensor_server.adapters.zarr import ZarrAdapter

        source_id = "plain_zarr"
        zarr_module = pytest.importorskip("zarr")
        adapter = ZarrAdapter(zarr_module.zeros((2, 2), chunks=(2, 2)), source_id)
        writable_server.register_source(source_id, adapter)

        assert writable_server.uploads.status(source_id)["state"] == "UNKNOWN"
        assert writable_server.uploads.discard(source_id, "x")["state"] == "UNKNOWN"

    def test_a_discarded_name_stays_taken(self, writable_server, client):
        """A tombstone holds its name like any other source.

        A named cache source's id is a hash of that name, so a retry would land
        on the tombstone -- and `create_tensor` refuses every collision rather
        than replacing what holds the id (`upload_session_test.py` has why).
        The retry goes under a new name; the tombstone keeps answering for the
        old one with its reason.
        """
        first = self._make_source(
            client, name="cache:retry-me", shape=(2, 2), chunk=(2, 2)
        )
        writable_server.uploads.discard(first.array_id, "gave up")

        with pytest.raises(flight.FlightServerError, match="already exists"):
            self._make_source(client, name="cache:retry-me", shape=(2, 2), chunk=(2, 2))

        assert client.get_upload_status(first.array_id)["state"] == "DISCARDED"
        assert client.get_upload_status(first.array_id)["reason"] == "gave up"

    def test_a_completed_upload_can_still_be_discarded(self, writable_server, client):
        """Disposal is not only for failures: dropping a finished result is the
        same operation."""
        desc = self._make_source(client, shape=(2, 2), chunk=(2, 2))
        self._put(client, desc, (0, 0), (2, 2))
        client.finish_upload(desc)
        assert client.get_upload_status(desc.array_id)["state"] == "READY"

        assert writable_server.uploads.discard(desc.array_id, "done with it")[
            "state"
        ] == ("DISCARDED")

    def test_zarr_backed_uploads_are_refused(self, writable_server, client):
        """A .zarr on disk and a catalog row are not this call's to release."""
        source_id = client.create_tensor(
            "ome_zarr:keepme", np.empty((4, 4), np.uint16), chunk_shape=(2, 2)
        ).array_id

        with pytest.raises(ValueError, match="not a cache-backed upload"):
            writable_server.uploads.discard(source_id, "nope")

        assert client.get_upload_status(source_id)["state"] == "PENDING"

    def test_the_poll_sees_the_reason(self, writable_server, client):
        """A discarded upload is terminal and the status says why, which is
        what a caller's poll loop stops on instead of running to a timeout."""
        desc = self._make_source(client)
        writable_server.uploads.discard(desc.array_id, "client disconnected")

        status = client.get_upload_status(desc.array_id)
        assert status["state"] == "DISCARDED"
        assert status["reason"] == "client disconnected"

    def test_the_record_tracks_when_it_last_moved(self, writable_server, client):
        """`updated_at` is what will bound a tombstone's life, and what tells a
        stalled upload from a slow one. Both halves read the same field, so both
        are checked here: progress writes it, and discard writes it.

        Poisons the field and checks it was replaced, rather than comparing two
        clock readings. `time.monotonic` resolves to ~15.6ms on Windows before
        CPython 3.13 (GetTickCount64), and both operations finish well inside one
        tick -- so `after > before` is not merely flaky there, it is false.
        """
        desc = self._make_source(client, shape=(4, 2), chunk=(2, 2))
        state = writable_server.sources.get(desc.array_id).upload
        assert state.updated_at > 0  # stamped at creation

        state.updated_at = -1.0
        self._put(client, desc, (0, 0), (2, 2))
        assert state.updated_at > 0

        state.updated_at = -1.0
        writable_server.uploads.discard(desc.array_id, "enough")
        assert state.updated_at > 0

    def test_a_second_discard_does_not_extend_the_tombstone(
        self, writable_server, client
    ):
        """Otherwise a retrying caller could keep a tombstone alive forever."""
        desc = self._make_source(client)
        writable_server.uploads.discard(desc.array_id, "first")
        first = writable_server.sources.get(desc.array_id).upload.updated_at

        writable_server.uploads.discard(desc.array_id, "second")

        assert writable_server.sources.get(desc.array_id).upload.updated_at == first

    def test_a_straggler_does_not_extend_the_tombstone(self):
        """The mark guard returns before touching the clock."""
        adapter = self._tombstone()
        at_discard = adapter.upload.updated_at

        adapter._mark_chunk(ChunkBounds(start=[0], stop=[2]))

        assert adapter.upload.updated_at == at_discard

    def test_the_clock_stays_off_the_wire(self, writable_server, client):
        """A monotonic reading means nothing in another process, so it is not in
        the status contract. A client-facing "discarded at" would be a wall clock
        and a separate field."""
        desc = self._make_source(client)
        writable_server.uploads.discard(desc.array_id, "x")

        assert "updated_at" not in client.get_upload_status(desc.array_id)


def _durable_upload(server):
    req_desc = TensorDescriptor(
        array_id="ome_zarr:owned",
        shape=[8, 8],
        dtype="uint8",
        chunk_shape=[4, 4],
        dim_labels=["y", "x"],
    )
    return server.uploads.create_tensor(req_desc)


def test_a_durable_upload_lands_in_the_servers_catalog():
    """The upload path catalogues a durable upload itself, like any other
    registering caller -- so list_sources sees it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server = catalog_server(
            location="grpc://localhost:0", writable=True, write_dir=Path(tmpdir)
        )
        try:
            response_desc = _durable_upload(server)
            assert response_desc.array_id in _catalog_ids(server.metadata_db)
        finally:
            server.shutdown()


def test_a_durable_upload_to_a_catalog_less_server_is_addressable_not_listed():
    """metadata_db=None is a deployment shape, not a broken one: the upload
    succeeds and its id comes back, and there is simply nowhere to list it."""
    import pyarrow.flight as flight
    from biopb_tensor_server.serving.server import TensorFlightServer

    with tempfile.TemporaryDirectory() as tmpdir:
        server = TensorFlightServer(
            location="grpc://localhost:0", writable=True, write_dir=Path(tmpdir)
        )
        try:
            response_desc = _durable_upload(server)
            assert response_desc.array_id  # readable by the id it hands back
            assert server.sources.get(response_desc.array_id.split("/")[0]) is not None
            with pytest.raises(flight.FlightUnavailableError, match="no catalog"):
                list(server.list_flights(None, b""))
        finally:
            server.shutdown()
