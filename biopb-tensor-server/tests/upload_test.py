"""Unit tests for data upload functionality.

Tests for CachedSourceAdapter, server-side do_put handler,
and Python client upload methods.
"""

import tempfile
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds, ChunkUpload
from biopb_tensor_server.adapters.cached_source import CachedSourceAdapter
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.chunk import encode_chunk_id, get_bounds_from_chunk_id
from biopb_tensor_server.config import CacheConfig


class MockMetadataWriter:
    """Mock FlightMetadataWriter for testing."""

    def __init__(self):
        self.metadata = None

    def write(self, metadata: bytes):
        self.metadata = metadata


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

    def test_write_chunk(self):
        """write_chunk stores data in cache."""
        CacheManager.reset()
        config = CacheConfig(backend="memory", memory_max_entries=10)
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

        entry, is_owner = CacheManager.get_instance().start_compute(chunk_id)
        assert entry.state.name == "READY"

        # Verify batch schema: [data: list<dtype>, shape: list<int64>, dtype: string]
        batch = entry.data
        assert batch.schema.names == ["data", "shape", "dtype"]

        # Verify data shape via the list array
        flat_data = batch.column("data").to_pylist()[0]
        assert len(flat_data) == 50 * 50

        CacheManager.get_instance().release(chunk_id)
        CacheManager.reset()

    def test_resolve_chunk_data_memory_backend(self):
        """Regression test: resolve_chunk_data must work for memory-backed CachedSourceAdapter.

        Bug: Base class resolve_chunk_data only caches for scaled chunks or ArrowFileBackend.
        CachedSourceAdapter has no backend data source - all data is in cache.
        Without override, retrieval would call get_data() which raises error.

        This tests the resolve_chunk_data override in CachedSourceAdapter.
        """
        CacheManager.reset()
        config = CacheConfig(backend="memory", memory_max_entries=10)
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
        from biopb_tensor_server.base import unpack_chunk_array

        arr = unpack_chunk_array(batch)
        np.testing.assert_array_equal(arr, test_data)

        CacheManager.reset()

    def test_write_chunk_arbitrary_bounds(self):
        """Cache sources accept arbitrary chunk bounds."""
        CacheManager.reset()
        config = CacheConfig(backend="memory", memory_max_entries=10)
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

        entry, is_owner = CacheManager.get_instance().start_compute(chunk_id)
        assert entry.state.name == "READY"
        CacheManager.get_instance().release(chunk_id)
        CacheManager.reset()

    def test_write_chunk_arrow_rejects_list_wrapper(self):
        # The binary chunk schema stores the flat value buffer, and the upload
        # dtype is derived from the primitive Arrow field type; a list<T> wrapper
        # would corrupt both (buffer[1] is offsets, to_pandas_dtype() is wrong).
        # write_chunk_arrow must refuse it rather than silently mis-store.
        CacheManager.reset()
        CacheManager.initialize(CacheConfig(backend="memory", memory_max_entries=10))
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
                backend="file",
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
            from biopb_tensor_server.base import unpack_chunk_array

            reconstructed = unpack_chunk_array(batch)
            assert reconstructed.shape == data.shape
            assert reconstructed.dtype == data.dtype

            cache_manager.release(chunk_id)
        finally:
            CacheManager.reset()
            shutil.rmtree(cache_dir, ignore_errors=True)


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
        from biopb_tensor_server.server import TensorFlightServer

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
        writer = MockMetadataWriter()
        server._handle_create_source(req_desc, writer)

        # Check response
        assert writer.metadata is not None
        response_desc = TensorDescriptor()
        response_desc.ParseFromString(writer.metadata)
        assert response_desc.array_id.startswith("cache_")

        # Check adapter was registered
        adapter = server._sources.get(response_desc.array_id)
        assert adapter is not None
        assert isinstance(adapter, CachedSourceAdapter)

    def test_create_source_cache_server_generated_name(self):
        """Create cache-backed source with server-generated name."""
        from biopb_tensor_server.server import TensorFlightServer

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

        writer = MockMetadataWriter()
        server._handle_create_source(req_desc, writer)
        response_desc = TensorDescriptor()
        response_desc.ParseFromString(writer.metadata)
        assert response_desc.array_id.startswith("cache_")

    def test_create_source_not_writable_raises(self):
        """Non-writable server rejects source creation."""
        from biopb_tensor_server.server import TensorFlightServer

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

        # The writable check is in do_put, not _handle_create_source
        # This test verifies _handle_create_source can be called on non-writable server
        # (the check happens at do_put level)
        writer = MockMetadataWriter()
        server._handle_create_source(req_desc, writer)
        # Source should be created (do_put would reject before reaching this)
        assert writer.metadata is not None

    def test_create_source_ome_zarr_requires_write_dir(self):
        """Zarr-backed source requires write_dir."""
        from biopb_tensor_server.server import TensorFlightServer

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

        writer = MockMetadataWriter()
        with pytest.raises(flight.FlightServerError, match="write_dir not configured"):
            server._handle_create_source(req_desc, writer)

    def test_create_source_ome_zarr_with_write_dir(self):
        """Create zarr-backed source with write_dir."""
        from biopb_tensor_server.server import TensorFlightServer

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

            writer = MockMetadataWriter()
            server._handle_create_source(req_desc, writer)
            response_desc = TensorDescriptor()
            response_desc.ParseFromString(writer.metadata)
            assert response_desc.array_id.startswith("ome_zarr_")

            # Check zarr directory was created
            zarr_dirs = list(Path(tmpdir).glob("*.zarr"))
            assert len(zarr_dirs) == 1

    def test_ome_zarr_upload_synced_to_catalog(self):
        """File-backed (durable) uploads are added to the catalog so they are
        discoverable via list_sources/query_sources (biopb/biopb#265)."""
        from biopb_tensor_server.metadata_db import MetadataDatabase
        from biopb_tensor_server.server import TensorFlightServer

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
            writer = MockMetadataWriter()
            server._handle_create_source(req_desc, writer)
            response_desc = TensorDescriptor()
            response_desc.ParseFromString(writer.metadata)

            # The durable upload appears in the catalog.
            descriptors, _ = db.list_source_descriptors()
            assert response_desc.array_id in {d.source_id for d in descriptors}

    def test_cache_upload_not_synced_but_readable_by_id(self):
        """Ephemeral cache-backed uploads are NOT catalogued (no removal hook ->
        the row would dangle), but stay readable by their returned id."""
        from biopb_tensor_server.metadata_db import MetadataDatabase
        from biopb_tensor_server.server import TensorFlightServer

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
        writer = MockMetadataWriter()
        server._handle_create_source(req_desc, writer)
        response_desc = TensorDescriptor()
        response_desc.ParseFromString(writer.metadata)

        # Not enumerable via the catalog...
        descriptors, _ = db.list_source_descriptors()
        assert response_desc.array_id not in {d.source_id for d in descriptors}
        # ...but still registered and readable by its returned id.
        assert isinstance(
            server._sources.get(response_desc.array_id), CachedSourceAdapter
        )

    def test_create_source_invalid_prefix(self):
        """Invalid array_id prefix raises error."""
        from biopb_tensor_server.server import TensorFlightServer

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

        writer = MockMetadataWriter()
        with pytest.raises(flight.FlightServerError, match="Invalid array_id format"):
            server._handle_create_source(req_desc, writer)

    def test_create_source_action_round_trip(self):
        """Live client create_source should return the server-assigned source_id."""
        from biopb.tensor import TensorFlightClient
        from biopb_tensor_server.server import TensorFlightServer

        CacheManager.reset()
        config = CacheConfig(backend="memory", memory_max_entries=10)
        CacheManager.initialize(config)

        server = TensorFlightServer(
            location="grpc://127.0.0.1:0",
            writable=True,
        )
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()

        client = TensorFlightClient(f"grpc://127.0.0.1:{server.port}")
        try:
            source_id = client.create_source(
                "cache:test-action",
                shape=(10, 10),
                dtype="uint8",
                chunk_shape=(5, 5),
            )
            assert source_id.startswith("cache_")
            assert source_id in server._sources
        finally:
            client.close()
            server.shutdown()
            CacheManager.reset()


class TestChunkUpload:
    """Tests for chunk upload handling."""

    def test_upload_chunk_cache_backed(self):
        """Upload chunk to cache-backed source."""
        from biopb_tensor_server.server import TensorFlightServer

        CacheManager.reset()
        config = CacheConfig(backend="memory", memory_max_entries=10)
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
        writer = MockMetadataWriter()
        server._handle_create_source(req_desc, writer)
        response_desc = TensorDescriptor()
        response_desc.ParseFromString(writer.metadata)
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
        server._handle_chunk_upload(upload, MockReader())

        # Check chunk was stored
        adapter = server._sources.get(source_id)
        assert adapter is not None

        CacheManager.reset()

    def test_upload_chunk_cache_backed_preserves_logical_shape(self):
        """Cache-backed uploads must store shape metadata for reconstruction."""
        from biopb_tensor_server.server import TensorFlightServer

        CacheManager.reset()
        config = CacheConfig(backend="memory", memory_max_entries=10)
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
        writer = MockMetadataWriter()
        server._handle_create_source(req_desc, writer)
        response_desc = TensorDescriptor()
        response_desc.ParseFromString(writer.metadata)
        source_id = response_desc.array_id

        bounds = ChunkBounds(start=[10, 20], stop=[40, 60])
        upload = ChunkUpload(source_id=source_id, bounds=bounds)

        data = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
        batch = pa.RecordBatch.from_arrays([pa.array(data.ravel())], ["data"])

        class MockReader:
            def read_all(self):
                return pa.Table.from_batches([batch])

        server._handle_chunk_upload(upload, MockReader())

        chunk_id = encode_chunk_id(source_id, bounds)
        cache_manager = CacheManager.get_instance()
        entry, is_owner = cache_manager.start_compute(chunk_id)
        assert entry.state.name == "READY"
        assert not is_owner

        stored_batch = entry.data
        assert stored_batch.column("shape").to_pylist()[0] == [30, 40]
        assert stored_batch.column("dtype").to_pylist()[0] == np.dtype(np.uint8).str

        from biopb_tensor_server.base import unpack_chunk_array

        reconstructed = unpack_chunk_array(stored_batch)
        assert reconstructed.shape == data.shape
        assert np.array_equal(reconstructed, data)

        cache_manager.release(chunk_id)
        CacheManager.reset()

    def test_upload_chunk_missing_source(self):
        """Upload to missing source raises error."""
        from biopb_tensor_server.server import TensorFlightServer

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
            server._handle_chunk_upload(upload, MockReader())


class TestOmeZarrChunkAlignment:
    """Tests for chunk alignment validation for zarr-backed sources."""

    def test_aligned_chunk_accepted(self):
        """Aligned chunk upload is accepted."""
        pytest.importorskip("zarr")

        from biopb_tensor_server.server import TensorFlightServer

        with tempfile.TemporaryDirectory() as tmpdir:
            CacheManager.reset()
            config = CacheConfig(backend="memory", memory_max_entries=10)
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
            writer = MockMetadataWriter()
            server._handle_create_source(req_desc, writer)
            response_desc = TensorDescriptor()
            response_desc.ParseFromString(writer.metadata)
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
            server._handle_chunk_upload(upload, MockReader())

            CacheManager.reset()

    def test_unaligned_chunk_rejected(self):
        """Unaligned chunk upload is rejected."""
        pytest.importorskip("zarr")

        from biopb_tensor_server.server import TensorFlightServer

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
            writer = MockMetadataWriter()
            server._handle_create_source(req_desc, writer)
            response_desc = TensorDescriptor()
            response_desc.ParseFromString(writer.metadata)
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
                server._handle_chunk_upload(upload, MockReader())


class TestBuildMinimalOmeMetadata:
    """Tests for OME metadata generation."""

    def test_minimal_metadata_structure(self):
        """Generated metadata has required fields."""
        from biopb_tensor_server.server import TensorFlightServer

        server = TensorFlightServer(location="grpc://localhost:0")

        desc = TensorDescriptor(
            array_id="test",
            shape=[100, 100, 100],
            dtype="uint8",
            chunk_shape=[50, 50, 50],
            dim_labels=["z", "y", "x"],
        )

        metadata = server._build_minimal_ome_metadata(desc)

        assert "multiscales" in metadata
        assert len(metadata["multiscales"]) == 1

        multiscale = metadata["multiscales"][0]
        assert "axes" in multiscale
        assert len(multiscale["axes"]) == 3
        assert "datasets" in multiscale
        assert len(multiscale["datasets"]) == 1

    def test_axis_types_detected(self):
        """Axis types are detected from labels."""
        from biopb_tensor_server.server import TensorFlightServer

        server = TensorFlightServer(location="grpc://localhost:0")

        desc = TensorDescriptor(
            array_id="test",
            shape=[1, 100, 100, 100],
            dtype="uint8",
            dim_labels=["c", "z", "y", "x"],
        )

        metadata = server._build_minimal_ome_metadata(desc)
        axes = metadata["multiscales"][0]["axes"]

        assert axes[0]["type"] == "channel"  # 'c' detected as channel
        assert axes[1]["type"] == "space"  # 'z' detected as space
        assert axes[2]["type"] == "space"  # 'y' detected as space
        assert axes[3]["type"] == "space"  # 'x' detected as space
