"""Tests for multifield support - sources with multiple tensors of different shapes."""

import threading
import time

import numpy as np
from biopb.tensor import TensorFlightClient

from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.base import (
    BackendAdapter,
    ChunkEndpoint,
    DataSourceDescriptor,
    TensorDescriptor,
)


class MockMultifieldAdapter(BackendAdapter):
    """Mock adapter simulating a multifield source with different-shaped tensors."""

    @classmethod
    def claim(cls, path, visited_identities):
        """Mock adapter does not participate in discovery."""
        return None

    def __init__(self, source_id: str, tensor_specs):
        """Initialize with multiple tensors.

        Args:
            source_id: Source identifier
            tensor_specs: List of (tensor_id, shape, dtype) tuples
        """
        self.source_id = source_id
        self.tensor_specs = tensor_specs
        self._tensor_adapters = {}

        # Create mock adapters for each tensor
        for idx, (tensor_id, shape, dtype) in enumerate(tensor_specs):
            self._tensor_adapters[tensor_id] = MockSingleTensorAdapter(
                f"{source_id}/{tensor_id}",  # Include source_id in array_id for chunk encoding
                shape,
                dtype,
                value=idx,  # Use index as value for deterministic testing
            )

        # Source-level metadata
        self._source_url = "mock://multifield"
        self._source_type = "mock-multifield"

    def get_tensor_descriptor(self) -> TensorDescriptor:
        """Return descriptor for the first tensor (default)."""
        first_spec = self.tensor_specs[0]
        return TensorDescriptor(
            array_id=first_spec[0],
            shape=list(first_spec[1]),
            chunk_shape=list(first_spec[1]),  # Single chunk
            dtype=first_spec[2],
        )

    def list_tensor_descriptors(self):
        """Return descriptors for all tensors - multifield override."""
        descriptors = []
        for tensor_id, shape, dtype in self.tensor_specs:
            descriptors.append(TensorDescriptor(
                array_id=tensor_id,
                shape=list(shape),
                chunk_shape=list(shape),  # Single chunk per tensor
                dtype=dtype,
            ))
        return descriptors

    def get_tensor_adapter(self, tensor_id: str) -> BackendAdapter:
        """Return adapter for specific tensor - multifield override."""
        if tensor_id in self._tensor_adapters:
            return self._tensor_adapters[tensor_id]
        raise ValueError(f"Unknown tensor: {tensor_id}")

    def get_source_descriptor(self) -> DataSourceDescriptor:
        """Build DataSourceDescriptor with correct source_id."""
        return DataSourceDescriptor(
            source_id=self.source_id,  # Use actual source_id, not tensor's
            source_url=self._source_url,
            source_type=self._source_type,
            tensors=self.list_tensor_descriptors(),
            metadata_json="",  # Not populated; returned via GetFlightInfo instead
        )

    def get_raw_chunk_endpoints(self):
        return iter([])

    def get_chunk_array(self, chunk_id: bytes) -> np.ndarray:
        return np.zeros((1, 1), dtype='uint8')

    def get_metadata(self) -> dict:
        return {"multifield": True, "n_tensors": len(self.tensor_specs)}


class MockSingleTensorAdapter(BackendAdapter):
    """Mock adapter for a single tensor within a multifield source."""

    @classmethod
    def claim(cls, path, visited_identities):
        """Mock adapter does not participate in discovery."""
        return None

    def __init__(self, array_id: str, shape: tuple, dtype: str, value: int = 0):
        # Parse array_id to get source_id and tensor_name
        # array_id format: source_id/tensor_name for multi-tensor
        if '/' in array_id:
            self.source_id, self._tensor_name = array_id.split('/', 1)
        else:
            self.source_id = array_id
            self._tensor_name = None
        self.shape = shape
        self.dtype = dtype
        self.value = value
        self._source_url = ""
        self._source_type = "mock-single"
        self._tensor_context = True  # Always in tensor context for mocks

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            shape=list(self.shape),
            chunk_shape=list(self.shape),
            dtype=self.dtype,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def get_raw_chunk_endpoints(self):
        # Single chunk covering entire tensor
        from biopb.tensor.ticket_pb2 import ChunkBounds

        from biopb_tensor_server.base import encode_chunk_id

        chunk_id = encode_chunk_id(self.array_id, b"0")
        yield ChunkEndpoint(
            chunk_id=chunk_id,
            bounds=ChunkBounds(start=[0] * len(self.shape), stop=list(self.shape)),
        )

    def get_chunk_array(self, chunk_id: bytes) -> np.ndarray:
        """Return mock chunk data as numpy array."""
        return np.full(self.shape, self.value, dtype=self.dtype)


class TestMultifieldSourceLevel:
    """Tests for source-level methods in multifield adapters."""

    def test_list_tensor_descriptors_returns_all_tensors(self):
        """list_tensor_descriptors() should return all tensor descriptors."""
        tensor_specs = [
            ("tensor_0", (64, 64), "uint8"),
            ("tensor_1", (128, 128), "uint8"),
            ("tensor_2", (32, 32), "uint8"),
        ]
        adapter = MockMultifieldAdapter("multifield-source", tensor_specs)

        descriptors = adapter.list_tensor_descriptors()

        assert len(descriptors) == 3
        assert descriptors[0].array_id == "tensor_0"
        assert descriptors[0].shape == [64, 64]
        assert descriptors[1].array_id == "tensor_1"
        assert descriptors[1].shape == [128, 128]
        assert descriptors[2].array_id == "tensor_2"
        assert descriptors[2].shape == [32, 32]

    def test_get_tensor_adapter_returns_correct_adapter(self):
        """get_tensor_adapter() should return adapter for specific tensor."""
        tensor_specs = [
            ("tensor_0", (64, 64), "uint8"),
            ("tensor_1", (128, 128), "uint8"),
        ]
        adapter = MockMultifieldAdapter("multifield-source", tensor_specs)

        tensor_adapter = adapter.get_tensor_adapter("tensor_1")

        desc = tensor_adapter.get_tensor_descriptor()
        assert desc.array_id == "multifield-source/tensor_1"  # Full path
        assert desc.shape == [128, 128]

    def test_get_source_descriptor_contains_all_tensors(self):
        """get_source_descriptor() should contain all tensor info."""
        tensor_specs = [
            ("tensor_0", (64, 64), "uint8"),
            ("tensor_1", (128, 128), "uint16"),
        ]
        adapter = MockMultifieldAdapter("multifield-source", tensor_specs)

        source_desc = adapter.get_source_descriptor()

        assert source_desc.source_id == "multifield-source"  # Uses actual source_id
        assert source_desc.source_url == "mock://multifield"
        assert source_desc.source_type == "mock-multifield"
        assert len(source_desc.tensors) == 2
        assert source_desc.tensors[0].array_id == "tensor_0"
        assert source_desc.tensors[1].array_id == "tensor_1"


class TestMultifieldServerClient:
    """Tests for server/client with multifield sources."""

    def test_list_sources_returns_all_tensors_in_descriptor(self):
        """list_sources() should return DataSourceDescriptor with all tensors."""
        tensor_specs = [
            ("pos_0", (64, 64), "uint8"),
            ("pos_1", (100, 100), "uint8"),
        ]
        adapter = MockMultifieldAdapter("multifield-test", tensor_specs)

        server = TensorFlightServer("grpc://localhost:8877")
        server.register_source("multifield-test", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient("grpc://localhost:8877")

            sources = client.list_sources()

            assert "multifield-test" in sources
            source_desc = sources["multifield-test"]
            assert len(source_desc.tensors) == 2
            # Client has all tensor shape info upfront
            assert source_desc.tensors[0].shape == [64, 64]
            assert source_desc.tensors[1].shape == [100, 100]

            client.close()
        finally:
            server.shutdown()

    def test_client_can_access_different_tensors_from_same_source(self):
        """Client should be able to access different tensors from a multifield source."""
        tensor_specs = [
            ("pos_0", (32, 32), "uint8"),  # Value 0
            ("pos_1", (64, 64), "uint8"),  # Value 1
        ]
        adapter = MockMultifieldAdapter("multifield-access", tensor_specs)

        server = TensorFlightServer("grpc://localhost:8876")
        server.register_source("multifield-access", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient("grpc://localhost:8876")

            # Access first tensor
            arr0 = client.get_tensor("multifield-access", "pos_0")
            assert arr0.shape == (32, 32)
            data0 = arr0.compute()
            assert data0.mean() == 0  # Value based on tensor_id

            # Access second tensor (different shape)
            arr1 = client.get_tensor("multifield-access", "pos_1")
            assert arr1.shape == (64, 64)
            data1 = arr1.compute()
            assert data1.mean() == 1  # Value based on tensor_id

            client.close()
        finally:
            server.shutdown()

    def test_single_tensor_source_still_works(self):
        """Single tensor sources should still work with new API."""
        tensor_specs = [
            ("single-tensor", (50, 50), "uint8"),
        ]
        adapter = MockMultifieldAdapter("single-source", tensor_specs)

        server = TensorFlightServer("grpc://localhost:8875")
        server.register_source("single-source", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient("grpc://localhost:8875")

            sources = client.list_sources()
            assert len(sources["single-source"].tensors) == 1

            # Access the single tensor
            arr = client.get_tensor("single-source", "single-tensor")
            assert arr.shape == (50, 50)

            client.close()
        finally:
            server.shutdown()


class TestMultifieldDifferentDtypes:
    """Tests for multifield with different data types."""

    def test_tensors_with_different_dtypes(self):
        """Tensors in a source can have different dtypes."""
        tensor_specs = [
            ("uint8_tensor", (64, 64), "uint8"),
            ("float32_tensor", (128, 128), "float32"),
            ("uint16_tensor", (32, 32), "uint16"),
        ]
        adapter = MockMultifieldAdapter("mixed-dtype-source", tensor_specs)

        descriptors = adapter.list_tensor_descriptors()

        assert descriptors[0].dtype == "uint8"
        assert descriptors[1].dtype == "float32"
        assert descriptors[2].dtype == "uint16"