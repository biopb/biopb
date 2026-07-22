"""Tests for multifield support - sources with multiple tensors of different shapes."""

import threading
import time

import numpy as np
import pytest
from biopb.tensor import TensorFlightClient
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.core.base import (
    DataSourceDescriptor,
    TensorAdapter,
    TensorDescriptor,
    strip_source_prefix,
)


class MockMultifieldAdapter(TensorAdapter):
    """Mock adapter simulating a multifield source with different-shaped tensors."""

    @classmethod
    def claim(cls, path, visited_identities):
        """Mock adapter does not participate in discovery."""
        return None

    @classmethod
    def create_from_config(cls, source, credentials_config=None):
        """Mock adapter is not created from config."""
        raise NotImplementedError("MockMultifieldAdapter is for testing only")

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
            descriptors.append(
                TensorDescriptor(
                    array_id=tensor_id,
                    shape=list(shape),
                    chunk_shape=list(shape),  # Single chunk per tensor
                    dtype=dtype,
                )
            )
        return descriptors

    def get_tensor_adapter(self, tensor_id: str) -> TensorAdapter:
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

    def get_metadata(self) -> dict:
        return {"multifield": True, "n_tensors": len(self.tensor_specs)}

    def get_data(self, bounds):
        """Mock get_data - raises since multifield adapter delegates to tensor adapters."""
        raise NotImplementedError(
            "MockMultifieldAdapter.get_data() should not be called directly"
        )


class MockSingleTensorAdapter(TensorAdapter):
    """Mock adapter for a single tensor within a multifield source."""

    @classmethod
    def claim(cls, path, visited_identities):
        """Mock adapter does not participate in discovery."""
        return None

    @classmethod
    def create_from_config(cls, source, credentials_config=None):
        """Mock adapter is not created from config."""
        raise NotImplementedError("MockSingleTensorAdapter is for testing only")

    def __init__(self, array_id: str, shape: tuple, dtype: str, value: int = 0):
        # Parse array_id to get source_id and tensor_name
        # array_id format: source_id/tensor_name for multi-tensor
        if "/" in array_id:
            self.source_id, self._tensor_name = array_id.split("/", 1)
        else:
            self.source_id = array_id
            self._tensor_name = None
        self.shape = shape
        self.dtype = dtype
        self.value = value
        self._source_url = ""
        self._source_type = "mock-single"

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            shape=list(self.shape),
            chunk_shape=list(self.shape),
            dtype=self.dtype,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds) -> np.ndarray:
        """Return mock data within bounds."""
        super().get_data(bounds)
        shape = tuple(
            int(stop - start)
            for start, stop in zip(bounds.start, bounds.stop, strict=True)
        )
        return np.full(shape, self.value, dtype=self.dtype)

    def get_metadata(self) -> dict:
        """Return mock metadata."""
        return {"mock_tensor": True, "value": self.value}


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

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("multifield-test", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

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

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("multifield-access", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # Access first tensor
            arr0 = client.get_tensor("multifield-access/pos_0")
            assert arr0.shape == (32, 32)
            data0 = arr0.compute()
            assert data0.mean() == 0  # Value based on tensor_id

            # Access second tensor (different shape)
            arr1 = client.get_tensor("multifield-access/pos_1")
            assert arr1.shape == (64, 64)
            data1 = arr1.compute()
            assert data1.mean() == 1  # Value based on tensor_id

            client.close()
        finally:
            server.shutdown()

    def test_get_descriptor_with_bare_source_id_resolves_default_tensor(self):
        """get_descriptor(source_id)/get_physical_scale() with no within-source
        field must resolve the source's default (first) tensor, not forward "" to
        the adapter.

        Regression for #44: the client sends tensor_id="" (proto3 default) when
        none is given; before the chokepoint default-resolution in
        get_flight_info, that "" reached MockMultifieldAdapter.get_tensor_adapter
        -- which (like aicsimageio's scene lookup) raises on an unknown tensor,
        surfacing as a Flight error for a documented-valid call.
        """
        tensor_specs = [
            ("pos_0", (32, 32), "uint8"),
            ("pos_1", (64, 64), "uint8"),
        ]
        adapter = MockMultifieldAdapter("multifield-default", tensor_specs)

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("multifield-default", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # Bare source_id -> must not raise "Unknown tensor: " and must come
            # back anchored on the first tensor (server-side default resolution).
            # Structural probe (checks shape/array_id, not metadata), so this bare
            # server needs no metadata catalog.
            desc = client.get_descriptor("multifield-default", with_metadata=False)
            # Shape unambiguously identifies the default (first) tensor; the
            # descriptor reports the globally-unique source_id/field array_id
            # (identity policy).
            assert desc.shape == [32, 32]
            assert desc.array_id == "multifield-default/pos_0"

            # get_physical_scale rides the same no-tensor_id path; the mock
            # advertises none, so None (cleanly), never a Flight error.
            assert client.get_physical_scale("multifield-default") is None

            client.close()
        finally:
            server.shutdown()

    def test_get_descriptor_enumeration_vs_probe(self):
        """Issue #75: enumeration is list_sources(); get_descriptor() is a single
        tensor probe that reaches any scene and never clobbers the full
        enumeration."""
        tensor_specs = [
            ("pos_0", (32, 32), "uint8"),
            ("pos_1", (64, 64), "uint8"),
            ("pos_2", (16, 16), "uint8"),
        ]
        adapter = MockMultifieldAdapter("multi", tensor_specs)

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("multi", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # Enumeration: list_sources() carries ALL scenes for the source.
            enumerated = client.list_sources()["multi"].tensors
            assert len(enumerated) == 3
            assert sorted(list(t.shape) for t in enumerated) == [
                [16, 16],
                [32, 32],
                [64, 64],
            ]

            # Probe: get_descriptor(array_id) fetches exactly the addressed scene
            # -- including a non-default one (the #75 symptom: pos_1/pos_2 were
            # unreachable through the old get_source path). It caches only the
            # descriptor, so it must NOT clobber the full enumeration cached above.
            # Structural probes (shape only) -> no metadata catalog needed.
            assert client.get_descriptor("multi/pos_1", with_metadata=False).shape == [
                64,
                64,
            ]
            assert client.get_descriptor("multi/pos_2", with_metadata=False).shape == [
                16,
                16,
            ]
            # Read the _sources cache directly -- re-calling list_sources() would
            # just refetch.
            assert len(client._sources["multi"].tensors) == 3

            client.close()
        finally:
            server.shutdown()

    def test_get_tensor_accepts_qualified_id(self):
        """Identity policy: a tensor is read by its globally-unique
        source_id/field array_id.
        """
        tensor_specs = [
            ("pos_0", (32, 32), "uint8"),
            ("pos_1", (64, 64), "uint8"),
        ]
        adapter = MockMultifieldAdapter("mf-liberal", tensor_specs)

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("mf-liberal", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # Addressed by the full source-qualified array_id.
            qualified = client.get_tensor("mf-liberal/pos_1")
            assert qualified.shape == (64, 64)
            assert qualified.compute().mean() == 1  # value based on field index

            # The wire descriptor reports the qualified array_id (structural probe).
            desc = client.get_descriptor("mf-liberal/pos_1", with_metadata=False)
            assert desc.array_id == "mf-liberal/pos_1"

            client.close()
        finally:
            server.shutdown()

    def test_get_tensor_array_id_addressing(self):
        """get_tensor/get_tensor_pb take a single array_id (identity policy); a
        bare multi-tensor source id is ambiguous and must be qualified."""
        tensor_specs = [
            ("pos_0", (32, 32), "uint8"),
            ("pos_1", (64, 64), "uint8"),
        ]
        adapter = MockMultifieldAdapter("mf", tensor_specs)

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("mf", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # Canonical single-arg form: a qualified array_id reaches the scene.
            assert client.get_tensor("mf/pos_1").shape == (64, 64)
            assert client.get_tensor_pb("mf/pos_1") is not None

            # A bare multi-tensor source id is ambiguous -> must specify (never a
            # silent default; the #75 lesson).
            with pytest.raises(ValueError, match="multiple tensors"):
                client.get_tensor("mf")

            client.close()
        finally:
            server.shutdown()

    def test_fetch_endpoints_via_get_flight_info_multi_tensor(self):
        """The SerializedTensor endpoint-fetch fallback must derive source_id
        from a multi-tensor qualified array_id ("source_id/field").

        Regression for the identity-policy alignment: previously it set the
        FlightCmd source_id to the *whole* array_id, so for "mf-fetch/pos_1" the
        server looked up source "mf-fetch/pos_1" and failed. It must split on the
        first "/" -> source "mf-fetch", and the server reduces the tensor_id to
        the "pos_1" field.
        """
        from biopb.tensor.client import _fetch_endpoints_via_get_flight_info
        from biopb.tensor.serialized_pb2 import SerializedTensor

        tensor_specs = [
            ("pos_0", (32, 32), "uint8"),
            ("pos_1", (64, 64), "uint8"),
        ]
        adapter = MockMultifieldAdapter("mf-fetch", tensor_specs)

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("mf-fetch", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            pb = SerializedTensor(location=f"grpc://localhost:{server.port}")
            # Qualified multi-tensor array_id; endpoints left empty triggers the
            # GetFlightInfo fallback under test.
            pb.tensor_descriptor.array_id = "mf-fetch/pos_1"

            chunk_ids, bounds = _fetch_endpoints_via_get_flight_info(pb)

            assert len(chunk_ids) > 0
            assert len(chunk_ids) == len(bounds)
            # Extent matches pos_1's 64x64 shape -> the qualified array_id
            # resolved to the correct within-source field (not pos_0's 32x32).
            max_stop = [max(b.stop[i] for b in bounds) for i in range(2)]
            assert max_stop == [64, 64]
        finally:
            server.shutdown()

    def test_single_tensor_source_still_works(self):
        """Single tensor sources should still work with new API."""
        tensor_specs = [
            ("single-tensor", (50, 50), "uint8"),
        ]
        adapter = MockMultifieldAdapter("single-source", tensor_specs)

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("single-source", adapter)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            sources = client.list_sources()
            assert len(sources["single-source"].tensors) == 1

            # Access the single tensor -- a bare source id resolves the sole tensor.
            arr = client.get_tensor("single-source")
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


class MockImage0Adapter(TensorAdapter):
    """Single-tensor source whose tensor is named "Image:0".

    Models a single-scene aicsimageio file: every such file names its one
    tensor "Image:0", so two distinct sources share a *non-unique* bare
    array_id. It also mirrors aicsimageio's two array_id forms (issue #45
    fault 2): list_tensor_descriptors() advertises the bare "Image:0", while
    the tensor-level get_tensor_descriptor() carries the source-qualified
    "source_id/Image:0".
    """

    @classmethod
    def claim(cls, path, visited_identities):
        return None

    @classmethod
    def create_from_config(cls, source, credentials_config=None):
        raise NotImplementedError("MockImage0Adapter is for testing only")

    def __init__(self, source_id, shape, physical_scale, physical_unit):
        self.source_id = source_id
        self._tensor_name = "Image:0"  # array_id property -> source_id/Image:0
        self._shape = shape
        self._phys_scale = physical_scale
        self._phys_unit = physical_unit
        self._source_url = f"mock://{source_id}"
        self._source_type = "mock-aics"

    def get_tensor_descriptor(self) -> TensorDescriptor:
        # Tensor-level: source-qualified array_id, like aicsimageio.
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=["z", "y", "x"],
            shape=list(self._shape),
            chunk_shape=list(self._shape),
            dtype="uint8",
        )

    def list_tensor_descriptors(self):
        # Source-level listing: bare array_id, like aicsimageio.
        return [
            TensorDescriptor(
                array_id=self._tensor_name,
                dim_labels=["z", "y", "x"],
                shape=list(self._shape),
                chunk_shape=list(self._shape),
                dtype="uint8",
            )
        ]

    def get_metadata(self) -> dict:
        return {}

    def _physical_scale(self):
        return list(self._phys_scale), list(self._phys_unit)

    def get_data(self, bounds) -> np.ndarray:
        super().get_data(bounds)
        shape = tuple(
            int(stop - start)
            for start, stop in zip(bounds.start, bounds.stop, strict=True)
        )
        return np.zeros(shape, dtype="uint8")


class TestFieldWithinSource:
    """_field_within_source reduces a request tensor_id to the within-source
    field the adapters key on (identity policy, server-free)."""

    def test_strips_source_qualified_prefix(self):
        assert (
            TensorFlightServer._field_within_source("src", "src/Image:0") == "Image:0"
        )

    def test_hierarchical_field_keeps_inner_slashes(self):
        # HCS: array_id = source/well/field; split only on the first '/'.
        assert (
            TensorFlightServer._field_within_source("plate", "plate/A01/0") == "A01/0"
        )

    def test_bare_field_passes_through(self):
        # Back-compat: a client sending just the field.
        assert TensorFlightServer._field_within_source("src", "Image:0") == "Image:0"

    def test_source_id_itself_resolves_to_default(self):
        assert TensorFlightServer._field_within_source("src", "src") is None

    def test_empty_resolves_to_default(self):
        assert TensorFlightServer._field_within_source("src", "") is None

    def test_field_equal_to_source_id_is_preserved_not_defaulted(self):
        # Precedence edge: a real field that happens to equal the source_id
        # (array_id "src/src" -> field "src") must survive, because the
        # ==source_id -> default check runs BEFORE the prefix strip. If they were
        # reordered, "src/src" would strip to "src" and then wrongly default.
        assert TensorFlightServer._field_within_source("src", "src/src") == "src"


class TestStripSourcePrefix:
    """strip_source_prefix: the pure, policy-free reduction shared by the server
    chokepoint and the adapters' _within_source_field (biopb/biopb#277 item F)."""

    def test_strips_prefix(self):
        assert strip_source_prefix("src", "src/Image:0") == "Image:0"

    def test_splits_only_first_slash(self):
        assert strip_source_prefix("plate", "plate/A01/0") == "A01/0"

    def test_bare_field_unchanged(self):
        assert strip_source_prefix("src", "Image:0") == "Image:0"

    def test_source_id_itself_unchanged_no_default_policy(self):
        # Unlike the server helper, the pure strip invents no None: a bare
        # source_id is returned as-is (the "default tensor" call is the caller's).
        assert strip_source_prefix("src", "src") == "src"

    def test_none_and_empty_pass_through(self):
        assert strip_source_prefix("src", None) is None
        assert strip_source_prefix("src", "") == ""


class TestDescriptorCacheCollision:
    """Regression for #45: cross-source descriptor cache collisions.

    Two single-scene-aicsimageio-like sources share the bare tensor id
    "Image:0". A descriptor cache keyed by the bare array_id collapses them to
    one entry, so get_physical_scale / get_source silently return another
    source's descriptor (wrong shape, dims, physical scale). The cache must be
    keyed per (source_id, array_id).
    """

    def test_same_bare_array_id_across_sources_returns_own_descriptor(self):
        srcA = MockImage0Adapter(
            "aics_aaa", (10, 256, 256), [2.0, 0.5, 0.5], ["um", "um", "um"]
        )
        srcB = MockImage0Adapter(
            "aics_bbb", (181, 1024, 1024), [4.0, 0.1, 0.1], ["um", "um", "um"]
        )

        server = TensorFlightServer("grpc://localhost:0")
        server.register_source("aics_aaa", srcA)
        server.register_source("aics_bbb", srcB)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        try:
            client = TensorFlightClient(f"grpc://localhost:{server.port}")

            # get_descriptor returns each source's own shape, never the other's
            # (structural probe -> no metadata catalog needed on this bare server).
            assert client.get_descriptor("aics_aaa", with_metadata=False).shape == [
                10,
                256,
                256,
            ]
            assert client.get_descriptor("aics_bbb", with_metadata=False).shape == [
                181,
                1024,
                1024,
            ]

            # get_physical_scale reads from the descriptor cache keyed on the
            # bare tensor id "Image:0" -- the exact collision in #45. Each must
            # return its OWN scale. With a bare-only key the second source reads
            # the first's cached entry and returns the wrong scale.
            scale_a, unit_a = client.get_physical_scale("aics_aaa/Image:0")
            assert scale_a == [2.0, 0.5, 0.5]
            assert unit_a == ["um", "um", "um"]

            scale_b, unit_b = client.get_physical_scale("aics_bbb/Image:0")
            assert scale_b == [4.0, 0.1, 0.1]
            assert unit_b == ["um", "um", "um"]

            # Both sources coexist in the cache under distinct composite keys.
            assert client._descriptor_key("aics_aaa", "Image:0") in client._descriptors
            assert client._descriptor_key("aics_bbb", "Image:0") in client._descriptors

            client.close()
        finally:
            server.shutdown()
