"""Serve-path metadata is read from the DuckDB catalog, not recomputed on the
adapter (biopb/biopb#253 core).

One scheme: ``GetFlightInfo(with_metadata)`` fills ``metadata_json`` from either
(1) the tensor adapter's ``get_tensor_metadata()`` -- per-tensor metadata the
source-level row cannot represent (HCS field OME, EMD per-signal) -- or, when that
is ``None``, (2) the source's ``sources.metadata_json`` catalog row (stored once at
registration). The catalog is mandatory: there is **no** adapter fallback, and a
DB-less server raises rather than recompute.
"""

import importlib.util
import json
import threading
import time

import pytest


def _zarr_available() -> bool:
    return importlib.util.find_spec("zarr") is not None


def _serve(server):
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    time.sleep(1)
    return t


def _meta_zarr_cls():
    """A ZarrAdapter whose get_metadata() is controllable + call-counted."""
    from biopb_tensor_server import ZarrAdapter

    class _MetaZarr(ZarrAdapter):
        def __init__(self, *a, meta, **k):
            super().__init__(*a, **k)
            self._meta = meta
            self.get_metadata_calls = 0

        def get_metadata(self):
            self.get_metadata_calls += 1
            return self._meta

    return _MetaZarr


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_serve_reads_metadata_from_catalog_without_recompute(simple_zarr_array):
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    _MetaZarr = _meta_zarr_cls()

    db = MetadataDatabase()
    server = TensorFlightServer("grpc://localhost:0", metadata_db=db)
    adapter = _MetaZarr(arr, "img", ["y", "x"], meta={"ome": {"channel": "DAPI"}})
    server.register_source("img", adapter)
    db.sync_source_added("img", adapter)  # stores the metadata in the catalog

    adapter.get_metadata_calls = 0  # serve must NOT recompute
    _serve(server)
    try:
        client = TensorFlightClient(f"grpc://localhost:{server.port}")
        desc = client.get_descriptor("img")  # GetFlightInfo(with_metadata)
        wrapped = json.loads(desc.metadata_json)
        assert wrapped["metadata"] == {"ome": {"channel": "DAPI"}}
        # single-wrapped (the raw dict, not another envelope)
        assert "metadata" not in wrapped["metadata"]
        assert adapter.get_metadata_calls == 0  # read from DuckDB, not the adapter
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_serve_null_row_yields_empty_metadata_no_adapter_recompute(simple_zarr_array):
    """A NULL catalog row (empty metadata at registration) yields empty
    metadata_json -- no fallback to the adapter. The catalog is authoritative:
    the adapter is never consulted on the serve path, even for a NULL row."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    _MetaZarr = _meta_zarr_cls()

    db = MetadataDatabase()
    server = TensorFlightServer("grpc://localhost:0", metadata_db=db)
    adapter = _MetaZarr(arr, "img", ["y", "x"], meta={})  # empty -> NULL row
    server.register_source("img", adapter)
    db.sync_source_added("img", adapter)

    adapter._meta = {"late": "meta"}  # the adapter now has metadata the row lacks
    adapter.get_metadata_calls = 0
    _serve(server)
    try:
        client = TensorFlightClient(f"grpc://localhost:{server.port}")
        desc = client.get_descriptor("img")
        assert not desc.metadata_json  # empty row -> empty, NOT the adapter's late meta
        assert adapter.get_metadata_calls == 0  # never recomputed on the serve path
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_serve_without_metadata_db_raises(simple_zarr_array):
    """A metadata request against a DB-less server (the embedded image-base cache)
    fails closed -- there is no catalog to read and no adapter fallback."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from pyarrow import flight

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    _MetaZarr = _meta_zarr_cls()

    server = TensorFlightServer("grpc://localhost:0")  # no metadata_db
    adapter = _MetaZarr(arr, "img", ["y", "x"], meta={"ome": {"channel": "GFP"}})
    server.register_source("img", adapter)
    _serve(server)
    try:
        client = TensorFlightClient(f"grpc://localhost:{server.port}")
        # GetFlightInfo(with_metadata) fails closed -> FlightInternalError
        with pytest.raises(flight.FlightInternalError, match="no metadata catalog"):
            client.get_descriptor("img")
        assert adapter.get_metadata_calls == 0  # never recomputed
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_get_tensor_metadata_default_is_none():
    """The base tensor adapter defers to the source-level catalog row (returns
    None); only a per-tensor format (HCS field, EMD signal) overrides it."""
    from biopb_tensor_server import ZarrAdapter
    from biopb_tensor_server.core.base import TensorAdapter

    # class-level: the default is inherited straight from TensorAdapter
    assert ZarrAdapter.get_tensor_metadata is TensorAdapter.get_tensor_metadata


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_serve_merges_per_tensor_delta_over_catalog(simple_zarr_array):
    """get_tensor_metadata() is a cheap per-tensor *delta* merged over the cached
    source-level row (biopb/biopb#253): an HCS-plate-like source serves the plate
    row (from the catalog) plus each field's own metadata (from the adapter)."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")

    class _PerTensorZarr(ZarrAdapter):
        def __init__(self, *a, field_meta, **k):
            super().__init__(*a, **k)
            self._field_meta = field_meta

        def get_metadata(self):
            return {"plate": {"rows": ["A"]}}  # source-level (cached in catalog)

        def get_tensor_metadata(self):
            return self._field_meta  # per-tensor delta, merged over the row

    db = MetadataDatabase()
    server = TensorFlightServer("grpc://localhost:0", metadata_db=db)
    adapter = _PerTensorZarr(arr, "plate", ["y", "x"], field_meta={"ome": "field"})
    server.register_source("plate", adapter)
    db.sync_source_added("plate", adapter)  # catalog holds the plate metadata

    _serve(server)
    try:
        client = TensorFlightClient(f"grpc://localhost:{server.port}")
        desc = client.get_descriptor("plate")
        wrapped = json.loads(desc.metadata_json)
        # cached plate row + per-tensor delta, merged
        assert wrapped["metadata"] == {"plate": {"rows": ["A"]}, "ome": "field"}
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_serve_no_delta_serves_catalog_row(simple_zarr_array):
    """get_tensor_metadata() None (the default) means no delta: the tensor is
    fully described by the cached source-level catalog row."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer, ZarrAdapter
    from biopb_tensor_server.core.metadata_db import MetadataDatabase

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")

    class _RowOnlyZarr(ZarrAdapter):
        def get_metadata(self):
            return {"plate": {"rows": ["A"]}}

        # get_tensor_metadata inherits the None default -> no delta

    db = MetadataDatabase()
    server = TensorFlightServer("grpc://localhost:0", metadata_db=db)
    adapter = _RowOnlyZarr(arr, "plate", ["y", "x"])
    server.register_source("plate", adapter)
    db.sync_source_added("plate", adapter)

    _serve(server)
    try:
        client = TensorFlightClient(f"grpc://localhost:{server.port}")
        desc = client.get_descriptor("plate")
        wrapped = json.loads(desc.metadata_json)
        assert wrapped["metadata"] == {"plate": {"rows": ["A"]}}  # row, no delta
        client.close()
    finally:
        server.shutdown()
