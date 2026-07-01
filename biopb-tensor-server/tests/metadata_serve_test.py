"""Serve-path metadata is read from the DuckDB catalog, not recomputed on the
adapter (biopb/biopb#253 core).

GetFlightInfo(with_metadata) reads sources.metadata_json (stored once at
registration) and re-wraps it, falling back to adapter.get_metadata() only when
there is no DB or the row is NULL.
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
    from biopb_tensor_server.metadata_db import MetadataDatabase

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
def test_serve_falls_back_to_adapter_when_row_is_null(simple_zarr_array):
    """A NULL catalog row (empty metadata at registration) falls back to the
    adapter -- covers the unresolved 'resolved-but-row-still-NULL' edge."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer
    from biopb_tensor_server.metadata_db import MetadataDatabase

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
        wrapped = json.loads(desc.metadata_json)
        assert wrapped["metadata"] == {"late": "meta"}  # from the adapter fallback
        assert adapter.get_metadata_calls >= 1
        client.close()
    finally:
        server.shutdown()


@pytest.mark.skipif(not _zarr_available(), reason="zarr not available")
def test_serve_no_metadata_db_uses_adapter(simple_zarr_array):
    """An embedded/test server without a metadata DB serves metadata via the
    adapter (the fallback path)."""
    import zarr
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server import TensorFlightServer

    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    _MetaZarr = _meta_zarr_cls()

    server = TensorFlightServer("grpc://localhost:0")  # no metadata_db
    adapter = _MetaZarr(arr, "img", ["y", "x"], meta={"ome": {"channel": "GFP"}})
    server.register_source("img", adapter)
    _serve(server)
    try:
        client = TensorFlightClient(f"grpc://localhost:{server.port}")
        desc = client.get_descriptor("img")
        wrapped = json.loads(desc.metadata_json)
        assert wrapped["metadata"] == {"ome": {"channel": "GFP"}}
        assert adapter.get_metadata_calls >= 1
        client.close()
    finally:
        server.shutdown()
