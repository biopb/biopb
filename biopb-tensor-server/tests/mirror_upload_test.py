"""An upload onto a mirrored source goes to the upstream that owns the source.

Two real servers in one process: an upstream with a ``write_dir`` serving ``img``,
and a mirror of it serving ``lab__img``. The producer talks to the mirror only.
"""

import threading
import time

import numpy as np
import pyarrow.flight as flight
import pytest
import zarr
from biopb.tensor import TensorFlightClient
from biopb_tensor_server import ZarrAdapter
from biopb_tensor_server.adapters.remote_tensor import RemoteTensorAdapter
from biopb_tensor_server.adapters.scratch import SCRATCH_SOURCE_ID

from tests import catalog_server, register_and_catalog


def _serve(server):
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(1)
    return server


@pytest.fixture
def pair(simple_zarr_array, tmp_path):
    """``(upstream, mirror, client)``; the client is bound to the mirror."""
    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = catalog_server(
        "localhost:0", writable=True, write_dir=tmp_path / "upstream"
    )
    register_and_catalog(upstream, "img", ZarrAdapter(arr, "img", ["y", "x"]))
    _serve(upstream)
    mirror = catalog_server("localhost:0", writable=True)
    register_and_catalog(
        mirror,
        "lab__img",
        RemoteTensorAdapter("lab__img", f"grpc://localhost:{upstream.port}", "img"),
    )
    _serve(mirror)
    client = TensorFlightClient(f"grpc://localhost:{mirror.port}")
    try:
        yield upstream, mirror, client
    finally:
        client.close()
        mirror.shutdown()
        upstream.shutdown()


def _declare(client, name="result", scheme="cache", source="lab__img"):
    return client.setup_array_upload(
        f"{scheme}://{source}/@fields/{name}",
        np.zeros((8, 8), dtype=np.uint8),
        chunk_shape=(4, 4),
        dim_labels=["y", "x"],
    )


def test_the_upload_lands_on_the_upstream_and_the_mirror_keeps_no_record(pair):
    upstream, mirror, client = pair
    desc = _declare(client)
    # Answered under the mirror's id, though the upstream minted it.
    assert desc.array_id == "lab__img/@fields/result"

    assert "@fields/result" in upstream.sources.attachments("img")
    assert mirror.sources.attachments("lab__img") == {}


def test_chunks_and_publish_go_through_and_the_mirror_reads_them_back(pair):
    upstream, _, client = pair
    desc = _declare(client)
    data = np.arange(64, dtype=np.uint8).reshape(8, 8)

    status = client.upload_array(desc, data)

    assert status["state"] == "READY"
    assert client.get_upload_status(desc.array_id)["state"] == "READY"
    # On the upstream's own read path ...
    upstream_client = TensorFlightClient(f"grpc://localhost:{upstream.port}")
    try:
        up = upstream_client.get_tensor("img/@fields/result").compute()
    finally:
        upstream_client.close()
    np.testing.assert_array_equal(up, data)
    # ... and through the mirror, which fetches it from there.
    np.testing.assert_array_equal(client.get_tensor(desc.array_id).compute(), data)


def test_progress_is_the_upstreams(pair):
    _, _, client = pair
    desc = _declare(client)

    status = client.get_upload_status(desc.array_id)

    assert status["state"] == "PENDING"
    assert status["expected_chunks"] == 4
    assert status["uploaded_chunks"] == 0


def test_a_discard_is_forwarded(pair):
    upstream, _, client = pair
    desc = _declare(client)

    client.set_upload_status(desc, "DISCARDED", "not needed")

    assert client.get_upload_status(desc.array_id)["state"] == "DISCARDED"
    upload = upstream.sources.attachments("img")["@fields/result"]
    assert upload.upload_status()["state"] == "DISCARDED"


def test_a_label_set_takes_the_same_path(pair):
    upstream, _, client = pair
    labels = np.zeros((128, 128), dtype=np.uint16)
    labels[10:20, 10:20] = 1
    desc = client.setup_array_upload(
        "zarr://lab__img/@labels/nuclei",
        labels,
        chunk_shape=(64, 64),
        dim_labels=["y", "x"],
    )
    assert desc.array_id == "lab__img/@labels/nuclei"

    assert client.upload_array(desc, labels)["state"] == "READY"

    assert "@labels/nuclei" in upstream.sources.attachments("img")


def test_a_read_only_upstream_refuses_in_its_own_words(simple_zarr_array):
    zarr_path, _, _ = simple_zarr_array
    arr = zarr.open_array(zarr_path, mode="r")
    upstream = catalog_server("localhost:0")  # not writable
    register_and_catalog(upstream, "img", ZarrAdapter(arr, "img", ["y", "x"]))
    _serve(upstream)
    mirror = catalog_server("localhost:0", writable=True)
    register_and_catalog(
        mirror,
        "lab__img",
        RemoteTensorAdapter("lab__img", f"grpc://localhost:{upstream.port}", "img"),
    )
    _serve(mirror)
    client = TensorFlightClient(f"grpc://localhost:{mirror.port}")
    try:
        with pytest.raises(flight.FlightError, match="not in write mode"):
            _declare(client)
    finally:
        client.close()
        mirror.shutdown()
        upstream.shutdown()


def test_an_upstream_scratch_source_is_not_uploadable_through_a_mirror(tmp_path):
    mirror = catalog_server("localhost:0", writable=True, write_dir=tmp_path)
    # A mirror never holds the upstream's scratch (mirrorable_upstream_id), but a
    # hand-built one must not forward to it either.
    mirror.register_source(
        "lab__scratch",
        RemoteTensorAdapter("lab__scratch", "grpc://localhost:1", SCRATCH_SOURCE_ID),
    )
    from biopb.tensor.descriptor_pb2 import TensorDescriptor

    with pytest.raises(flight.FlightServerError, match="scratch"):
        mirror.uploads.add_tensor(
            TensorDescriptor(
                array_id="cache://lab__scratch/@fields/x",
                shape=[8, 8],
                dtype="uint8",
                chunk_shape=[4, 4],
                dim_labels=["y", "x"],
            )
        )
