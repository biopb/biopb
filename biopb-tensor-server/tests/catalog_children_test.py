"""``catalog_children``: one level of the catalog tree, over a real server and the
SDK's method for it."""

import threading
import time
from types import SimpleNamespace

import pyarrow.flight as flight
import pytest
from biopb.tensor._session import CatalogClient
from biopb.tensor.client import TensorFlightClient
from biopb_tensor_server import TensorFlightServer
from biopb_tensor_server.serving.metadata_db import CatalogRecord, MetadataDatabase

from tests.test_metadata_db import MockAdapter

URLS = {
    "a": "file:///lab/x/a.zarr",
    "b": "file:///lab/x/b.zarr",
    "c": "file:///lab/y/z/c.zarr",
    "top": "file:///top.zarr",
    "g": "grpc://host:1/remote/g.zarr",
}


def _serve_db(db, token=None):
    server = TensorFlightServer("localhost:0", metadata_db=db, token=token)
    threading.Thread(target=server.serve, daemon=True).start()
    time.sleep(0.5)
    return server


def _serve(urls, token=None):
    db = MetadataDatabase()
    for source_id, url in urls.items():
        db.sync_source_added(
            source_id, MockAdapter(source_id, url, "zarr", [4, 4], "uint8")
        )
    return _serve_db(db, token)


@pytest.fixture(scope="module")
def client():
    server = _serve(URLS)
    try:
        yield TensorFlightClient(f"grpc://localhost:{server.port}")
    finally:
        server.shutdown()


class TestCatalogChildren:
    def test_the_top_lists_folders_then_sources(self, client):
        level = client.catalog_children()
        assert level["key"] == ""
        assert [f["name"] for f in level["folders"]] == ["grpc://host:1/remote", "lab"]
        assert level["sources"] == [{"name": "top.zarr", "source_id": "top"}]
        assert level["truncated"] is False

    def test_a_folder_lists_what_is_directly_under_it(self, client):
        level = client.catalog_children("lab")
        assert level["folders"] == [
            {"name": "x", "key": "lab/x"},
            {"name": "y/z", "key": "lab/y/z"},
        ]
        assert level["sources"] == []
        assert client.catalog_children("lab/x")["sources"] == [
            {"name": "a.zarr", "source_id": "a"},
            {"name": "b.zarr", "source_id": "b"},
        ]

    def test_a_dropped_root_arrives_marked_over_the_wire(self):
        # Through the action and the SDK, not just the database: the field is an
        # extra key of the JSON level, which both hand back untouched.
        db = MetadataDatabase()
        db.ensure_root("d", "dnd://scratch")
        db.sync_source_added(
            "b",
            MockAdapter("b", "x", "zarr", [4, 4], "uint8"),
            CatalogRecord(None, {}, "d", "b.zarr"),
        )
        server = _serve_db(db)
        try:
            level = TensorFlightClient(
                f"grpc://localhost:{server.port}"
            ).catalog_children()
        finally:
            server.shutdown()
        assert level["folders"] == [
            {"name": "scratch", "key": "scratch", "dropped": True}
        ]

    def test_an_endpoint_is_one_folder(self, client):
        (folder,) = [
            f for f in client.catalog_children()["folders"] if "host" in f["name"]
        ]
        assert folder["key"] == "grpc://host:1/remote"
        assert client.catalog_children(folder["key"])["sources"] == [
            {"name": "g.zarr", "source_id": "g"}
        ]

    def test_a_key_with_nothing_under_it_is_an_empty_level(self, client):
        assert client.catalog_children("nowhere") == {
            "key": "nowhere",
            "folders": [],
            "sources": [],
            "truncated": False,
        }

    def test_the_limit_cuts_a_level_and_says_so(self, client):
        level = client.catalog_children("lab/x", limit=1)
        assert [s["source_id"] for s in level["sources"]] == ["a"]
        assert level["truncated"] is True

    def test_folders_share_the_limit_and_come_first(self, client):
        level = client.catalog_children("", limit=2)
        assert len(level["folders"]) == 2
        assert level["sources"] == []
        assert level["truncated"] is True

    @pytest.mark.parametrize(
        "body",
        [
            b"[]",
            b"not json",
            b'{"key": 5}',
            b'{"limit": "ten"}',
            b'{"limit": true}',
            b'{"limit": 0}',
            b'{"key": "\\ud800"}',
        ],
    )
    def test_a_malformed_request_is_refused(self, client, body):
        action = flight.Action("catalog_children", body)
        with pytest.raises(flight.FlightServerError, match="catalog_children"):
            list(client._client.do_action(action))

    def test_a_limit_below_one_is_refused_before_the_call(self, client):
        with pytest.raises(ValueError, match="limit"):
            client.catalog_children("", 0)

    def test_it_is_advertised(self, client):
        names = {a.type for a in client._client.list_actions()}
        assert "catalog_children" in names


def test_the_server_token_gates_it():
    server = _serve(URLS, token="secret")
    try:
        loc = f"grpc://localhost:{server.port}"
        with pytest.raises(flight.FlightUnauthenticatedError):
            TensorFlightClient(loc).catalog_children()
        level = TensorFlightClient(loc, token="secret").catalog_children()
        assert level["sources"][0]["source_id"] == "top"
    finally:
        server.shutdown()


def test_a_server_that_predates_it_is_named():
    class Old:
        def do_action(self, action, options=None):
            raise flight.FlightServerError("Unknown action: catalog_children")

    state = SimpleNamespace(client=Old(), call_options=None)
    with pytest.raises(RuntimeError, match="catalog_children"):
        CatalogClient(state).children()
