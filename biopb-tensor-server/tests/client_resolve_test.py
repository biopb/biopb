"""Unit tests for the SDK ``TensorFlightClient.resolve()`` and the directive
error that steers callers to it (cloud-storage phase 2, agent-facing API).

These exercise the resolve trigger and its caches without a live Flight server:
``_fetch_tensor_descriptor`` (the per-tensor GetFlightInfo) and ``list_sources``
are stubbed, and the client is built with ``object.__new__`` so no connection is
opened. The end-to-end resolve-on-serve path is covered by the server suite.
"""

import pytest

from biopb.tensor.client import TensorFlightClient
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor


def _bare_client():
    """A client instance with no network: only the caches resolve() touches."""
    client = object.__new__(TensorFlightClient)
    client._sources = {}
    client._descriptors = {}
    return client


def _resolved_tensor(array_id):
    return TensorDescriptor(array_id=array_id, shape=[4, 4], dtype="<f4")


class TestResolve:
    def test_returns_full_source_descriptor_from_relist(self, monkeypatch):
        # resolve() triggers a per-tensor fetch, then re-lists to enumerate ALL
        # fields (get_descriptor alone returns only the default tensor).
        client = _bare_client()
        fetched = []
        monkeypatch.setattr(
            client,
            "_fetch_tensor_descriptor",
            lambda sid, tid=None: fetched.append(sid) or _resolved_tensor(sid),
        )
        full = DataSourceDescriptor(
            source_id="cloud_x",
            tensors=[_resolved_tensor("cloud_x/f0"), _resolved_tensor("cloud_x/f1")],
        )
        monkeypatch.setattr(client, "list_sources", lambda: {"cloud_x": full})

        out = client.resolve("cloud_x")

        assert fetched == ["cloud_x"]  # resolve-on-serve was triggered
        assert out is full
        assert len(out.tensors) == 2  # the complete field set, not just default

    def test_truncated_catalog_falls_back_to_single_tensor(self, monkeypatch):
        # When the source is beyond the list_sources() cap, resolve() still yields
        # a usable descriptor (the resolved default tensor) and seeds the cache so
        # a following get_tensor() doesn't re-fetch.
        client = _bare_client()
        td = _resolved_tensor("cloud_x")
        monkeypatch.setattr(
            client, "_fetch_tensor_descriptor", lambda sid, tid=None: td
        )
        monkeypatch.setattr(client, "list_sources", lambda: {})  # truncated

        out = client.resolve("cloud_x")

        assert out.source_id == "cloud_x"
        assert list(out.tensors) == [td]
        assert client._sources["cloud_x"] is out  # cache seeded for reuse


class TestUnresolvedDirectiveError:
    def test_get_tensor_context_points_at_resolve(self):
        # A bare get_tensor() on an unresolved (empty-tensors) source must fail
        # with a directive message naming client.resolve(), not a bare "no tensors".
        client = _bare_client()
        client._sources = {
            "cloud_x": DataSourceDescriptor(source_id="cloud_x")  # no tensors
        }
        with pytest.raises(ValueError) as exc:
            client._get_tensor_context("cloud_x")
        msg = str(exc.value)
        assert "unresolved" in msg
        assert "client.resolve('cloud_x')" in msg


class TestPhysicalScaleUnresolvedGuard:
    """F1: get_physical_scale must not silently recall a whole cloud file."""

    def test_unresolved_source_raises_instead_of_recalling(self, monkeypatch):
        client = _bare_client()
        client._sources = {
            "cloud_x": DataSourceDescriptor(source_id="cloud_x")  # unresolved
        }
        recalled = []
        monkeypatch.setattr(
            client,
            "_fetch_tensor_descriptor",
            lambda *a, **k: recalled.append(a) or _resolved_tensor("cloud_x"),
        )
        with pytest.raises(ValueError, match="client.resolve"):
            client.get_physical_scale("cloud_x")
        assert recalled == []  # no GetFlightInfo / download was triggered

    def test_resolved_source_still_fetches_scale(self, monkeypatch):
        # A resolved source is unaffected: the cheap one-shot fetch still runs.
        client = _bare_client()
        td = TensorDescriptor(
            array_id="r",
            shape=[2, 2],
            dtype="<f4",
            physical_scale=[0.5, 0.25],
            physical_unit=["um", "um"],
        )
        client._sources = {
            "r": DataSourceDescriptor(source_id="r", tensors=[td])
        }
        monkeypatch.setattr(
            client, "_fetch_tensor_descriptor", lambda *a, **k: td
        )
        scale, unit = client.get_physical_scale("r")
        assert scale == [0.5, 0.25]
        assert unit == ["um", "um"]
