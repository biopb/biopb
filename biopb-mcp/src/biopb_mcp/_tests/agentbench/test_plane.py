"""The run-scoped data plane, and the property the whole arrangement rests on.

Split by cost, and the split matters. The hermetic half runs with the ordinary
suite: the plane must be **writable** and must use **its own** cache and write
directories, and both are one-line mistakes that would only surface as a failed
paid run — or, worse, as a benchmark quietly reading and writing the
developer's own catalog.

The other half spawns a real server, so it is marked `bench` and never runs in
CI. It is worth its seconds because it is the only place the isolation
argument is *checked* rather than argued: the id the server answers with is
minted from its own randomness and is unrelated to the name the harness
uploaded under, which is never sent anywhere an agent can read.
`test_a_reupload_under_the_same_name_is_refused` checks the server's half: a
name is taken while its store is on disk, so even the name would not let you
swap the data out from under the id. The fingerprint check in `bench/_engine`
stays as the mechanism that would notice if either ever gave.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from . import _plane

# --- hermetic ---------------------------------------------------------------


def test_the_plane_gets_its_own_cache_and_write_directories(tmp_path):
    """Never the developer's. `_session._write_config` applies the same
    discipline to the MCP config tree, and for the same reason: a benchmark
    that wrote into a real catalog would be discovered by its side effects."""
    config = json.loads(_plane._write_plane_config(tmp_path).read_text())

    assert config["sources"] == [], "everything it serves arrives by upload"
    for key in (config["cache"]["file_cache_dir"], config["server"]["write_dir"]):
        assert Path(key).is_dir(), f"{key} was declared but not created"
        assert Path(key).is_relative_to(tmp_path)


def test_the_plane_is_writable(tmp_path):
    """The server defaults to read-only, and a read-only plane would fail every
    skill step that uploads a result (`drift-correction` step 7,
    `stitch-tiles` step 7) rather than measure it.

    Asserted **under `server`**, where the loader reads it from
    (`core/config.py`, ``data.get("server", {})``). Spelled at the top level it
    is ignored, and a test that checked it there would be checking the
    harness's own dict.
    """
    config = json.loads(_plane._write_plane_config(tmp_path).read_text())
    assert config["server"]["writable"] is True
    assert config["server"]["write_dir"], "an upload needs somewhere to put a store"


def test_a_machine_without_the_server_says_so_rather_than_failing(monkeypatch):
    """biopb-mcp cannot depend on biopb-tensor-server — it is never on PyPI —
    so a machine without it must report `tensor` cases as unavailable, the same
    discipline as a missing API key."""
    monkeypatch.setattr(_plane.importlib.util, "find_spec", lambda name: None)
    assert "not installed" in _plane.plane_unavailable()


def test_nothing_starts_until_something_asks():
    """Conditional, so a run whose cases all present `array` behaves exactly as
    it did before any of this existed."""
    assert _plane.running_plane() is None


# --- against a real server --------------------------------------------------


@pytest.fixture(scope="module")
def plane():
    if why := _plane.plane_unavailable():
        pytest.skip(why)
    started = _plane.start_plane()
    try:
        yield started
    finally:
        started.stop()


@pytest.mark.bench
def test_an_uploaded_fixture_comes_back_as_the_bytes_that_went_in(plane):
    volume = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    array_id = plane.upload("a-case-stack", volume, chunks=(2, 4, 4))

    got = np.asarray(plane.client.get_tensor(array_id))
    assert got.shape == volume.shape
    assert np.array_equal(got, volume)


@pytest.mark.bench
def test_the_chunking_the_case_asked_for_is_the_chunking_it_gets(plane):
    """Where laziness is the point the chunking *is* the thing under test: a
    route that only fails at a chunk boundary is not exercised by a
    single-chunk array."""
    array_id = plane.upload(
        "chunked", np.zeros((4, 8, 8), np.float32), chunks=(1, 4, 8)
    )
    assert tuple(plane.client.get_descriptor(array_id).chunk_shape) == (1, 4, 8)


@pytest.mark.bench
def test_an_agent_holding_the_id_cannot_name_the_source(plane):
    """The isolation property, stated as what is *absent*: the id is minted
    from the server's own randomness, and nothing the agent can see carries the
    name — not the descriptor, not the source url, not the layer."""
    array_id = plane.upload("secretly-named", np.zeros((2, 2), np.float32))
    descriptor = plane.client.get_descriptor(array_id)

    assert plane.secret not in array_id
    assert plane.secret not in str(descriptor)
    assert "secretly-named" not in str(descriptor)


@pytest.mark.bench
def test_a_reupload_under_the_same_name_is_refused(plane):
    """A name is taken for as long as its store is on disk: a second upload
    under it is refused rather than replacing the fixture in place, and what
    the id serves is unchanged."""
    import pyarrow.flight as flight

    original = np.zeros((4, 4), np.float32)
    array_id = plane.upload("overwrite-me", original)
    before = plane.fingerprint(array_id)

    with pytest.raises(flight.FlightServerError, match="already exists"):
        plane.upload("overwrite-me", np.ones((4, 4), np.float32))
    assert plane.fingerprint(array_id) == before


@pytest.mark.bench
def test_a_discarded_fixture_stops_being_readable(plane):
    """What the id *cannot* protect against, pinned so it stays visible.

    A published tensor can be discarded, which takes its store with it, so an
    agent holding the id can make the fixture go away even though it cannot
    name the source or swap the bytes. Hence `_engine.contaminated` treating
    unreadable as contaminated: "the bytes changed" and "the bytes are gone"
    are one flag.
    """
    import pyarrow.flight as flight

    array_id = plane.upload("droppable", np.zeros((4, 4), np.float32))
    assert plane.fingerprint(array_id)

    plane.client.set_upload_status(array_id, "DISCARDED")

    with pytest.raises(flight.FlightError):
        plane.fingerprint(array_id)
