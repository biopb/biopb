"""Can a real session be brought up, driven, and reaped — with no model at all?

This is the floor the interaction layer stands on. §5 is the least isolable
tier in the suite: a red run's cause space includes the skill body, the model,
the tool schemas, the kernel, Qt, dask and the fixture. These tests exist so
that when the stack is the problem, *they* fail — separately, deterministically,
and before a single token is spent.

They also pin the three environment facts the harness forces rather than
inherits, each of which silently changes what a run tests: a real napari viewer
rather than the headless fallback, no tensor plane so the fixture reaches the
agent as a layer and nothing else, and a config tree of our own so the catalog
under test is the shipped one.

Marked `interaction` and deselected by default. Slow (a kernel, napari and dask
per test) but free — no key, no network beyond loopback.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import _session
from ._session import SessionUnavailable, live_session

pytestmark = pytest.mark.interaction


@pytest.fixture(scope="module")
def session():
    """One session for the module. Bring-up is the expensive part and none of
    these tests dirty it in a way the next would notice."""
    if reason := _session.why_unavailable():
        pytest.skip(reason)
    try:
        with live_session() as live:
            yield live
    except SessionUnavailable as exc:
        pytest.skip(str(exc))


def test_the_nine_tools_are_there_with_their_schemas(session):
    """What the agent is handed is the server's own advertisement — the point
    of driving real MCP rather than a stand-in. If this ever shrinks, an agent
    lost a capability the skill bodies assume."""
    names = {t.name for t in session.tools}
    assert names == {
        "find_skills",
        "start_kernel",
        "take_screenshot",
        "execute_code",
        "poll_job",
        "inspect_object",
        "interrupt_kernel",
        "restart_kernel",
        "server_status",
    }, f"tool surface changed: {sorted(names)}"

    execute = next(t for t in session.tools if t.name == "execute_code")
    assert "python_code" in (execute.input_schema.get("properties") or {}), (
        "execute_code's parameter name is part of what an agent must get right"
    )


def test_the_server_instructions_reach_the_client(session):
    """`instructions` is shipped prose the agent reads before anything else,
    and it is the field a generic mcp-proxy drops -- which is why the shim
    vendors its own bridge. A run that never saw it is not the real thing."""
    assert len(session.instructions) > 500, session.instructions[:200]
    assert "find_skills" in session.instructions


def test_the_skill_body_comes_from_the_shipped_catalog(session):
    """The property that makes this layer worth its cost. This reads
    `drift-correction` through the same `_skills.py` the runtime uses, so
    deleting or editing the file changes what a run is scored against — which is
    exactly what a hand-transcribed procedure could never do."""
    found = session.call("find_skills", query="stage drift in a time lapse")
    assert "drift-correction" in found.text, found.text[:400]

    body = session.read_resource("skill://drift-correction")
    assert 'reference="previous"' in body, "step 3 is missing from the body"
    assert "REF_CHANNEL" in body, "step 2's parameter is missing from the body"
    assert len(body) > 4000, f"body is only {len(body)} chars"


def test_the_viewer_is_real_and_not_the_headless_sentinel(session):
    """`display_mode: auto` degrades silently when no display is found, and
    `QT_QPA_PLATFORM=offscreen` is not enough either -- napari builds, then
    `add_image` dies in vispy's GL probe. Either way step 2 could not happen,
    so the harness refuses to run there and this is where that is stated."""
    real, detail = session.has_real_viewer()
    assert real, detail


def test_there_is_no_tensor_plane(session):
    """Forced, not incidental. A developer box often has a data plane up, and
    then the agent can wander into whatever catalog that machine holds -- so a
    finding might not reproduce anywhere else."""
    assert session.client_is_none(), (
        "client is live; this run would depend on the machine's own catalog"
    )


def test_an_array_makes_the_round_trip(session):
    """The scraping contract every interaction case depends on: the harness can
    put a fixture into the kernel and read a result back out. Via files, not
    literals -- a fixture movie is megabytes and tool output is truncated."""
    sent = np.random.default_rng(0).random((6, 2, 24, 24)).astype(np.float32)
    session.put_array("movie", sent)

    got = session.get_array("movie")
    assert got is not None and np.array_equal(got, sent)

    session.setup("corrected = movie * 2.0")
    assert np.allclose(session.get_array("corrected"), sent * 2.0)


def test_a_missing_result_reads_as_absent_not_as_an_error(session):
    """A run that left nothing behind is an ordinary outcome -- the agent gave
    up, or never got there. It must come back as "nothing to score", which is
    what `Outcome.passed` already refuses to treat as a pass."""
    assert session.get_array("no_such_name_at_all") is None


def test_the_fixture_can_be_a_napari_layer(session):
    """How a §5 fixture actually reaches the agent, with no tensor plane: as a
    layer on the viewer, which every skill's Parameters table accepts as a
    source. This is the call that fails without a GL context."""
    movie = np.random.default_rng(1).random((4, 2, 16, 16)).astype(np.float32)
    session.put_array("_fx", movie)
    out = session.setup(
        "viewer.add_image(_fx, name='timelapse', channel_axis=None)\n"
        "print('LAYERS', [lyr.name for lyr in viewer.layers])"
    )
    assert "timelapse" in out.text, out.text

    # And it survives as data the agent can read back off the layer.
    back = session.get_array("viewer.layers['timelapse'].data")
    assert back is not None and back.shape == movie.shape


def test_tool_calls_are_recorded_for_the_gate_spy(session):
    """Structural assertions (§5) ask whether a blocking question preceded the
    expensive call. That needs a record of what was called and when, and setup
    the harness itself did must not appear in it as agent behaviour."""
    before = len(session.calls)
    session.call("server_status")
    agent_calls = [c for c in session.calls[before:] if c[0] >= 0]
    assert [c[1] for c in agent_calls] == ["server_status"]

    session.setup("pass")
    assert session.calls[-1][0] == -1, "harness setup must not read as a turn"
