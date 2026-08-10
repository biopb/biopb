"""Can a real session be brought up, driven, and reaped — with no model at all?

This is the floor the whole layer stands on. It is the least isolable tier in
the suite: a red run's cause space includes the skill body, the model, the tool
schemas, the kernel, Qt, dask and the fixture. These tests exist so that when
the stack is the problem, *they* fail — separately, deterministically, and
before a single token is spent.

They also pin the three environment facts the harness forces rather than
inherits, each of which silently changes what a run tests: a real napari viewer
(on the user's display or the launcher's own Xvfb), no tensor plane so an
`array` fixture reaches the agent as a layer and nothing else, and a config tree
of our own so the catalog under test is the shipped one.

The last two tests are **per case**, and they are the ones that pay for
themselves: a fixture that will not reach the viewer, or a deliverable name the
harness cannot read back, wastes a paid conversation and looks exactly like a
model failing the task. They run over the same case list the run itself will
use, so `--bench-fixtures=synthetic` does not smoke-test the curated cases it is
not going to run.

Marked `bench` and deselected by default. Slow (a kernel, napari and dask per
session) but free — no key, no network beyond loopback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ..agentbench import _plane, _session
from ..agentbench._conversation import Trace, scrape
from ..agentbench._session import SessionUnavailable, live_session
from ._case import TENSOR_HANDLE
from ._engine import load_fixture, uploaded_ids
from .cases import drift_correction

pytestmark = pytest.mark.bench


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


@pytest.fixture
def bench_case(request):
    """One case this run would pay for. Parametrized in `conftest.py`."""
    ok, why = request.param.available()
    if not ok:
        pytest.skip(f"fixture: {why}")
    return request.param


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


def test_the_agent_can_reach_a_skill_body_and_not_only_the_harness(session):
    """The gap the test above cannot see, because it uses the harness's own
    accessor.

    `read_resource` is a method on `LiveSession`; for a long time it was *only*
    that. The agent is driven over chat-completions and is handed `tools`, so a
    resource — which is not a tool — had no verb behind it. `find_skills`
    returned a `uri` and the handshake said to read it, and nothing could.

    Measured consequence: a `skill+silent` arm that used `pystackreg` purely
    because `checklist:` named it, having never read the procedure. So this
    asserts the body arrives through `call`, the same door the model uses.
    """
    names = {t.name for t in session.agent_tools}
    assert {"read_resource", "list_resources"} <= names, sorted(names)
    assert {t.name for t in session.tools} < names, (
        "agent_tools must extend the server's advertisement, not replace it"
    )

    found = session.call("find_skills", query="stage drift in a time lapse")
    uri = next(
        part.strip('", ')
        for part in found.text.split()
        if part.strip('", ').startswith("skill://")
    )
    body = session.call("read_resource", uri=uri)
    assert not body.is_error, body.text
    assert 'reference="previous"' in body.text, (
        "the agent reached the resource but not the procedure inside it"
    )


def test_a_uri_that_does_not_resolve_is_an_error_result_not_a_crash(session):
    """An agent has to be able to read a bad uri and try something else. Raising
    out of `call` would end the run instead."""
    out = session.call("read_resource", uri="")
    assert out.is_error and "uri" in out.text

    unknown = session.call("read_resource", uri="skill://no-such-skill")
    assert "no-such-skill" in unknown.text or "catalog" in unknown.text.lower()


def test_the_ablation_survives_the_new_verb():
    """The one way this change could quietly void the benchmark.

    `noskill` withholds the catalog, not the filesystem. If `read_resource`
    reached the body around `load_catalog()`, an ablated arm could read
    `skill://<id>` straight back and the 2x2 would be measuring nothing. It does
    not, and it is the server's own gate rather than one the harness re-states —
    but the cost of that being wrong is every skill number in the layer, so it
    is worth a session of its own.
    """
    if reason := _session.why_unavailable():
        pytest.skip(reason)
    try:
        with live_session(skills_enabled=False) as live:
            assert "read_resource" in {t.name for t in live.agent_tools}
            out = live.call("read_resource", uri="skill://drift-correction")
            assert 'reference="previous"' not in out.text, (
                "the ablation arm just read the skill it is supposed to lack"
            )
            assert "catalog" in out.text.lower(), out.text[:200]
    except SessionUnavailable as exc:
        pytest.skip(str(exc))


def test_the_kernel_runs_the_shipped_package_not_the_checkout(session):
    """The answer key is not reachable because it is not *there*.

    Running from a checkout puts `_tests/` — every case's `truth`, tolerances
    and persona — inside the installed package, one `os.path.dirname` from any
    agent that looks. The wheel excludes it, so the child imports that instead.
    Also the more faithful run: it is what a user has.
    """
    out = session.call(
        "execute_code",
        python_code=(
            "import biopb_mcp, importlib.util, os.path\n"
            "print('pkg:', biopb_mcp.__file__)\n"
            "print('tests:', importlib.util.find_spec('biopb_mcp._tests'))\n"
            # isdir, not load_catalog(): reading the catalog *from the kernel*
            # is indistinguishable from an ablated arm doing the same, and it
            # would leave that residue in the tripwire for the next test.
            "print('skills_dir:', os.path.isdir(os.path.join(\n"
            "    os.path.dirname(biopb_mcp.__file__), 'mcp', '_skills_data')))\n"
        ),
    )
    assert not out.is_error, out.text
    assert "tests: None" in out.text, f"the test tree is importable:\n{out.text}"
    assert "/biopb_mcp/_tests/" not in out.text
    # Staging must not have cost the child the catalog it is measured on.
    assert "skills_dir: True" in out.text, out.text


def test_reading_the_answer_key_does_not_go_unrecorded(session):
    """`execute_code` is arbitrary Python, so a run *can* open the fixture that
    defines its own answer. Nothing stops it and nothing should — but a run that
    did it must not score like one that did not.

    This is the exact route a measured `skill+asked` arm took to its procedure:
    `os.path.dirname(biopb_mcp.__file__)`, then open what it found.
    """
    before = len(session.peeked())
    # By absolute path, because staging already removed the *importable* route:
    # the checkout is still on this disk and a determined run can still open it,
    # which is the residual the tripwire exists to cover.
    answer_key = Path(drift_correction.__file__).resolve()
    out = session.call(
        "execute_code", python_code=f"print(open({str(answer_key)!r}).read()[:40])"
    )
    assert not out.is_error, out.text

    peeked = session.peeked()
    assert len(peeked) > before, "the fixture was read and nothing recorded it"
    assert any("drift_correction" in e["path"] for e in peeked), peeked[-3:]
    assert all(e["pid"] != session.child_pid for e in peeked), (
        "agent code runs in the kernel, not the session child"
    )


def test_the_session_serving_a_skill_is_not_mistaken_for_peeking(session):
    """The other half, and the one that decides whether this is usable: reading
    `_skills_data` is how `skill://` is served. If that counted, every skill arm
    would flag itself and the signal would be worth nothing."""
    session.call("read_resource", uri="skill://drift-correction")
    assert not [e for e in session.peeked() if "_skills_data" in e["path"]], (
        "serving a skill body registered as the agent peeking at it"
    )


def test_the_viewer_is_real(session):
    """A working viewer is what this layer scores against; on a display-less box
    the launcher's own Xvfb provides it. `QT_QPA_PLATFORM=offscreen` is not a
    substitute -- napari builds, then `add_image` dies in vispy's GL probe --
    which is why the harness forces a real GL platform."""
    real, detail = session.has_real_viewer()
    assert real, detail


def test_there_is_no_tensor_plane(session):
    """Forced, not incidental. A developer box often has a data plane up, and
    then the agent can wander into whatever catalog that machine holds -- so a
    finding might not reproduce anywhere else.

    A case presented on the plane gets the *run's* plane, one server for the
    whole invocation. The gate stays per case regardless: a session is handed a
    `tensor_url` only when its own case uploaded something, so this holds even
    once some other case has brought a plane up."""
    assert session.client_is_none(), (
        "client is live; this run would depend on the machine's own catalog"
    )


def test_an_array_makes_the_round_trip(session):
    """The scraping contract every case depends on: the harness can put a
    fixture into the kernel and read a result back out. Via files, not literals
    -- a fixture movie is megabytes and tool output is truncated."""
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
    """How an `array` fixture actually reaches the agent, with no tensor plane:
    as a layer on the viewer, which every skill's Parameters table accepts as a
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
    """Structural assertions ask whether a blocking question preceded the
    expensive call. That needs a record of what was called and when, and setup
    the harness itself did must not appear in it as agent behaviour."""
    before = len(session.calls)
    session.call("server_status")
    agent_calls = [c for c in session.calls[before:] if c[0] >= 0]
    assert [c[1] for c in agent_calls] == ["server_status"]

    session.setup("pass")
    assert session.calls[-1][0] == -1, "harness setup must not read as a turn"


# --- per case, and the reason this file is worth its wall-clock -------------


def test_this_cases_fixture_reaches_a_viewer_and_its_results_come_back(bench_case):
    """The whole setup path and the whole scrape path, with nobody driving them.

    This is what a paid run does before the agent says anything and after it
    stops: bring up a session, upload what the case presents on the plane, add
    every layer, and — at the other end — read the names the verifier will be
    handed. A case that fails either half is scored `no-result`, which is
    indistinguishable from an agent that failed, twenty minutes and one
    conversation later.

    One session for both halves. Two would double the smoke bill of a run over
    a catalogue, and neither half dirties what the other needs.
    """
    if reason := _session.why_unavailable():
        pytest.skip(reason)
    if any(layer.lazy for layer in bench_case.layers) and (
        why := _plane.plane_unavailable()
    ):
        pytest.skip(f"this case is presented on a data plane, and {why}")

    fixture = bench_case.build_fixture()
    ids = uploaded_ids(bench_case, fixture)
    plane = _plane.running_plane() if ids else None
    with live_session(
        skills_enabled=True,
        plugins=bench_case.plugins,
        tensor_url=plane.url if plane is not None else "",
    ) as session:
        load_fixture(session, bench_case, fixture, ids)
        names = session.setup("print([lyr.name for lyr in viewer.layers])").text
        for layer in bench_case.layers:
            assert layer.name in names, (
                f"`{layer.name}` never reached the viewer; the kernel reports {names}"
            )
        if ids:
            handles = session.setup(f"print(sorted({TENSOR_HANDLE}))").text
            for layer in bench_case.layers:
                if layer.lazy:
                    assert layer.name in handles, (
                        f"`{layer.name}` is presented on the plane but its id is "
                        f"not in {TENSOR_HANDLE}, which the task tells the agent "
                        "to read"
                    )

        wanted = list(bench_case.collect.values())
        session.setup(
            "import numpy as _np\n"
            + "\n".join(f"{name} = _np.zeros((3, 2))" for name in wanted)
        )
        trace = Trace(agent="none", respondent="none", task="smoke")
        got = scrape(session, trace, dict(bench_case.collect))
        for key in bench_case.collect:
            assert key in got, f"`{key}` was bound in the kernel but not scraped"
            assert np.asarray(got[key]).shape == (3, 2)
