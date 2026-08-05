"""That the stack works, with no model in it.

The cheapest thing that can go wrong here is also the most expensive to
discover late: a session that will not come up, or a fixture that will not
reach the viewer, wastes a paid run and looks exactly like a model failing the
task. This spends nothing — no API key is read and no completion is requested —
and `conftest.py` makes a failure here *skip* the run rather than merely
precede it.

Marked `tasks` because it needs the real session (a GL-capable display, the
kernel, dask). It is the one member of the marked set that costs only time.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..agentbench import _plane, _session
from ._runner import TENSOR_HANDLE, load_fixture, uploaded_ids
from .cases import CASES

pytestmark = pytest.mark.tasks


@pytest.fixture(params=CASES, ids=lambda c: c.case_id)
def case(request):
    ok, why = request.param.available()
    if not ok:
        pytest.skip(f"fixture: {why}")
    return request.param


def test_the_fixture_builds_and_withholds_its_truth(case):
    """Before any session: the tree on this machine is the one the case was
    written against, and the answer is not in what the agent will be handed."""
    fixture = case.build_fixture()
    assert fixture.citation, "a curated fixture with no citation should not build"
    for layer in case.layers:
        assert layer.key in fixture.data, (
            f"{case.case_id} puts `{layer.key}` on the viewer, but the fixture "
            "has no such data key"
        )
    assert not (set(fixture.data) & set(fixture.truth)), (
        "a truth key is also a data key -- a truth the run can see is not a truth"
    )


def test_the_fixture_reaches_a_real_viewer(case):
    """The whole setup path, with nobody driving it.

    This is what a paid run does before the agent says anything: bring up a
    session, upload what the case presents on the plane, add every layer, and
    confirm the kernel sees them. If it cannot happen, no run should start.
    """
    if reason := _session.why_unavailable():
        pytest.skip(reason)
    if any(layer.lazy for layer in case.layers) and (why := _plane.plane_unavailable()):
        pytest.skip(f"this case is presented on a data plane, and {why}")

    fixture = case.build_fixture()
    ids = uploaded_ids(case, fixture)
    plane = _plane.running_plane() if ids else None
    with _session.live_session(
        skills_enabled=True,
        plugins=case.plugins,
        tensor_url=plane.url if plane is not None else "",
    ) as session:
        load_fixture(session, case, fixture, ids)
        names = session.setup("print([lyr.name for lyr in viewer.layers])").text
        for layer in case.layers:
            assert layer.name in names, (
                f"`{layer.name}` never reached the viewer; the kernel reports {names}"
            )
        if ids:
            handles = session.setup(f"print(sorted({TENSOR_HANDLE}))").text
            for layer in case.layers:
                if layer.lazy:
                    assert layer.name in handles, (
                        f"`{layer.name}` is presented on the plane but its id is "
                        f"not in {TENSOR_HANDLE}, which the task tells the agent "
                        "to read"
                    )


def test_a_result_can_be_scraped_back_out(case):
    """The other end of the loop: a name bound in the kernel comes back as an
    array the verifier can score. A run whose answer cannot be read is scored
    as `no-result`, which would be indistinguishable from an agent that failed.
    """
    if reason := _session.why_unavailable():
        pytest.skip(reason)
    from ..agentbench._conversation import Trace, scrape

    with _session.live_session(skills_enabled=True) as session:
        wanted = list(case.collect.values())
        session.setup(
            "import numpy as _np\n"
            + "\n".join(f"{name} = _np.zeros((3, 2))" for name in wanted)
        )
        trace = Trace(agent="none", respondent="none", task="smoke")
        got = scrape(session, trace, dict(case.collect))
        for key in case.collect:
            assert key in got, f"`{key}` was bound in the kernel but not scraped"
            assert np.asarray(got[key]).shape == (3, 2)
