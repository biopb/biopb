"""The broker that hands a session's view to an open viewer page."""

import asyncio
import json
import threading
import urllib.error
import urllib.request

import pytest

from biopb_control._viewer_broker import (
    ShowFailed,
    ViewerBroker,
    ViewerUnavailable,
)

from .test_control_front import (
    control,  # noqa: F401 - fixture
    upstream,  # noqa: F401 - fixture
    web_bundle,  # noqa: F401 - fixture
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"data"


def _run(coro):
    return asyncio.run(coro)


def test_a_request_with_no_page_connected_is_refused_plainly():
    async def go():
        with pytest.raises(ViewerUnavailable, match="open the web viewer"):
            await ViewerBroker().show("id=x", True, 512, timeout=1)

    _run(go())


def test_a_parked_page_receives_the_request_and_its_answer_returns():
    async def go():
        broker = ViewerBroker()
        poll = asyncio.create_task(broker.next("tab-a", timeout=5))
        await asyncio.sleep(0.05)
        asking = asyncio.create_task(broker.show("id=x&z=2", True, 256, timeout=5))
        job = await poll
        assert job["view"] == "id=x&z=2" and job["max_edge"] == 256 and job["image"]
        assert broker.resolve(
            job["req"], True, b"PNG", partial=True, notes=["labels: 404"]
        )
        got = await asking
        assert (got.visible, got.png, got.partial, got.notes) == (
            True,
            b"PNG",
            True,
            ["labels: 404"],
        )

    _run(go())


def test_the_most_recently_parked_page_is_the_one_asked():
    async def go():
        broker = ViewerBroker()
        first = asyncio.create_task(broker.next("old", timeout=5))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(broker.next("new", timeout=5))
        await asyncio.sleep(0.05)
        asking = asyncio.create_task(broker.show("id=x", True, 64, timeout=5))
        job = await second
        assert not first.done()
        broker.resolve(job["req"], True, b"x")
        await asking
        first.cancel()

    _run(go())


def test_a_poll_that_times_out_answers_none_and_leaves_the_tab_unparked():
    async def go():
        broker = ViewerBroker()
        assert await broker.next("t", timeout=0.1) is None
        assert broker.tab_count() == 0

    _run(go())


def test_a_disconnected_page_frees_its_slot():
    async def go():
        broker = ViewerBroker()

        async def gone():
            return True

        assert await broker.next("t", timeout=5, disconnected=gone) is None
        assert broker.tab_count() == 0

    _run(go())


def test_a_page_error_surfaces_as_a_failed_show():
    async def go():
        broker = ViewerBroker()
        poll = asyncio.create_task(broker.next("t", timeout=5))
        await asyncio.sleep(0.05)
        asking = asyncio.create_task(broker.show("id=x", True, 64, timeout=5))
        job = await poll
        broker.resolve(job["req"], True, error="no canvas")
        with pytest.raises(ShowFailed, match="no canvas"):
            await asking

    _run(go())


def test_a_page_that_never_answers_times_out():
    async def go():
        broker = ViewerBroker()
        poll = asyncio.create_task(broker.next("t", timeout=5))
        await asyncio.sleep(0.05)
        with pytest.raises(ShowFailed, match="did not answer"):
            await broker.show("id=x", True, 64, timeout=0.3)
        await poll

    _run(go())


def test_an_answer_nobody_waits_for_is_refused():
    assert ViewerBroker().resolve("c99", True, b"x") is False


def _call(url, data=None, method=None):
    req = urllib.request.Request(url, data=data, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()


def test_the_routes_carry_a_show_end_to_end(control):  # noqa: F811
    polled = {}

    def page():
        status, body = _call(f"{control}/api/viewer/next?client=t1")
        polled["job"] = json.loads(body)
        _call(
            f"{control}/api/viewer/answer/{polled['job']['req']}?partial=1&note=hi&visible=1",
            data=_PNG,
            method="POST",
        )

    thread = threading.Thread(target=page)
    thread.start()
    import time

    time.sleep(0.3)
    status, body = _call(
        f"{control}/api/viewer/show",
        data=json.dumps({"view": "id=a", "image": True, "max_edge": 128}).encode(),
        method="POST",
    )
    thread.join(5)
    answer = json.loads(body)
    assert status == 200 and polled["job"]["view"] == "id=a"
    import base64

    assert base64.b64decode(answer["png"]) == _PNG
    assert (
        answer["visible"] is True
        and answer["partial"] is True
        and answer["notes"] == ["hi"]
    )


def test_a_show_with_no_page_is_a_409(control):  # noqa: F811
    with pytest.raises(urllib.error.HTTPError) as exc:
        _call(f"{control}/api/viewer/show", data=b'{"view": "id=a"}', method="POST")
    assert exc.value.code == 409
    assert b"web viewer" in exc.value.read()


def test_a_show_without_a_view_is_a_400(control):  # noqa: F811
    with pytest.raises(urllib.error.HTTPError) as exc:
        _call(f"{control}/api/viewer/show", data=b"{}", method="POST")
    assert exc.value.code == 400


def test_an_answer_that_is_not_a_png_fails_the_show(control):  # noqa: F811
    polled = {}

    def page():
        _, body = _call(f"{control}/api/viewer/next?client=t2")
        polled["job"] = json.loads(body)
        _call(
            f"{control}/api/viewer/answer/{polled['job']['req']}",
            data=b"",
            method="POST",
        )

    thread = threading.Thread(target=page)
    thread.start()
    import time

    time.sleep(0.3)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _call(
            f"{control}/api/viewer/show",
            data=b'{"view": "id=a", "image": true}',
            method="POST",
        )
    thread.join(5)
    assert exc.value.code == 504 and b"sent no image" in exc.value.read()


def test_two_requests_in_a_row_share_one_tab():
    async def go():
        broker = ViewerBroker()

        async def page():
            for _ in range(2):
                job = await broker.next("t", timeout=5)
                await asyncio.sleep(0.2)
                broker.resolve(job["req"], True, b"x")

        poll = asyncio.create_task(page())
        await asyncio.sleep(0.05)
        first = asyncio.create_task(broker.show("id=a", True, 64, timeout=5))
        second = asyncio.create_task(broker.show("id=b", True, 64, timeout=5))
        assert (await first).png == b"x" and (await second).png == b"x"
        await poll

    _run(go())


def test_a_hidden_page_still_takes_the_view_and_answers_without_an_image():
    async def go():
        broker = ViewerBroker()
        poll = asyncio.create_task(broker.next("t", timeout=5, visible=False))
        await asyncio.sleep(0.05)
        asking = asyncio.create_task(broker.show("id=x", True, 64, timeout=5))
        job = await poll
        assert job["view"] == "id=x"
        broker.resolve(job["req"], False)
        got = await asking
        assert (got.visible, got.png) == (False, None)

    _run(go())


def test_a_view_without_an_image_is_acknowledged_with_none():
    async def go():
        broker = ViewerBroker()
        poll = asyncio.create_task(broker.next("t", timeout=5))
        await asyncio.sleep(0.05)
        asking = asyncio.create_task(broker.show("id=x", False, 64, timeout=5))
        job = await poll
        assert job["image"] is False
        broker.resolve(job["req"], True)
        got = await asking
        assert (got.visible, got.png) == (True, None)

    _run(go())


def test_a_visible_page_is_preferred_over_a_later_hidden_one():
    async def go():
        broker = ViewerBroker()
        shown = asyncio.create_task(broker.next("front", timeout=5))
        await asyncio.sleep(0.05)
        hidden = asyncio.create_task(broker.next("back", timeout=5, visible=False))
        await asyncio.sleep(0.05)
        asking = asyncio.create_task(broker.show("id=x", False, 64, timeout=5))
        job = await shown
        assert not hidden.done()
        broker.resolve(job["req"], True)
        await asking
        hidden.cancel()

    _run(go())
