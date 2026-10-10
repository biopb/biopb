"""Hands a session's view of its results to a browser tab of the web viewer.

A session with no napari window cannot show the user anything. The control's SPA
can, so a request goes from the session to the control, and the control hands it
to an open viewer page, which moves to that view and acknowledges -- with a PNG
of what it drew, if the session asked for one and the page is on screen.

The page asks for work by long-polling :meth:`ViewerBroker.next`; each parked
poll is also its heartbeat, so a closed tab drops out when its poll ends. A
hidden page keeps polling and moves its state, but it does not repaint, so it
says it is hidden and sends no image. Every connected page gets a request; the
first visible one to answer is the answer.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

#: How long one parked poll waits before answering "nothing", under the shortest
#: idle timeout a reverse proxy is likely to apply.
POLL_SECONDS = 25.0

#: How long a request waits for a tab to be parked, to ride out the instant
#: between one poll ending and the next arriving.
PARK_GRACE = 1.5

#: Largest PNG the control accepts back.
MAX_PNG_BYTES = 16 * 1024 * 1024

#: How often a parked poll checks that its page is still there.
_STEP = 1.0


class ViewerUnavailable(Exception):
    """No viewer page is connected to take the request."""


class ShowFailed(Exception):
    """The page took the request and could not answer it."""


@dataclass
class Shown:
    #: Was the page on screen when it answered? A hidden one moved its state
    #: but drew nothing.
    visible: bool
    #: What it drew, when an image was asked for and the page could draw one.
    png: Optional[bytes] = None
    partial: bool = False
    notes: list = field(default_factory=list)


@dataclass
class _Parked:
    fut: asyncio.Future
    visible: bool


class ViewerBroker:
    def __init__(self) -> None:
        # tab id -> its parked poll, oldest park first: the last entry is the
        # tab that polled most recently.
        self._parked: dict[str, _Parked] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._arrived = asyncio.Event()
        # One request at a time: a tab is un-parked while it works, so a second
        # request would find no page and wrongly report that none is open.
        self._one = asyncio.Lock()
        self._ids = itertools.count(1)

    def tab_count(self) -> int:
        """How many tabs have a poll parked."""
        return len(self._parked)

    async def next(
        self,
        tab_id: str,
        timeout: float = POLL_SECONDS,
        visible: bool = True,
        disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> Optional[dict]:
        """Park until a request is assigned to *tab_id*; ``None`` on timeout.

        *visible* is whether the page is on screen as it parks. *disconnected* is
        polled so a tab that went away frees its slot at once
        rather than at the end of the wait.
        """
        old = self._parked.pop(tab_id, None)
        if old is not None and not old.fut.done():
            # One poll per tab: a reload racing its predecessor's poll wins.
            old.fut.set_result(None)
        fut = asyncio.get_running_loop().create_future()
        self._parked[tab_id] = _Parked(fut, visible)
        self._arrived.set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while not fut.done():
                left = deadline - loop.time()
                if left <= 0:
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(fut), min(_STEP, left))
                except asyncio.TimeoutError:
                    if disconnected is not None and await disconnected():
                        break
            return fut.result() if fut.done() else None
        finally:
            parked = self._parked.get(tab_id)
            if parked is not None and parked.fut is fut:
                del self._parked[tab_id]

    def _live(self) -> list[asyncio.Future]:
        return [p.fut for p in self._parked.values() if not p.fut.done()]

    async def show(
        self, view: str, image: bool, max_edge: int, timeout: float = 25.0
    ) -> Shown:
        """Have a viewer page move to *view*, and draw it if *image* is asked for.

        *view* is the viewer's own query string; the page applies it exactly as
        it would a shared link.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        async with self._one:
            return await self._show(view, image, max_edge, deadline)

    async def _show(
        self, view: str, image: bool, max_edge: int, deadline: float
    ) -> Shown:
        loop = asyncio.get_running_loop()
        tabs = self._live()
        if not tabs:
            # The instant between one poll ending and the next arriving.
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), PARK_GRACE)
            except asyncio.TimeoutError:
                pass
            tabs = self._live()
        if not tabs:
            raise ViewerUnavailable(
                "no viewer page is connected: ask the user to open the web viewer "
                "in a browser tab (build the link with user_base_url(), not a bare "
                '/viewer: see read_doc("web-viewer"))'
            )
        # Every tab gets it: nothing says which machine the user is sitting at.
        pending: dict[str, asyncio.Future] = {}
        for tab in tabs:
            req = f"c{next(self._ids)}"
            fut = loop.create_future()
            self._pending[req] = pending[req] = fut
            tab.set_result(
                {"req": req, "view": view, "image": image, "max_edge": max_edge}
            )
        try:
            return await self._first_answer(list(pending.values()), image, deadline)
        finally:
            for req in pending:
                self._pending.pop(req, None)

    @staticmethod
    async def _first_answer(
        futs: list[asyncio.Future], image: bool, deadline: float
    ) -> Shown:
        """The first visible tab's answer, else a hidden tab's, else the first error.

        Waits for a tab that can show the user, since a hidden one answers at
        once and says nothing about the others; stops at the deadline.
        """
        loop = asyncio.get_running_loop()
        waiting = set(futs)
        hidden: Optional[Shown] = None
        failure: Optional[Exception] = None
        while waiting:
            left = max(0.1, deadline - loop.time())
            done, waiting = await asyncio.wait(
                waiting, timeout=left, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                break
            for fut in done:
                try:
                    shown = fut.result()
                except ShowFailed as exc:
                    failure = failure or exc
                    continue
                if not shown.visible:
                    hidden = hidden or shown
                elif image and shown.png is None:
                    failure = failure or ShowFailed("the viewer page sent no image")
                else:
                    return shown
        if hidden is not None:
            return hidden
        if failure is not None:
            raise failure
        raise ShowFailed("the viewer page did not answer in time (a busy tab?)")

    def resolve(
        self,
        req: str,
        visible: bool,
        png: Optional[bytes] = None,
        partial: bool = False,
        notes: Optional[list] = None,
        error: Optional[str] = None,
    ) -> bool:
        """The page's answer to *req*; false if nothing is waiting for it."""
        fut = self._pending.get(req)
        if fut is None or fut.done():
            return False
        if error:
            fut.set_exception(ShowFailed(error))
        else:
            fut.set_result(Shown(visible, png, partial, notes or []))
        return True
