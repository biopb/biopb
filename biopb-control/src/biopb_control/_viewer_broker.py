"""Matches a session's request for a picture of the viewer with a browser tab.

A session with no napari window cannot see its own results. The control's SPA
can draw them, so a request goes from the session to the control, and the control
hands it to an open viewer page, which renders it and posts the PNG back.

The page asks for work by long-polling :meth:`ViewerBroker.next`; each parked
poll is also its heartbeat, so a closed or hidden tab drops out when its poll
ends. Only a tab with a poll parked right now can be given a request, and the
page only polls while it is visible -- a hidden tab does not repaint, so a
capture sent to one would return a stale frame.
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
    """No visible viewer page can take the request."""


class CaptureFailed(Exception):
    """The page took the request and could not answer it."""


@dataclass
class Capture:
    png: bytes
    partial: bool = False
    notes: list = field(default_factory=list)


class ViewerBroker:
    def __init__(self) -> None:
        # tab id -> its parked poll, oldest park first: the last entry is the
        # tab that polled most recently.
        self._parked: dict[str, asyncio.Future] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._arrived = asyncio.Event()
        # One capture at a time: a tab is un-parked while it renders, so a second
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
        disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> Optional[dict]:
        """Park until a request is assigned to *tab_id*; ``None`` on timeout.

        *disconnected* is polled so a tab that went away frees its slot at once
        rather than at the end of the wait.
        """
        old = self._parked.pop(tab_id, None)
        if old is not None and not old.done():
            # One poll per tab: a reload racing its predecessor's poll wins.
            old.set_result(None)
        fut = asyncio.get_running_loop().create_future()
        self._parked[tab_id] = fut
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
            if self._parked.get(tab_id) is fut:
                del self._parked[tab_id]

    def _pick(self) -> Optional[asyncio.Future]:
        live = [f for f in self._parked.values() if not f.done()]
        return live[-1] if live else None

    async def capture(self, view: str, max_edge: int, timeout: float = 25.0) -> Capture:
        """Have a viewer page render *view* and return what it drew.

        *view* is the viewer's own query string; the page applies it exactly as
        it would a shared link.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        async with self._one:
            return await self._capture(view, max_edge, deadline)

    async def _capture(self, view: str, max_edge: int, deadline: float) -> Capture:
        loop = asyncio.get_running_loop()
        tab = self._pick()
        if tab is None:
            # The instant between one poll ending and the next arriving.
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), PARK_GRACE)
            except asyncio.TimeoutError:
                pass
            tab = self._pick()
        if tab is None:
            raise ViewerUnavailable(
                "no visible viewer page is connected: ask the user to open the "
                "web viewer in a browser tab and keep it visible (build the link "
                'with user_base_url(), not a bare /viewer: see read_doc("web-viewer"))'
            )
        req = f"c{next(self._ids)}"
        fut = asyncio.get_running_loop().create_future()
        self._pending[req] = fut
        tab.set_result({"req": req, "view": view, "max_edge": max_edge})
        try:
            return await asyncio.wait_for(fut, max(0.1, deadline - loop.time()))
        except asyncio.TimeoutError:
            raise CaptureFailed(
                "the viewer page did not answer in time (a hidden or busy tab?)"
            ) from None
        finally:
            self._pending.pop(req, None)

    def resolve(
        self,
        req: str,
        png: bytes,
        partial: bool = False,
        notes: Optional[list] = None,
        error: Optional[str] = None,
    ) -> bool:
        """The page's answer to *req*; false if nothing is waiting for it."""
        fut = self._pending.get(req)
        if fut is None or fut.done():
            return False
        if error:
            fut.set_exception(CaptureFailed(error))
        else:
            fut.set_result(Capture(png, partial, notes or []))
        return True
