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
import time
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

_STEP = 0.5

#: Seconds a tab that stopped polling stays on record.
_FORGET = 60.0


class ViewerUnavailable(Exception):
    """No visible viewer page can take the request."""


class CaptureFailed(Exception):
    """The page took the request and could not answer it."""


@dataclass
class _Tab:
    parked: Optional[asyncio.Future] = None
    last_seen: float = 0.0


@dataclass
class Capture:
    png: bytes
    partial: bool = False
    notes: list = field(default_factory=list)


class ViewerBroker:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._tabs: dict[str, _Tab] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._ids = itertools.count(1)

    def tab_count(self) -> int:
        """How many tabs have a poll parked."""
        return sum(1 for t in self._tabs.values() if _is_parked(t))

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
        now = self._clock()
        # A page load mints a fresh id, so tabs that stopped polling are dropped
        # rather than accumulated.
        self._tabs = {
            k: t
            for k, t in self._tabs.items()
            if _is_parked(t) or now - t.last_seen < _FORGET
        }
        tab = self._tabs.setdefault(tab_id, _Tab())
        if _is_parked(tab):
            # One poll per tab: a reload racing its predecessor's poll wins.
            tab.parked.set_result(None)
        fut = asyncio.get_running_loop().create_future()
        tab.parked = fut
        tab.last_seen = self._clock()
        deadline = self._clock() + timeout
        try:
            while not fut.done():
                left = deadline - self._clock()
                if left <= 0:
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(fut), min(_STEP, left))
                except asyncio.TimeoutError:
                    if disconnected is not None and await disconnected():
                        break
            return fut.result() if fut.done() else None
        finally:
            if tab.parked is fut:
                tab.parked = None
            tab.last_seen = self._clock()

    def _pick(self) -> Optional[_Tab]:
        parked = [t for t in self._tabs.values() if _is_parked(t)]
        return max(parked, key=lambda t: t.last_seen, default=None)

    async def capture(self, view: str, max_edge: int, timeout: float = 25.0) -> Capture:
        """Have a viewer page render *view* and return what it drew.

        *view* is the viewer's own query string; the page applies it exactly as
        it would a shared link.
        """
        deadline = self._clock() + timeout
        tab = self._pick()
        waited = 0.0
        while tab is None and waited < PARK_GRACE:
            await asyncio.sleep(0.05)
            waited += 0.05
            tab = self._pick()
        if tab is None:
            raise ViewerUnavailable(
                "no visible viewer page is connected: ask the user to open the "
                "web viewer (/viewer) in a browser tab and keep it visible"
            )
        req = f"c{next(self._ids)}"
        fut = asyncio.get_running_loop().create_future()
        self._pending[req] = fut
        tab.parked.set_result({"req": req, "view": view, "max_edge": max_edge})
        try:
            return await asyncio.wait_for(fut, max(0.1, deadline - self._clock()))
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


def _is_parked(tab: _Tab) -> bool:
    return tab.parked is not None and not tab.parked.done()
