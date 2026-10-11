"""Who this session answers to: one lease, held by an agent or by the chat loop.

Runs **in the MCP server process**. A session serves one holder at a time. An
agent takes the lease through ``attach`` (the stdio shim, over ``/api/lease``);
the chat loop takes it with its first turn. A held lease is refused to the other
kind, so a chat pane and an attached agent never write to one kernel together.

A lease is time-bound because neither holder can be trusted to say goodbye: a
``kill -9``'d shim and a closed browser tab both just stop. The agent's shim
renews on a short beat; chat renews on turns only, with a long window, and is
kept alive while a turn runs. Expiry is lazy -- noticed by whoever asks next --
and goes through the same change hook as a release, so there is one path for
"the holder is gone".

The kernel's own one-agent claim (``_writers``) is keyed on the client that
wrote last and outlives whoever made it. A change of holder clears it, so the
new holder is not measured against the old one's claim.
"""

import threading
import time

from . import _writers

#: Seconds a lease survives without a renewal, by holder kind.
TTL = {"agent": 30.0, "chat": 300.0}

#: The chat loop's token. One writer however many windows are open.
CHAT_TOKEN = "biopb-chat"

_lock = threading.Lock()
_holder = None  # "agent" | "chat" | None
_token = None
_since = 0.0
_renewed = 0.0

# Called as ``fn(previous_kind, new_kind)`` after the holder changed or lapsed,
# outside the lock. A kind may be None.
_hooks = []
# kind -> callable returning whether that holder is mid-work and so not idle.
_busy_probes = {}


def on_change(fn):
    """Register *fn(previous_kind, new_kind)* for every change of holder."""
    if fn not in _hooks:
        _hooks.append(fn)


def set_busy_probe(kind, probe):
    """Keep a *kind* lease alive while *probe()* is true."""
    _busy_probes[kind] = probe


def _reset():
    """Forget everything, for tests."""
    global _holder, _token, _since, _renewed
    with _lock:
        _holder, _token, _since, _renewed = None, None, 0.0, 0.0
    _hooks.clear()
    _busy_probes.clear()


def _fire(changes):
    for previous, new in changes:
        _writers.clear_claim()
        for fn in list(_hooks):
            fn(previous, new)


def _expire_locked(now):
    """Drop a lapsed holder; returns the change to announce, if any."""
    global _holder, _token
    if _holder is None:
        return []
    probe = _busy_probes.get(_holder)
    if probe is not None and probe():
        return []
    if now - _renewed < TTL[_holder]:
        return []
    previous, _holder, _token = _holder, None, None
    return [(previous, None)]


def _snapshot_locked(now):
    if _holder is None:
        return {"holder": None}
    return {
        "holder": _holder,
        "age": round(now - _since, 1),
        "idle": round(now - _renewed, 1),
    }


def snapshot():
    """``{"holder": kind | None, ...}``: who holds it and for how long."""
    now = time.monotonic()
    with _lock:
        changes = _expire_locked(now)
        snap = _snapshot_locked(now)
    _fire(changes)
    return snap


def acquire(kind, token, force=False):
    """Take or refresh the lease for *token*; ``{"ok": bool, **snapshot}``.

    The holder asking again with its own token renews. A different holder is
    refused unless *force*, which replaces it; the replaced holder's hook sees
    the change.
    """
    global _holder, _token, _since, _renewed
    if kind not in TTL:
        raise ValueError(f"unknown lease kind: {kind!r}")
    now = time.monotonic()
    with _lock:
        changes = _expire_locked(now)
        ok = _holder is None or _token == token or force
        if ok:
            if _token != token or _holder != kind:
                changes.append((_holder, kind))
                _holder, _token, _since = kind, token, now
            _renewed = now
        snap = _snapshot_locked(now)
    _fire(changes)
    return {"ok": ok, **snap}


def renew(token):
    """Extend *token*'s lease; False if it no longer holds one."""
    global _renewed
    now = time.monotonic()
    with _lock:
        changes = _expire_locked(now)
        ok = _token is not None and _token == token
        if ok:
            _renewed = now
    _fire(changes)
    return ok


def release(token):
    """Give up *token*'s lease; False if it held none."""
    global _holder, _token
    now = time.monotonic()
    with _lock:
        changes = _expire_locked(now)
        ok = _token is not None and _token == token
        if ok:
            changes.append((_holder, None))
            _holder, _token = None, None
    _fire(changes)
    return ok
