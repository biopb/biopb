"""The control plane's single web origin — a Starlette/uvicorn ASGI app on 8813.

This is the single-origin web front of the control plane
(``biopb-mcp/ARCHITECTURE.md``). It replaces the
earlier stdlib ``ThreadingHTTPServer`` control API with a real ASGI app **on the
same port**, and routes by namespace so no two upstreams share a path prefix:

- ``GET  /health``                -> ``{"control": "ok", "data_plane": {...}}`` —
                                     the control's own liveness (what
                                     ``_control_client`` and the installer poll).
                                     Bare, kept byte-for-byte.
- ``POST /api/data_plane/{ensure,stop,restart}`` -> supervisor verbs: ensure the
                                     plane is up (bounded wait), stop it, or bounce
                                     it, each returning the snapshot. ``biopb-mcp``
                                     calls ``ensure`` in place of shelling out
                                     ``biopb server start``; the dashboard drives all
                                     three. Under ``/api/`` — control *verbs about*
                                     the plane live there.
- ``GET  /api/status``            -> the control's own liveness + the data-plane
                                     snapshot + a live-session count (what the
                                     dashboard polls).
- ``GET  /api/sessions``          -> the live MCP sessions from the registry, each
                                     with its ``/session/<id>/observe`` link.
- ``POST /api/sessions/new``      -> launch an agentless ``biopb mcp view``
                                     session on this machine's display. Refused
                                     unless this control is loopback-bound and
                                     has a display of its own; the child is
                                     detached and self-registering, so the
                                     control launches it without owning it.
- ``GET  /`` (and every other non-API, non-proxy GET) -> the built ``web/``
                                     SPA bundle (``static_dir``). The control is
                                     the **single web origin**: it serves the
                                     bundle's static assets and falls back to
                                     ``index.html`` for deep links, so the
                                     dashboard (``/``), the dataviewer
                                     (``/viewer``), and each session's observe
                                     shell (``/session/<id>/observe``) are all
                                     React routes of that one SPA — no build-time
                                     namespacing, base ``/``. (No bundle ->
                                     API-only.) ``url_prefix`` republishes the
                                     whole origin under a reverse-proxy path
                                     prefix at run time; see
                                     ``docs/url-prefix.md``.
- ``/data_plane/{api,livez,...}`` is reverse-proxied to the supervised tensor
  server's HTTP sidecar — a ``Mount`` that strips its prefix, so the sidecar
  (which serves ``/api/*`` at its own root) needs no knowledge of
  the ``/data_plane`` namespace. The sidecar no longer serves static assets (the
  control owns the whole UI), so there is no ``/data_plane/viewer`` mount. Auth
  headers pass straight through; the sidecar re-validates.

The three ``/api/*`` namespaces therefore never collide: the control's own API is
``/api/*``, the data plane's is ``/data_plane/api/*``, and (later) each session's
is ``/session/<id>/api/*``.

Keeping the control lean (invariant I2) still holds: the ASGI stack
(starlette/uvicorn/httpx) is light and pulls in no napari/dask/Qt/
pyarrow, and the tensor server is still a *supervised subprocess* the control
never imports — the proxy reaches it over loopback like any other client.

- ``/session/<id>/observe`` serves the control's own SPA shell (the React
  ObservePage), while ``/session/<id>/api/*`` is reverse-proxied to the shim-owned
  MCP session child on its dynamic loopback port, resolved per-request from the
  filesystem registry (``biopb._sessions``); an unknown or dead session
  yields a clean 404 (and the dead record is pruned). Unlike the data-plane proxy,
  the ``/api/*`` hop drops both ``Host`` and ``Origin``: httpx then sets ``Host``
  to the loopback target (satisfying the child's own loopback Host guard) and the
  absent ``Origin`` passes its Origin guard — so the trusted control→child hop is
  accepted regardless of which external hostname the browser used to reach the
  control. (Rebinding/token protection for the origin as a whole is a
  follow-up, same as the data-plane proxy's.)
- ``/session/<id>/console/*`` is the same hop for the **user console** — a code
  cell on the observe page that runs in that session's kernel — and is proxied
  **only when this control is loopback-bound** (``_session_proxy_roots``). It is
  a separate root precisely so that "can a browser reach an RCE here?" stays one
  checkable statement: ``api`` always, ``console`` local-mode only, ``/mcp``
  never.

This module lands the namespaced origin, the data-plane API proxy, per-session
observe routing, and the control-served SPA bundle — the full single-origin
front.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from html import escape as _escape_html
from pathlib import Path

import httpx
import uvicorn
from biopb import (
    _agents,
    _algorithms,
    _kernel_plugins,
    _locations,
    _sessions,
    _web_auth,
)
from biopb._lifecycle.daemon import detach_kwargs
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from ._supervisor import DataPlaneSupervisor

logger = logging.getLogger(__name__)

# Headroom (seconds) between the server's ensure wait and the client's HTTP
# timeout: the server must send its verdict BEFORE the client's urlopen times
# out, else the client treats a working-but-slow control plane as unreachable.
_RESPONSE_MARGIN = 5.0
_MIN_ENSURE_WAIT = 1.0

# Response headers we must not copy verbatim from the upstream tensor server:
# hop-by-hop headers and framing that StreamingResponse re-derives itself.
_HOP_BY_HOP = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"}
)

# The only session-child surface the control will proxy: its data API (matched by
# first path segment). The observe *page* is now the SPA shell the control serves
# itself (/session/<id>/observe -> index.html); only its /api/* data calls reach
# the child. This is an ALLOWLIST on purpose — the child also serves /mcp (the
# agent RCE transport) on the same port, and this hop strips its only auth
# (Host/Origin), so anything not explicitly allowed must be refused. A denylist
# would be unsafe: httpx normalizes dot-segments, so a traversal like
# `api/../mcp` (or its %2e%2e form, already decoded by the ASGI server) collapses
# to /mcp past a naive "startswith('mcp')" check.
_SESSION_ALLOWED_ROOTS = frozenset({"api"})

# The **conditionally** proxied root: the user console, a code cell on the observe
# page that runs in the session's kernel (biopb-mcp ``docs/user-console.md``).
# Kept out of the set above rather than added to it, because the two are gated
# differently and the difference is the whole point: `api` is always proxied,
# `console` only when this control is loopback-bound.
#
# Why a separate root at all. The allowlist exists to keep the child's /mcp — an
# RCE on the same port — off this origin. An execute route folded into `api`
# would put arbitrary code back through exactly that hole, silently: the
# allowlist would still be there, still enforced, and no longer true. A distinct
# root keeps the statement checkable — `api` always, `console` local-mode only,
# `/mcp` never — and makes "is RCE reachable from the browser?" one boolean.
#
# That boolean assumes the root is **POST-only**, and `session_proxy` enforces
# it rather than trusting the child to: the CSRF gate skips safe methods, so a
# cross-site GET to any proxied root is forwarded unchecked.
#
# Known limitation: this reads the control's own **bind**, so a loopback control
# deliberately published by a reverse proxy (the topology biopb-mcp CLAUDE.md
# points at for untrusted networks) reads as local and gets the console. That
# operator is already responsible for the token in front of the data plane; a
# control-side opt-out flag is the follow-up if the reverse-proxy topology stops
# being the exception.
_SESSION_CONSOLE_ROOT = "console"

# The built-in chat client's one write route (biopb-mcp ``mcp/_chat_api.py``).
# Gated identically to the console and for the identical reason: a chat turn
# runs arbitrary code in the session kernel, so it is the same RCE the allowlist
# above exists to keep off this origin. Its *reads* are not here -- they live
# under `api`, which is both correct (a conversation is a read like the job
# list) and required, since the POST-only assumption above would forward a
# cross-site GET to this root unchecked.
_SESSION_CHAT_ROOT = "chat"

# The execute-capable roots, which `session_proxy` narrows to POST. Both are
# here for the same reason the console was: the CSRF gate skips safe methods, so
# a cross-site GET to either is forwarded unchecked, and the root's claim must
# not rest on the child's method list. Naming the set rather than testing one
# root means a third such root inherits the narrowing by being added here.
_SESSION_POST_ONLY_ROOTS = frozenset({_SESSION_CONSOLE_ROOT, _SESSION_CHAT_ROOT})


def _session_proxy_roots(console_enabled: bool) -> frozenset[str]:
    """The session-child path roots this control will proxy.

    One source for both the proxy's own gate and the auth middleware, so the
    guard and the thing it guards cannot disagree about what is reachable.

    The flag reads "console" for history but means **this control is
    loopback-bound**: it is computed from the bind, not from any feature switch,
    and it gates every execute-capable root together. Whether a given one is
    actually served is the child's own decision (``observe.console_enabled``,
    ``observe.chat_enabled``), which is the half this control does not and
    should not know.
    """
    if console_enabled:
        return _SESSION_ALLOWED_ROOTS | {_SESSION_CONSOLE_ROOT, _SESSION_CHAT_ROOT}
    return _SESSION_ALLOWED_ROOTS


# HTTP methods that change state (so they carry a CSRF risk); safe verbs
# (GET/HEAD/OPTIONS) don't.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Every /api/ route is now gated, `/api/data_plane/ensure` included. It used to be
# exempted (biopb/biopb#424 item 2) because biopb-mcp's _control_client had no way
# to obtain the token — the control handed back the plane's endpoint but never a
# credential — so gating this idempotent route would have locked the mcp client out
# of a token-gated deployment. That exemption was an unauthenticated state-change,
# safe only while a local control was necessarily tokenless; #468's optional local
# token falsified that. The credential handoff (biopb/biopb#470) unblocks the fix:
# the control writes the token to an owner-only file and _control_client carries it,
# so this route can be gated like the rest and the exemption is gone.

# Data-plane log tail (the dashboard /logs page polls it). Bound BOTH the returned
# line count and the bytes read off the end of the file, so tailing a multi-GB log
# never loads it whole: we seek to the final _LOG_TAIL_MAX_BYTES and keep the last
# N lines of that window.
_LOG_TAIL_DEFAULT_LINES = 200
_LOG_TAIL_MAX_LINES = 2000
_LOG_TAIL_MAX_BYTES = 512 * 1024


def _tail_file(path: Path, max_lines: int, max_bytes: int) -> tuple[list[str], bool]:
    """Return ``(lines, truncated)`` for the tail of *path*.

    Reads at most the final *max_bytes* and returns at most *max_lines* lines from
    the end. ``truncated`` is True when older content exists that was not returned
    (the byte window didn't reach the file start, or the line cap trimmed more).

    The child (tensor server) and its native libraries emit arbitrary bytes, so
    decode UTF-8 with ``errors="replace"`` rather than risk a decode error. When
    the byte window starts mid-file its first line is almost certainly a fragment,
    so drop it.
    """
    size = path.stat().st_size
    read_bytes = min(size, max_bytes)
    with path.open("rb") as f:
        if read_bytes < size:
            f.seek(size - read_bytes)
        data = f.read(read_bytes)
    partial = read_bytes < size
    lines = data.decode("utf-8", "replace").splitlines()
    if partial and lines:
        lines = lines[1:]  # drop the leading fragment
    truncated = partial
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated = True
    return lines, truncated


def _is_proxied_session_path(path: str, roots=_SESSION_ALLOWED_ROOTS) -> bool:
    """True for ``/session/<id>/<root>/...`` where ``<root>`` is in *roots* —
    i.e. exactly what ``session_proxy`` forwards to the child.

    Takes the *same* root set the proxy's own gate uses, so the guard and the
    thing it guards cannot drift: any path the proxy would forward (including a
    bare ``/session/<id>/api`` with no further segment) is gated, and a root
    added to the set is covered on both sides at once. Not
    ``/session/<id>/observe`` (the SPA shell), and not a bare ``/session/<id>``.
    """
    if not path.startswith("/session/"):
        return False
    rest = path[len("/session/") :]  # "<id>/<sub_path...>" (session ids are slash-free)
    slash = rest.find("/")
    if slash == -1:
        return False  # bare /session/<id>
    return rest[slash + 1 :].split("/")[0] in roots


# --- publishing this origin under a path prefix (biopb/biopb#728) ---------- #
#
# The control is normally the origin root, but a portal can publish it under a
# path prefix -- the driver is an Open OnDemand interactive app, whose
# `/node/<host>/<port>/` route passes the full, untouched path to the backend and
# rewrites nothing in the response. The prefix carries the compute node's
# hostname and a per-session port, both allocated at job start, so there is no
# build-time answer (`vite build --base=...` cannot bake it) and it has to be
# learned at run time.
#
# Two halves: `_URLPrefixMiddleware` takes the prefix *off* the request path so
# every route matches unchanged, and `_rewrite_shell_html` puts it *back* into the
# served index.html so the browser asks for prefixed URLs to begin with.
#
# The prefix comes from explicit configuration ONLY -- never from a request header
# such as X-Forwarded-Prefix. A request-controlled `<base href>` lets any caller
# repoint every relative URL in the document at an origin of their choosing, which
# is a considerably worse bug than the one being fixed. Nothing needs inferring:
# an OnDemand `before.sh` knows $host and $port before the job starts.


# What a prefix segment may contain: the unreserved + sub-delim URL path
# characters, and nothing else. Deliberately excludes three classes, each of
# which would let a *configured* prefix mean something other than "a path on this
# origin":
#
#   - ``\`` and whitespace/controls. WHATWG URL parsing resolves
#     ``<base href="/\evil.com/">`` to ``http://evil.com/`` — a backslash after
#     the leading slash enters the authority, so every relative URL in the
#     document (and every ``new URL(x, document.baseURI)`` the SPA runs) leaves
#     the origin. Browsers strip tabs and newlines *before* parsing, so those
#     smuggle a backslash into the same position.
#   - ``?`` and ``#``. A query or fragment in a ``<base href>`` silently changes
#     what every relative URL resolves to.
#   - ``%``. ``scope["path"]`` reaches the middleware percent-*decoded* while the
#     shell carries the prefix *encoded*, so an encoded prefix cannot be both at
#     once. Barring it keeps the two representations identical by construction.
#
# ``:`` is legal in a path segment but excluded too, so that the likely operator
# slip -- pasting a whole URL, ``--url-prefix https://host/biopb`` -- fails loudly
# instead of quietly becoming the path ``/https:/host/biopb``.
#
# This is hardening, not a patched exploit: the prefix comes from configuration,
# and whoever sets it can already pass --static-dir or PYTHONPATH. It is what
# makes "the prefix can only ever name a path on this origin" a property of the
# code rather than of the operator's care.
_SAFE_PREFIX_SEGMENT = re.compile(r"[A-Za-z0-9._~!$&'()*+,;=@-]+")


def normalize_url_prefix(value: str | None) -> str | None:
    """Canonicalize a configured URL prefix to ``/a/b``, or ``None`` for no prefix.

    One leading slash, no trailing slash, empty segments dropped; ``None``,
    ``""`` and ``"/"`` all mean "serve at the root". Applied by
    :func:`build_app` — the single consumer — so every entry point (``python -m
    biopb_control run``, the foreground CLI, tests) normalizes by the same rule.

    Raises :class:`ValueError` for anything that is not a plain same-origin path
    (see :data:`_SAFE_PREFIX_SEGMENT`, and ``.``/``..``, which would make the
    served ``<base href>`` and the path the middleware strips disagree). Callers
    surface it as a configuration error — refusing to start beats serving a
    document whose every relative URL points somewhere unintended.
    """
    if not value:
        return None
    segments = [s for s in value.strip().split("/") if s]
    if not segments:
        return None
    for segment in segments:
        if segment in (".", "..") or not _SAFE_PREFIX_SEGMENT.fullmatch(segment):
            raise ValueError(
                f"invalid URL prefix segment {segment!r} in {value!r}: a prefix "
                "must be a plain path on this origin (letters, digits and "
                "._~-!$&'()*+,;=@ per segment)"
            )
    return "/" + "/".join(segments)


class _URLPrefixMiddleware:
    """Strip the configured URL prefix off incoming request paths.

    For ``http`` and ``websocket`` scopes whose path lies under *prefix*, rewrite
    ``scope["path"]`` (and ``raw_path``) to the remainder. Every route below then
    sees byte-for-byte the request it would see at the origin root, so nothing
    else in this module knows the prefix exists — the existing route table, the
    two proxy ``Mount``s and the auth gate are all covered by that one property.

    Two constraints hold this in place:

    - It must be the **outermost** middleware. :class:`_ControlAuthMiddleware`
      decides what to gate by reading ``scope["path"]`` directly, so an unstripped
      ``/node/h/p/api/data_plane/restart`` would sail past its
      ``startswith("/api/")`` check — an auth bypass, not merely a 404.
    - An unprefixed request must pass through **untouched**, not 404: biopb-mcp's
      ``_control_client`` and the installer poll ``http://127.0.0.1:8813/health``
      over loopback with no prefix, and they keep working while a prefix is
      configured for the portal.

    Deliberately **not** the ASGI ``root_path`` convention (leave the path whole,
    name the prefix in ``scope["root_path"]``), and not a hybrid either.
    ``Mount`` composes ``root_path + matched_path`` for its sub-app while
    ``get_route_path`` subtracts ``root_path`` from ``path`` only when the path
    still starts with it — so a stripped path plus a ``root_path`` makes that
    subtraction silently no-op inside ``/data_plane`` and ``/session/{id}``, and
    the sub-app sees its own mount prefix again (its routes stop matching).
    The un-stripped variant would work for routing but hands
    ``_ControlAuthMiddleware`` a prefixed path, which is the bypass above. Nothing
    here builds absolute URLs from ``root_path`` — the browser side is carried by
    the rewritten shell — so stripping outright is both the simpler and the
    correct half.
    """

    def __init__(self, app: ASGIApp, prefix: str) -> None:
        self.app = app
        self._prefix = prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            rest = self._strip(scope.get("path", ""))
            if rest is not None:
                scope = dict(scope)
                scope["path"] = rest
                # raw_path is the still-encoded path; strip the same bytes when it
                # carries the prefix verbatim, and otherwise leave it alone —
                # routing reads scope["path"], so a percent-encoded prefix costs
                # nothing but a stale raw_path.
                raw = scope.get("raw_path")
                encoded = self._prefix.encode("utf-8")
                if isinstance(raw, bytes) and raw.startswith(encoded):
                    scope["raw_path"] = raw[len(encoded) :] or b"/"
        await self.app(scope, receive, send)

    def _strip(self, path: str) -> str | None:
        """*path* without the prefix, or ``None`` when it is not prefixed."""
        if path == self._prefix:
            return "/"
        if path.startswith(self._prefix + "/"):
            return path[len(self._prefix) :]
        return None


# Root-absolute ``src=``/``href=`` values in the SPA shell. ``<base href>`` has no
# effect on these — it resolves only *relative* URLs — so they are rewritten
# outright. The ``(?!/)`` leaves protocol-relative ``//host/...`` alone.
_ROOT_ABSOLUTE_REF = re.compile(r"""\b(src|href)=(["'])/(?!/)""")
_HEAD_OPEN = re.compile(r"<head[^>]*>", re.IGNORECASE)


def _rewrite_shell_html(shell: str, prefix: str) -> str:
    """Return the SPA shell (*index.html*) rearranged to live under *prefix*.

    Three edits, confined to ``index.html`` — no JS or CSS is touched, because the
    built bundle needs none: its lazy route chunks are relative module specifiers
    (``import("./DashboardPage-*.js")``), which resolve against the importing
    module's URL and so follow the prefix for free.

    - ``<base href="<prefix>/">`` first in ``<head>``, so every *relative* URL in
      the document and every runtime ``new URL(x, document.baseURI)`` lands under
      the prefix;
    - each root-absolute ``src=``/``href=`` rewritten to ``<prefix>/…`` (the entry
      chunk, the stylesheet, the icons) — what ``<base>`` cannot do;
    - ``window.__BIOPB_BASE__``, the runtime hook the SPA reads in place of the
      build-time ``import.meta.env.BASE_URL``.

    The rewrite is computed once and served to *every* request — this never sees
    the request path — so an unprefixed ``http://127.0.0.1:8813/`` gets the
    prefixed document too. Its assets still load (the browser asks for
    ``<prefix>/assets/…`` on the same origin and the middleware strips the prefix
    straight back off), but the *app* must not take ``__BIOPB_BASE__`` at face
    value there: a router basename of ``<prefix>`` against a location of ``/``
    renders an empty tree. ``web/packages/app/src/base.ts`` therefore honours the
    prefix only when ``location.pathname`` is actually under it, which is what
    keeps that root — the one ``biopb ui`` opens — working alongside the portal
    route.
    """
    # Escape for each context the prefix lands in, even though
    # normalize_url_prefix has already confined it to path characters: json.dumps
    # quotes the script literal (and `</` must not close the tag early), and the
    # three attribute sites take HTML escaping. Two layers, so neither the
    # charset nor the escaping is load-bearing on its own.
    literal = json.dumps(prefix).replace("</", "<\\/")
    attr = _escape_html(prefix, quote=True)
    injected = f'<base href="{attr}/"><script>window.__BIOPB_BASE__={literal};</script>'
    rewritten = _ROOT_ABSOLUTE_REF.sub(
        lambda m: f"{m.group(1)}={m.group(2)}{attr}/", shell
    )
    head = _HEAD_OPEN.search(rewritten)
    if head is None:  # no <head> to open: the injection still has to come first
        return injected + rewritten
    return rewritten[: head.end()] + injected + rewritten[head.end() :]


class _ControlAuthMiddleware:
    """Gate the control's web API at the single origin — both the
    control's **own** ``/api/*`` and each session's proxied ``/session/<id>/api/*``.

    A pure-ASGI middleware (not ``BaseHTTPMiddleware``) so it touches only the
    guarded API paths and leaves the streaming ``/data_plane`` proxy, the observe
    SPA shell, and the static bundle to pass straight through untouched — wrapping
    those in ``BaseHTTPMiddleware`` would interfere with the proxies'
    ``StreamingResponse`` + background-close.

    Policy, mirroring the tensor sidecar so the two agree:

    - **Token configured** → require a valid ``Authorization: Bearer`` /
      ``X-Biopb-Token`` (401 otherwise). This is the whole point of the single
      origin: the token that already gates the data plane now also gates the
      control's stop/restart verbs and the session enumeration.
    - **No token** (local mode, all listeners loopback-bound) → require a
      **loopback Host** (421 otherwise), so a DNS-rebinding page can't drive the
      token-less origin.
    - **Unsafe method** (POST/…) → additionally refuse a forgeable cross-site
      request (403) — a token header or a same-origin ``Sec-Fetch-Site`` passes,
      a browser's cross-site POST does not (CSRF).

    ``/session/<id>/api/*`` gets the *same* policy (biopb/biopb#424): the observe
    API drives mutating kernel verbs (interrupt/restart, job cancel), the proxy
    hop deliberately strips Host/Origin toward the child (so the child cannot
    judge the browser origin itself), and session ids are guessable
    (``<timestamp>-<pid>``) — so a guessed id must not be drivable cross-site or
    via DNS-rebinding. The ``/observe`` shell (a plain SPA GET serving only the
    app bundle) stays open. ``/data_plane/*`` keeps its own gate (the sidecar
    re-validates the forwarded token), so it is not touched here.

    ``session_roots`` is the proxy's own root set, so whatever that forwards is
    what this gates — including ``/session/<id>/console/*`` when the console is
    enabled, which is the one path where the request being gated is arbitrary
    code. Note the gate is **necessary but not sufficient** for the console: it
    judges the caller, not the topology, and would happily authorize an execute
    on a public origin. What keeps the console off a public origin is that the
    root is not proxied there at all (:func:`_session_proxy_roots`).
    """

    def __init__(
        self,
        app: ASGIApp,
        token: str | None,
        session_roots=_SESSION_ALLOWED_ROOTS,
    ) -> None:
        self.app = app
        self._token = token
        self._session_roots = session_roots

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._guarded(scope["path"]):
            get = Headers(scope=scope).get
            denial = self._deny(scope["method"], get)
            if denial is not None:
                await denial(scope, receive, send)
                return
        await self.app(scope, receive, send)

    def _guarded(self, path: str) -> bool:
        if path.startswith("/api/"):
            return True
        return _is_proxied_session_path(path, self._session_roots)

    def _deny(self, method: str, get: _web_auth.HeaderGetter) -> Response | None:
        """The response to send if the request is refused, else ``None``."""
        if self._token:
            if not _web_auth.token_valid(get, self._token):
                return JSONResponse(
                    {"error": "invalid or missing token"}, status_code=401
                )
        elif not _web_auth.host_is_loopback(get("host")):
            return JSONResponse({"error": "invalid Host header"}, status_code=421)
        if method in _UNSAFE_METHODS and _web_auth.is_forgeable_cross_site(get):
            return JSONResponse(
                {"error": "cross-site request refused"}, status_code=403
            )
        return None


def _bounded_ensure_wait(ensure_timeout: float, client_timeout: float) -> float:
    """How long ``/data_plane/ensure`` should wait for the plane to come up.

    Bounded strictly below the client's HTTP timeout (by ``_RESPONSE_MARGIN``) so
    the server always answers first; also capped by the server's own configured
    ``ensure_timeout``, and floored at ``_MIN_ENSURE_WAIT``. A missing/invalid
    client hint (``<= 0``) falls back to the configured ``ensure_timeout`` (the
    client then relies on its own urlopen timeout being generous).
    """
    if client_timeout <= 0:
        return ensure_timeout
    return max(_MIN_ENSURE_WAIT, min(ensure_timeout, client_timeout - _RESPONSE_MARGIN))


def _loopback_url(host: str, port: int, scheme: str = "http") -> str:
    """A loopback-reachable base URL for a server that may bind a wildcard.

    A tensor server bound to ``0.0.0.0`` / ``::`` is reached over its loopback
    address; anything else (an explicit host) is used as given. An IPv6 literal
    is bracketed so the ``:port`` suffix stays unambiguous. Mirrors the
    supervisor's liveness-probe convention.
    """
    reachable = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(host, host)
    if ":" in reachable:  # IPv6 literal must be bracketed in a URL (e.g. [::1])
        reachable = f"[{reachable}]"
    return f"{scheme}://{reachable}:{port}"


# How long each per-session kernel probe may take. The dashboard polls the
# session list every few seconds and the probes run concurrently, so this is
# kept short: a slow or wedged child yields "unknown" rather than stalling the
# whole list.
_KERNEL_PROBE_TIMEOUT = 0.6

# Timeouts for the reverse proxies into the sidecar / session children. Not
# ``None`` (biopb#420): a wedged upstream that accepts the connection but never
# answers must fail eventually, not hang the request forever. The ``read`` bound
# is per read-event, not total, and is set generously — every upstream buffers
# its whole response before sending (no long-poll / SSE / chunked-with-gaps path,
# so a large slice/render streams without inter-chunk stalls), so 300s only trips
# on a genuinely stuck upstream, never on legitimately large or slow-computed
# transfers. ``connect``/``write``/``pool`` are short since every hop is loopback.
_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)


def _kernel_state(health: dict) -> str:
    """Map a session child's ``/api/status`` health dict to a dashboard kernel
    state.

    The kernel is the heavy component and starts on demand, so the useful bit is
    attached-or-not. A *live* kernel always reports its live state (a stale
    ``start_error`` from an earlier, since-recovered attempt never masks it):
    ``ready`` (booted past its bootstrap probe), ``busy`` (executing), or
    ``starting`` (process up, not yet ready). A kernel that is not alive is
    ``error`` if it failed / died, else ``none`` (never started this session).
    """
    if health.get("alive"):
        if not health.get("ready"):
            return "starting"
        return "busy" if health.get("busy") else "ready"
    if health.get("start_error") or health.get("dead"):
        return "error"
    return "none"


# What a session probe reports when the child cannot be reached or understood.
# Every field degrades to its least-claiming value: an unknown kernel, no chat,
# and no stop offered — never a button that would 404.
_PROBE_UNKNOWN = {"kernel": "unknown", "chat": False, "agentless": False}


async def _probe_session(client: httpx.AsyncClient, rec: dict) -> dict:
    """Best-effort ``{kernel, chat, agentless}`` for one session.

    A single cheap loopback GET to the child's ``/api/status`` — which returns
    ``KernelHost.health()`` with no kernel round-trip and whose ``api`` observe
    root the control already proxies. Never raises: a missing port, an
    unreachable/slow child, a non-200, or unparseable JSON all degrade to
    :data:`_PROBE_UNKNOWN` so the session list is never blocked or truncated by a
    probe. httpx sets ``Host`` from the target (satisfying the child's loopback
    guard) and sends no ``Origin`` (passing its Origin guard), like the session
    proxy.

    The two booleans come off the same response rather than extra requests, and
    answer different questions. ``chat_enabled`` says what that session's page
    leads with, which is how the dashboard labels its link. ``agentless`` says
    who owns the reap — only a ``biopb mcp view`` viewer ends itself, a
    shim-owned child is its shim's to reap — which is how the dashboard decides
    whether to offer a stop. Both absent on an older child, which reads as
    False: an observe link and no stop button, the behaviour that predates them.
    """
    port = rec.get("port")
    if not port:
        return dict(_PROBE_UNKNOWN)
    url = _loopback_url(rec.get("host", "127.0.0.1"), port) + "/api/status"
    try:
        resp = await client.get(url, timeout=_KERNEL_PROBE_TIMEOUT)
        if resp.status_code != 200:
            return dict(_PROBE_UNKNOWN)
        health = resp.json()
        return {
            "kernel": _kernel_state(health),
            "chat": bool(health.get("chat_enabled")),
            "agentless": bool(health.get("agentless")),
        }
    except Exception:  # noqa: BLE001 - a probe is decorative; never fail the list
        return dict(_PROBE_UNKNOWN)


# --- launching a viewer session ------------------------------------------- #
#
# Invariant I1 (ARCHITECTURE.md) says the control observes sessions and never
# spawns them, and its reason is a *display* one: a session spawned from the
# control's frozen environment would put the agent's napari viewer somewhere the
# user is not (biopb/biopb-mcp#98). That reason covers a session serving an MCP
# client — whose spawner is that client's shim anyway — but not an agentless
# `biopb mcp view` viewer, whose only natural spawner is the person at the
# machine, and whose only way to exist until now was a terminal command. So the
# control may *launch* one, under two conditions that keep #98 shut:
#
#   * it refuses unless it could plausibly reach the user's screen
#     (:func:`_session_launch_gate`), and
#   * it launches `--view` specifically, never a plain http session. A non-view
#     session with a stale DISPLAY falls back to a virtual display and renders
#     where nobody can see it — #98 exactly. `--view` refuses instead.
#
# What it does not do is own the result: the child is detached, self-registers,
# and self-de-registers, so the *registry* still only ever observes, and a
# control restart does not close the user's viewer.

# How long POST /api/sessions/new waits for a launched viewer to publish itself.
# Generous because `--view` starts the kernel and opens the napari window
# *before* it registers, so this covers a cold Qt/napari import and not just the
# http stack the stdio shim waits on. Expiring is not a failure — the child is
# still coming up and the dashboard's own poll picks it up when it lands.
_VIEWER_START_TIMEOUT = 150.0
_VIEWER_POLL_INTERVAL = 0.25

# Characters of this launch's own output echoed back when the viewer dies before
# registering. That tail is the whole diagnosis (a dead display, a broken
# install) and the only one the dashboard can show.
_VIEWER_LOG_TAIL = 2000

# How many past launches' logs to keep. Matches the shim's session-log retention
# (``transport.session_log_keep``): enough to look back over a couple of failed
# attempts, not an unbounded pile of Qt chatter.
_VIEWER_LOG_KEEP = 5

# Cap on same-second name collisions before giving up on a log for this launch.
# Two dashboard launches inside one second is already a double-click; a hundred
# is a bug, and a viewer that starts without a log beats one that does not start.
_VIEWER_LOG_ATTEMPTS = 100


def _display_available() -> bool:
    """Whether this process could put a window on the user's screen.

    macOS and Windows always have a window server; on Linux it takes an X11 or
    Wayland session in *this* process's environment, because that is what a
    launched child inherits. A duplicate of biopb-mcp's ``_has_display`` rather
    than a call into it: the control may not import biopb-mcp (I2).

    Only ever used to decide whether to *offer* a launch. A set-but-dead
    ``DISPLAY`` (a control that outlived the login session that started it) is
    not catchable this cheaply and passes — which is safe, because the child
    re-checks, and fails and exits without registering rather than rendering
    somewhere invisible.
    """
    if sys.platform == "darwin" or os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _session_launch_gate(console_enabled: bool) -> tuple[bool, str | None]:
    """Whether this control may launch a viewer, and if not, why not.

    Both halves are properties of this process — its bind and its environment —
    so this is settled once at startup, not per request.

    ``console_enabled`` is the same "this control is loopback-bound" bit that
    gates the console and chat proxies, and it is required here for the same
    kind of reason: a remote browser cannot see a napari window that opens on
    the server. The message is returned rather than logged because the dashboard
    shows it in place of the button — a missing control with no explanation is
    the thing this is meant to avoid.
    """
    if not console_enabled:
        return False, (
            "this control is not loopback-bound, so a viewer it started would "
            "open on the server's display, not yours"
        )
    if not _display_available():
        return False, (
            "no display is available to this control plane "
            "($DISPLAY/$WAYLAND_DISPLAY are unset); start it from a desktop "
            "session, or run `biopb mcp view` in a terminal that has one"
        )
    return True, None


def _viewer_argv() -> list[str]:
    """The command that starts an agentless viewer session.

    ``--port 0`` so N viewers never collide on the configured MCP port. Run as a
    module of *this* interpreter — the supervisor's idiom for the data plane —
    so it resolves through the environment the control was installed into and
    needs no console script on PATH. Importing nothing of it here keeps I2.
    """
    return [sys.executable, "-m", "biopb_mcp.mcp", "--view", "--port", "0"]


def _prune_viewer_logs(log_dir, keep: int) -> None:
    """Keep only the newest *keep* launch logs. Best-effort.

    *keep* is passed rather than defaulted from the module constant: a default
    argument binds once at definition, so the constant would be frozen into this
    signature and could never be overridden.

    Run before this launch's file exists, so the count it leaves room for
    includes the one about to be created. A prune failure never affects the
    launch.
    """
    try:
        logs = sorted(
            log_dir.glob("viewer-*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old in logs[max(keep - 1, 0) :]:
        try:
            old.unlink()
        except OSError:
            pass


def _open_viewer_log():
    """Create this launch's own logfile; return ``(handle, path)``.

    **One file per launch**, not one shared file appended to. A shared log
    interleaves concurrent viewers, and lines that cannot be attributed to a
    process are no use for diagnosing a session that is still running — which is
    the case the failure tail does not cover. The shim reached the same
    conclusion for its per-session logs.

    Exclusive-create rather than a bare timestamp: two launches in the same
    second would otherwise land on one name and reintroduce the interleaving in
    miniature. On any failure the caller still spawns, with the child's output
    discarded — a viewer that starts without a log beats one that does not start.
    """
    try:
        log_dir = _locations.mcp_viewer_log_dir()
    except OSError:
        logger.warning("No viewer log dir; discarding child output", exc_info=True)
        return None, None
    _prune_viewer_logs(log_dir, _VIEWER_LOG_KEEP)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1, _VIEWER_LOG_ATTEMPTS + 1):
        suffix = "" if n == 1 else f"-{n}"
        path = log_dir / f"viewer-{stamp}{suffix}.log"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        except OSError:
            break
        # Binary + unbuffered, like every other owned-child log here: the fd is
        # inherited by the child (and its kernel), which emits arbitrary bytes
        # from native Qt/GL/dask writers, so it must not be a text wrapper.
        return os.fdopen(fd, "wb", buffering=0), path
    logger.warning("Could not create a viewer log in %s; discarding output", log_dir)
    return None, None


def _viewer_log_tail(path) -> str:
    """This launch's own output. The whole file *is* this launch, so no anchor
    is needed — that is what one file per launch buys."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", "replace").strip()[-_VIEWER_LOG_TAIL:]


def _launch_viewer(timeout: float) -> dict:
    """Start an agentless viewer session; wait for it to publish itself.

    Registration is the readiness signal, and it is an exact one: ``--view``
    runs its eager ``host.ensure_started()`` *before* ``_register_view_session``
    (biopb-mcp ``mcp/__main__.py``), so a record appearing means a napari window
    really opened, and a child that dies first never registers. The record is
    matched on the child's own pid — a viewer registers ``os.getpid()`` — so a
    session someone else starts concurrently is never mistaken for this one.

    Detached (:func:`detach_kwargs`) and then forgotten: the ``Popen`` handle is
    held only long enough to notice an early exit, never to reap or restart. A
    viewer is the user's window, and a control restart must not close it.

    Returns ``{"state": "started"|"starting"|"failed", ...}``; only ``failed``
    carries ``error`` and ``log``.
    """
    log_fh, log_path = _open_viewer_log()
    argv = _viewer_argv()
    logger.info("Launching viewer session: %s (log: %s)", " ".join(argv), log_path)
    # The environment is inherited: it carries the DISPLAY/XAUTHORITY/
    # WAYLAND_DISPLAY (or the Aqua session, or the Windows station) that decides
    # where the window lands. That inheritance is the whole risk #98 named and
    # the whole reason for the gate above. The one addition tells the child where
    # its own output went, so `server_status` can name the file rather than
    # guessing the canonical one -- the same thing the shim does for its child.
    env = None
    if log_path is not None:
        env = {**os.environ, _locations.MCP_SESSION_LOG_ENV: str(log_path)}
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_fh is not None else subprocess.DEVNULL,
            env=env,
            **detach_kwargs(),
        )
    except OSError as exc:
        logger.exception("Could not launch a viewer session")
        return {"state": "failed", "error": str(exc), "log": "", "log_path": None}
    finally:
        if log_fh is not None:
            log_fh.close()  # the child holds its own dup

    deadline = time.monotonic() + timeout
    while True:
        for rec in _sessions.list_sessions():
            session_id = rec.get("session_id")
            if session_id and rec.get("pid") == proc.pid:
                logger.info("Viewer session %s is up (pid %s)", session_id, proc.pid)
                return {
                    "state": "started",
                    "session_id": session_id,
                    "observe_url": f"/session/{session_id}/observe",
                }
        code = proc.poll()
        if code is not None:
            logger.error("Viewer session exited with code %s before starting", code)
            return {
                "state": "failed",
                "error": f"the viewer exited with code {code} before it opened",
                "log": _viewer_log_tail(log_path) if log_path else "",
                # Named so a tail that was truncated, or empty because no log
                # could be opened, still leads somewhere.
                "log_path": str(log_path) if log_path else None,
            }
        if time.monotonic() >= deadline:
            # Still alive, just slow (a cold napari import on a loaded box). Say
            # so instead of failing: the dashboard polls /api/sessions anyway and
            # will show it the moment it registers.
            logger.info("Viewer session still starting after %.0fs", timeout)
            return {
                "state": "starting",
                "log_path": str(log_path) if log_path else None,
            }
        time.sleep(_VIEWER_POLL_INTERVAL)


def build_app(
    supervisor: DataPlaneSupervisor,
    ensure_timeout: float,
    data_web_url: str,
    token: str | None = None,
    static_dir: str | Path | None = None,
    console_enabled: bool = False,
    url_prefix: str | None = None,
) -> Starlette:
    """Build the control-plane ASGI app.

    ``data_web_url`` is the loopback base URL of the supervised tensor server's
    HTTP sidecar; the ``/data_plane`` namespace reverse-proxies there. ``token``
    is the data-plane access token (``None`` in local mode, where every listener
    is loopback-bound); the ``/api/*`` gate enforces it when set, else falls back
    to a loopback Host check. ``static_dir`` is the built ``web/`` bundle (``web/packages/app/
    dist``); when present the control serves it at its root as the single web
    origin — the dashboard (``/``), the dataviewer (``/viewer``), and each
    session's observe shell (``/session/<id>/observe``) are all React routes of
    that one SPA. Split out from :func:`serve_control_api` so it is unit-testable
    against a fake upstream without binding uvicorn.

    ``console_enabled`` proxies ``/session/<id>/console/*`` — the user console,
    which **executes code in that session's kernel**. Default off, and the
    decision is the caller's because only it knows this control's bind address:
    :func:`serve_control_api` derives it from a loopback bind, so a
    network-reachable control never carries the console however it is
    configured downstream. Deliberately *not* delegated to the session child —
    the proxy hop strips Host and Origin, so the child cannot tell a browser
    from this trusted loopback hop and cannot make this call.

    ``url_prefix`` publishes this origin under a path prefix (``/node/<host>/
    <port>`` — an Open OnDemand interactive app) rather than at ``/``: requests
    under it are stripped before routing and the served SPA shell is rewritten to
    point back at it. ``None`` (the default) is the plain root origin and changes
    nothing. It is normalized here, the single consumer.
    """
    session_roots = _session_proxy_roots(console_enabled)
    # Whether this control may launch a viewer session for the dashboard, and
    # the sentence explaining it when it may not (both settled here: they read
    # this process's bind and environment, neither of which changes).
    can_start_session, start_session_blocked = _session_launch_gate(console_enabled)
    url_prefix = normalize_url_prefix(url_prefix)

    # The built SPA bundle the control serves at its root (None / missing ->
    # API-only: the control still answers /health + /api/* + the proxies, but
    # serves no web UI). Resolved once; index.html is the SPA shell every
    # non-API, non-proxy GET falls back to.
    web_root = Path(static_dir) if static_dir else None
    if web_root is not None and not (web_root / "index.html").is_file():
        logger.warning("web bundle not found at %s; serving API only", web_root)
        web_root = None

    # Under a URL prefix the shell is served rewritten (see _rewrite_shell_html).
    # Computed once here rather than per request: the bundle is static, and every
    # non-asset GET hands back this same document. An unreadable index.html
    # degrades to the plain FileResponse instead of failing the whole app.
    shell_html: str | None = None
    if web_root is not None and url_prefix:
        index = web_root / "index.html"
        try:
            shell_html = _rewrite_shell_html(
                index.read_text(encoding="utf-8"), url_prefix
            )
        except (OSError, UnicodeDecodeError):
            logger.exception("could not rewrite %s for %s", index, url_prefix)

    # One pooled client to the sidecar for the process lifetime. Held in a
    # closure (not ``app.state``) because the proxy runs inside a *mounted*
    # sub-app whose ``request.app`` is the sub-app, not this one -- ``app.state``
    # would read the wrong app's state. Closed by the app lifespan below. The
    # generous ``_PROXY_TIMEOUT`` (not ``None``) keeps large slice responses
    # flowing while ensuring a wedged sidecar fails cleanly (biopb#420).
    proxy_client = httpx.AsyncClient(base_url=data_web_url, timeout=_PROXY_TIMEOUT)

    # A second pooled client for the session proxy. No base_url: each session's
    # target is a *different* dynamic loopback port resolved per-request from the
    # registry, so the proxy builds absolute URLs (httpx pools connections per
    # host:port automatically). Also closed by the lifespan below.
    session_client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)

    # --- control-owned endpoints (sync: they take the supervisor lock and do a
    # blocking TCP liveness probe, so Starlette runs them in its threadpool) --- #

    def health(_request: Request) -> JSONResponse:
        # `auth_required` is the SPA's public probe: the browser bundle + this
        # endpoint stay unauthenticated always, and the app reads this to decide
        # whether to gate itself behind the unlock page. It tracks the *token*,
        # not the network mode: always true in remote (which requires one), and
        # true in local mode too when an optional token was supplied.
        # `console_enabled` rides the same public probe for the same reason: the
        # observe page must know whether to offer a code cell before it renders
        # one, and an editor whose every POST 404s is worse than no editor. It
        # discloses nothing a caller cannot already infer -- reaching this
        # endpoint from off-box *is* the evidence that the bind is public and the
        # console therefore off.
        return JSONResponse(
            {
                "control": "ok",
                "auth_required": token is not None,
                "console_enabled": console_enabled,
                "data_plane": supervisor.snapshot(),
            }
        )

    def data_plane_ensure(request: Request) -> JSONResponse:
        # The client passes ?client_timeout=<its HTTP timeout>; cap our wait
        # below it so we return a verdict before the client gives up (and wrongly
        # treats a slow-but-working control plane as unreachable).
        try:
            client_timeout = float(request.query_params.get("client_timeout", "0"))
        except ValueError:
            client_timeout = 0.0
        wait = _bounded_ensure_wait(ensure_timeout, client_timeout)
        # ensure()/_spawn_locked count a spawn failure toward the backoff and do
        # not raise, but wrap defensively so any unexpected error still returns a
        # clean JSON verdict (with the snapshot reflecting the counted failure)
        # rather than an unhandled 500.
        try:
            supervisor.ensure()
            supervisor.wait_until_up(wait)
            return JSONResponse({"data_plane": supervisor.snapshot()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("data_plane/ensure failed")
            return JSONResponse(
                {"error": str(exc), "data_plane": supervisor.snapshot()},
                status_code=500,
            )

    def data_plane_stop(_request: Request) -> JSONResponse:
        # Full teardown of the data plane (want=False): the control stays up, but
        # its supervised child is stopped and won't be respawned until an ensure.
        try:
            supervisor.stop()
            return JSONResponse({"data_plane": supervisor.snapshot()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("data_plane/stop failed")
            return JSONResponse(
                {"error": str(exc), "data_plane": supervisor.snapshot()},
                status_code=500,
            )

    def data_plane_restart(request: Request) -> JSONResponse:
        # Bounce the plane: stop() (want=False, so a racing supervision tick backs
        # off instead of seeing the down port as a conflict) then ensure() flips
        # want back on and spawns a fresh child, bounded like /ensure so we answer
        # before the client's HTTP timeout.
        try:
            client_timeout = float(request.query_params.get("client_timeout", "0"))
        except ValueError:
            client_timeout = 0.0
        wait = _bounded_ensure_wait(ensure_timeout, client_timeout)
        try:
            supervisor.stop()
            supervisor.ensure()
            supervisor.wait_until_up(wait)
            return JSONResponse({"data_plane": supervisor.snapshot()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("data_plane/restart failed")
            return JSONResponse(
                {"error": str(exc), "data_plane": supervisor.snapshot()},
                status_code=500,
            )

    def api_data_plane_logs(request: Request) -> JSONResponse:
        # The dashboard /logs page polls this: the tail of the data-plane
        # subprocess's stdout/stderr log (the file the supervisor writes the tensor
        # server to). Read is bounded in both lines and bytes (see _tail_file), so
        # tailing a huge log stays cheap; no-store so each poll sees fresh output.
        # Never raises -- a bad read degrades to an error field, not a 500 trace.
        try:
            n = int(request.query_params.get("lines", _LOG_TAIL_DEFAULT_LINES))
        except (TypeError, ValueError):
            n = _LOG_TAIL_DEFAULT_LINES
        n = max(1, min(n, _LOG_TAIL_MAX_LINES))
        headers = {"Cache-Control": "no-store"}
        path = supervisor.log_path
        if path is None:
            return JSONResponse(
                {
                    "path": None,
                    "exists": False,
                    "lines": [],
                    "truncated": False,
                    "note": "data plane logs to the control's stderr "
                    "(no log file configured)",
                },
                headers=headers,
            )
        try:
            if not path.exists():
                return JSONResponse(
                    {
                        "path": str(path),
                        "exists": False,
                        "lines": [],
                        "truncated": False,
                    },
                    headers=headers,
                )
            lines, truncated = _tail_file(path, n, _LOG_TAIL_MAX_BYTES)
            return JSONResponse(
                {
                    "path": str(path),
                    "exists": True,
                    "size": path.stat().st_size,
                    "lines": lines,
                    "truncated": truncated,
                },
                headers=headers,
            )
        except OSError as exc:
            logger.info("data plane log read failed: %s", exc)
            return JSONResponse(
                {
                    "path": str(path),
                    "exists": False,
                    "lines": [],
                    "error": f"could not read log: {exc}",
                },
                status_code=500,
                headers=headers,
            )

    def api_status(_request: Request) -> JSONResponse:
        # What the dashboard polls: the control is up (it answered), the data
        # plane's supervisor snapshot, and how many sessions are live. Sync (the
        # snapshot probes the port and list_sessions() touches the filesystem), so
        # Starlette runs it in the threadpool.
        #
        # __version__ is read here, not imported at module scope: __init__ binds
        # it and *then* imports _run -> _control, so a module-level `from . import
        # __version__` only works while those two stay in that order.
        from . import __version__

        # `can_start_session` rides here (not /health) because the button it
        # gates lives on the token-gated dashboard, and the reason rides with it
        # so the page can say *why* there is no button instead of just not
        # having one.
        return JSONResponse(
            {
                "control": "ok",
                "version": __version__,
                "data_plane": supervisor.snapshot(),
                "sessions": len(_sessions.list_sessions()),
                "can_start_session": can_start_session,
                "start_session_blocked": start_session_blocked,
            }
        )

    async def api_sessions(_request: Request) -> JSONResponse:
        # The live MCP sessions, newest first, projected to what the dashboard
        # needs — the id, when it started, its loopback port, the control-relative
        # observe link, and a best-effort "kernel" state (the heavy on-demand
        # component, probed concurrently over one cheap GET each; see
        # _probe_session). list_sessions() self-heals (prunes dead/reused records)
        # on read, so a stale session never lingers on the page. Async so the
        # per-session probes fan out concurrently rather than serializing.
        records = [rec for rec in _sessions.list_sessions() if rec.get("session_id")]
        probes = await asyncio.gather(
            *(_probe_session(session_client, rec) for rec in records)
        )
        sessions = [
            {
                "session_id": rec["session_id"],
                "started_at": rec.get("started_at"),
                "port": rec.get("port"),
                "observe_url": f"/session/{rec['session_id']}/observe",
                "kernel": probe["kernel"],
                # Whether that page will lead with the chat client: the child
                # mounts it (only a `biopb mcp view` viewer does) AND this
                # control will proxy /chat/*. Both halves, as ObservePage needs
                # both — answered here so the dashboard needs no second probe.
                "chat": probe["chat"] and console_enabled,
                # Whether the session serves a stop verb. Not gated on the bind
                # the way chat is: the route lives under `api`, which is proxied
                # everywhere, and stopping a session is no more destructive than
                # the kernel restart already there.
                "can_stop": probe["agentless"],
            }
            for rec, probe in zip(records, probes, strict=True)
        ]
        return JSONResponse({"sessions": sessions})

    def api_session_new(request: Request) -> JSONResponse:
        # Launch an agentless viewer on this machine's display. Sync: it spawns
        # and then blocks polling the registry, so Starlette runs it in the
        # threadpool. Gated at startup rather than here (nothing it reads can
        # change), and 409 rather than 403 — the request is fine, this
        # deployment just cannot serve it.
        if not can_start_session:
            return JSONResponse({"error": start_session_blocked}, status_code=409)
        # The client passes ?client_timeout=<its HTTP timeout>; bound our wait
        # below it so a slow-but-working launch comes back as "starting" rather
        # than as a browser-side timeout with a viewer still coming up behind it.
        try:
            client_timeout = float(request.query_params.get("client_timeout", "0"))
        except ValueError:
            client_timeout = 0.0
        wait = _bounded_ensure_wait(_VIEWER_START_TIMEOUT, client_timeout)
        try:
            return JSONResponse(_launch_viewer(wait))
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("session launch failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    def api_agents(_request: Request) -> JSONResponse:
        # The supported MCP clients and whether biopb is registered with each.
        # Reads are subprocess-free (biopb._agents), so the dashboard can poll
        # this without spawning anything; still sync (filesystem), so Starlette
        # runs it in the threadpool.
        try:
            return JSONResponse({"agents": _agents.statuses()})
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("api/agents failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    def _agent_action(request: Request, action) -> JSONResponse:
        # Register/unregister biopb with one client, returning its fresh status.
        # A bad request (unknown client, unparseable client config) is the
        # caller's fault -> 400; anything else -> 500. Both write user config
        # files (Claude Code via its CLI), so these are token-gated /api/* verbs.
        agent_id = request.path_params["agent_id"]
        try:
            return JSONResponse({"agent": action(agent_id)})
        except _agents.AgentError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("agent %s failed", getattr(action, "__name__", "action"))
            return JSONResponse({"error": str(exc)}, status_code=500)

    def agent_register(request: Request) -> JSONResponse:
        return _agent_action(request, _agents.register)

    def agent_unregister(request: Request) -> JSONResponse:
        return _agent_action(request, _agents.unregister)

    def api_algorithms(_request: Request) -> JSONResponse:
        # The configured algorithm-plane servers (biopb.image ProcessImage
        # servicers listed in the biopb-mcp config) with a live health + ops
        # probe. Read-only inspection — no lifecycle control (the pending
        # algorithm plane).
        # Sync: statuses() reads a config file and makes blocking gRPC calls (run
        # concurrently, bounded by one probe timeout), so Starlette runs it in the
        # threadpool. Polled on demand (a dashboard button), not on the interval,
        # because it dials external servers.
        #
        # `plugins` folds in the kernel-namespace "bring your own tool" surface
        # (biopb/biopb-mcp#92) -- a static, stdlib-only listing of the ~/.config/
        # biopb/kernel/ files and biopb_mcp.namespace packages, read (never
        # executed, invariant I2) via _kernel_plugins. It renders in the same
        # algorithm-plane card; a summary read failure degrades to empty rather
        # than 500-ing the servers view alongside it.
        try:
            servers = _algorithms.statuses()
        except Exception as exc:  # noqa: BLE001 - report, never crash the handler
            logger.exception("api/algorithms failed")
            return JSONResponse({"error": str(exc)}, status_code=500)
        try:
            plugins = _kernel_plugins.summary()
        except Exception:  # noqa: BLE001 - inspector is never-raise; belt-and-braces
            logger.exception("api/algorithms plugin summary failed")
            plugins = {"dir": "", "files": [], "entry_points": []}
        return JSONResponse({"servers": servers, "plugins": plugins})

    def api_mcp_config(_request: Request) -> JSONResponse:
        # The biopb-mcp settings editor's backing read: the raw on-disk config +
        # its path + the JSON Schema (labels/help/bounds), mirroring the tensor
        # sidecar's GET /api/config so the same schema-driven admin UI renders it.
        # The control OWNS this because the config is global (~/.config/biopb/
        # mcp-config.json) while mcp sessions are ephemeral/dynamic-port -- none of
        # them owns the file. biopb_mcp is soft-imported (only for the schema): the
        # lean control does not hard-depend on it (invariant I2), but a real biopb
        # deployment always co-installs it. mcp_config_path lives in core biopb.
        from biopb._locations import mcp_config_path

        try:
            from biopb_mcp._config_schema import build_mcp_config_schema
        except Exception as exc:  # noqa: BLE001 - biopb-mcp not installed here
            return JSONResponse(
                {"error": f"biopb-mcp is not installed: {exc}"}, status_code=501
            )
        p = mcp_config_path()
        raw: dict = {}
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            except (OSError, ValueError) as exc:
                return JSONResponse(
                    {"error": f"config on disk is unreadable: {exc}"}, status_code=500
                )
        # no-store: a config editor must always see the live file, never a cached
        # GET (a stale empty {} cached before the file was populated would render
        # the wrong config and clobber it on save).
        return JSONResponse(
            {"path": str(p), "config": raw, "schema": build_mcp_config_schema()},
            headers={"Cache-Control": "no-store"},
        )

    async def api_mcp_config_save(request: Request) -> JSONResponse:
        # Validate + write the biopb-mcp config. Validation calls biopb-mcp's own
        # config_problems -- the exact check its load path runs, cross-field rules
        # included -- so "the form accepted it" == "biopb-mcp will accept it", with
        # no jsonschema dependency in the lean control and nothing for this handler
        # to restate. The difference is only what happens next: biopb-mcp clamps to
        # defaults at load, this endpoint rejects, because a human is here to fix
        # it (biopb/biopb#34). Changes apply to the NEXT session (each session
        # reads config fresh at bootstrap), so there is no server to restart --
        # unlike the data plane.
        try:
            from biopb_mcp._config import config_problems, save_config
        except Exception as exc:  # noqa: BLE001 - biopb-mcp not installed here
            return JSONResponse(
                {"error": f"biopb-mcp is not installed: {exc}"}, status_code=501
            )
        from biopb._locations import mcp_config_path

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {"detail": "Request body is not valid JSON"}, status_code=422
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {"detail": "Config body must be a JSON object"}, status_code=422
            )

        errors = [p.as_dict() for p in config_problems(body)]
        if errors:
            errors.sort(key=lambda d: d["path"])
            return JSONResponse(
                {"detail": "Config failed validation", "errors": errors},
                status_code=422,
            )
        try:
            save_config(body)
        except OSError as exc:
            return JSONResponse(
                {"error": f"could not write config: {exc}"}, status_code=500
            )
        return JSONResponse({"saved": True, "path": str(mcp_config_path())})

    def _serve_shell() -> Response:
        # The SPA shell (index.html) every non-API GET falls back to; the React
        # router then renders the right surface for the URL. web_root is checked
        # by the caller, so index.html exists here.
        assert web_root is not None
        if shell_html is not None:
            return HTMLResponse(shell_html)  # rewritten for url_prefix
        return FileResponse(web_root / "index.html")

    async def spa(request: Request) -> Response:
        # Catch-all for the single web origin: serve a real static file from the
        # bundle when the path names one (/assets/<hash>, /favicon.ico, …), else
        # the SPA shell so a deep link like /viewer or /admin boots the router.
        # Registered LAST, after every API route and proxy mount, so it never
        # shadows them.
        if web_root is None:
            return JSONResponse({"error": "web bundle not installed"}, status_code=404)
        rel = request.path_params["path"].lstrip("/")
        if rel:
            candidate = (web_root / rel).resolve()
            # Contain traversal: only serve files that resolve inside web_root.
            if web_root.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
        return _serve_shell()

    # --- reverse proxy into the tensor server's HTTP sidecar ---------------- #
    # Handlers forward the *mount-relative* path (``Mount`` has already stripped
    # the ``/data_plane[/viewer]`` prefix into ``path_params``), so the sidecar
    # always sees a root-relative path regardless of which mount matched.

    async def proxy(request: Request) -> Response:
        target = "/" + request.path_params["path"]
        # Append the query only when present -- an empty one would render a bare
        # trailing "?" that changes the path the sidecar sees.
        if request.url.query:
            target = f"{target}?{request.url.query}"
        # Drop Host so httpx sets it from base_url; forward everything else
        # (Authorization / X-Biopb-Token pass through, the sidecar re-validates).
        headers = [(k, v) for k, v in request.headers.raw if k.lower() != b"host"]
        # Request bodies here are small JSON (e.g. POST /api/slice params); read
        # fully so GETs carry no chunked body. Responses (images) are streamed.
        body = await request.body()
        upstream = proxy_client.build_request(
            request.method, target, headers=headers, content=body
        )
        try:
            resp = await proxy_client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            # Any upstream/transport failure -- refused connect, an upstream that
            # accepts then dies mid-response (RemoteProtocolError/ReadError), or a
            # read/connect timeout on a wedged sidecar -- is a gateway error, not a
            # control-plane bug: surface a clean 502, never a 500 traceback
            # (biopb#420). A failure *after* the headers stream (in aiter_raw) can't
            # be turned into a 502 anymore, but the timeout still bounds the hang.
            logger.info("data plane proxy to %s failed: %s", target, exc)
            return JSONResponse({"error": "data plane not reachable"}, status_code=502)
        # HTTP headers are latin-1 on the wire (RFC 9110 / ASGI). A header value
        # may carry a legitimate high byte (e.g. a non-ASCII Content-Disposition
        # filename); decoding it as UTF-8 would raise and 500 the proxy. latin-1
        # is total and round-trips -- Starlette re-encodes response headers as
        # latin-1 too.
        resp_headers = [
            (k, v)
            for k, v in resp.headers.raw
            if k.decode("latin-1").lower() not in _HOP_BY_HOP
        ]
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={k.decode("latin-1"): v.decode("latin-1") for k, v in resp_headers},
            background=BackgroundTask(resp.aclose),
        )

    async def session_proxy(request: Request) -> Response:
        # The outer Mount captured {session_id}; the inner catch-all captured the
        # rest into {path} (both survive in path_params). Resolve the session to a
        # live loopback target via the registry — an unknown/dead one is a clean
        # 404 (and the dead record is pruned by resolve()).
        session_id = request.path_params["session_id"]
        sub_path = request.path_params["path"]
        # Allowlist the session data API only — the observe page itself is
        # the control-served SPA shell (session_observe below), so only /api/*
        # proxies here (plus /console/* where the console is enabled). The
        # child's /mcp agent transport is deliberately off this origin — agents
        # reach it directly on the child's own loopback port (stdio shim bridge /
        # `biopb mcp view`), never via the control — and this hop strips /mcp's
        # entire auth (Host/Origin), so exposing it would be an RCE hole on the
        # public origin. Require an allowed first segment AND reject any
        # parent-traversal, so no path (raw, encoded, or dot-collapsed by httpx)
        # can escape an allowed root into /mcp.
        segments = sub_path.split("/")
        if segments[0] not in session_roots or ".." in segments:
            return JSONResponse({"error": "not found"}, status_code=404)
        # The execute-capable roots are POST-only *here*, not merely in the
        # children that happen to serve them that way. The CSRF gate upstream
        # only inspects unsafe methods -- correct, since safe verbs must not
        # change state -- so a cross-site GET (`<img
        # src=".../console/execute?code=...">`) is forwarded unchecked, exactly
        # as a GET to /api/jobs is. That is harmless only while nothing under
        # these roots acts on a GET, which is a promise about code living in
        # another package. Pinning the method here makes the roots' claim
        # ("reaching an RCE requires a request a hostile page cannot forge") true
        # at the layer that makes it, and fences off a future GET route that
        # would silently reopen it. Checked before resolving the session, so it
        # discloses nothing about which ids exist.
        if segments[0] in _SESSION_POST_ONLY_ROOTS and request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        rec = _sessions.resolve(session_id)
        if rec is None:
            return JSONResponse(
                {"error": f"session {session_id!r} not found or ended"},
                status_code=404,
            )
        base = _loopback_url(rec.get("host", "127.0.0.1"), rec["port"])
        target = base + "/" + sub_path
        if request.url.query:
            target = f"{target}?{request.url.query}"
        # Drop Host AND Origin: httpx sets Host from the target (127.0.0.1:<port>,
        # matching the child's loopback Host allowlist) and an absent Origin
        # passes the child's Origin guard, so the trusted control->child hop is
        # accepted whatever external host the browser used. Everything else
        # forwards verbatim.
        headers = [
            (k, v)
            for k, v in request.headers.raw
            if k.lower() not in (b"host", b"origin")
        ]
        body = await request.body()
        upstream = session_client.build_request(
            request.method, target, headers=headers, content=body
        )
        try:
            resp = await session_client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            # Same as the data-plane proxy: any upstream/transport failure or
            # timeout on a wedged session child is a clean 502, not a 500
            # traceback (biopb#420).
            logger.info("session proxy to %s failed: %s", target, exc)
            return JSONResponse({"error": "session not reachable"}, status_code=502)
        resp_headers = [
            (k, v)
            for k, v in resp.headers.raw
            if k.decode("latin-1").lower() not in _HOP_BY_HOP
        ]
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers={k.decode("latin-1"): v.decode("latin-1") for k, v in resp_headers},
            background=BackgroundTask(resp.aclose),
        )

    async def session_observe(request: Request) -> Response:
        # The observe *page* is the control-served SPA shell (React ObservePage);
        # only its /api/* data calls proxy to the child (session_proxy above).
        # Resolve the session first so an unknown/dead id is a clean 404 rather
        # than a shell wired to a session that no longer exists.
        if web_root is None:
            return JSONResponse({"error": "web bundle not installed"}, status_code=404)
        session_id = request.path_params["session_id"]
        if _sessions.resolve(session_id) is None:
            return JSONResponse(
                {"error": f"session {session_id!r} not found or ended"},
                status_code=404,
            )
        return _serve_shell()

    # One sub-app proxying to the sidecar root, mounted at /data_plane (the data
    # plane's /api, health). It strips its prefix, so the sidecar needs no
    # knowledge of the namespace. (The dataviewer's static assets are no longer
    # proxied out of the sidecar — the control serves the whole SPA itself, so
    # there is no /data_plane/viewer mount.)
    sidecar = Starlette(
        routes=[
            Route(
                "/{path:path}",
                proxy,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
        ]
    )

    # A session sub-app mounted under /session/{session_id}: /observe serves the
    # control's SPA shell, everything else (/api/*) proxies to the child.
    session_app = Starlette(
        routes=[
            Route("/observe", session_observe, methods=["GET"]),
            Route(
                "/{path:path}",
                session_proxy,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
        ]
    )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/sessions", api_sessions, methods=["GET"]),
        Route("/api/sessions/new", api_session_new, methods=["POST"]),
        Route("/api/data_plane/ensure", data_plane_ensure, methods=["POST"]),
        Route("/api/data_plane/stop", data_plane_stop, methods=["POST"]),
        Route("/api/data_plane/restart", data_plane_restart, methods=["POST"]),
        Route("/api/data_plane/logs", api_data_plane_logs, methods=["GET"]),
        Route("/api/agents", api_agents, methods=["GET"]),
        Route("/api/agents/{agent_id}/register", agent_register, methods=["POST"]),
        Route("/api/agents/{agent_id}/unregister", agent_unregister, methods=["POST"]),
        Route("/api/algorithms", api_algorithms, methods=["GET"]),
        Route("/api/mcp_config", api_mcp_config, methods=["GET"]),
        Route("/api/mcp_config", api_mcp_config_save, methods=["PUT"]),
        Mount("/data_plane", sidecar),
        # Per-session observe: /session/<id>/observe (SPA shell) + /session/<id>/
        # api/* (proxied). The {session_id} convertor is slash-free (session ids
        # are), so it stops at the first slash and the remainder falls to the
        # session sub-app.
        Mount("/session/{session_id}", session_app),
        # The single web origin's catch-all: static asset or SPA shell. LAST so
        # every API route and proxy mount above wins first.
        Route(
            "/{path:path}",
            spa,
            methods=["GET", "HEAD"],
        ),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        try:
            yield
        finally:
            await proxy_client.aclose()
            await session_client.aclose()

    # The /api/* auth gate wraps the whole app but acts only on /api/* (pure ASGI,
    # so the streaming proxies pass through untouched). The prefix stripper goes
    # OUTSIDE it: the gate reads scope["path"] directly, so a still-prefixed
    # /node/h/p/api/... would not match its startswith("/api/") and would reach
    # the verb ungated.
    middleware = []
    if url_prefix:
        middleware.append(Middleware(_URLPrefixMiddleware, prefix=url_prefix))
    middleware.append(
        Middleware(_ControlAuthMiddleware, token=token, session_roots=session_roots)
    )
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


class _ControlServer:
    """Shutdown handle the caller holds for teardown.

    Wraps the :class:`uvicorn.Server` (run in a background thread) so ``_run``'s
    teardown keeps its existing ``server.shutdown()`` call. Signalling
    ``should_exit`` unwinds uvicorn's serve loop from another thread.
    """

    def __init__(self, server: uvicorn.Server) -> None:
        self._server = server

    def shutdown(self) -> None:
        self._server.should_exit = True


def serve_control_api(
    host: str,
    port: int,
    supervisor: DataPlaneSupervisor,
    ensure_timeout: float,
    data_web_url: str | None = None,
) -> tuple[_ControlServer, threading.Thread]:
    """Start the control-plane web origin on ``host:port`` in a background thread.

    Binds the listening socket **eagerly, in the caller's thread**, so a port
    clash surfaces here (a control plane already running) instead of in a
    detached uvicorn thread — preserving the old stdlib server's fail-fast
    contract. ``data_web_url`` defaults to the supervised tensor server's sidecar
    (loopback of its configured web host/port); pass it explicitly in tests.

    Returns ``(server, thread)``; the caller stops it with ``server.shutdown()``.
    """
    spec = supervisor._spec
    if data_web_url is None:
        data_web_url = _loopback_url(spec.web_host, spec.web_port)

    # The data-plane token gates the control's own /api/* too (single
    # origin). None in local mode -> the gate falls back to a loopback Host check
    # instead.
    #
    # The user console (arbitrary code in a session's kernel) rides this origin
    # only when the origin is same-machine. Derived from *this* listener's bind
    # through the shared predicate, not from --remote or the plane's bind: what
    # decides is who can reach this web front. Deliberately not gated by the
    # token instead — the data-plane token authorizes reading pixels and is
    # readable from the local credential file by design (biopb/biopb#470); fine
    # for viewing, and not a credential to trade for a shell. Remote console, if
    # ever wanted, needs its own.
    console_enabled = not _web_auth.host_is_public_bind(host)
    app = build_app(
        supervisor,
        ensure_timeout,
        data_web_url,
        token=spec.token,
        static_dir=spec.static_dir,
        console_enabled=console_enabled,
        url_prefix=spec.url_prefix,
    )
    if not console_enabled:
        logger.info(
            "session console disabled: control bound to %s (not loopback)", host
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform == "win32":
        # On Windows SO_REUSEADDR lets a *second* bind to the same port SUCCEED and
        # then delivers incoming connections to one of the sockets nondeterministically
        # -- so it would defeat the single-owner guarantee a concurrent `control start`
        # relies on (you could end up with two live controls on 8813). SO_EXCLUSIVEADDRUSE
        # instead makes the second bind fail with EADDRINUSE, which is the behavior POSIX
        # SO_REUSEADDR already gives here (it only reuses a TIME_WAIT port, never a live
        # bind). So the eager bind stays the true arbiter of "one control per port" on
        # both platforms.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))  # raises OSError on clash -> caller reports it
    sock.listen(128)

    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        # No websocket routes: the control proxies HTTP only, so uvicorn need not
        # load a websockets impl at all.
        ws="none",
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        # uvicorn skips signal-handler installation off the main thread, so this
        # is safe to run in a daemon thread alongside the supervision loop.
        server.run(sockets=[sock])

    thread = threading.Thread(target=_run, name="control-api", daemon=True)
    thread.start()
    logger.info("Control plane origin listening on http://%s:%d", host, port)
    return _ControlServer(server), thread
