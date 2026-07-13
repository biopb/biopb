"""Shared, stdlib-only web-auth predicates for the single-origin front.

The control (``biopb-control``), the tensor HTTP sidecar
(``biopb-tensor-server``), and the observe UI (``biopb-mcp``) all need the *same*
token / same-origin / loopback decisions to gate their web surfaces, but none can
import another — invariant I2 forbids the control from importing the tensor server
or ``biopb-mcp``, and ``biopb-mcp`` cannot import the (PyPI-absent) tensor server.
So the **policy** lives here in the core ``biopb`` SDK — the one place all three
already depend — as framework-agnostic predicates over a header *getter*, and each
package keeps a thin binding to its own web stack (a FastAPI dependency, a
Starlette middleware, an MCP route wrapper).

Kept **stdlib-only** (like ``_config_control`` / ``_config_sessions`` / ``_proc``)
so importing it never drags a web framework into the SDK/CLI/client import path.

The control binds these as a Starlette middleware and the tensor sidecar as
FastAPI dependencies (``check_token`` / ``_require_same_origin`` / the render-WS
guard), so the two are provably in agreement rather than carrying copies that can
drift.
"""

from __future__ import annotations

import re
import secrets
from typing import Callable, Optional

# A case-insensitive header lookup, e.g. Starlette ``request.headers.get`` — the
# only coupling to the caller's framework, so these predicates stay pure.
HeaderGetter = Callable[[str], Optional[str]]

# Hosts honored when no token is configured (local mode, loopback-only). Any
# port is allowed; only the host part is checked.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Sec-Fetch-Site values that are NOT a cross-site request (so not a CSRF vector).
_SAFE_FETCH_SITES = frozenset({"same-origin", "none"})

_BEARER_PREFIX = "Bearer "


def extract_bearer(get: HeaderGetter) -> str:
    """The token from ``Authorization: Bearer <t>`` or the ``X-Biopb-Token``
    header (Bearer wins), or ``""`` if neither is present. Mirrors the sidecar's
    two accepted schemes."""
    auth = get("authorization") or ""
    if auth.startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX) :]
    return get("x-biopb-token") or ""


def token_valid(get: HeaderGetter, expected: Optional[str]) -> bool:
    """Timing-safe check that the request carries ``expected``.

    A falsy ``expected`` (no token configured — local mode, where every listener
    is loopback-bound) means "no token is enforced" and returns ``True``,
    matching the sidecar's ``self.token is None`` path. Callers that want a
    loopback backstop for that no-token case apply :func:`host_is_loopback`
    themselves.
    """
    if not expected:
        return True
    provided = extract_bearer(get)
    return secrets.compare_digest(provided.encode(), expected.encode())


def token_valid_with_query(
    get: HeaderGetter, query_get: HeaderGetter, expected: Optional[str]
) -> bool:
    """Like :func:`token_valid`, but also accepts the token from a ``token``
    query parameter when no header carries it.

    Browsers cannot set request headers on a WebSocket handshake, so the token
    arrives as ``?token=<t>``; header schemes still take precedence. ``query_get``
    is a getter over the query params (e.g. Starlette ``ws.query_params.get``).
    Mirrors the sidecar's ``_ws_authorized`` and the control's render-WS proxy,
    both of which pass the token this way.
    """
    if not expected:
        return True
    provided = extract_bearer(get) or (query_get("token") or "")
    return secrets.compare_digest(provided.encode(), expected.encode())


_TOKEN_CHARS = re.compile(r"[A-Za-z0-9_\-]+")
_TOKEN_MIN_LEN = 16
_TOKEN_MAX_LEN = 128


def valid_token(token: Optional[str]) -> bool:
    """Whether ``token`` is a well-formed access token.

    The single shared rule for what counts as a *usable* token — 16–128 URL-safe
    characters (``[A-Za-z0-9_-]``), surrounding whitespace ignored. The tensor
    ``launch`` and the control's ``_resolve_mode`` both apply it so the two layers
    cannot disagree on whether a supplied token is acceptable: without a shared
    rule, one layer could enforce a ``--token`` the other rejects and silently
    regenerates, leaving the browser holding a token the data plane no longer
    accepts. This is a *shape* check on a locally-supplied secret, not a
    timing-sensitive comparison against a request (that is :func:`token_valid`),
    so a plain ``fullmatch`` is fine.
    """
    if not token:
        return False
    token = token.strip()
    return _TOKEN_MIN_LEN <= len(token) <= _TOKEN_MAX_LEN and bool(
        _TOKEN_CHARS.fullmatch(token)
    )


def has_token_header(get: HeaderGetter) -> bool:
    """Whether the request presents either token header at all."""
    return bool(get("authorization") or get("x-biopb-token"))


def is_forgeable_cross_site(get: HeaderGetter) -> bool:
    """Whether an unsafe (state-changing) request looks like a forgeable
    cross-origin one — the CSRF signal.

    A request carrying a token header is **not** forgeable: a cross-origin
    ``no-cors`` ``fetch`` cannot set ``Authorization`` / ``X-Biopb-Token``.
    Otherwise a browser that stamped ``Sec-Fetch-Site`` cross-site (i.e. not
    ``same-origin`` / ``none``) is the CSRF vector. A non-browser client sends no
    ``Sec-Fetch-Site`` (``None``) and cannot be driven by a victim's browser, so
    it is not forgeable. Mirrors the sidecar's ``_require_same_origin``.
    """
    if has_token_header(get):
        return False
    sfs = get("sec-fetch-site")
    return sfs is not None and sfs not in _SAFE_FETCH_SITES


def host_is_loopback(host_header: Optional[str]) -> bool:
    """Whether a ``Host`` header names a loopback address (any port).

    The rebinding backstop for token-less mode: with no token, only a
    same-machine caller (which resolves the origin to ``127.0.0.1`` / ``::1`` /
    ``localhost``) is honored, so a DNS-rebinding page that points an external
    name at the loopback bind is refused. An IPv6 literal's brackets are stripped
    so ``[::1]`` and ``[::1]:8813`` both match.
    """
    if not host_header:
        return False
    if host_header.startswith("["):  # bracketed IPv6, optionally ``]:port``
        end = host_header.find("]")
        host = host_header[1:end] if end != -1 else host_header[1:]
    elif ":" in host_header:
        host = host_header.rsplit(":", 1)[0]
    else:
        host = host_header
    return host in _LOOPBACK_HOSTS
