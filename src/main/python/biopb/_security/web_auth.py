"""Shared, stdlib-only web-auth predicates (token / same-origin / loopback).

Framework-agnostic predicates over a header getter, shared by the control, the
tensor sidecar and the observe UI (which cannot import each other); each binds
them to its own web stack.
"""

from __future__ import annotations

import re
import secrets
from typing import Callable, Optional

# A case-insensitive header lookup, e.g. Starlette ``request.headers.get``.
HeaderGetter = Callable[[str], Optional[str]]

# Hosts honored when no token is configured; only the host part is checked.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Sec-Fetch-Site values that are NOT a cross-site request (so not a CSRF vector).
_SAFE_FETCH_SITES = frozenset({"same-origin", "none"})

_BEARER_PREFIX = "Bearer "


def extract_bearer(get: HeaderGetter) -> str:
    """The token from ``Authorization: Bearer <t>`` or the ``X-Biopb-Token``
    header (Bearer wins), or ``""`` if neither is present."""
    auth = get("authorization") or ""
    if auth.startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX) :]
    return get("x-biopb-token") or ""


def token_valid(get: HeaderGetter, expected: Optional[str]) -> bool:
    """Timing-safe check that the request carries ``expected``.

    A falsy ``expected`` (no token configured) returns ``True``; callers wanting
    a loopback backstop apply :func:`host_is_loopback` themselves.
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

    Browsers cannot set headers on a WebSocket handshake; header schemes still
    take precedence. ``query_get`` is a getter over the query params.
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

    16-128 URL-safe characters (``[A-Za-z0-9_-]``), surrounding whitespace
    ignored. Shared by the tensor ``launch`` and the control so the two layers
    agree on what a supplied token is. A shape check, not the timing-safe request
    comparison (:func:`token_valid`).
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

    A request carrying a token header is not forgeable (a cross-origin
    ``no-cors`` fetch cannot set it). Otherwise a ``Sec-Fetch-Site`` other than
    ``same-origin`` / ``none`` is the CSRF vector; no header (non-browser) is not.
    """
    if has_token_header(get):
        return False
    sfs = get("sec-fetch-site")
    return sfs is not None and sfs not in _SAFE_FETCH_SITES


def host_is_loopback(host_header: Optional[str]) -> bool:
    """Whether a ``Host`` header names a loopback address (any port).

    The DNS-rebinding backstop for token-less mode. IPv6 brackets are stripped,
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


def host_is_public_bind(host: str) -> bool:
    """Whether a listener *bind address* is network-reachable (not loopback-only).

    Takes a bare bind address (unlike :func:`host_is_loopback`, which parses a
    ``Host`` header). Fail-closed: anything but a loopback literal is public.
    """
    return host not in _LOOPBACK_HOSTS
