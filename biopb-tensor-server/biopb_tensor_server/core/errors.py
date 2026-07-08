"""Custom exceptions for the tensor server.

Kept deliberately small; subclass standard exceptions so existing ``except``
guards in the read paths degrade gracefully when a new error type is introduced.
"""

from __future__ import annotations


class SourceUnresolvedError(ValueError):
    """A source's descriptor is not resolved (shape/dtype unknown).

    Raised at the read-planning boundary when a tensor has no concrete shape or
    dtype yet -- e.g. a cloud/synced-folder source whose pixels have not been
    hydrated, so its descriptor cannot be filled. Subclasses ``ValueError`` so
    the existing ``except ValueError`` guards in the server still catch it and
    turn it into a Flight error rather than a 500; the server adds explicit
    handling to surface a legible "open to resolve" message.

    A bare ``SourceUnresolvedError`` is **permanent** -- the source cannot be
    resolved as-is (unsupported type, parse/format failure, no adapter). Use
    ``SourceResolveRetriableError`` for transient causes where a retry may
    succeed.
    """


class SourceResolveRetriableError(SourceUnresolvedError):
    """Resolution failed *transiently* -- a retry may succeed.

    Raised when hydrating/resolving a cloud source fails for a recall/IO/offline
    reason (an ``OSError`` opening or probing the now-resident path) rather than
    a permanent one (bad format, no adapter). Subclasses
    ``SourceUnresolvedError`` so every existing ``except SourceUnresolvedError``
    / ``except ValueError`` guard still catches it; the server boundary checks
    for this subclass *first* and maps it to a retriable status (UNAVAILABLE),
    while a bare ``SourceUnresolvedError`` maps to a permanent status.
    """


class WriteNotSupportedError(Exception):
    """The source format has no write contract (it is read-only).

    Raised by the default ``SourceAdapter.put_chunk`` for formats that only read
    (the common case). The DoPut boundary wraps it into a Flight error. Kept off
    the ``ValueError`` hierarchy on purpose so it is never swallowed by the
    read-path ``except ValueError`` guards -- a write rejection is unrelated to
    read planning.
    """
