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

    # The Flight boundary can only surface an unresolved source as the *retriable*
    # ``FlightUnavailableError`` (UNAVAILABLE): pyarrow exposes no
    # ``FAILED_PRECONDITION`` class, and the Python client's resolve-steering keys
    # on that class + an "unresolved" message (biopb/tensor/_session.py). A blind
    # retry is harmless -- ``GetFlightInfo`` never resolves on serve -- and an
    # explicit ``resolve()`` is the real recovery. So this code matches the class
    # the boundary picks (UNAVAILABLE), keeping ``extra_info``'s retryability in
    # step with it; the permanent-vs-transient split below is a *domain*
    # distinction (class identity), not a separate wire status.
    grpc_code = "UNAVAILABLE"


class SourceResolveRetriableError(SourceUnresolvedError):
    """Resolution failed *transiently* -- a retry may succeed.

    Raised when hydrating/resolving a cloud source fails for a recall/IO/offline
    reason (an ``OSError`` opening or probing the now-resident path) rather than
    a permanent one (bad format, no adapter). Subclasses
    ``SourceUnresolvedError`` so every existing ``except SourceUnresolvedError``
    / ``except ValueError`` guard still catches it. Both surface at the Flight
    boundary as the same retriable ``UNAVAILABLE`` (see the base's ``grpc_code``
    note -- pyarrow has no ``FAILED_PRECONDITION`` class and the client's
    resolve-steering keys on ``FlightUnavailableError``); this subclass exists to
    mark the transient cause for callers and logging, not to carry a distinct wire
    status.
    """

    grpc_code = "UNAVAILABLE"


class TensorResolutionError(ValueError):
    """A field/tensor within a source could not be resolved to an adapter.

    The base for the *terminal client-error* taxonomy on the read path
    (``get_tensor_adapter``): a bad ``array_id`` is the caller's mistake, not a
    server bug. Subclasses ``ValueError`` on purpose -- like
    ``SourceUnresolvedError`` -- so the read paths' existing ``except ValueError``
    guards still catch it and every adapter that has not yet adopted the typed
    taxonomy degrades gracefully.

    Carries a canonical gRPC status-code name (``grpc_code``) and a
    machine-readable ``reason`` slug. pyarrow's Flight-in-Python exposes no
    ``NOT_FOUND``/``INVALID_ARGUMENT`` exception class, so the Flight boundary
    maps this to the best *coarse-retryability* class (a terminal
    ``FlightServerError``, never the "server bug, don't retry"
    ``FlightInternalError``) and serializes ``{"code", "reason"}`` into the
    error's ``extra_info`` -- the class carries retryability, ``extra_info``
    carries the precise code the class hierarchy cannot express.
    """

    grpc_code = "INTERNAL"

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class TensorNotFound(TensorResolutionError):
    """The requested tensor field does not exist in the source.

    Canonical gRPC ``NOT_FOUND``. Raised by every ``get_tensor_adapter`` (the
    single-tensor base included) for an unknown *nonempty* field, so a typo'd
    ``array_id`` is representable before it is mapped instead of silently
    returning the wrong tensor. The ``#44`` no-field defaults (empty / bare
    ``source_id`` / the source's own id) still resolve to the default tensor.
    """

    grpc_code = "NOT_FOUND"


class InvalidTensorId(TensorResolutionError):
    """The tensor id is structurally malformed for this source.

    Canonical gRPC ``INVALID_ARGUMENT``. Distinct from :class:`TensorNotFound`
    (a well-formed id that names no existing tensor): the id itself cannot be
    parsed -- e.g. an HCS field that is not ``well/field`` or whose field index
    is not an integer.
    """

    grpc_code = "INVALID_ARGUMENT"


class UnknownResolutionError(TensorResolutionError):
    """A resolution-path exception that could not be classified.

    Canonical gRPC ``UNKNOWN`` -- the code gRPC reserves for "errors raised by
    APIs that do not return enough error information." The Flight read boundary
    coerces a bare ``ValueError`` / ``KeyError`` / ``AttributeError`` /
    ``TypeError`` from an adapter's resolution to this rather than fabricating a
    specific :class:`TensorNotFound` (which would misattribute a possible server
    bug to the caller) or leaking a "server bug, don't retry"
    ``FlightInternalError``. Still terminal -- it rides ``FlightServerError`` like
    the rest of the taxonomy, so: don't blindly retry, but don't blame the client
    either. Once every adapter raises the typed taxonomy, a genuine field miss no
    longer reaches this fallback, so what remains here is honestly *unclassified*.
    """

    grpc_code = "UNKNOWN"


class WriteNotSupportedError(Exception):
    """The source format has no write contract (it is read-only).

    Raised by the default ``SourceAdapter.put_chunk`` for formats that only read
    (the common case). The DoPut boundary wraps it into a Flight error. Kept off
    the ``ValueError`` hierarchy on purpose so it is never swallowed by the
    read-path ``except ValueError`` guards -- a write rejection is unrelated to
    read planning.
    """
