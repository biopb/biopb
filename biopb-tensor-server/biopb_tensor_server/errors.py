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
    """
