"""One spelling for a server location.

Arrow preserves whatever location string it is handed -- ``pyarrow.flight
.Location`` round-trips the input verbatim, and each binding's constructors pick
their own form (Java's ``Location.forGrpcInsecure`` emits ``grpc+tcp://``, while
Python callers write ``grpc://``). So a location is an opaque user string, not an
identity, and anything that keys on one has to canonicalize first.

This is that canonicalization, and it is a cross-SDK contract: the on-disk chunk
cache names its per-server directory ``sha256(canonical_location(...))``, so a
second SDK reading that tree has to derive the same string. The rules, in full:

- A missing scheme reads as ``grpc+tcp``.
- ``grpc`` -> ``grpc+tcp`` and ``grpcs`` -> ``grpc+tls``; ``grpc+tcp``,
  ``grpc+tls`` and ``grpc+unix`` are already canonical.
- Scheme and host lowercase; an IPv6 host keeps its brackets.
- The port is kept as written and never defaulted -- a location with no port
  cannot be connected to anyway, so inventing 8815 would only merge two
  distinct spellings of something broken.
- A trailing ``/`` goes; any other path (a ``grpc+unix`` socket) is kept.

Names are *not* resolved. ``localhost`` and ``127.0.0.1`` stay distinct, since
resolving would put a DNS lookup on the chunk-fetch path to merge two spellings
of a server the disk cache skips anyway (it is remote-only).
"""

from typing import Optional
from urllib.parse import urlsplit

_DEFAULT_SCHEME = "grpc+tcp"
_SCHEME_ALIASES = {"grpc": "grpc+tcp", "grpcs": "grpc+tls"}

# Loopback spellings folded onto one host by same_location() -- unlike
# canonical_location(), which keeps them apart (see its docstring).
_LOOPBACK_ALIASES = frozenset({"localhost", "::1", "0.0.0.0"})


def normalize_flight_location(location: str) -> str:
    """Normalize the public TLS spelling to Arrow Flight's URI scheme."""
    if location[:8].lower() == "grpcs://":
        return "grpc+tls://" + location[8:]
    return location


_GRPC_SCHEMES = frozenset({"grpc", "grpcs", "grpc+tcp", "grpc+tls"})


def realign_transport_scheme(location: str, *, tls: bool) -> Optional[str]:
    """Rewrite *location*'s scheme to match an actual transport decision.

    Every gRPC-shaped scheme this SDK or Arrow speaks (``grpc``, ``grpcs``,
    ``grpc+tcp``, ``grpc+tls``) is accepted and rewritten to the one the
    transport is actually using -- ``grpc+tls://`` for TLS, ``grpc://``
    otherwise -- regardless of which shorthand was supplied (an operator
    typing the public ``grpcs://`` spelling for a server that turns out to
    serve plaintext, or vice versa). Returns ``None`` for anything else (no
    scheme at all, or one this isn't), so a caller can refuse it rather than
    advertise a nonsense address.
    """
    prefix_end = location.find("://")
    if prefix_end == -1 or location[:prefix_end].lower() not in _GRPC_SCHEMES:
        return None
    return ("grpc+tls://" if tls else "grpc://") + location[prefix_end + 3 :]


def canonical_location(location: str) -> str:
    """The canonical spelling of *location*.

    Falls back to the stripped input on anything unparseable. Callers use this to
    key caches and to inspect a host, so raising here would fail a chunk fetch
    over a cosmetic detail; an odd string that only ever matches itself is the
    safer failure.
    """
    raw = location.strip()
    if "://" not in raw:
        raw = f"{_DEFAULT_SCHEME}://{raw}"
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return location.strip()

    scheme = parts.scheme.lower()
    scheme = _SCHEME_ALIASES.get(scheme, scheme)
    if ":" in host:  # IPv6, unbracketed by urlsplit
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}"


def same_location(a: str, b: str) -> bool:
    """Whether *a* and *b* name the same server, loopback aliases folded in.

    For "is this already the server I'm talking to" rather than a cache key: a
    same-machine deployment mixes ``localhost``/``127.0.0.1``/``0.0.0.0``/``::1``
    freely, so (unlike :func:`canonical_location`) those fold onto one host here.
    Falls back to False on anything unparseable.
    """

    def key(location: str):
        try:
            parts = urlsplit(canonical_location(location))
        except ValueError:
            return None
        host = parts.hostname or ""
        if host in _LOOPBACK_ALIASES:
            host = "127.0.0.1"
        return parts.scheme, host, parts.port

    ka = key(a)
    return ka is not None and ka == key(b)


def location_host(location: str) -> str:
    """The hostname of *location*, lowercased and without IPv6 brackets.

    ``""`` when there is none to read -- an empty or unparseable location, or a
    ``grpc+unix`` socket, none of which name a host to compare.
    """
    raw = location.strip()
    if "://" not in raw:
        raw = f"{_DEFAULT_SCHEME}://{raw}"
    try:
        return (urlsplit(raw).hostname or "").lower()
    except ValueError:
        return ""
