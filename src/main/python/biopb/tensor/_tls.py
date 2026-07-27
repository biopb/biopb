"""Trust-on-first-use (TOFU) certificate pinning for the tensor Flight client.

A ``grpcs://`` server on a private/firewalled LAN typically presents a
self-signed or private-CA certificate with no publicly-trusted anchor. Rather
than make the operator distribute a root CA into every client's trust store, the
client pins the server's certificate on **first** connect — the SSH host-key
model (biopb/biopb#604):

- first connect to a ``host:port``: fetch the presented leaf cert, record its PEM
  in the pin store, and use that exact cert as the trusted root;
- later connects: use the pinned cert as the trusted root. If the server now
  presents a *different* cert (a rotation, or a MITM), the fingerprints diverge
  and we raise :class:`TlsPinMismatchError` with the fix — exactly like SSH's
  "REMOTE HOST IDENTIFICATION HAS CHANGED" warning.

The security boundary is the same as SSH: the *first* handshake is trusted
implicitly, so TOFU protects against an attacker who arrives *after* pinning, not
one already in the path at first connect — an accepted trade on a trusted LAN.

Resolution runs once, in the process that constructs ``TensorFlightClient``, and
yields plain PEM bytes. Those bytes are what the connection pool passes to every
worker's ``FlightClient`` as ``tls_root_certs`` — so dask workers never re-run
TOFU or touch the pin store; they receive the resolved cert as ordinary data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlsplit

from biopb._locations import tls_known_hosts

logger = logging.getLogger(__name__)

# Arrow Flight's TLS scheme (``grpcs://`` is normalized to this before resolve).
_TLS_SCHEME = "grpc+tls"

# Bound the first-connect cert fetch so an unreachable host fails fast rather
# than hanging client construction.
_FETCH_TIMEOUT_S = 10.0


class TlsPinMismatchError(Exception):
    """The server's certificate no longer matches the one pinned for this host.

    A legitimate cause is cert rotation; a malicious one is a man-in-the-middle.
    Either way the client refuses to connect until the operator confirms and
    clears the stale pin (the message names the store and the ``host:port`` key).
    """


def _host_port(location: str) -> Optional[tuple[str, int]]:
    """Extract ``(host, port)`` from a ``grpc+tls://host:port`` location.

    Returns ``None`` for any non-TLS scheme (plaintext ``grpc://`` / a bare
    ``host:port``), which is the signal that no pinning applies.
    """
    parts = urlsplit(location)
    if parts.scheme != _TLS_SCHEME or not parts.hostname or not parts.port:
        return None
    return parts.hostname, parts.port


def _fetch_server_cert(host: str, port: int) -> bytes:
    """Fetch the leaf certificate a TLS server presents, as PEM bytes.

    An intentionally *unverified* handshake — there is no trust anchor yet (that
    is what TOFU is bootstrapping). We only read the presented cert; no data is
    exchanged over the connection.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=_FETCH_TIMEOUT_S) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            der = ssock.getpeercert(binary_form=True)
    if not der:
        raise ssl.SSLError(f"{host}:{port} presented no certificate")
    return ssl.DER_cert_to_PEM_cert(der).encode("ascii")


def _fingerprint(pem: bytes) -> str:
    """SHA-256 of the certificate's DER body — a scheme/whitespace-stable id."""
    der = ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))
    return hashlib.sha256(der).hexdigest()


def _load_pins(store: Path) -> Dict[str, str]:
    """Read the ``host:port -> PEM`` pin store; an absent/corrupt file is empty."""
    try:
        with open(store, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_pin(store: Path, key: str, pem: bytes) -> None:
    """Add ``key -> pem`` to the pin store, written owner-only and atomically."""
    store.parent.mkdir(parents=True, exist_ok=True)
    pins = _load_pins(store)
    pins[key] = pem.decode("ascii")
    # Atomic replace so a concurrent reader never sees a half-written file.
    fd, tmp = tempfile.mkstemp(dir=store.parent, prefix=".tls-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(pins, f, indent=2, sort_keys=True)
        # The pins are public certs, not secrets, but keeping the store
        # owner-only stops another local uid from rewriting a pin to defeat TOFU.
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        os.replace(tmp, store)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_tls_root_certs(location: str) -> Optional[bytes]:
    """Resolve the trusted root cert (PEM bytes) for a Flight *location* via TOFU.

    Returns ``None`` for a non-TLS location — a plaintext ``grpc://`` connection
    needs no cert. For a ``grpc+tls://`` location, returns the pinned certificate
    (pinning it now if this is the first connect). Raises
    :class:`TlsPinMismatchError` if the server presents a cert that differs from
    the pinned one.

    The returned bytes are handed to ``pyarrow.flight.FlightClient(...,
    tls_root_certs=...)`` — using the pinned leaf itself as the trust anchor, so
    verification succeeds iff the server presents that exact certificate.
    """
    hp = _host_port(location)
    if hp is None:
        return None
    host, port = hp
    key = f"{host}:{port}"
    store = tls_known_hosts()

    presented = _fetch_server_cert(host, port)
    pins = _load_pins(store)
    pinned = pins.get(key)

    if pinned is None:
        _save_pin(store, key, presented)
        logger.info(
            "TOFU: pinned certificate for %s (%s). Delete the entry in %s to "
            "re-pin after a legitimate cert change.",
            key,
            _fingerprint(presented)[:16],
            store,
        )
        return presented

    if _fingerprint(presented) != _fingerprint(pinned.encode("ascii")):
        raise TlsPinMismatchError(
            f"TLS certificate for {key} does not match the pinned one "
            f"(pinned {_fingerprint(pinned.encode('ascii'))[:16]}, "
            f"server now {_fingerprint(presented)[:16]}). If the server's "
            f"certificate was legitimately rotated, remove the '{key}' entry "
            f"from {store} and reconnect; otherwise this may be a "
            f"man-in-the-middle and you should not connect."
        )
    return pinned.encode("ascii")
