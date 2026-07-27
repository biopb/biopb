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

Resolution yields plain PEM bytes. Those bytes are what the connection pool
passes to every worker's ``FlightClient`` as ``tls_root_certs`` — so a worker
executing a chunk-fetch task receives the resolved cert as ordinary graph data
and never touches the pin store. A worker that opens its *own* client (the
``tensor_from_pb`` path) resolves once per process and then reads the memo below.

Trust and hostname verification are separate concerns, and only the first is
TOFU's job: pinning supplies the trust *anchor*, but gRPC still verifies the
dialed hostname against the certificate's SANs. A cert that pins fine can
therefore still fail the handshake if the client dials a name the server did not
put in its SANs, so :func:`resolve_tls_root_certs` probes for exactly that case
and logs the fix rather than leaving an opaque gRPC error.

TOFU is the *default*, not the only mode. A caller that already knows what the
server should present can say so, which removes the trust-on-*first*-use hole
(biopb/biopb#604 item 4 — a downstream server mounting a remote plane configures
this per upstream):

- ``ca_pem`` — trust this PEM (a private CA, or the server's own leaf) and skip
  TOFU entirely. The pin store is not consulted or written: the configured anchor
  is the source of truth, and a second one that can go stale independently would
  only produce mismatch errors naming a file the operator never edited.
- ``expected_fingerprint`` — verified-first-use. The presented leaf must match
  this SHA-256 digest on *every* connect, so an attacker in the path at first
  connect is refused rather than pinned. Also bypasses the pin store, for the
  same reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import ssl
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

from biopb._locations import tls_known_hosts

logger = logging.getLogger(__name__)

# Arrow Flight's TLS scheme (``grpcs://`` is normalized to this before resolve).
_TLS_SCHEME = "grpc+tls"

# Bound the first-connect cert fetch so an unreachable host fails fast rather
# than hanging client construction.
_FETCH_TIMEOUT_S = 10.0

# Per-process memo. Resolution is called on paths that can hit a pooled,
# already-open connection (``_get_thread_client``'s fast path) and is evaluated
# eagerly as an argument there, so without this every ``GetFlightInfo`` would
# open a throwaway TLS handshake just to re-derive a value it already has -- and
# a momentary failure of that side handshake would fail a call the healthy
# pooled connection could have served.
#
# Keyed by the endpoint AND the trust material, not the endpoint alone: one
# process can front several upstreams, and two of them naming the same
# ``host:port`` with different configured anchors must not read each other's
# result (biopb/biopb#604 item 4).
#
# The memo is deliberately not invalidated: a server that rotates its cert
# mid-process keeps failing against the memoized pin until the client restarts,
# which is the same "confirm, then clear the pin" ceremony a mismatch requires
# anyway.
_MemoKey = Tuple[str, Optional[str], Optional[str]]
_memo_lock = threading.Lock()
_memo: Dict[_MemoKey, bytes] = {}


class TlsPinMismatchError(Exception):
    """The server's certificate is not the one this client expected.

    Raised both for a TOFU pin that no longer matches and for a configured
    ``expected_fingerprint`` that the presented cert fails. A legitimate cause is
    cert rotation; a malicious one is a man-in-the-middle. Either way the client
    refuses to connect until the operator confirms, and the message names what to
    update -- the pin store entry, or the configured fingerprint.
    """


def _normalize_fingerprint(value: str) -> str:
    """Canonicalize a user-supplied SHA-256 fingerprint for comparison.

    Accepts the two spellings an operator is likely to paste -- our own
    colon-grouped ``AB:CD:...`` display form and bare hex -- and is
    case-insensitive, so a fingerprint copied out of any of the tools that print
    one compares equal.
    """
    return value.replace(":", "").replace(" ", "").strip().lower()


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


def _warn_if_dialed_name_not_in_cert(host: str, port: int, pem: bytes) -> None:
    """Log an actionable warning if *host* is not covered by the cert's SANs.

    gRPC verifies the dialed name against the certificate even when the trust
    anchor is a TOFU pin, so a cert whose SANs omit the name the client uses
    pins successfully and then fails every handshake with an opaque TLS error.
    We reproduce that check here — one verifying handshake against the pinned
    cert — purely to name the cause and the fix.

    Diagnostic only: on anything other than a definite hostname-verification
    failure (including any failure of this probe itself) it stays silent, so it
    can never turn a working connection into a broken one.
    """
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cadata=pem.decode("ascii"))
        with socket.create_connection((host, port), timeout=_FETCH_TIMEOUT_S) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                return
    except ssl.SSLCertVerificationError as e:
        # verify_message is the OpenSSL reason; code 62 is hostname mismatch.
        if "hostname mismatch" not in str(e).lower():
            return
        logger.warning(
            "TLS certificate for %s:%s is pinned, but does not list '%s' among "
            "its subject-alternative names, so the connection will fail "
            "hostname verification. Re-mint the server's certificate with that "
            "name (`biopb-tensor-server cert init --force --san %s`) and clear "
            "the '%s:%s' entry from the client's pin store, or dial a name the "
            "certificate does list.",
            host,
            port,
            host,
            host,
            host,
            port,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must never break the caller
        return


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


def resolve_tls_root_certs(
    location: str,
    *,
    ca_pem: Optional[bytes] = None,
    expected_fingerprint: Optional[str] = None,
) -> Optional[bytes]:
    """Resolve the trusted root cert (PEM bytes) for a Flight *location*.

    Returns ``None`` for a non-TLS location — a plaintext ``grpc://`` connection
    needs no cert. For a ``grpc+tls://`` location the anchor comes from whichever
    mode the caller selected:

    - *ca_pem* — trust exactly these PEM bytes (a private CA, or the server's own
      leaf). Returned unchanged; no network probe, no pin store.
    - *expected_fingerprint* — fetch the presented leaf and require its SHA-256 to
      equal this. Unlike TOFU this rejects a wrong cert on the *first* connect
      too. No pin store.
    - neither — TOFU: return the pinned cert, pinning it now if this is the first
      connect to this ``host:port``.

    Raises :class:`TlsPinMismatchError` when the server presents a cert that
    contradicts the pin or the configured fingerprint.

    The returned bytes are handed to ``pyarrow.flight.FlightClient(...,
    tls_root_certs=...)`` — with a leaf anchor, verification succeeds iff the
    server presents that exact certificate.

    The result is memoized per process, per ``host:port`` *and* per trust
    material: callers evaluate this eagerly on paths that often reuse an
    already-open pooled connection, and repeating the handshake there would be
    pure cost.
    """
    hp = _host_port(location)
    if hp is None:
        return None
    host, port = hp
    key = f"{host}:{port}"
    fingerprint = (
        _normalize_fingerprint(expected_fingerprint) if expected_fingerprint else None
    )
    memo_key: _MemoKey = (
        key,
        hashlib.sha256(ca_pem).hexdigest() if ca_pem else None,
        fingerprint,
    )

    with _memo_lock:
        memoized = _memo.get(memo_key)
    if memoized is not None:
        return memoized

    if ca_pem:
        resolved = ca_pem
    elif fingerprint:
        resolved = _resolve_against_fingerprint(host, port, fingerprint)
    else:
        resolved = _resolve_uncached(host, port, key)
    _warn_if_dialed_name_not_in_cert(host, port, resolved)
    with _memo_lock:
        _memo[memo_key] = resolved
    return resolved


def _resolve_against_fingerprint(host: str, port: int, fingerprint: str) -> bytes:
    """Verified-first-use: accept the presented cert only if it matches *fingerprint*.

    The configured digest is the whole trust decision, so this deliberately does
    not touch the pin store: writing one would create a second anchor that can
    later disagree with the config, and a mismatch report naming a state file the
    operator never edited is worse than no report.
    """
    presented = _fetch_server_cert(host, port)
    actual = _fingerprint(presented)
    if actual != fingerprint:
        raise TlsPinMismatchError(
            f"TLS certificate for {host}:{port} does not match the configured "
            f"fingerprint (expected {fingerprint[:16]}, server presented "
            f"{actual[:16]}). If the server's certificate was legitimately "
            f"rotated, update the configured fingerprint to the new value; "
            f"otherwise this may be a man-in-the-middle and you should not "
            f"connect."
        )
    return presented


def clear_pin_cache() -> None:
    """Forget every memoized resolution, so the next connect re-runs TOFU.

    Needed after a *legitimate* server cert rotation (together with clearing the
    stale entry in the pin store) if the client process is long-lived, and by
    tests that drive the pin state machine within one process.
    """
    with _memo_lock:
        _memo.clear()


def _resolve_uncached(host: str, port: int, key: str) -> bytes:
    """Do the actual TOFU exchange for ``host:port`` — fetch, compare, pin."""
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
