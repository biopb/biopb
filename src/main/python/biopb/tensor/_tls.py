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

Pinning supplies the anchor; it does **not** exempt the certificate from validity
checking. gRPC verifies notAfter on the anchor even when the anchor is the
presented leaf, so an expired pinned cert fails every handshake — reported by the
transport as a bare "failed to connect to all addresses", with the reason only in
gRPC's own stderr log. :class:`TlsCertExpiredError` is raised here instead, on
the resolution path that already has the answer (biopb/biopb#913).

Resolution yields a :class:`TlsTrust` — plain data (PEM bytes, an optional
hostname override, and a key id). That is what the connection pool passes to
every worker's ``FlightClient``, so a worker executing a chunk-fetch task
receives the resolved trust as ordinary graph data and never touches the pin
store. A worker that opens its *own* client (the ``tensor_from_pb`` path)
resolves once per process and then reads the memo below.

Trust and hostname verification are separate checks, and pinning only answers the
first: it supplies the trust *anchor*, but gRPC still matches the **dialed** name
against the certificate's SANs. A cert that pins fine therefore still fails every
handshake if the client dials a name the server did not put in its SANs — the
normal shape for a container that minted its cert before anyone knew what name
clients would use (biopb/biopb#606). Where the anchor is the presented leaf
itself, that second check is redundant — the presented cert must *be* the pinned
one, so there is no "different but validly-issued cert" for a name check to
exclude — and :func:`resolve_tls_trust` substitutes a name the cert does list via
``override_hostname``. See :func:`_resolve_hostname_override` for the exact
predicate and the two things this deliberately does not do.

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
from typing import Dict, NamedTuple, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from biopb._locations import tls_known_hosts

logger = logging.getLogger(__name__)

# Arrow Flight's TLS scheme (``grpcs://`` is normalized to this before resolve).
_TLS_SCHEME = "grpc+tls"

# Bound the first-connect cert fetch so an unreachable host fails fast rather
# than hanging client construction.
_FETCH_TIMEOUT_S = 10.0

# ``X509_V_ERR_CERT_HAS_EXPIRED`` — OpenSSL's verify code for a certificate past
# its notAfter, read off ``SSLCertVerificationError.verify_code``. The code rather
# than the message text: the wording is OpenSSL's to change, the number is ABI.
_VERIFY_CODE_EXPIRED = 10

# The dialed name is not in the certificate: ``X509_V_ERR_HOSTNAME_MISMATCH`` and
# ``X509_V_ERR_IP_ADDRESS_MISMATCH``. **Both**, because which one you get depends
# on what was dialed, and the second is not a corner case: the co-located sidecar
# always dials an address (127.0.0.1 / ::1), and OpenSSL words that one "IP
# address mismatch" — so a check for the string "hostname mismatch" silently
# declined to help exactly the caller that needed it most (biopb/biopb#916).
_VERIFY_CODES_NAME_MISMATCH = (62, 64)

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
_memo_lock = threading.Lock()
_memo: Dict[str, TlsTrust] = {}


class TlsTrust(NamedTuple):
    """Everything a ``FlightClient`` needs to trust one endpoint — plain data.

    Plain data on purpose: this rides into the lazy chunk-fetch graph and is
    unpickled in dask workers that must never consult the pin store or hold
    credentials of their own.

    Attributes:
        root_certs: PEM trust anchor, passed as ``tls_root_certs``. ``None`` for a
            plaintext location (or to fall back to the system trust store).
        override_hostname: name to match against the cert's SANs *instead of* the
            dialed one, or ``None`` to verify the dialed name as usual. Only ever
            set when the anchor is the presented leaf — see
            :func:`_resolve_hostname_override`.
        key_id: opaque discriminator for "which trust decision is this",
            ``endpoint | anchor digest | fingerprint``. The connection pool keys
            its per-thread ``FlightClient`` on it, so two upstreams that share a
            ``host:port`` but not an anchor cannot be served each other's
            connection. It is the memo key itself, so pool and memo partition the
            world identically rather than by coincidence.
    """

    root_certs: Optional[bytes] = None
    override_hostname: Optional[str] = None
    key_id: Optional[str] = None

    def client_kwargs(self) -> Dict[str, object]:
        """TLS keyword arguments for ``pyarrow.flight.FlightClient``.

        One construction site for the TLS arguments, so the facade's own client
        and every worker's pooled client cannot drift apart -- notably, cannot
        end up with the anchor but not the override.
        """
        kwargs: Dict[str, object] = {}
        if self.root_certs:
            kwargs["tls_root_certs"] = self.root_certs
        if self.override_hostname:
            kwargs["override_hostname"] = self.override_hostname
        return kwargs


#: Resolution result for a plaintext location: no anchor, no override, no key.
NO_TLS = TlsTrust()


def _trust_key_id(
    endpoint: str, ca_pem: Optional[bytes], fingerprint: Optional[str]
) -> str:
    """Build the memo/pool discriminator for one trust decision.

    Digests the configured CA over its **raw bytes**, deliberately not through
    :func:`_fingerprint`: that one round-trips PEM->DER and so rejects a
    multi-cert bundle outright, which a private-CA chain legitimately is. Nothing
    here is a trust decision -- it only has to separate distinct anchors -- but it
    is the full digest rather than a prefix, because a collision would mean
    handing one upstream's connection to another.
    """
    anchor = hashlib.sha256(ca_pem).hexdigest() if ca_pem else "-"
    return f"{endpoint}|{anchor}|{fingerprint or '-'}"


class TlsPinMismatchError(Exception):
    """The server's certificate is not the one this client expected.

    Raised both for a TOFU pin that no longer matches and for a configured
    ``expected_fingerprint`` that the presented cert fails. A legitimate cause is
    cert rotation; a malicious one is a man-in-the-middle. Either way the client
    refuses to connect until the operator confirms, and the message names what to
    update -- the pin store entry, or the configured fingerprint.
    """


class TlsCertExpiredError(Exception):
    """The server's certificate is past its notAfter, so no handshake can succeed.

    A separate failure from :class:`TlsPinMismatchError`: the cert is the expected
    one, it has simply run out. Raised because the transport would otherwise
    report it as an unexplained connection failure — see the module docstring.
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


def _probe_peer(
    host: str, port: int, pem: bytes, *, check_hostname: bool
) -> Tuple[dict, bytes]:
    """Handshake against *host:port* trusting only *pem*; return the peer cert.

    Returns ``(getpeercert() dict, DER bytes)``. The chain is always verified
    (``CERT_REQUIRED`` against *pem* alone); *check_hostname* selects whether the
    dialed name is also matched, which is how the caller separates "is the name
    covered" from "what names does this cert actually carry". Raises on any
    handshake failure, including the hostname mismatch the caller is looking for.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = check_hostname
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cadata=pem.decode("ascii"))
    with socket.create_connection((host, port), timeout=_FETCH_TIMEOUT_S) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            return ssock.getpeercert() or {}, ssock.getpeercert(binary_form=True) or b""


def _anchor_der(pem: bytes) -> Optional[bytes]:
    """DER body of *pem* iff it holds exactly one certificate, else ``None``.

    A multi-cert bundle -- what a private-CA anchor often is -- is not a leaf, and
    ``PEM_cert_to_DER_cert`` rejects it outright (``binascii.Error``, a
    ``ValueError``). Both facts mean the same thing to the caller, so an
    unparseable anchor is simply "not a single leaf".
    """
    try:
        return ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))
    except ValueError:
        return None


def _pick_override(san_entries: Sequence[Tuple[str, str]]) -> Optional[str]:
    """Choose a name from the cert's SANs to verify against instead of the dialed one.

    Prefers a DNS name over an IP: gRPC matches an IP SAN only when the target
    *looks* like an IP, so a DNS entry is the more portable substitute.

    Wildcards are skipped rather than used. Substituting ``*.lab.local`` would
    mean putting a name that cannot exist on the wire as SNI (see
    :func:`_resolve_hostname_override`), and synthesizing a matching label
    instead would be inventing a hostname -- the warning is the honest outcome.
    """
    dns = [value for kind, value in san_entries if kind == "DNS"]
    ips = [value for kind, value in san_entries if kind == "IP Address"]
    for name in dns:
        if "*" not in name:
            return name
    return ips[0] if ips else None


def _raise_if_expired(host: str, port: int, exc: Exception, *, tofu: bool) -> None:
    """Re-raise an OpenSSL "certificate has expired" verdict as an actionable error.

    The probe below is the one place a client learns *why* the handshake will
    fail: gRPC's own attempt reports only "failed to connect to all addresses"
    and leaves ``certificate has expired`` in its stderr log. Expiry is not
    recoverable here the way a name mismatch is — there is no substitution that
    makes an expired cert verify — so this raises rather than warns.

    The remediation is two-sided, and both sides are required: the server re-mints
    and the client drops the anchor it recorded for the old one.
    """
    if getattr(exc, "verify_code", None) != _VERIFY_CODE_EXPIRED:
        return
    remediation = (
        f"then clear the '{host}:{port}' entry from {tls_known_hosts()}"
        if tofu
        else "then update the configured TLS fingerprint to the new certificate"
    )
    raise TlsCertExpiredError(
        f"The TLS certificate for {host}:{port} has expired. Pinning it does not "
        f"exempt it from expiry -- the handshake is refused, and the transport "
        f"reports only a generic connection failure. Re-mint the server's "
        f"certificate (`biopb-tensor-server cert init --force`) and {remediation}."
    ) from exc


def _resolve_hostname_override(
    host: str, port: int, pem: bytes, *, tofu: bool
) -> Optional[str]:
    """Return a name to substitute into hostname verification, or ``None``.

    gRPC matches the **dialed** name against the cert's SANs even when the anchor
    is a pin, so a cert that pins fine still fails every handshake if its SANs
    omit that name — an address counting as a name here, matched against the IP
    SANs. When the anchor *is* the presented leaf we substitute a name
    the cert does list (``override_hostname``), which costs nothing: hostname
    verification exists to stop a MITM presenting a *different* validly-issued
    cert, and pinning the leaf has already foreclosed that -- the presented cert
    must be byte-identical to the one we pinned.

    **That reasoning depends on the anchor being the leaf**, which is why the
    substitution is gated on exactly that (``presented DER == anchor DER``) rather
    than on which resolution mode we are in. Pin a private *CA* instead and the
    SAN check becomes load-bearing again -- any host in that PKI could impersonate
    any other -- so a CA anchor keeps it and gets the warning.

    Two things this deliberately does not do:

    - it does not disable verification. ``override_hostname`` changes *which* name
      is matched; the chain is still verified against the anchor, so a garbage
      override fails the handshake rather than skipping it. (``disable_server_
      verification`` would skip the chain entirely, making the pin decorative.)
    - it does not fabricate names. No usable SAN means the warning stands.

    One caveat worth knowing: gRPC passes this value as SNI as well as verifying
    against it, so the server sees the substituted name in the handshake. Direct
    to a self-signed server that is a non-event, but an SNI-routing intermediary
    in the path would route by the cert's name instead of the dialed one -- such a
    deployment should configure an explicit anchor (``tls_ca_pem``), which never
    reaches here.

    Diagnostic-only failure mode, with one exception: any error other than a
    definite hostname mismatch leaves this silent and overrideless, so it can
    never turn a working connection into a broken one. The exception is an
    expired certificate (:func:`_raise_if_expired`), which is raised — that
    connection is already broken, and this is the only place the reason is
    legible.
    """
    try:
        _probe_peer(host, port, pem, check_hostname=True)
        return None  # the dialed name is covered -- OpenSSL says so; nothing to do
    except ssl.SSLCertVerificationError as e:
        # Expiry first: it is fatal and unfixable from here, unlike a name
        # mismatch (biopb/biopb#913).
        _raise_if_expired(host, port, e, tofu=tofu)
        # Let OpenSSL make the "is the dialed name covered" call rather than
        # comparing the SANs ourselves: it is what already implements wildcards,
        # IP SANs, case and trailing dots, and a naive membership test would
        # override certs that in fact match. Read as a verify code, not as
        # message text -- see _VERIFY_CODES_NAME_MISMATCH.
        if getattr(e, "verify_code", None) not in _VERIFY_CODES_NAME_MISMATCH:
            return None
    except Exception:  # noqa: BLE001 - a diagnostic must never break the caller
        return None

    # The dialed name is not covered. Re-handshake without the name check to read
    # what the cert *does* carry -- and to compare the presented leaf against our
    # anchor. Second round trip, paid once per process and only on this already-
    # broken path.
    anchor_der = _anchor_der(pem)
    try:
        peercert, presented_der = _probe_peer(host, port, pem, check_hostname=False)
    except Exception:  # noqa: BLE001
        peercert, presented_der = {}, b""

    anchor_is_leaf = bool(anchor_der) and presented_der == anchor_der
    override = (
        _pick_override(peercert.get("subjectAltName") or ()) if anchor_is_leaf else None
    )
    if override is not None:
        logger.info(
            "TLS certificate for %s:%s does not list '%s' among its "
            "subject-alternative names; verifying against '%s' instead, which it "
            "does list. Safe here because the trust anchor is that exact "
            "certificate, so no other certificate can satisfy the chain.",
            host,
            port,
            host,
            override,
        )
        return override

    _warn_no_usable_name(host, port, tofu=tofu, anchor_is_leaf=anchor_is_leaf)
    return None


def _warn_no_usable_name(
    host: str, port: int, *, tofu: bool, anchor_is_leaf: bool
) -> None:
    """Log the actionable fix when verification will fail and can't be fixed here.

    The remediation is mode-specific: a TOFU pin is cleared from the pin store, a
    configured fingerprint is updated in config, and a certificate this client did
    not pin is not re-minted with ``cert init`` at all (biopb/biopb#606).
    """
    if not anchor_is_leaf:
        logger.warning(
            "TLS certificate for %s:%s does not list '%s' among its "
            "subject-alternative names, so the connection will fail hostname "
            "verification. The configured trust anchor for this endpoint is not "
            "that certificate itself, so the name check is load-bearing here and "
            "is not substituted away: reissue the server's certificate with '%s' "
            "in its SANs, or dial a name it does list.",
            host,
            port,
            host,
            host,
        )
        return

    remediation = (
        f"clear the '{host}:{port}' entry from the client's pin store"
        if tofu
        else "update the configured TLS fingerprint to the new certificate"
    )
    logger.warning(
        "TLS certificate for %s:%s does not list '%s' among its "
        "subject-alternative names and carries no name that could be verified "
        "instead, so the connection will fail hostname verification. Re-mint the "
        "server's certificate with that name (`biopb-tensor-server cert init "
        "--force --san %s`) and %s, or dial a name the certificate does list.",
        host,
        port,
        host,
        host,
        remediation,
    )


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


def resolve_tls_trust(
    location: str,
    *,
    ca_pem: Optional[bytes] = None,
    expected_fingerprint: Optional[str] = None,
) -> TlsTrust:
    """Resolve the :class:`TlsTrust` for a Flight *location*.

    Returns :data:`NO_TLS` for a non-TLS location — a plaintext ``grpc://``
    connection needs no cert and this never touches the network. For a
    ``grpc+tls://`` location the anchor comes from whichever mode the caller
    selected:

    - *ca_pem* — trust exactly these PEM bytes (a private CA, or the server's own
      leaf). Returned unchanged; no network at all, and no pin store. Staying
      offline is the point of configuring an anchor, so this mode also skips the
      hostname probe below and never carries an override — which is the
      conservative answer anyway, since the usual ``ca_pem`` is a real CA whose
      SAN check is load-bearing.
    - *expected_fingerprint* — fetch the presented leaf and require its SHA-256 to
      equal this. Unlike TOFU this rejects a wrong cert on the *first* connect
      too. No pin store.
    - neither — TOFU: return the pinned cert, pinning it now if this is the first
      connect to this ``host:port``.

    The two modes that already reach the network then derive
    ``override_hostname`` if the dialed name is missing from the cert's SANs (see
    :func:`_resolve_hostname_override`) — no extra handshake unless the
    connection would otherwise have failed.

    Raises :class:`TlsPinMismatchError` when the server presents a cert that
    contradicts the pin or the configured fingerprint, and
    :class:`TlsCertExpiredError` when the anchor it resolved has expired — which
    a pin does not excuse, so the handshake would fail anyway with a far less
    legible error. A configured *ca_pem* stays offline and is therefore not
    checked for either.

    ``root_certs`` is handed to ``pyarrow.flight.FlightClient(...,
    tls_root_certs=...)`` — with a leaf anchor, verification succeeds iff the
    server presents that exact certificate.

    The result is memoized per process, per ``host:port`` *and* per trust
    material: callers evaluate this eagerly on paths that often reuse an
    already-open pooled connection, and repeating the handshake there would be
    pure cost.
    """
    hp = _host_port(location)
    if hp is None:
        return NO_TLS
    host, port = hp
    key = f"{host}:{port}"
    fingerprint = (
        _normalize_fingerprint(expected_fingerprint) if expected_fingerprint else None
    )
    key_id = _trust_key_id(key, ca_pem, fingerprint)

    with _memo_lock:
        memoized = _memo.get(key_id)
    if memoized is not None:
        return memoized

    override: Optional[str] = None
    if ca_pem:
        resolved = ca_pem
    else:
        if fingerprint:
            resolved = _resolve_against_fingerprint(host, port, fingerprint)
        else:
            resolved = _resolve_uncached(host, port, key)
        override = _resolve_hostname_override(
            host, port, resolved, tofu=not fingerprint
        )

    trust = TlsTrust(resolved, override, key_id)
    with _memo_lock:
        _memo[key_id] = trust
    return trust


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
    """Forget every memoized resolution, so the next *resolve* re-runs TOFU.

    For tests that drive the pin state machine within one process, and for any
    caller that also rebuilds its clients. It is deliberately **not** the
    in-process answer to a cert rotation, and nothing in the SDK calls it: it
    clears this memo only, while the connection pool goes on handing out
    ``FlightClient``s already built with the old anchor. The documented rotation
    ceremony stays "clear the pin-store entry and reconnect" — a restart, which
    clears memo and pool together (biopb/biopb#606).
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
