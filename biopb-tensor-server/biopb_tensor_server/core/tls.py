"""Self-signed TLS certificate generation for the tensor Flight server.

The remote-mode encryption story (biopb/biopb#604) is deliberately CA-free: a
client trusts the server by pinning its certificate on first connect (TOFU, see
``biopb.tensor._tls``), so the server only needs a *self-signed leaf* — its own
cert is its own trust anchor. There is no private CA to manage, no chain to
build. This module mints that leaf with the right subject-alternative names
(every address a client might dial: ``localhost``, the hostname/FQDN, and the
host's LAN IPs) and a long validity, and caches it in the state tree so it is
generated once and reused.

**The SANs still matter under TOFU.** Pinning supplies the *trust anchor*, but
gRPC keeps hostname verification on, so the name a client actually dials must
appear in this cert's SANs or the handshake fails — after a successful pin, which
reads as "it pinned but won't connect". :func:`collect_san_hosts` can only see the
host's own view of itself, so a name that lives elsewhere (a NAT/VPN address, a
CNAME, a reverse-proxy hostname) has to be named explicitly:
``cert init --force --san <name>`` / ``serve --tls --san <name>``.

Lives beside the token machinery (both are resolved by the ``serve``/``launch``
CLI) rather than in the control plane, so a **headless** tensor server with no
control installed can still stand up TLS on its own (case 2 of biopb/biopb#604).

``cryptography`` is imported lazily inside :func:`generate_self_signed_cert` so
importing this module — for the stdlib-only path/SAN/fingerprint helpers — costs
nothing on a plaintext server that never generates a cert.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import logging
import os
import socket
import ssl
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from biopb._locations import tls_server_cert, tls_server_key

logger = logging.getLogger(__name__)

# ~27 months, the total span (the cert is backdated one day against clock skew
# and the span measured from there). Long because rotating this cert invalidates
# every client's TOFU pin, and 825 specifically because it was the CA/Browser
# Forum's leaf ceiling when this was chosen -- a conservative bound, not a
# requirement anything here enforces.
#
# **A pin does not exempt the cert from validity checking.** gRPC validates
# notAfter on the trust anchor even when the anchor *is* the presented leaf, so
# this span ends in a total data-plane outage for every pinned client, reported
# to them as a generic "failed to connect to all addresses"
# (biopb/biopb#913). :func:`cert_expiry_warning` is what gives an operator
# notice before that day; the client raises
# :class:`biopb.tensor._tls.TlsCertExpiredError` on the day itself.
_DEFAULT_VALIDITY_DAYS = 825

# How long before notAfter the server starts saying so. A month is enough notice
# to re-mint and re-pin every client without being so early it becomes noise.
_EXPIRY_WARN_DAYS = 30

# Ceiling on each name-resolution probe below. Generous for a working
# resolver, and short enough that a broken one costs a startup blip rather
# than a stall.
_NAME_PROBE_TIMEOUT_S = 5.0


def split_san_values(values: List[str]) -> Tuple[List[str], List[str]]:
    """Partition operator-supplied SAN strings into ``(dns_names, ip_addresses)``.

    A value that parses as an IP literal becomes an ``iPAddress`` SAN, everything
    else a ``dNSName``. This is what backs ``--san`` on ``cert init`` /
    ``serve --tls``: the auto-collected names cover the host's own view of itself,
    but a client may dial a name this host cannot see (a NAT/VPN address, a CNAME,
    a reverse-proxy hostname), and gRPC verifies the *dialed* name against the SANs
    even though the trust anchor came from a TOFU pin.
    """
    dns: List[str] = []
    ips: List[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        try:
            ipaddress.ip_address(value)
        except ValueError:
            dns.append(value)
        else:
            ips.append(value)
    return dns, ips


def _bounded(label: str, probe, default):
    """Run a name-resolution *probe*, giving up after ``_NAME_PROBE_TIMEOUT_S``.

    ``getfqdn`` / ``getaddrinfo`` take no timeout argument and block in libc until
    the resolver answers. On a host whose resolver is slow about its *own* name —
    a macOS runner doing mDNS is the measured case, ~60 s per call — that would
    stall `serve --tls` startup before the port ever opens. A SAN we could not
    resolve is a cert one name thinner, which is recoverable (`--san`); a server
    that will not start is not, so the probe is bounded and its result optional.

    The worker is a daemon thread, deliberately: if the resolver never answers,
    the thread stays parked in libc forever and must not hold up interpreter exit.
    """
    result = [default]

    def _run():
        try:
            result[0] = probe()
        except OSError:
            pass

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(_NAME_PROBE_TIMEOUT_S)
    if worker.is_alive():
        logger.warning(
            "%s took longer than %.0fs while collecting certificate SANs; "
            "continuing without it. Name the address clients dial with --san if "
            "it is missing from the certificate.",
            label,
            _NAME_PROBE_TIMEOUT_S,
        )
    return result[0]


@lru_cache(maxsize=1)
def _host_identity() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """The probe half of :func:`collect_san_hosts`, resolved once per process.

    A process's own identity does not change under it, so the (possibly slow,
    see :func:`_bounded`) probes run once and every later caller reads the cache.

    Returns tuples because the result is cached: :func:`collect_san_hosts` copies
    them into fresh lists so no caller can mutate the shared value.
    """
    dns: List[str] = ["localhost"]
    ips: List[str] = ["127.0.0.1", "::1"]

    try:
        hostname = socket.gethostname()  # a syscall, not a lookup
    except OSError:
        hostname = ""
    fqdn = _bounded("getfqdn()", socket.getfqdn, "")
    for name in (hostname, fqdn):
        if name and name not in dns:
            dns.append(name)

    # Primary LAN IP without sending a packet: a UDP socket "connected" to an
    # off-net address just picks the outbound interface, and getsockname reads it.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            primary = s.getsockname()[0]
        if primary and primary not in ips:
            ips.append(primary)
    except OSError:
        pass

    # Any additional addresses the hostname resolves to (multi-homed hosts).
    resolved = (
        _bounded(
            "getaddrinfo()",
            lambda: [info[4][0] for info in socket.getaddrinfo(hostname, None)],
            [],
        )
        if hostname
        else []
    )
    for addr in resolved:
        if addr and addr not in ips:
            ips.append(addr)

    return tuple(dns), tuple(ips)


def collect_san_hosts() -> Tuple[List[str], List[str]]:
    """Best-effort ``(dns_names, ip_addresses)`` for the server's certificate.

    Gathers every address a client on the LAN might use to reach this host:
    ``localhost`` + the hostname + the FQDN as DNS names, and the loopback plus
    the primary outbound LAN IP as IP addresses. Every probe is wrapped — name
    resolution can fail on an offline box — so this never raises; a thin result
    (just loopback) still yields a usable cert for same-host use.

    Only what this host can see about *itself*: a name that lives elsewhere (a
    NAT/VPN address, a CNAME, a reverse-proxy hostname) has to be named with
    ``--san``, and must be, because gRPC verifies the dialed name against these.
    """
    dns, ips = _host_identity()
    return list(dns), list(ips)


def generate_self_signed_cert(
    dns_names: List[str],
    ip_addresses: List[str],
    days: int = _DEFAULT_VALIDITY_DAYS,
) -> Tuple[bytes, bytes]:
    """Generate a self-signed cert + key (both PEM bytes) for the given SANs.

    ``cryptography`` is imported here (not at module load) so a plaintext server
    never pays for it, and its absence surfaces only on the TLS path -- with an
    actionable message -- rather than at import. It is an opt-in extra precisely
    because it drags a Rust/OpenSSL build surface the default install avoids
    (biopb/biopb#355). Malformed IP strings are skipped, not fatal.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as e:
        raise RuntimeError(
            "TLS certificate generation requires the 'cryptography' package, "
            "which is an opt-in extra. Install it with "
            "`pip install 'biopb-tensor-server[tls]'`, or serve plaintext "
            "(without --tls), or supply a ready cert via --tls-cert/--tls-key."
        ) from e

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cn = dns_names[0] if dns_names else "localhost"
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    san: List[x509.GeneralName] = [x509.DNSName(d) for d in dns_names]
    for ip in ip_addresses:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            logger.debug("skipping unparseable SAN IP %r", ip)

    # Backdated one day against client/server clock skew; `days` is the total
    # span measured from there, so it matches the documented validity exactly.
    not_before = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=1
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_before + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        # An end-entity cert, explicitly: it is its own trust anchor only because
        # a client pins it, never because it may sign anything. Stating that (plus
        # a serverAuth EKU and a matching KeyUsage) keeps stricter TLS stacks --
        # which may reject a leaf that omits them -- from refusing the handshake.
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def cert_fingerprint(cert_pem: bytes) -> str:
    """SHA-256 of the cert's DER body — the value a client's TOFU pin matches on.

    Stdlib only (no ``cryptography``), so callers can print it cheaply.
    """
    der = ssl.PEM_cert_to_DER_cert(cert_pem.decode("ascii"))
    return hashlib.sha256(der).hexdigest()


def format_fingerprint(fingerprint: str) -> str:
    """Colon-group a hex fingerprint for display (``AB:CD:...``).

    Operators eyeball-compare this against a client's pin, so every printout of a
    fingerprint uses the full digest in this one grouped form -- a truncated
    prefix is not something a human should be asked to trust.
    """
    return ":".join(
        fingerprint[i : i + 2].upper() for i in range(0, len(fingerprint), 2)
    )


def cert_expiry_warning(cert_pem: bytes) -> Optional[str]:
    """Message naming *cert_pem*'s expiry when it is past or near, else ``None``.

    A self-signed cert a client pinned is still rejected once it is past its
    notAfter -- pinning supplies the trust anchor, it does not turn validity
    checking off (biopb/biopb#913) -- and nothing else watches the date, so
    ``serve --tls`` / ``cert init`` say it here.

    Returns a message rather than logging one so the CLI can render it in its own
    style. ``cryptography`` is optional on this path: reading an existing cert
    otherwise needs none (the BYO ``--tls-cert`` route deliberately runs without
    the extra), so an absent or unparseable cert is simply "nothing to say".
    """
    try:
        from cryptography import x509

        not_after = x509.load_pem_x509_certificate(cert_pem).not_valid_after_utc
    except Exception:  # noqa: BLE001 - advisory only; never break the TLS path
        return None

    remaining = not_after - datetime.datetime.now(datetime.timezone.utc)
    stamp = not_after.strftime("%Y-%m-%d")
    if remaining.total_seconds() <= 0:
        return (
            f"This TLS certificate expired on {stamp}. Clients will fail the "
            f"handshake with a generic connection error -- a pinned certificate "
            f"is still checked against its expiry. Re-mint it with `cert init "
            f"--force`; every client must then clear its pin for this endpoint."
        )
    if remaining.days <= _EXPIRY_WARN_DAYS:
        return (
            f"This TLS certificate expires on {stamp} ({remaining.days} days). "
            f"Once it does, clients fail the handshake with a generic connection "
            f"error. Re-mint it with `cert init --force` before then; every "
            f"client must clear its pin for this endpoint afterwards."
        )
    return None


def _write_cert_files(cert_path: Path, key_path: Path, cert_pem: bytes, key_pem: bytes):
    """Persist cert (public) + key (owner-only) to disk, creating the dir."""
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    if os.name == "posix":
        # The key is a secret; the cert is public and left world-readable.
        os.chmod(key_path, 0o600)


def ensure_server_cert(
    cert_path: Path | None = None,
    key_path: Path | None = None,
    *,
    regenerate: bool = False,
    extra_sans: List[str] | None = None,
) -> Tuple[bytes, bytes]:
    """Return the server's TLS cert + key (PEM bytes), generating them if needed.

    Reuses an existing cert/key pair at the given paths (defaulting to the state
    tree). When either file is missing, or ``regenerate`` is set, a fresh
    self-signed cert is minted for this host's SANs — plus any *extra_sans* the
    operator named — and written to disk, so ``serve --tls`` on a fresh host just
    works and ``cert init --force`` re-mints.

    *extra_sans* only affects a cert that is **generated** here: an existing pair
    is returned untouched (adding a name to a minted cert means re-minting it,
    i.e. ``cert init --force --san <name>``).
    """
    cert_path = cert_path or tls_server_cert()
    key_path = key_path or tls_server_key()

    if not regenerate and cert_path.exists() and key_path.exists():
        return cert_path.read_bytes(), key_path.read_bytes()

    dns_names, ip_addresses = collect_san_hosts()
    if extra_sans:
        extra_dns, extra_ips = split_san_values(extra_sans)
        dns_names += [d for d in extra_dns if d not in dns_names]
        ip_addresses += [i for i in extra_ips if i not in ip_addresses]
    cert_pem, key_pem = generate_self_signed_cert(dns_names, ip_addresses)
    _write_cert_files(cert_path, key_path, cert_pem, key_pem)
    logger.info(
        "generated self-signed TLS cert %s (fingerprint %s, SANs dns=%s ip=%s)",
        cert_path,
        cert_fingerprint(cert_pem)[:16],
        dns_names,
        ip_addresses,
    )
    return cert_pem, key_pem
