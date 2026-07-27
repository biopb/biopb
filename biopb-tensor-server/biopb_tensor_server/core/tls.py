"""Self-signed TLS certificate generation for the tensor Flight server.

The remote-mode encryption story (biopb/biopb#604) is deliberately CA-free: a
client trusts the server by pinning its certificate on first connect (TOFU, see
``biopb.tensor._tls``), so the server only needs a *self-signed leaf* — its own
cert is its own trust anchor. There is no private CA to manage, no chain to
build. This module mints that leaf with the right subject-alternative names
(every address a client might dial: ``localhost``, the hostname/FQDN, and the
host's LAN IPs) and a long validity, and caches it in the state tree so it is
generated once and reused.

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
from pathlib import Path
from typing import List, Tuple

from biopb._locations import tls_server_cert, tls_server_key

logger = logging.getLogger(__name__)

# ~27 months. Comfortably under the 825-day ceiling browsers enforce for
# publicly-trusted leaves; irrelevant to a TOFU pin (which ignores validity) but
# a sane default for anyone who does import the cert into a trust store.
_DEFAULT_VALIDITY_DAYS = 825


def collect_san_hosts() -> Tuple[List[str], List[str]]:
    """Best-effort ``(dns_names, ip_addresses)`` for the server's certificate.

    Gathers every address a client on the LAN might use to reach this host:
    ``localhost`` + the hostname + the FQDN as DNS names, and the loopback plus
    the primary outbound LAN IP as IP addresses. Every probe is wrapped — name
    resolution can fail on an offline box — so this never raises; a thin result
    (just loopback) still yields a usable cert for same-host use.
    """
    dns: List[str] = ["localhost"]
    ips: List[str] = ["127.0.0.1", "::1"]

    for probe in (socket.gethostname, socket.getfqdn):
        try:
            name = probe()
        except OSError:
            continue
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
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if addr and addr not in ips:
                ips.append(addr)
    except OSError:
        pass

    return dns, ips


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

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
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
) -> Tuple[bytes, bytes]:
    """Return the server's TLS cert + key (PEM bytes), generating them if needed.

    Reuses an existing cert/key pair at the given paths (defaulting to the state
    tree). When either file is missing, or ``regenerate`` is set, a fresh
    self-signed cert is minted for this host's SANs and written to disk — so
    ``serve --tls`` on a fresh host just works, and ``cert init --force`` re-mints.
    """
    cert_path = cert_path or tls_server_cert()
    key_path = key_path or tls_server_key()

    if not regenerate and cert_path.exists() and key_path.exists():
        return cert_path.read_bytes(), key_path.read_bytes()

    dns_names, ip_addresses = collect_san_hosts()
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
