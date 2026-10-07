"""Validation for operator-supplied TLS material (``--tls-cert`` / ``--tls-key``).

One rule shared by the control CLIs and the tensor server, so a bad file fails
on the command that was typed rather than crash-looping the supervised child.
:func:`read_pem` actually opens the file (``is_file()`` passes for an unreadable
private key). Stdlib only (no ``cryptography``), so it checks only what bytes
show: readable, non-empty, PEM, not passphrase-protected.
"""

from __future__ import annotations

import hashlib
import logging
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Every PEM object starts with this, whatever it holds.
_PEM_PREAMBLE = b"-----BEGIN"

#: A passphrase-protected key: PKCS#8 or traditional OpenSSL spelling.
_ENCRYPTED_MARKERS = (b"ENCRYPTED PRIVATE KEY", b"Proc-Type: 4,ENCRYPTED")


class TlsMaterialError(Exception):
    """TLS material that cannot be served, described with its fix.

    The message is an actionable sentence that callers render verbatim.
    """


def expand_user_path(value: str | Path) -> Path:
    """*value* with a leading ``~`` expanded, or as given when there is no home.

    ``expanduser`` raises when no home is set, which would hide the missing file.
    """
    path = Path(value)
    try:
        return path.expanduser()
    except RuntimeError:
        return path


def read_pem(path: Path, label: str) -> bytes:
    """Read *path* as PEM, or raise :class:`TlsMaterialError` naming the fault.

    *label* names the file to the operator (the flag or the config key).
    """
    path = Path(path)
    if path.is_dir():
        raise TlsMaterialError(f"{label} is a directory, not a PEM file: {path}")
    if not path.is_file():
        raise TlsMaterialError(f"{label} not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as e:
        # Usually a key readable only by root; the message says whose permission.
        raise TlsMaterialError(
            f"{label} could not be read: {path} ({e.strerror}). The data plane "
            f"reads this file itself, as the user that starts it."
        ) from e
    if not data.strip():
        raise TlsMaterialError(f"{label} is empty: {path}")
    if _PEM_PREAMBLE not in data:
        raise TlsMaterialError(
            f"{label} is not PEM: {path} (no '-----BEGIN' block). A DER or "
            f"PKCS#12 file has to be converted first — `openssl x509 -inform "
            f"der` for a certificate, `openssl pkey -inform der` for a key."
        )
    if any(marker in data for marker in _ENCRYPTED_MARKERS):
        raise TlsMaterialError(
            f"{label} is passphrase-protected: {path}. Nothing in the serving "
            f"path can prompt for one, so supply a decrypted copy — `openssl "
            f"pkey -in {path.name} -out <plaintext>` — kept mode 0600."
        )
    return data


@dataclass(frozen=True)
class TlsAnchor:
    """What a client trusts of a TLS server: a CA/leaf PEM, or a leaf's SHA-256.

    At most one is set; neither leaves the client to trust on first use.
    """

    ca_pem: Optional[bytes] = None
    fingerprint: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.ca_pem or self.fingerprint)


def choose_anchor(
    ca_pem: Optional[bytes], fingerprint: Optional[str], *, source: str
) -> TlsAnchor:
    """The one anchor to trust when a config may name both: the CA wins.

    Same rule as ``TensorFlightClient``'s constructor; *source* names who set
    them in the warning.
    """
    fingerprint = (fingerprint or "").strip() or None
    if ca_pem and fingerprint:
        logger.warning(
            "%s sets both a CA and a fingerprint; the CA is used and the "
            "fingerprint is ignored.",
            source,
        )
        fingerprint = None
    return TlsAnchor(ca_pem=ca_pem or None, fingerprint=fingerprint)


def leaf_pem(bundle_pem: bytes) -> bytes:
    """The first certificate in *bundle_pem* — the leaf — as its own PEM block.

    A ``--tls-cert`` may be a chain (leaf first). Returns the input unchanged
    when it holds one certificate or nothing recognizable (callers validate with
    :func:`read_pem` first).
    """
    marker = b"-----END CERTIFICATE-----"
    end = bundle_pem.find(marker)
    if end == -1:
        return bundle_pem
    return bundle_pem[: end + len(marker)] + b"\n"


def fingerprint(cert_pem: bytes) -> str:
    """SHA-256 of a certificate's DER body — the identity clients check against.

    Over the DER, so line endings and whitespace cannot change the answer. One
    certificate only: pass a chain through :func:`leaf_pem` first.
    """
    der = ssl.PEM_cert_to_DER_cert(cert_pem.decode("ascii"))
    return hashlib.sha256(der).hexdigest()


def format_fingerprint(hex_digest: str) -> str:
    """Colon-group a hex fingerprint for display (``AB:CD:...``).

    Always the full digest, since operators eyeball-compare these.
    """
    return ":".join(hex_digest[i : i + 2].upper() for i in range(0, len(hex_digest), 2))
