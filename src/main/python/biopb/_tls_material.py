"""Validation for operator-supplied TLS material (``--tls-cert`` / ``--tls-key``).

Three entry points resolve the same pair — ``biopb control start`` / ``run``,
``python -m biopb_control run``, and the tensor server's own ``serve`` /
``launch`` — and only the last of them actually reads the files. Anything the
first two miss therefore surfaces inside the *supervised child*, which exits 2 on
every spawn and crash-loops on backoff while the control reports a clean start,
with the one useful sentence in tensor-server.log (biopb/biopb#913). One rule,
here, so the three cannot drift and the fault lands on the command that was
typed.

``is_file()`` is not that rule: it passes for a file that can be stat'd and not
read, which is the *normal* state of a private key (``0600``, often owned by
root). :func:`read_pem` opens the file, because opening it is the thing the
server will do.

Stdlib only, deliberately: the BYO path exists so a plane can serve TLS without
the ``cryptography`` extra, so nothing here may need it. That bounds this to what
a byte-level look can decide — readable, non-empty, PEM, not passphrase-protected
— and leaves the certificate's own contents (SANs, expiry, whether the key
matches the cert) to whoever does have a parser.
"""

from __future__ import annotations

from pathlib import Path

#: Every PEM object starts with this, whatever it holds.
_PEM_PREAMBLE = b"-----BEGIN"

#: A passphrase-protected key, in either spelling: PKCS#8 says
#: "ENCRYPTED PRIVATE KEY", the traditional OpenSSL format uses a Proc-Type
#: header instead.
_ENCRYPTED_MARKERS = (b"ENCRYPTED PRIVATE KEY", b"Proc-Type: 4,ENCRYPTED")


class TlsMaterialError(Exception):
    """TLS material that cannot be served, described with its fix.

    Carries a complete, actionable sentence: callers render it verbatim (a CLI
    prints it and exits 2), so the message is written for the operator rather
    than for a traceback.
    """


def read_pem(path: Path, label: str) -> bytes:
    """Read *path* as PEM, or raise :class:`TlsMaterialError` naming the fault.

    *label* is how the caller names this file to the operator — the flag
    (``--tls-cert``) for a value that came from the command line, the config key
    (``tls_cert``) for one that came from the file, since the tensor server
    accepts it from either.

    Returns the bytes so the one caller that needs them does not read twice; the
    two preflight callers discard them.
    """
    path = Path(path)
    if path.is_dir():
        raise TlsMaterialError(f"{label} is a directory, not a PEM file: {path}")
    if not path.is_file():
        raise TlsMaterialError(f"{label} not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as e:
        # Overwhelmingly a permission problem on a key: readable by root, and the
        # data plane runs as whoever started it. Name that, since the errno alone
        # ("Permission denied") does not say whose permission was missing.
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
