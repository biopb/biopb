"""Operator-supplied TLS material is validated by opening it (biopb/biopb#913).

Three entry points resolve the same ``--tls-cert`` / ``--tls-key`` pair and only
the tensor server actually serves it, so a fault the two control entry points
miss surfaces in a supervised child that crash-loops on backoff. The rule they
share lives in :mod:`biopb._tls_material`; these are its cases.

``is_file()`` is what this replaces, and the case that motivated it is the
*normal* state of a private key: mode 0600, often owned by another user. It
stats fine and reads not at all.
"""

import os
import sys

import pytest
from biopb._tls_material import TlsMaterialError, read_pem

CERT = b"-----BEGIN CERTIFICATE-----\nZmFrZQ==\n-----END CERTIFICATE-----\n"
KEY = b"-----BEGIN PRIVATE KEY-----\nZmFrZQ==\n-----END PRIVATE KEY-----\n"


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_a_pem_file_is_returned_verbatim(tmp_path):
    # The one caller that serves the material reads it through this, so the bytes
    # have to come back untouched rather than be re-read afterwards.
    assert read_pem(_write(tmp_path, "c.pem", CERT), "tls_cert") == CERT
    assert read_pem(_write(tmp_path, "k.pem", KEY), "tls_key") == KEY


def test_a_missing_file_names_itself(tmp_path):
    with pytest.raises(TlsMaterialError, match="not found"):
        read_pem(tmp_path / "absent.pem", "--tls-cert")


def test_a_directory_is_not_a_pem_file(tmp_path):
    # `read_bytes()` on a directory raises IsADirectoryError, whose strerror
    # would otherwise read as a permission-ish failure.
    with pytest.raises(TlsMaterialError, match="is a directory"):
        read_pem(tmp_path, "--tls-cert")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads anything"
)
def test_an_unreadable_key_is_refused_not_merely_stat_ed(tmp_path):
    """The case `is_file()` passes and everything after it fails.

    A key readable only by root is the ordinary shape of one, and the operator
    who typed the command has to hear about it there -- not from a data plane
    that exits 2 on every spawn.
    """
    path = _write(tmp_path, "k.pem", KEY)
    path.chmod(0o000)
    try:
        assert path.is_file()  # the check this replaces would have passed
        with pytest.raises(TlsMaterialError, match="could not be read"):
            read_pem(path, "--tls-key")
    finally:
        path.chmod(0o600)


def test_an_empty_file_is_refused(tmp_path):
    # A placeholder created by `touch`, or a half-finished copy.
    with pytest.raises(TlsMaterialError, match="is empty"):
        read_pem(_write(tmp_path, "c.pem", b"   \n"), "--tls-cert")


def test_der_material_is_refused_with_the_conversion(tmp_path):
    """gRPC takes PEM only, and says nothing useful about a DER file."""
    with pytest.raises(TlsMaterialError, match="not PEM") as excinfo:
        read_pem(_write(tmp_path, "c.der", b"\x30\x82\x01\x0a\xff\x00"), "--tls-cert")
    assert "openssl" in str(excinfo.value)


@pytest.mark.parametrize(
    "body",
    [
        b"-----BEGIN ENCRYPTED PRIVATE KEY-----\nZmFrZQ==\n"
        b"-----END ENCRYPTED PRIVATE KEY-----\n",
        b"-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nZmFrZQ==\n"
        b"-----END RSA PRIVATE KEY-----\n",
    ],
    ids=["pkcs8", "traditional"],
)
def test_a_passphrase_protected_key_is_refused(tmp_path, body):
    """Both spellings: PKCS#8 renames the block, OpenSSL adds a header."""
    with pytest.raises(TlsMaterialError, match="passphrase-protected"):
        read_pem(_write(tmp_path, "k.pem", body), "--tls-key")
