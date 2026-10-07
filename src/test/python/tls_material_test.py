"""Operator-supplied TLS material is validated by opening it (biopb/biopb#913).

Three entry points resolve the same ``--tls-cert`` / ``--tls-key`` pair and only
the tensor server actually serves it, so a fault the two control entry points
miss surfaces in a supervised child that crash-loops on backoff. The rule they
share lives in :mod:`biopb._security.tls_material`; these are its cases.

``is_file()`` is what this replaces, and the case that motivated it is the
*normal* state of a private key: mode 0600, often owned by another user. It
stats fine and reads not at all.
"""

import os
import sys
from pathlib import Path

import pytest
from biopb._security.tls_material import (
    TlsAnchor,
    TlsMaterialError,
    choose_anchor,
    expand_user_path,
    read_pem,
)

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


class TestExpandUserPath:
    """``~`` means the home directory on POSIX (``HOME``) and Windows
    (``USERPROFILE``); both are set here so either platform's rule is met."""

    def test_a_leading_tilde_is_the_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert expand_user_path("~/ca.pem") == tmp_path / "ca.pem"

    def test_a_path_without_one_is_unchanged(self):
        assert expand_user_path("relative/ca.pem") == Path("relative/ca.pem")

    def test_no_home_leaves_the_path_as_given(self, monkeypatch):
        # Windows with USERPROFILE unset raises; the caller should get the
        # ordinary "not found" naming this path instead.
        def no_home(self):
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(Path, "expanduser", no_home)
        assert expand_user_path("~/ca.pem") == Path("~/ca.pem")


class TestChooseAnchor:
    """One CA-or-fingerprint rule for every config that may name both."""

    PEM = b"-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"

    def test_neither_is_no_anchor(self):
        assert not choose_anchor(None, None, source="x")
        assert not choose_anchor(b"", "  ", source="x")

    def test_a_ca_alone(self):
        assert choose_anchor(self.PEM, None, source="x") == TlsAnchor(ca_pem=self.PEM)

    def test_a_fingerprint_alone_is_stripped(self):
        anchor = choose_anchor(None, "  ab:cd  ", source="x")
        assert anchor == TlsAnchor(fingerprint="ab:cd")

    def test_the_ca_wins_and_the_source_is_named(self, caplog):
        with caplog.at_level("WARNING"):
            anchor = choose_anchor(self.PEM, "ab:cd", source="credentials profile 'p'")
        assert anchor == TlsAnchor(ca_pem=self.PEM)
        assert "credentials profile 'p' sets both" in caplog.text
        assert "fingerprint is ignored" in caplog.text
