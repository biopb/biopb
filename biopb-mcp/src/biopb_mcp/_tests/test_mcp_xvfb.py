"""Tests for the launcher-owned Xvfb host (mcp/_xvfb.py, issue #90).

The pipe/handshake plumbing is exercised against a fake ``Xvfb`` (a tiny
executable script that speaks just the ``-displayfd`` protocol), so these run
fast and everywhere POSIX. One test drives the real binary, skipped where it
is not installed.
"""

import os
import re
import stat
import sys

import pytest

from biopb_mcp.mcp import _xvfb

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the Xvfb fallback is POSIX/Linux-only"
)


def _fake_xvfb(tmp_path, body):
    """Write an executable stand-in for Xvfb and point the module at it."""
    path = tmp_path / "Xvfb"
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# Speaks the -displayfd protocol like the real server: report the display
# once "ready", then serve (sleep) until killed.
_WELL_BEHAVED = """\
import os, sys, time
fd = int(sys.argv[sys.argv.index("-displayfd") + 1])
os.write(fd, b"99\\n")
os.close(fd)
time.sleep(600)
"""

_EXITS_WITHOUT_REPORTING = """\
import sys
sys.exit(3)
"""


class TestAvailable:
    def test_true_when_on_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", _fake_xvfb(tmp_path, ""))
        assert _xvfb.available() is True

    def test_false_when_missing(self, monkeypatch):
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", "definitely-not-a-binary")
        assert _xvfb.available() is False


class TestStart:
    def test_returns_reported_display_and_live_process(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", _fake_xvfb(tmp_path, _WELL_BEHAVED))
        proc, display = _xvfb.start()
        try:
            assert display == ":99"
            assert proc.poll() is None  # still serving
        finally:
            _xvfb.stop(proc)
        assert proc.poll() is not None

    def test_missing_binary_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", "definitely-not-a-binary")
        with pytest.raises(RuntimeError) as exc:
            _xvfb.start()
        # The message must be actionable: name the package to install.
        assert "apt install xvfb" in str(exc.value)

    def test_early_exit_raises_with_exit_code(self, tmp_path, monkeypatch):
        # EOF on the display pipe (the server died during startup) must raise,
        # not hang or return garbage.
        monkeypatch.setattr(
            _xvfb, "XVFB_BINARY", _fake_xvfb(tmp_path, _EXITS_WITHOUT_REPORTING)
        )
        with pytest.raises(RuntimeError) as exc:
            _xvfb.start()
        assert "exited during startup" in str(exc.value)
        assert "3" in str(exc.value)

    def test_unparsable_report_raises(self, tmp_path, monkeypatch):
        body = _WELL_BEHAVED.replace('b"99\\n"', 'b"not-a-number\\n"')
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", _fake_xvfb(tmp_path, body))
        with pytest.raises(RuntimeError) as exc:
            _xvfb.start()
        assert "unparsable" in str(exc.value)

    def test_timeout_raises(self, tmp_path, monkeypatch):
        # A wedged server that never reports must not block bring-up forever.
        body = "import time\ntime.sleep(600)\n"
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", _fake_xvfb(tmp_path, body))
        with pytest.raises(RuntimeError) as exc:
            _xvfb.start(timeout=0.5)
        assert "did not report" in str(exc.value)


class TestStop:
    def test_none_and_double_stop_are_noops(self, tmp_path, monkeypatch):
        _xvfb.stop(None)  # nothing to do
        monkeypatch.setattr(_xvfb, "XVFB_BINARY", _fake_xvfb(tmp_path, _WELL_BEHAVED))
        proc, _ = _xvfb.start()
        _xvfb.stop(proc)
        _xvfb.stop(proc)  # idempotent


@pytest.mark.skipif(not _xvfb.available(), reason="Xvfb not installed")
def test_real_xvfb_round_trip():
    """The real server: start on a free display, then reap it."""
    proc, display = _xvfb.start()
    try:
        assert re.fullmatch(r":\d+", display)
        assert proc.poll() is None
    finally:
        _xvfb.stop(proc)
    assert proc.poll() is not None
