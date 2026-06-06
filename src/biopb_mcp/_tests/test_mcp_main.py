"""Tests for the launcher's transport selection and kernel-log helper.

These exercise the pure plumbing in ``biopb_mcp.mcp.__main__`` (arg parsing and
the stdio kernel-log file) without starting a real kernel or viewer.
"""

import sys

import pytest

from biopb_mcp.mcp.__main__ import (
    _config_defaults,
    _has_display,
    _open_kernel_log,
    _parse_args,
    _resolve_headless,
)


class TestParseArgs:
    def test_defaults_come_from_config(self):
        opts = _parse_args([], default_transport="http", default_port=8765)
        assert opts.transport == "http"
        assert opts.port == 8765

    def test_config_default_can_be_stdio(self):
        opts = _parse_args([], default_transport="stdio", default_port=8765)
        assert opts.transport == "stdio"

    def test_transport_flag_overrides(self):
        opts = _parse_args(
            ["--transport", "stdio"], default_transport="http", default_port=1
        )
        assert opts.transport == "stdio"

    def test_port_flag_overrides(self):
        opts = _parse_args(
            ["--port", "9000"], default_transport="http", default_port=8765
        )
        assert opts.port == 9000

    def test_unknown_transport_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(
                ["--transport", "ftp"],
                default_transport="http",
                default_port=8765,
            )


class TestConfigDefaults:
    def test_clean_config_passes_through(self):
        assert _config_defaults({"transport": "http", "port": 9000}) == (
            "http",
            9000,
        )

    def test_unknown_transport_falls_back_to_stdio(self):
        transport, _ = _config_defaults({"transport": "ftp"})
        assert transport == "stdio"

    def test_stringified_port_is_coerced_to_int(self):
        _, port = _config_defaults({"port": "8765"})
        assert port == 8765

    def test_garbage_port_falls_back(self):
        _, port = _config_defaults({"port": "not-a-number"})
        assert port == 8765

    def test_empty_config_uses_documented_defaults(self):
        assert _config_defaults({}) == ("stdio", 8765)


class TestOpenKernelLog:
    def test_uses_configured_path(self, tmp_path):
        path = tmp_path / "k.log"
        f = _open_kernel_log({"kernel_log": str(path)})
        try:
            f.write(b"hello\n")
            f.flush()
        finally:
            f.close()
        assert path.read_bytes() == b"hello\n"

    def test_empty_path_defaults_under_config_dir(self, tmp_path, monkeypatch):
        # _open_kernel_log does `from .._config import get_config_dir` at call
        # time, so patching the source module is what takes effect.
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "get_config_dir", lambda: tmp_path)

        f = _open_kernel_log({"kernel_log": ""})
        try:
            assert (tmp_path / "kernel.log").exists()
        finally:
            f.close()

    def test_falls_back_to_stderr_on_open_error(self):
        # An unwritable path must not crash the launcher; it degrades to the
        # stderr byte sink (sys.stderr.buffer when present, else sys.stderr).
        f = _open_kernel_log(
            {"kernel_log": "/nonexistent_dir/deep/path/kernel.log"}
        )
        assert f is getattr(sys.stderr, "buffer", sys.stderr)


class TestHasDisplay:
    def test_linux_gates_on_display_env(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _has_display() is False
        monkeypatch.setenv("DISPLAY", ":0")
        assert _has_display() is True

    def test_linux_wayland_counts(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert _has_display() is True

    def test_macos_always_has_display(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _has_display() is True


class TestResolveHeadless:
    def test_explicit_headless_always_true(self):
        assert _resolve_headless("headless", True) is True
        assert _resolve_headless("headless", False) is True

    def test_explicit_visible_always_false(self):
        # The launcher fails fast separately when visible + no display.
        assert _resolve_headless("visible", False) is False
        assert _resolve_headless("visible", True) is False

    def test_auto_follows_display(self):
        assert _resolve_headless("auto", True) is False
        assert _resolve_headless("auto", False) is True
