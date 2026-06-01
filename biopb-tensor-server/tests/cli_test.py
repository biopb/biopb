import tempfile
from pathlib import Path
from types import SimpleNamespace

import biopb_tensor_server.cli as cli
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.config import CacheConfig


class _FakeServer:
    def __init__(self):
        self.shutdown_calls = 0

    def serve(self):
        raise KeyboardInterrupt()

    def shutdown(self):
        self.shutdown_calls += 1


class _FakeStoppable:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_serve_stops_monitoring_resources_on_keyboard_interrupt(monkeypatch):
    server = _FakeServer()
    source_manager = _FakeStoppable()
    watcher = _FakeStoppable()
    server_config = SimpleNamespace(host="127.0.0.1", port=8815, log_level="INFO")

    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *args, **kwargs: (server, source_manager, watcher),
    )

    cli.serve(config=Path("unused.toml"))

    assert source_manager.stop_calls == 1
    assert watcher.stop_calls == 1
    assert server.shutdown_calls == 1


def test_graceful_shutdown_releases_file_cache_lock(tmp_path):
    """Shutdown must close the cache so the file-backend process lock is removed.

    Otherwise the lock file persists and the next start treats a clean exit as a
    crash (and could falsely block a concurrent same-user start).
    """
    cache_dir = tmp_path / "cache"
    config = CacheConfig(backend="file", file_cache_dir=cache_dir)
    CacheManager.initialize(config)
    lock_path = cache_dir / "lock"
    assert lock_path.exists()  # lock held while server "runs"

    try:
        cli._graceful_shutdown(source_manager=None, watcher=None, flight_server=None)
        assert not lock_path.exists()  # released on shutdown
    finally:
        mgr = CacheManager.get_instance()
        if mgr is not None:
            mgr.close()


def test_serve_releases_cache_lock_on_keyboard_interrupt(monkeypatch, tmp_path):
    """End-to-end: serve()'s shutdown path releases the cache lock."""
    cache_dir = tmp_path / "cache"
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert lock_path.exists()

    server = _FakeServer()
    server_config = SimpleNamespace(host="127.0.0.1", port=8815, log_level="INFO")
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *a, **k: (server, _FakeStoppable(), _FakeStoppable()),
    )

    cli.serve(config=Path("unused.toml"))

    assert not lock_path.exists()
