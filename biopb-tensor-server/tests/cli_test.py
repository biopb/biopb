from pathlib import Path
from types import SimpleNamespace

import biopb_tensor_server.cli as cli
import pytest
import typer
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.recovery import ProcessLock
from biopb_tensor_server.core.config import CacheConfig


def _cache_lock_is_free(lock_path: Path) -> bool:
    """Whether the cache lock at `lock_path` has been released cleanly.

    Release is no longer observable as the lock file disappearing: exclusion is
    an OS lock on an open descriptor and the file is deliberately permanent
    (unlinking it would let a racing acquirer lock a different file by the same
    name). What "released" means now is that another owner can take it -- and,
    since a clean release also removes the `.owner` record, that the next owner
    does not see a crash (biopb/biopb#544).
    """
    probe = ProcessLock(lock_path)
    if not probe.acquire():
        return False
    clean = not probe.is_stale()
    probe.release()
    return clean


_VALID_TOKEN = "a" * 32  # 32 URL-safe chars: passes _web_auth.valid_token


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

    def stop(self, join_timeout=None):
        # _graceful_shutdown passes a short join_timeout to source_manager.stop();
        # accept-and-ignore it here (also used as the watcher, called with no arg).
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
        lambda *args, **kwargs: (server, source_manager, watcher, None),
    )

    cli.serve(config=Path("unused.json"))

    assert source_manager.stop_calls == 1
    assert watcher.stop_calls == 1
    assert server.shutdown_calls == 1


def test_launch_installs_sigterm_handler_before_blocking_and_runs_finally(
    monkeypatch,
):
    """`launch` must install the SIGTERM->KeyboardInterrupt handler.

    Regression for biopb/biopb#516: under the control supervisor, `launch`
    relied on uvicorn to handle SIGTERM, but uvicorn reverts SIGTERM to the
    default (terminate) disposition when its loop closes, so the process was
    signal-killed (exit 143) before the `finally` reached `_graceful_shutdown`
    and the cache process lock leaked on every restart. Owning the handler
    routes SIGTERM through `except KeyboardInterrupt`/`finally` instead. The
    handler must be installed *before* the blocking HTTP server starts.
    """
    order: list[str] = []

    flight_server = SimpleNamespace(serve=lambda: None)
    source_manager = _FakeStoppable()
    watcher = _FakeStoppable()
    server_config = SimpleNamespace(host="127.0.0.1", port=8815, log_level="INFO")

    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *args, **kwargs: None)
    # Loopback bind with no token -> local mode (no token printing).
    monkeypatch.setattr(cli, "_resolve_launch_token", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *args, **kwargs: (flight_server, source_manager, watcher, None),
    )
    monkeypatch.setattr(
        cli, "_install_sigterm_handler", lambda: order.append("install_sigterm")
    )
    # Stand in for uvicorn returning after a SIGTERM-driven graceful stop.
    monkeypatch.setattr(
        cli,
        "run_http_server",
        lambda **kwargs: order.append("run_http_server"),
    )
    monkeypatch.setattr(
        cli,
        "_graceful_shutdown",
        lambda *args, **kwargs: order.append("graceful_shutdown"),
    )

    # Pass explicit values rather than typer OptionInfo defaults (calling the
    # command as a plain function bypasses typer's default resolution; a truthy
    # OptionInfo for open_browser would otherwise launch a real browser).
    cli.launch(
        config=Path("unused.json"),
        log_level=None,
        log_scope_biopb=True,
        web_port=8816,
        web_host="127.0.0.1",
        token=None,
        open_browser=False,
        web_url="http://localhost:5173",
        cors_origins=None,
        log_file=None,
    )

    assert order == ["install_sigterm", "run_http_server", "graceful_shutdown"]


def test_graceful_shutdown_releases_file_cache_lock(tmp_path):
    """Shutdown must close the cache so the file-backend process lock is removed.

    Otherwise the lock file persists and the next start treats a clean exit as a
    crash (and could falsely block a concurrent same-user start).
    """
    cache_dir = tmp_path / "cache"
    config = CacheConfig(backend="file", file_cache_dir=cache_dir)
    CacheManager.initialize(config)
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)  # held while server "runs"

    try:
        cli._graceful_shutdown(source_manager=None, watcher=None, flight_server=None)
        assert _cache_lock_is_free(lock_path)  # released on shutdown
    finally:
        mgr = CacheManager.get_instance()
        if mgr is not None:
            mgr.close()


def test_graceful_shutdown_releases_lock_before_slow_source_manager(tmp_path):
    """The cache lock must be released BEFORE the (up-to-5s) source-manager join,
    so a mid-teardown SIGKILL still finds it released (biopb/biopb#300). A slow or
    raising source_manager.stop() must not keep the lock from being released.
    """
    cache_dir = tmp_path / "cache"
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)

    order = []
    state = {}

    class _Flight:
        def shutdown(self):
            order.append("flight")

    class _SourceManager:
        def stop(self, join_timeout=None):
            order.append("source_manager")
            # Graceful shutdown passes a short join bound (the daemon thread may
            # be blocked in an upstream re-list); assert it is not the 5s default.
            state["join_timeout"] = join_timeout
            # The cache lock must already be gone by the time this slow step runs.
            state["lock_at_stop"] = not _cache_lock_is_free(lock_path)
            raise RuntimeError("boom")  # a failure here must not matter

    try:
        cli._graceful_shutdown(
            source_manager=_SourceManager(),
            watcher=None,
            flight_server=_Flight(),
        )
        # Lock released before the flight drain and the join.
        assert order == ["flight", "source_manager"]
        assert state["lock_at_stop"] is False  # released before the join ran
        assert state["join_timeout"] == 1  # short bound, not the 5s default
        assert _cache_lock_is_free(lock_path)  # released despite source_manager raising
    finally:
        mgr = CacheManager.get_instance()
        if mgr is not None:
            mgr.close()


def test_graceful_shutdown_bounds_a_hanging_flight_drain(tmp_path, monkeypatch):
    """A wedged Flight drain must not keep the cache lock (biopb/biopb#300 follow-up).

    On a caching proxy an in-flight do_get can be gated on a dead/slow upstream,
    so FlightServerBase.shutdown() (which takes no timeout) blocks unbounded.
    Graceful shutdown must still (a) release/unlink the cache process lock and
    (b) return within roughly the bound instead of hanging with it.
    """
    import threading
    import time

    cache_dir = tmp_path / "cache"
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)

    # Shrink the drain bound so the test is fast; the fake hangs far beyond it.
    monkeypatch.setattr(cli, "_FLIGHT_DRAIN_TIMEOUT_S", 0.3)

    release = threading.Event()  # never set until teardown -> shutdown() hangs
    entered = threading.Event()

    class _HangingFlight:
        def shutdown(self):
            entered.set()
            release.wait(timeout=30)  # blocks well beyond the 0.3s bound

    try:
        start = time.monotonic()
        cli._graceful_shutdown(
            source_manager=None,
            watcher=None,
            flight_server=_HangingFlight(),
        )
        elapsed = time.monotonic() - start

        # The drain was actually entered and is STILL stuck...
        assert entered.is_set()
        # ...yet the lock is gone (released BEFORE the drain) and we returned
        # promptly rather than blocking on the wedged shutdown().
        assert _cache_lock_is_free(lock_path)
        assert elapsed < 5  # ~0.3s bound, nowhere near the 30s hang
    finally:
        release.set()  # let the daemon drain thread unwind
        mgr = CacheManager.get_instance()
        if mgr is not None:
            mgr.close()


def test_serve_releases_cache_lock_on_keyboard_interrupt(monkeypatch, tmp_path):
    """End-to-end: serve()'s shutdown path releases the cache lock."""
    cache_dir = tmp_path / "cache"
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)

    server = _FakeServer()
    server_config = SimpleNamespace(host="127.0.0.1", port=8815, log_level="INFO")
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *a, **k: (server, _FakeStoppable(), _FakeStoppable(), None),
    )

    cli.serve(config=Path("unused.json"))

    assert _cache_lock_is_free(lock_path)


def test_serve_releases_cache_lock_when_setup_fails(monkeypatch, tmp_path):
    """A failure in _setup_flight_server after cache init still releases the lock.

    Regression for biopb/biopb#515: cache init acquires the file-backend process
    lock, and an early exit after that (e.g. a bad static source) used to run
    *before/outside* serve()'s try/finally, so `_graceful_shutdown` never ran and
    the lock file was orphaned -- the next start then treated it as a stale lock
    and paid a crash-recovery scan. The setup call now lives inside the try, so
    the finally releases the lock on every exit path, not just a clean return.
    """
    cache_dir = tmp_path / "cache"
    CacheManager.initialize(CacheConfig(backend="file", file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)  # held once cache init ran

    server_config = SimpleNamespace(host="127.0.0.1", port=8815, log_level="INFO")
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    def _boom(*a, **k):
        # Stand in for a post-cache-init failure inside _setup_flight_server.
        raise typer.Exit(1)

    monkeypatch.setattr(cli, "_setup_flight_server", _boom)

    try:
        with pytest.raises(typer.Exit):
            cli.serve(config=Path("unused.json"))
        # Released by the finally's graceful shutdown despite the early exit.
        assert _cache_lock_is_free(lock_path)
    finally:
        mgr = CacheManager.get_instance()
        if mgr is not None:
            mgr.close()


def test_setup_empty_sources_serves_empty_catalog(tmp_path):
    """An empty source set reaches SERVING with an empty catalog, not exit(1).

    Regression for biopb/biopb#515: `_setup_flight_server` used to `raise
    typer.Exit(1)` when no static/monitored sources were configured. An empty
    catalog is a valid runtime state (sources arrive via runtime add_source,
    DoPut, or a monitored dir that fills later), and under the control plane an
    exit(1) reads as a crash -> restart loop. The server must boot and serve an
    empty catalog (health SERVING, source_count 0).
    """
    import json

    from pyarrow import flight

    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 0},
                "cache": {"backend": "memory"},
                "sources": [],
            }
        )
    )

    config = cli.load_config(config_path)
    server, source_manager, watcher, precache_worker = cli._setup_flight_server(
        config, port=0
    )
    try:
        assert server.is_ready is True
        assert source_manager is not None  # an empty manager, not None

        (raw,) = list(server.do_action(None, flight.Action("health", b"")))
        health = json.loads(bytes(raw))
        assert health["status"] == "SERVING"
        assert health["source_count"] == 0

        # And the empty catalog lists no flights.
        assert list(server.list_flights(None, None)) == []
    finally:
        if watcher is not None:
            watcher.stop()
        if precache_worker is not None:
            precache_worker.stop()
        if source_manager is not None:
            source_manager.stop(join_timeout=1)
        server.shutdown()


class TestResolveLaunchToken:
    """`launch`'s token decision is fail-closed on every public listener.

    The flight bind (server.host) is the mode switch; the sidecar's own bind
    (--web-host) must never be public *and* unauthenticated (biopb/biopb#424
    follow-up: the ``--web-host 0.0.0.0`` + loopback ``server.host`` footgun).
    """

    def test_local_mode_is_tokenless(self):
        # Loopback flight + loopback sidecar + no token supplied → local mode.
        assert cli._resolve_launch_token("127.0.0.1", "127.0.0.1", None, "") is None

    def test_public_flight_autogenerates_token(self):
        # Public flight bind with no token supplied → auto-generate (not open).
        tok = cli._resolve_launch_token("0.0.0.0", "0.0.0.0", None, "")
        assert tok and cli._web_auth.valid_token(tok)

    def test_supplied_token_is_honored(self):
        assert (
            cli._resolve_launch_token("0.0.0.0", "0.0.0.0", _VALID_TOKEN, "")
            == _VALID_TOKEN
        )

    def test_env_token_is_honored(self):
        assert (
            cli._resolve_launch_token("0.0.0.0", "127.0.0.1", None, _VALID_TOKEN)
            == _VALID_TOKEN
        )

    def test_public_sidecar_loopback_flight_no_token_is_forbidden(self):
        # The reported hole: a public HTTP sidecar with a loopback flight bind
        # resolves to no token → would serve the data API unauthenticated. Refuse.
        with pytest.raises(typer.Exit) as exc:
            cli._resolve_launch_token("127.0.0.1", "0.0.0.0", None, "")
        assert exc.value.exit_code == 1

    def test_public_sidecar_allowed_when_token_supplied(self):
        # A public sidecar is fine once a token is enforced across both listeners.
        assert (
            cli._resolve_launch_token("127.0.0.1", "0.0.0.0", _VALID_TOKEN, "")
            == _VALID_TOKEN
        )

    def test_public_flight_and_sidecar_is_authenticated(self):
        # Public flight auto-generates a token, so a co-public sidecar is covered.
        tok = cli._resolve_launch_token("0.0.0.0", "0.0.0.0", None, "")
        assert tok and cli._web_auth.valid_token(tok)

    def test_empty_web_host_counts_as_public(self):
        # An empty bind address means "all interfaces" — treat it as public.
        with pytest.raises(typer.Exit):
            cli._resolve_launch_token("127.0.0.1", "", None, "")

    def test_malformed_supplied_token_falls_through_to_mode(self):
        # A too-short --token is not a usable token; on a loopback flight bind it
        # falls through to local mode (tokenless), not a silent accept.
        assert cli._resolve_launch_token("127.0.0.1", "127.0.0.1", "short", "") is None


def test_setup_static_only_serves_immediately_with_freshness(tmp_path):
    """A static-only config reaches SERVING and reports a freshness timestamp.

    Progressive discovery: _setup_flight_server no longer blocks on a scan. With
    no monitored dirs there is nothing to background, so the launcher drives the
    first-scan-complete path directly -- the server is SERVING, not scanning, and
    last_full_scan_finished_at is stamped so a client sees an established catalog.
    """
    import json

    from biopb_tensor_server.fixtures import create_zarr_array
    from pyarrow import flight

    zarr_path, _, _ = create_zarr_array(str(tmp_path))
    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 0},
                "cache": {"backend": "memory"},
                "sources": [
                    {
                        "type": "zarr",
                        "url": zarr_path,
                        "dim_labels": ["y", "x"],
                    }
                ],
            }
        )
    )

    config = cli.load_config(config_path)
    server, source_manager, watcher, precache_worker = cli._setup_flight_server(
        config, port=0
    )
    try:
        assert server.is_ready is True

        (raw,) = list(server.do_action(None, flight.Action("health", b"")))
        health = json.loads(bytes(raw))
        assert health["status"] == "SERVING"
        assert health["full_scan_in_progress"] is False
        assert health["last_full_scan_finished_at"] is not None
        assert health["source_count"] == 1
    finally:
        if watcher is not None:
            watcher.stop()
        if precache_worker is not None:
            precache_worker.stop()
        server.shutdown()


# --- config errors are a refusal, not a traceback (biopb/biopb#34) ------------


def test_validate_reports_a_bad_knob_and_exits_1(tmp_path, capsys):
    """`validate` is the strict surface: a human asked, so report and fail.

    The load path clamps the same value (a supervised server must still come up),
    which is exactly why this command validates the *raw* file rather than
    inspecting a loaded config -- otherwise it would report a clean bill on a
    config whose bad value had just been defaulted away.
    """
    import json

    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps({"server": {"port": 8815}, "pyramid": {"downscale_factor": 0}})
    )

    with pytest.raises(typer.Exit) as exc:
        cli.validate(config=config_path)
    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "downscale_factor" in out
    # The section name survives rich's markup parser ("[pyramid]" is not a tag).
    assert "pyramid" in out


def test_serve_starts_with_a_bad_knob_clamped_to_its_default(tmp_path, monkeypatch):
    """A bad knob must not stop the server: the plane is supervised, so refusing
    to load would be restarted straight back into the same failure with the cause
    buried in a log (biopb/biopb#34). The value is defaulted instead, so nothing
    invalid reaches GetFlightInfo either."""
    import json

    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps({"server": {"port": 8815}, "pyramid": {"downscale_factor": 0}})
    )

    loaded = {}
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    def _capture(config, port=None, **kwargs):
        loaded["config"] = config
        return _FakeServer(), _FakeStoppable(), _FakeStoppable(), None

    monkeypatch.setattr(cli, "_setup_flight_server", _capture)
    cli.serve(config=config_path)

    from biopb_tensor_server.core.config import PyramidConfig

    assert loaded["config"].pyramid.downscale_factor == PyramidConfig().downscale_factor


def test_serve_refuses_legacy_toml_naming_the_migration_command(tmp_path, capsys):
    config_path = tmp_path / "biopb.toml"
    config_path.write_text("[server]\nport = 8815\n")

    with pytest.raises(typer.Exit) as exc:
        cli.serve(config=config_path)
    assert exc.value.exit_code == 1
    assert "migrate-config" in capsys.readouterr().out
