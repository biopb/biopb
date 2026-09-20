from pathlib import Path
from types import SimpleNamespace

import biopb_tensor_server.cli as cli
import pytest
import typer
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.recovery import ProcessLock
from biopb_tensor_server.core.config import CacheConfig
from biopb_tensor_server.serving.tls import cert_fingerprint


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


def _fake_server_config(**overrides):
    """Stand-in ServerConfig for tests that monkeypatch `load_config`.

    Carries every field the CLI commands read off the config, so adding one to
    ServerConfig surfaces here as a deliberate edit rather than an AttributeError
    scattered across half the module.
    """
    base = {
        "host": "127.0.0.1",
        "port": 8815,
        "log_level": "INFO",
        "tls": False,
        "tls_cert": None,
        "tls_key": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _run_serve(config, **overrides):
    """Invoke `serve` as a plain function with explicit option defaults.

    Calling a typer command directly bypasses typer's default resolution, so each
    Option would otherwise arrive as an unresolved `OptionInfo` -- harmless where a
    value is only formatted, but `serve` now resolves the flight token eagerly and
    an `OptionInfo` token would blow up in `valid_token`. Mirror the CLI defaults.
    """
    kwargs = {
        "config": config,
        "log_level": None,
        "log_scope_biopb": True,
        "host": None,
        "port": None,
        "writable": None,
        "token": None,
        "tls": None,
        "tls_cert": None,
        "tls_key": None,
        "san": None,
        "log_file": None,
    }
    kwargs.update(overrides)
    return cli.serve(**kwargs)


def _run_launch(config, **overrides):
    """Invoke `launch` as a plain function with explicit option defaults.

    Same reason as :func:`_run_serve`: calling a typer command directly bypasses
    typer's default resolution, so every Option would arrive as an OptionInfo.
    """
    kwargs = {
        "config": config,
        "log_level": None,
        "log_scope_biopb": True,
        "host": None,
        "port": None,
        "writable": None,
        "web_port": 8816,
        "web_host": "127.0.0.1",
        "token": None,
        "tls": None,
        "tls_cert": None,
        "tls_key": None,
        "san": None,
        "cors_origins": None,
        "log_file": None,
    }
    kwargs.update(overrides)
    return cli.launch(**kwargs)


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
        # accept-and-ignore it here.
        self.stop_calls += 1


def test_serve_stops_monitoring_resources_on_keyboard_interrupt(monkeypatch):
    server = _FakeServer()
    source_manager = _FakeStoppable()
    server_config = _fake_server_config()

    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *args, **kwargs: (server, source_manager, None),
    )

    _run_serve(Path("unused.json"))

    assert source_manager.stop_calls == 1
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
    server_config = _fake_server_config()

    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *args, **kwargs: None)
    # Loopback bind with no token -> local mode (no token printing).
    monkeypatch.setattr(cli, "_resolve_launch_token", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *args, **kwargs: (flight_server, source_manager, None),
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

    _run_launch(Path("unused.json"))

    assert order == ["install_sigterm", "run_http_server", "graceful_shutdown"]


@pytest.mark.parametrize(
    "argv, forwarded",
    [
        ([], None),  # no opinion -> the config file decides
        (["--writable"], True),  # explicit on
        (["--no-writable"], False),  # explicit off, overriding a config that says on
    ],
)
def test_writable_flag_is_three_state_through_typer(monkeypatch, argv, forwarded):
    """Absence of --writable must mean "no opinion", not "off".

    `_setup_flight_server` resolves `writable=None` to the config file's
    `server.writable`, so the flag has to be able to *not* be given. Declared as
    a plain `bool` it could not: omitting it arrived as False, which silently
    pinned every config-driven deployment to read-only -- `server.writable: true`
    had no effect anywhere, the control plane's supervised data plane included,
    since it passes no flag by design (biopb#1085).

    This goes through typer's own parsing via CliRunner. The sibling helpers here
    call the command as a plain function and supply their own defaults, which is
    exactly why the suite could not see it.
    """
    from typer.testing import CliRunner

    captured: dict = {}
    monkeypatch.setattr(cli, "load_config", lambda path: _fake_server_config())
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_install_sigterm_handler", lambda: None)
    monkeypatch.setattr(cli, "run_http_server", lambda **k: None)
    monkeypatch.setattr(cli, "_graceful_shutdown", lambda *a, **k: None)

    def _capture_setup(cfg, **kwargs):
        captured.update(kwargs)
        return (SimpleNamespace(serve=lambda: None), _FakeStoppable(), None)

    monkeypatch.setattr(cli, "_setup_flight_server", _capture_setup)

    result = CliRunner().invoke(cli.app, ["launch", "--config", "unused.json", *argv])
    assert result.exit_code == 0, result.output
    assert captured["writable"] is forwarded


def test_launch_forwards_flight_overrides_and_resolves_token_against_host(
    monkeypatch,
):
    """`launch` mirrors `serve`: --host/--port/--writable override the config's
    flight bind, and the token mode switch follows the *overridden* host. A public
    --host with no token must auto-generate one (fail-closed), not bind open.
    """
    captured: dict = {}
    # Config binds loopback; the override makes the flight plane public.
    server_config = _fake_server_config()
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_install_sigterm_handler", lambda: None)

    def _capture_setup(cfg, **kwargs):
        captured.update(kwargs)
        return (SimpleNamespace(serve=lambda: None), _FakeStoppable(), None)

    monkeypatch.setattr(cli, "_setup_flight_server", _capture_setup)
    monkeypatch.setattr(
        cli,
        "run_http_server",
        lambda **kwargs: captured.update(sidecar_token=kwargs.get("token")),
    )
    monkeypatch.setattr(cli, "_graceful_shutdown", lambda *a, **k: None)

    _run_launch(Path("unused.json"), host="0.0.0.0", port=9001, writable=True)

    # Overrides reached the flight server...
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9001
    assert captured["writable"] is True
    # ...and the public override drove fail-closed token auto-gen, enforced on
    # both the flight plane and the sidecar (same effective token).
    tok = captured["token"]
    assert tok and cli._web_auth.valid_token(tok)
    assert captured["sidecar_token"] == tok


def test_graceful_shutdown_releases_file_cache_lock(tmp_path):
    """Shutdown must close the cache so the file-backend process lock is removed.

    Otherwise the lock file persists and the next start treats a clean exit as a
    crash (and could falsely block a concurrent same-user start).
    """
    cache_dir = tmp_path / "cache"
    config = CacheConfig(file_cache_dir=cache_dir)
    CacheManager.initialize(config)
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)  # held while server "runs"

    try:
        cli._graceful_shutdown(source_manager=None, flight_server=None)
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
    CacheManager.initialize(CacheConfig(file_cache_dir=cache_dir))
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
    CacheManager.initialize(CacheConfig(file_cache_dir=cache_dir))
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
    CacheManager.initialize(CacheConfig(file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)

    server = _FakeServer()
    server_config = _fake_server_config()
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "_setup_flight_server",
        lambda *a, **k: (server, _FakeStoppable(), None),
    )

    _run_serve(Path("unused.json"))

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
    CacheManager.initialize(CacheConfig(file_cache_dir=cache_dir))
    lock_path = cache_dir / "lock"
    assert not _cache_lock_is_free(lock_path)  # held once cache init ran

    server_config = _fake_server_config()
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    def _boom(*a, **k):
        # Stand in for a post-cache-init failure inside _setup_flight_server.
        raise typer.Exit(1)

    monkeypatch.setattr(cli, "_setup_flight_server", _boom)

    try:
        with pytest.raises(typer.Exit):
            _run_serve(Path("unused.json"))
        # Released by the finally's graceful shutdown despite the early exit.
        assert _cache_lock_is_free(lock_path)
    finally:
        mgr = CacheManager.get_instance()
        if mgr is not None:
            mgr.close()


def test_file_cache_on_network_dir_refuses_to_start(tmp_path, monkeypatch):
    """A file cache configured on network/cloud storage refuses to start.

    The Arrow file backend mmaps its segments and assumes local-POSIX semantics;
    on NFS/CIFS an evicted-but-mapped segment can SIGBUS/ESTALE, and a cloud
    Files-On-Demand folder recalls a dehydrated segment on mmap read
    (biopb/biopb#571 follow-up). The launcher classifies the cache dir at
    startup and, on a positive network/cloud signal, exits rather than serve
    unsafe reads -- the on-disk cache is required infrastructure, not an
    optional accelerator with a fallback.
    """
    import json

    from biopb_tensor_server.cache import CacheManager

    cache_dir = tmp_path / "cache"  # a real local dir...
    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 0},
                "cache": {"file_cache_dir": str(cache_dir)},
                "sources": [],
            }
        )
    )
    # ...that we make the classifier report as network storage.
    monkeypatch.setattr(
        cli, "unsafe_cache_dir_reason", lambda _p: "a network filesystem (nfs4)"
    )

    config = cli.load_config(config_path)
    CacheManager.reset()
    try:
        with pytest.raises(typer.Exit) as exc:
            cli._setup_flight_server(config, port=0)
        assert exc.value.exit_code == 1
        # The cache dir was never created (the server exited before touching disk).
        assert not cache_dir.exists()
    finally:
        CacheManager.reset()


def test_file_cache_on_local_dir_stays_file(tmp_path):
    """The control case: a local cache dir keeps the Arrow file backend."""
    import json

    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.cache.file_backend import ArrowFileBackend

    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "server": {"host": "127.0.0.1", "port": 0},
                "cache": {"file_cache_dir": str(cache_dir)},
                "sources": [],
            }
        )
    )

    config = cli.load_config(config_path)
    CacheManager.reset()
    server, source_manager, precache_worker = cli._setup_flight_server(config, port=0)
    try:
        mgr = CacheManager.get_instance()
        assert isinstance(mgr.backend, ArrowFileBackend)  # tmp_path is local disk
    finally:
        if precache_worker is not None:
            precache_worker.stop()
        if source_manager is not None:
            source_manager.stop(join_timeout=1)
        server.shutdown()
        CacheManager.reset()


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
                "cache": {"file_cache_dir": str(tmp_path / "cache")},
                "sources": [],
            }
        )
    )

    config = cli.load_config(config_path)
    server, source_manager, precache_worker = cli._setup_flight_server(config, port=0)
    try:
        assert server.is_ready is True
        assert source_manager is not None  # an empty manager, not None

        (raw,) = list(server.do_action(None, flight.Action("health", b"")))
        health = json.loads(bytes(raw))
        assert health["status"] == "SERVING"
        assert health["source_count"] == 0

        # And the catalog holds no sources.
        assert server._metadata_db.query("SELECT source_id FROM sources").num_rows == 0
    finally:
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

    def test_allow_no_token_serves_public_sidecar_open(self):
        # The deliberate escape hatch: with allow_no_token, a public flight + public
        # sidecar and no token is served OPEN (None) instead of auto-generating /
        # refusing. Insecure-by-request, off by default.
        assert cli._resolve_launch_token("0.0.0.0", "0.0.0.0", None, "", True) is None

    def test_allow_no_token_overrides_the_public_sidecar_refusal(self):
        # Loopback flight + public sidecar normally refuses; the override turns that
        # into a warning + tokenless serve rather than typer.Exit.
        assert cli._resolve_launch_token("127.0.0.1", "0.0.0.0", None, "", True) is None

    def test_allow_no_token_does_not_override_a_supplied_token(self):
        # The override only matters when no token is supplied; a real token wins.
        assert (
            cli._resolve_launch_token("0.0.0.0", "0.0.0.0", _VALID_TOKEN, "", True)
            == _VALID_TOKEN
        )


class TestResolveFlightToken:
    """`serve`'s token decision — the shared ladder `launch` builds on.

    The flight bind is the single mode switch: loopback → local mode (tokenless);
    a public bind fails *closed* by auto-generating a token rather than serving the
    data API open (biopb/biopb#515 follow-up: `serve` used to fail open here).
    """

    def test_loopback_is_tokenless(self):
        assert cli._resolve_flight_token("127.0.0.1", None, "") is None

    def test_public_bind_autogenerates_token(self):
        # The gap this closes: a public flight bind with no token no longer serves
        # the gRPC data API unauthenticated — it auto-generates one.
        tok = cli._resolve_flight_token("0.0.0.0", None, "")
        assert tok and cli._web_auth.valid_token(tok)

    def test_supplied_token_is_honored(self):
        assert cli._resolve_flight_token("0.0.0.0", _VALID_TOKEN, "") == _VALID_TOKEN

    def test_env_token_is_honored(self):
        assert cli._resolve_flight_token("0.0.0.0", None, _VALID_TOKEN) == _VALID_TOKEN

    def test_supplied_token_beats_env(self):
        env = "b" * 32
        assert cli._resolve_flight_token("0.0.0.0", _VALID_TOKEN, env) == _VALID_TOKEN

    def test_malformed_supplied_token_falls_through_to_mode(self):
        # A too-short --token is not usable; on a loopback bind it falls through to
        # local mode (tokenless), not a silent accept — and on a public bind it
        # auto-generates rather than binding open.
        assert cli._resolve_flight_token("127.0.0.1", "short", "") is None
        tok = cli._resolve_flight_token("0.0.0.0", "short", "")
        assert tok and cli._web_auth.valid_token(tok)

    def test_allow_no_token_serves_public_bind_open(self):
        # The escape hatch: a public flight bind with no token is served OPEN (None)
        # instead of auto-generating one. Off by default, so the fail-closed default
        # is unchanged unless explicitly requested.
        assert cli._resolve_flight_token("0.0.0.0", None, "", True) is None

    def test_allow_no_token_does_not_override_a_supplied_token(self):
        # A real token still wins over the override.
        assert (
            cli._resolve_flight_token("0.0.0.0", _VALID_TOKEN, "", True) == _VALID_TOKEN
        )


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
                "cache": {"file_cache_dir": str(tmp_path / "cache")},
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
    server, source_manager, precache_worker = cli._setup_flight_server(config, port=0)
    try:
        assert server.is_ready is True

        (raw,) = list(server.do_action(None, flight.Action("health", b"")))
        health = json.loads(bytes(raw))
        assert health["status"] == "SERVING"
        assert health["full_scan_in_progress"] is False
        assert health["last_full_scan_finished_at"] is not None
        assert health["source_count"] == 1
    finally:
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
        return _FakeServer(), _FakeStoppable(), None

    monkeypatch.setattr(cli, "_setup_flight_server", _capture)
    _run_serve(config_path)

    from biopb_tensor_server.core.config import PyramidConfig

    assert loaded["config"].pyramid.downscale_factor == PyramidConfig().downscale_factor


# --- TLS material resolution ------------------------------------------------
# There is no config side to merge against any more (biopb/biopb#604): the flags
# are the whole story, and a bare --tls-cert/--tls-key still implies TLS.


def test_a_missing_cert_path_is_refused(tmp_path):
    """A cert path that vanished between typer's `exists=True` and the read.

    Without this the pair would reach `read_bytes()` and surface as a traceback
    instead of an actionable message.
    """
    missing = tmp_path / "nope.pem"
    key = tmp_path / "k.pem"
    key.write_text("key")
    with pytest.raises(typer.Exit) as exc:
        cli._resolve_tls_material(True, missing, key, None)
    assert exc.value.exit_code == 2


def _launch_capturing_tls(monkeypatch, server_config, **launch_kwargs):
    """Run `launch` with the plumbing stubbed, returning what TLS reached where."""
    captured: dict = {}
    monkeypatch.setattr(cli, "load_config", lambda path: server_config)
    monkeypatch.setattr(cli, "get_log_level_from_env", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_install_sigterm_handler", lambda: None)
    monkeypatch.setattr(cli, "_resolve_launch_token", lambda *a, **k: None)

    def _capture_setup(cfg, **kwargs):
        captured["flight_cert"] = kwargs.get("tls_cert_chain")
        return (SimpleNamespace(serve=lambda: None), _FakeStoppable(), None)

    monkeypatch.setattr(cli, "_setup_flight_server", _capture_setup)
    monkeypatch.setattr(cli, "run_http_server", lambda **kw: captured.update(kw))
    monkeypatch.setattr(cli, "_graceful_shutdown", lambda *a, **k: None)
    _run_launch(Path("unused.json"), **launch_kwargs)
    return captured


def test_launch_points_the_sidecar_at_grpcs_and_hands_it_the_cert(
    monkeypatch, tmp_path
):
    """The whole point of #604 case 1: `launch` can serve TLS *and* keep its sidecar.

    The sidecar is co-located, so it gets the served cert as an explicit trust
    anchor rather than pinning it off the wire.
    """
    # PEM-shaped, not a placeholder string: the BYO pair is validated by opening
    # it and looking at the bytes (biopb/biopb#913).
    cert_pem = b"-----BEGIN CERTIFICATE-----\nZmFrZQ==\n-----END CERTIFICATE-----\n"
    key_pem = b"-----BEGIN PRIVATE KEY-----\nZmFrZQ==\n-----END PRIVATE KEY-----\n"
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_bytes(cert_pem)
    key.write_bytes(key_pem)

    captured = _launch_capturing_tls(
        monkeypatch, _fake_server_config(), tls_cert=cert, tls_key=key
    )

    assert captured["flight_cert"] == cert_pem  # the plane serves it...
    # ...and the sidecar verifies that exact leaf on every connect. A fingerprint
    # rather than the PEM, because the PEM resolves offline and so forfeits the
    # hostname override this loopback dial needs (biopb/biopb#916).
    assert captured["tls_fingerprint"] == cert_fingerprint(cert_pem)
    assert captured["flight_location"].startswith("grpcs://")


def test_launch_without_tls_keeps_the_sidecar_on_plaintext(monkeypatch):
    captured = _launch_capturing_tls(monkeypatch, _fake_server_config())
    assert captured["flight_cert"] is None
    assert captured["tls_fingerprint"] is None
    assert captured["flight_location"].startswith("grpc://")


# --- the bind is the CLI's, not the config's (biopb/biopb#604) ---------------


def test_retired_bind_keys_warn_loudly_instead_of_being_ignored(caplog):
    """Silently dropping these would be the worst possible break.

    `"host": "0.0.0.0"` was someone's *remote* deployment; quietly falling back
    to the loopback default would take their server off the network with no
    signal at all. The message has to name the flag that replaced the key.
    """
    import logging

    from biopb_tensor_server.core.config import parse_config

    with caplog.at_level(logging.WARNING):
        parse_config({"server": {"host": "0.0.0.0", "port": 9000, "tls": True}})
    joined = "\n".join(r.getMessage() for r in caplog.records)
    for key, flag in (("host", "--host"), ("port", "--port"), ("tls", "--tls")):
        assert f"server.{key}" in joined
        assert flag in joined


def test_a_config_bind_cannot_move_the_plane(caplog):
    """The warning is not cosmetic: the value really is gone."""
    from biopb_tensor_server.core.config import ServerConfig, parse_config

    cfg = parse_config({"server": {"host": "0.0.0.0", "port": 9000}})
    assert not hasattr(cfg, "host")
    assert not hasattr(cfg, "port")
    assert set(vars(cfg)) == set(vars(ServerConfig()))


def test_the_default_bind_is_loopback():
    """Fail-safe. The old config default was 0.0.0.0, which made a plane public
    unless something said otherwise -- the wrong direction for a default."""
    assert cli.DEFAULT_FLIGHT_HOST == "127.0.0.1"


# --- validate checks the trust anchors it will actually use (biopb/biopb#608) -


def _config_with_ca(tmp_path, ca_path, source_url="grpc://lab-store:8815/img"):
    """A config naming one credentials profile with `tls_ca_file`.

    The default source url is the SINGLE-source form on purpose: expansion
    returns before resolving credentials for that shape, so nothing else in
    `validate` would ever read the profile.
    """
    import json

    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"url": source_url, "credentials_profile": "lab-store"},
                ],
                "credentials": {
                    "profiles": [
                        {
                            "name": "lab-store",
                            "storage_type": "biopb-tensor",
                            "tls_ca_file": str(ca_path),
                        }
                    ]
                },
            }
        )
    )
    return config_path


def test_validate_rejects_a_trust_anchor_it_cannot_read(tmp_path, capsys):
    """Serve-time is the wrong moment to learn the anchor you asked for is absent."""
    config_path = _config_with_ca(tmp_path, tmp_path / "typo.pem")

    with pytest.raises(typer.Exit) as exc:
        cli.validate(config=config_path)

    assert exc.value.exit_code == 1
    out = capsys.readouterr().out
    assert "tls_ca_file" in out
    assert "credentials.profiles" in out
    assert "✓ Config valid" not in out


def test_validate_rejects_an_empty_trust_anchor(tmp_path, capsys):
    ca = tmp_path / "empty.pem"
    ca.write_bytes(b"   \n")
    config_path = _config_with_ca(tmp_path, ca)

    with pytest.raises(typer.Exit) as exc:
        cli.validate(config=config_path)

    assert exc.value.exit_code == 1
    assert "empty" in capsys.readouterr().out


def test_validate_accepts_a_readable_trust_anchor(tmp_path, capsys):
    ca = tmp_path / "lab-ca.pem"
    ca.write_bytes(b"-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----\n")
    config_path = _config_with_ca(tmp_path, ca)

    cli.validate(config=config_path)  # no Exit

    assert "✓ Config valid" in capsys.readouterr().out


def test_validate_checks_a_profile_no_source_references(tmp_path, capsys):
    """An unused profile is still config the server will read the moment a source
    names it -- and a typo found now costs nothing to fix."""
    import json

    config_path = tmp_path / "biopb.json"
    config_path.write_text(
        json.dumps(
            {
                "sources": [],
                "credentials": {
                    "profiles": [
                        {
                            "name": "lab-store",
                            "storage_type": "biopb-tensor",
                            "tls_ca_file": str(tmp_path / "typo.pem"),
                        }
                    ]
                },
            }
        )
    )

    with pytest.raises(typer.Exit) as exc:
        cli.validate(config=config_path)

    assert exc.value.exit_code == 1
    assert "tls_ca_file" in capsys.readouterr().out
