"""Bootstrap executed *inside* the MCP child kernel.

Injected via IPython ``exec_lines`` so it runs before the kernel services any
tool calls.  It enables the Qt event loop, configures dask in the process
where compute actually happens, opens a visible napari viewer with the Tensor
Browser widget, and populates the ``execute_code`` namespace.

A failure here does not abort the kernel (exec_lines errors are swallowed by
IPython), so ``bootstrap`` prints a ``BOOTSTRAP_ERROR`` sentinel that the
host's health probe detects via the absence of ``viewer`` in the namespace.
"""

import logging
import os
import traceback

logger = logging.getLogger(__name__)


class _HeadlessViewer:
    """Stand-in bound to ``viewer`` when the kernel runs without a display.

    Any attribute access raises a clear, quotable error so agent code that
    reaches for the viewer (``viewer.add_image(...)``, ``viewer.layers`` …)
    surfaces a self-describing message — relayed to the user by the model —
    instead of a cryptic ``AttributeError`` on ``None``.  Falsy so kernel
    snippets can guard with ``if viewer:``.
    """

    _MSG = (
        "napari viewer unavailable: this biopb-mcp kernel started headless "
        "(no display). Data access (client), compute (ops), and execute_code "
        "still work — there is just no viewer window or screenshot."
    )

    def __getattr__(self, name):
        raise RuntimeError(self._MSG)

    def __repr__(self):
        return "<headless: no napari viewer (no display)>"

    def __bool__(self):
        return False


def _configure_dask(config: dict):
    """Set up dask in the kernel process.

    The kernel never owns a cluster: the session child (``biopb_mcp.mcp``) spins
    and owns the ``LocalCluster`` (see :class:`_cluster.DaskClusterHost`) and
    injects its address. Returns a distributed ``Client`` (or ``None`` for the
    in-process scheduler):

    * ``"distributed"`` + an address (``BIOPB_DASK_ADDRESS`` injected by the
      session child, or an external ``dask.address``) -> a ``Client`` attached
      to that scheduler.
    * ``"distributed"`` + no address -> the session child has no cluster
      (disabled or a spin failure), so degrade to the in-process ``threads``
      scheduler rather than spinning a competing kernel-local one.
    * ``"threads"`` / ``"synchronous"`` -> in-process scheduler.

    ``interrupt_kernel`` can stop an in-flight ``compute()`` in any distributed
    mode (it holds a real ``Client``). A failure attaching degrades gracefully to
    ``threads`` rather than aborting the bootstrap.
    """
    import dask

    from .._config import get_setting

    scheduler = get_setting(config, "dask.scheduler")
    num_workers = get_setting(config, "dask.num_workers") or None
    # The session-child-injected address (its owned cluster) wins over the
    # configured external one; either takes the plain Client(address) attach path.
    address = os.environ.get("BIOPB_DASK_ADDRESS") or get_setting(
        config, "dask.address"
    )

    if scheduler == "distributed":
        try:
            from dask.distributed import Client

            if address:
                client = Client(address)
                logger.info("Dask attached to distributed scheduler at %s", address)
                return client

            # No address: the session child owns the cluster and would have
            # injected BIOPB_DASK_ADDRESS. None here means it has none (disabled
            # or a spin failure) -> in-process threads, not a competing
            # kernel-local cluster.
            logger.info("No injected dask address; using in-process threads scheduler")
            scheduler = "threads"
        except Exception:
            # A missing `distributed` install or an unreachable address --
            # degrade to the in-process scheduler so the bootstrap (and the
            # viewer) survives.
            logger.exception(
                "Distributed dask unavailable; "
                "falling back to in-process threads scheduler"
            )
            scheduler = "threads"

    dask.config.set(scheduler=scheduler, num_workers=num_workers)
    logger.info("Dask scheduler: %s, num_workers: %s", scheduler, num_workers)
    return None


def _make_cache_plugin(location, token, cache_bytes):
    """Build a dask ``WorkerPlugin`` that pins each worker's chunk-cache budget.

    Lives here, not in the tensor SDK: it is dask-specific glue and a rare edge
    case. It only matters when the MCP kernel talks *directly* to a **remote**
    tensor server under the multi-process distributed cluster, where each worker
    would otherwise replicate the client cache. The usual path is a local
    server/proxy (localhost -> the tensor client keeps no cache), where this is a
    no-op.

    Registering the returned plugin runs ``biopb.tensor.client.configure_cache``
    (the SDK's per-process cache primitive) on every worker -- current and future
    -- so the budget stays fixed across the cluster; the plugin is ``name``-tagged
    so re-registration replaces rather than stacks. Returns ``None`` when
    ``distributed`` is unavailable so callers can no-op.
    """
    try:
        from distributed.diagnostics.plugin import WorkerPlugin
    except Exception:
        return None

    class _CacheConfigPlugin(WorkerPlugin):
        name = "biopb-cache-config"  # named -> idempotent re-registration

        def __init__(self, location, token, cache_bytes):
            self._args = (location, token, cache_bytes)

        def setup(self, worker):
            from biopb.tensor.client import configure_cache

            configure_cache(*self._args)

    return _CacheConfigPlugin(location, token, cache_bytes)


def _register_cache_plugin(dask_client, url, token, config: dict, planned_workers=None):
    """Split the data-plane chunk-cache budget across dask workers.

    Divides ``dask.cache_budget`` evenly across the workers and installs a
    worker-init plugin so each worker (current and future) caps its per-process
    *copy* cache at ``budget // n_workers`` -- bounding the aggregate cache that
    would otherwise be replicated per worker.

    This budget now applies on localhost too. It bounds only the strong cache of
    chunks that cost real RAM (``do_get`` / over-budget copies); on localhost
    those are rare (the mmap fast path dominates), and its views are cached
    *weakly* -- shared with the OS page cache, so not replicated per worker and
    needing no budget. So the old "localhost clamps to 0" special-case is gone;
    each worker just applies this budget uniformly.

    No-op without a distributed client. Best-effort: a failure here must not
    break the connect flow that invokes it. Called from
    ``TensorConnection.on_connect`` with the final ``(url, token)`` (the token is
    only known after connect).
    """
    if dask_client is None:
        return
    try:
        from dask.utils import parse_bytes

        from .._config import get_setting

        n_workers = max(
            1,
            planned_workers or len(dask_client.scheduler_info().get("workers", {})),
        )

        budget_cfg = get_setting(config, "dask.cache_budget")
        budget = (
            int(budget_cfg)
            if isinstance(budget_cfg, int | float)
            else parse_bytes(budget_cfg)
        )
        per_worker = max(0, budget // n_workers)

        plugin = _make_cache_plugin(url, token, per_worker)
        if plugin is None:
            return
        dask_client.register_plugin(plugin)
        logger.info(
            "Chunk-cache plugin: %d B/worker x %d workers (%s)",
            per_worker,
            n_workers,
            url,
        )
    except Exception:
        logger.exception("Failed to register chunk-cache budget plugin")


def _install_window_close_hook(viewer):
    """Signal the launcher when the user closes the napari window.

    The launcher inherits the *write* end of a pipe via ``BIOPB_WINDOW_CLOSE_FD``
    (set by ``KernelHost._launch``, name = ``_kernel.ENV_WINDOW_CLOSE_FD``); a
    reader thread there reaps this kernel back to idle on the byte we write. We
    connect to the Qt main window's ``destroyed`` signal — the same
    ``viewer.window._qt_window`` the closed-window probe (``viewer_window_alive``)
    keys off — which fires once the C++ window is deleted (a user X-close deletes
    it). Idempotent and fully best-effort: a missing fd, an absent window, or any
    wiring/IO failure must never break the bootstrap.
    """
    fd_str = os.environ.get("BIOPB_WINDOW_CLOSE_FD")
    if not fd_str:
        return
    try:
        fd = int(fd_str)
    except ValueError:
        return

    fired = {"done": False}

    def _notify(*_args):
        if fired["done"]:
            return
        fired["done"] = True
        try:
            os.write(fd, b"x")
        except OSError:
            pass

    try:
        viewer.window._qt_window.destroyed.connect(_notify)
    except Exception:
        logger.exception("Failed to install napari window-close hook")


def _make_token_report_hook():
    """Return a ``(url, token) -> None`` callback that reports the connection to
    the launcher, or ``None`` when no report pipe is configured.

    The launcher inherits the *write* end of a pipe via ``BIOPB_TOKEN_REPORT_FD``
    (set by ``KernelHost._launch``, name = ``_kernel.ENV_TOKEN_REPORT_FD``) and
    caches the latest token in the MCP-server process so it can re-inject it into
    the next kernel's env — persisting a token the user entered in the Tensor
    Browser across ``restart_kernel`` without it ever touching disk (issue #86).
    Wired into ``TensorConnection.on_connect`` so it fires on every successful
    connect. One ``url\\ttoken`` line per connect (a single small write, atomic
    under PIPE_BUF). Fully best-effort: a missing fd or any IO error is swallowed.
    """
    fd_str = os.environ.get("BIOPB_TOKEN_REPORT_FD")
    if not fd_str:
        return None
    try:
        fd = int(fd_str)
    except ValueError:
        return None

    def _report(url, token):
        line = f"{url or ''}\t{token or ''}\n".encode()
        try:
            os.write(fd, line)
        except OSError:
            pass

    return _report


def _start_update_check(viewer, config):
    """Kick off the kernel-start update reminder (issue #87), GUI branch only.

    Runs the network version check on a daemon thread so it can never delay
    window paint, then marshals a window-only reminder popup to the Qt main
    thread via ``run_on_main`` (which the popup returns from immediately — it
    ``.show()``s rather than ``.exec()``s). Fully best-effort and fail-open: the
    check itself swallows every error, and this wrapper swallows the rest, so it
    never disturbs a working session. The caller invokes this only when a real
    napari window exists (never headless — "only when a napari window exists").

    This is a *notify-only* reminder: it tells the user to run the install/
    upgrade script. biopb does not self-update (a graceful cross-platform apply
    needs a staging step we don't handle yet — see issue #87).
    """
    import threading

    def _worker():
        try:
            from ._jobs import run_on_main
            from ._update import check_for_update
            from ._update_apply import handle_choice
            from ._update_popup import show_update_popup

            info = check_for_update(config)
            if info is None:
                return

            logger.info("biopb update available: %s -> %s", info.current, info.latest)

            def _on_choice(action):
                handle_choice(action, info, config)

            # Returns as soon as the box is shown (non-blocking); button clicks
            # are handled later on the main thread via the popup's signals.
            run_on_main(show_update_popup, info, _on_choice, viewer)
        except Exception:
            logger.debug("update check failed (fail-open)", exc_info=True)

    threading.Thread(target=_worker, name="biopb-update-check", daemon=True).start()


# Load-bearing namespace names a user plugin (#92) must not shadow. The merge
# guard skips any of these; the startup-file save/restore protects all except the
# two owned by the background dask-attach thread (below), which it must not race.
_RESERVED_NAMES = frozenset(
    {
        "viewer",
        "client",
        "np",
        "da",
        "ops",
        "run_on_main",
        "_conn",
        "_jobs",
        "_dask_client",
        "_dask_attach_done",
        "_viewer_window_alive",
        "_resync_view",
    }
)
# Written only by the daemon attach thread (its sole-writer invariant, step 3), so
# a plugin exec must NOT snapshot+restore them — that would race the attach and
# could revert a just-attached Client.
_ATTACH_OWNED = frozenset({"_dask_client", "_dask_attach_done"})


def _public_names(mapping: dict) -> dict:
    """The names a module/mapping plugin contributes: ``__all__`` if declared, else
    every public (non-``_``) name that is not itself an imported module (so a
    plugin's ``import numpy as np`` doesn't leak ``np`` into the namespace)."""
    import types

    declared = mapping.get("__all__")
    if isinstance(declared, list | tuple):
        return {k: mapping[k] for k in declared if k in mapping}
    return {
        k: v
        for k, v in mapping.items()
        if not k.startswith("_") and not isinstance(v, types.ModuleType)
    }


def _merge_names(ip, names: dict, *, source: str) -> None:
    """Merge plugin-contributed *names* into the kernel namespace, skipping any
    reserved load-bearing name (warned, never silently)."""
    ns = ip.user_ns
    for key, value in names.items():
        if key in _RESERVED_NAMES:
            logger.warning(
                "kernel plugin %s would shadow reserved name %r; skipped",
                source,
                key,
            )
            continue
        ns[key] = value


def _load_startup_files(ip, plugin_dir) -> None:
    """Exec each ``*.py`` in *plugin_dir* directly in the kernel namespace.

    IPython ``startup/`` semantics: a file's top-level defs land beside ``viewer``/
    ``client``/``ops``, and its functions resolve those live handles as globals at
    call time -- exactly like agent ``execute_code`` (which also runs in this
    namespace), so ``client`` refreshed per-job is seen. Fail-open per file. Any
    load-bearing name the file overwrites is restored (warned); the two
    attach-thread-owned names are left untouched to avoid racing that thread.
    """
    try:
        paths = sorted(plugin_dir.glob("*.py"))
    except OSError:
        return
    ns = ip.user_ns
    protect = _RESERVED_NAMES - _ATTACH_OWNED
    for path in paths:
        if path.name.startswith("_"):
            continue
        saved = {k: ns[k] for k in protect if k in ns}
        try:
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
            exec(code, ns)  # noqa: S102 - user plugin; this kernel is RCE by design
            logger.info("Loaded kernel plugin file: %s", path.name)
        except Exception:
            logger.exception("kernel plugin file %s failed to load", path.name)
        finally:
            for key, value in saved.items():
                if ns.get(key) is not value:
                    logger.warning(
                        "kernel plugin %s overwrote reserved name %r; restored",
                        path.name,
                        key,
                    )
                    ns[key] = value


def _load_entry_point_plugins(ip) -> None:
    """Load ``biopb_mcp.namespace`` entry-point packages into the namespace.

    An entry point resolving to a ``register(namespace)`` callable is called with a
    read-through snapshot of the live namespace (it can inspect the handles and add
    names); one resolving to a module or mapping has its public names merged. All
    merges pass the reserved-name guard; fail-open per entry point.
    """
    from biopb._kernel_plugins import NAMESPACE_ENTRY_POINT_GROUP

    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return
    try:
        eps = list(entry_points(group=NAMESPACE_ENTRY_POINT_GROUP))
    except Exception:
        logger.debug("kernel plugin: entry-point discovery failed", exc_info=True)
        return

    import types
    from collections.abc import Mapping

    for ep in eps:
        try:
            obj = ep.load()
        except Exception:
            logger.exception("kernel plugin entry point %r failed to import", ep.name)
            continue
        try:
            if isinstance(obj, types.ModuleType):
                _merge_names(ip, _public_names(vars(obj)), source=ep.name)
            elif isinstance(obj, Mapping):
                # A returned mapping is a namespace-like source (like a module), so
                # filter it the same way: public names / honor __all__, drop the
                # odd dunder. (A register() hook, by contrast, writes literally.)
                _merge_names(ip, _public_names(dict(obj)), source=ep.name)
            elif callable(obj):
                # A read-through snapshot: register() sees the live handles, and we
                # merge only what it newly bound (guarded) rather than let it write
                # straight into user_ns and clobber a built-in.
                snapshot = dict(ip.user_ns)
                obj(snapshot)
                writes = {
                    k: v
                    for k, v in snapshot.items()
                    if ip.user_ns.get(k, _MISSING) is not v
                }
                _merge_names(ip, writes, source=ep.name)
            else:
                logger.warning(
                    "kernel plugin entry point %r is not a register()/module/mapping;"
                    " ignored",
                    ep.name,
                )
                continue
            logger.info("Loaded kernel plugin entry point: %s", ep.name)
        except Exception:
            logger.exception("kernel plugin entry point %r failed to load", ep.name)


def _load_namespace_plugins(ip, config) -> None:
    """Load user "bring your own tool" plugins into the kernel namespace (#92).

    Two sources, both fail-open per unit so one bad plugin never breaks the
    bootstrap (the ``build_ops`` / skills precedent): ``*.py`` files under
    ``~/.config/biopb/kernel/`` and installed ``biopb_mcp.namespace`` entry points.
    Called after the built-in handles exist (step 7) so plugins can reference them.
    Gated by ``services.namespace_enabled``.
    """
    from .._config import get_setting

    if not get_setting(config, "services.namespace_enabled", True):
        logger.info("kernel plugins disabled (services.namespace_enabled=false)")
        return
    from biopb._locations import mcp_plugin_dir

    try:
        _load_startup_files(ip, mcp_plugin_dir())
    except Exception:
        logger.exception("kernel plugin: startup-file load failed")
    try:
        _load_entry_point_plugins(ip)
    except Exception:
        logger.exception("kernel plugin: entry-point load failed")


# Sentinel for "key absent" in the entry-point snapshot diff (a plugin may bind a
# value that equals None, so `.get(k)` alone can't distinguish absent from None).
_MISSING = object()


def bootstrap():
    """Entry point called from the kernel's exec_lines."""
    try:
        _bootstrap_impl()
    except Exception:
        tb = traceback.format_exc()
        # Stash the traceback in the kernel namespace so the host's health
        # probe can fetch and surface it.  exec_lines output is otherwise
        # swallowed by IPython, leaving the probe with only "viewer absent".
        try:
            from IPython import get_ipython

            get_ipython().user_ns["_BOOTSTRAP_ERROR"] = tb
        except Exception:
            pass
        print("BOOTSTRAP_ERROR: " + tb)


def _bootstrap_impl():
    from biopb import _algorithms
    from IPython import get_ipython

    from .._config import get_setting, load_config

    ip = get_ipython()
    config = load_config()

    # Headless (compute-only) mode: the launcher sets BIOPB_HEADLESS when no
    # display is available (or display_mode forces it), so we skip Qt/napari
    # entirely rather than crash on a missing display.  client/ops/execute_code
    # still work; `viewer` is a self-describing sentinel.
    headless = bool(os.environ.get("BIOPB_HEADLESS"))

    # 1. Qt integration must be enabled before the viewer is created so napari
    #    shares the kernel's integrated Qt event loop (programmatic %gui qt).
    #    Do it FIRST — before the heavy core imports below (dask.array, and on
    #    some platforms napari, get pulled in there) and long before the ~10 s
    #    napari.Viewer(). enable_gui("qt") is cheap (~0.1 s) and needs none of
    #    those deps, so popping the splash here covers the *whole* slow stretch;
    #    showing it after the imports (as before) left several seconds of blank
    #    screen the splash was meant to hide (issue #386). Best-effort: show_splash
    #    fails open to _NullSplash when Qt is unavailable, and the headless branch
    #    (no Qt loop) keeps the _NullSplash default below.
    from ._splash import _NullSplash, show_splash

    splash = _NullSplash()  # replaced below when a real one can be shown
    if not headless:
        ip.enable_gui("qt")
        splash = show_splash()

    # Heavy core imports, now covered by the splash. dask.array is the slow one
    # here; napari is pulled in transitively on some platforms, so this is the
    # phase the "Loading napari…" cue is for (the later `import napari` is then a
    # no-op — see step 4). numpy/da are bound for the execute_code namespace.
    splash.message("Loading napari…")
    import dask.array as da
    import numpy as np

    from .._connection import TensorConnection
    from . import _jobs
    from ._process_ops import build_ops

    # 2. Data-access service (dask-free), shared by the widget and the agent
    #    namespace. Created before dask so the viewer can come up without waiting
    #    on the distributed Client attach below.
    conn = TensorConnection(config)

    # 3. Attach dask on a background thread so the viewer opens immediately. The
    #    cluster is session-child-owned and may still be registering workers, and
    #    even a bare Client(address) connect costs a round-trip; the viewer never
    #    needs the distributed cluster (its interactive reads pin to a
    #    single-process scheduler, issue #8) — only the agent's explicit
    #    da.compute() uses the distributed default, which is set once the Client
    #    attaches. Until then `_dask_client` is None; interrupt_kernel /
    #    server_status guard for that.
    import threading

    ip.user_ns["_dask_client"] = None
    # False until the attach thread resolves (to a Client or, for threads mode /
    # a degrade, None). Lets server_status distinguish "still attaching" from
    # "no distributed cluster".
    ip.user_ns["_dask_attach_done"] = False

    # The connect hook and the attach thread race to register the chunk-cache
    # plugin; whichever runs second (both hold this lock) registers it, since it
    # needs both a ready Client and a live (url, token). register_plugin is named
    # / idempotent so a double-register is harmless. The kernel always attaches to
    # the session child's cluster, so the budget splits across its live worker
    # count (see _register_cache_plugin).
    _dask_lock = threading.Lock()
    _dask_state = {
        "client": None,
        "connected": False,
        "url": None,
        "token": None,
    }

    def _register_cache_if_ready():
        # Caller holds _dask_lock. Splits dask.cache_budget across the worker
        # processes (localhost workers clamp it to 0 themselves). No-op until both
        # a Client and a connection exist.
        client = _dask_state["client"]
        if client is None or not _dask_state["connected"]:
            return
        _register_cache_plugin(client, _dask_state["url"], _dask_state["token"], config)

    # on_connect fires (in the kernel) after every successful connect with the
    # final (url, token): it bounds the dask chunk cache (token only known
    # post-connect) and reports the token up to the launcher so it survives a
    # kernel restart (issue #86). The report hook is None when no report pipe is
    # configured (Windows, or a bare unit test).
    _report_token = _make_token_report_hook()

    def _on_connect(url, token):
        with _dask_lock:
            _dask_state.update(url=url, token=token, connected=True)
            _register_cache_if_ready()
        if _report_token is not None:
            _report_token(url, token)

    conn.on_connect = _on_connect

    def _attach_dask():
        client = _configure_dask(config)
        with _dask_lock:
            _dask_state["client"] = client
            ip.user_ns["_dask_client"] = client
            ip.user_ns["_dask_attach_done"] = True
            _register_cache_if_ready()

    threading.Thread(target=_attach_dask, name="biopb-dask-attach", daemon=True).start()

    # 4. Visible napari viewer + Tensor Browser (auto-connects on its own tick).
    #    compute_scheduler pins the viewer's serial slice reads to a
    #    single-process scheduler so they share the main-process chunk cache
    #    instead of scattering across the distributed cluster (issue #8).
    #    Headless: no viewer — `viewer` is a self-describing sentinel instead.
    compute_scheduler = get_setting(config, "viewer.compute_scheduler")
    if headless:
        viewer = _HeadlessViewer()
        logger.info("Headless mode: no napari viewer (no display).")
        # No widget exists to drive the initial connect (the GUI branch's
        # TensorBrowserWidget runs the same conn.auto_connect policy off a
        # worker thread), so drive it here. On a daemon thread: connect() blocks
        # on network I/O and we must not stall kernel bring-up (this runs in
        # exec_lines, ahead of start_kernel returning). execute_code refreshes
        # `client` from `_conn.client` per job, so a connect that lands after the
        # kernel is ready is still seen. (threading imported at step 3.)
        threading.Thread(
            target=conn.auto_connect,
            name="biopb-headless-connect",
            daemon=True,
        ).start()
    else:
        # Enable napari async slicing via its NAPARI_ASYNC env override, set
        # BEFORE importing napari. The settings singleton reads the env at load,
        # and the viewer's _LayerSlicer captures the flag once at construction
        # (_layer_slicer.py: ``self._force_sync = not ...async_``) -- so the env
        # var is the only reliable hook; assigning the settings object after
        # import is too late (the settings load resets it). Async slicing
        # fetches slices off the Qt main thread so a zoom into a not-yet-cached
        # level doesn't freeze the viewer (vispy keeps the current coarse
        # texture until the finer slice resolves); take_screenshot force-syncs a
        # slice before capturing so the agent still sees the requested frame
        # (resync_view_for_capture).
        os.environ["NAPARI_ASYNC"] = (
            "1" if get_setting(config, "viewer.async_slicing") else "0"
        )

        try:
            # napari was already pulled in by the core imports above (splash is
            # showing "Loading napari…" for that phase), so this import just
            # binds the name — the real cost is napari.Viewer() below.
            import napari

            from ..tensor_browser import TensorBrowserWidget

            splash.message("Opening viewer…")  # the slow step
            viewer = napari.Viewer()
            tbw = TensorBrowserWidget(
                viewer, connection=conn, compute_scheduler=compute_scheduler
            )
            viewer.window.add_dock_widget(tbw, name="Tensor Browser")
            # Hand the splash off to the viewer window (closes once it's shown).
            splash.finish(viewer)
            # Tear the kernel down to idle when the user closes the window: signal
            # the launcher's reader thread over the inherited window-close pipe.
            _install_window_close_hook(viewer)

            # Kernel-start update reminder (issue #87): once a window exists, check
            # in the background whether a newer release-v* deployment is available
            # and, if so, remind the user to run the upgrade script. GUI branch only;
            # never blocks window paint.
            _start_update_check(viewer, config)

        except Exception:
            # Happy path: finish() hands the splash off to the viewer window (it
            # closes once the window shows). If a step above fails first, close it
            # so it can't linger before the kernel is torn down, then re-raise for
            # bootstrap()'s BOOTSTRAP_ERROR handler.
            splash.close()
            raise

    # 5. ProcessImage ops: thin Run() callables for each configured servicer.
    #    client_getter reads conn.client lazily so the async-connecting tensor
    #    client is picked up at call time.
    max_msg_bytes = get_setting(config, "grpc.max_message_size_mb") * 1024 * 1024
    channel_options = [
        ("grpc.max_receive_message_length", max_msg_bytes),
        ("grpc.max_send_message_length", max_msg_bytes),
    ]
    try:
        ops = build_ops(
            client_getter=lambda: conn.client,
            server_urls=_algorithms.servers_from_config(config),
            op_names_timeout=get_setting(config, "timeout.get_op_names"),
            run_timeout=get_setting(config, "timeout.process_image"),
            channel_options=channel_options,
        )
    except Exception:
        logger.exception("Failed to build ProcessImage ops")
        ops = {}

    # 6. Async job runner: execute_code runs in a background kernel thread so
    #    the main thread / Qt loop stays free for screenshot/status mid-job.
    #    install() stores the shell, installs the thread-aware stdout streams,
    #    and clears any prior job state.
    _jobs.install(ip)
    # The agent-facing `viewer` is a main-thread marshaling proxy so arbitrary
    # job-thread code (viewer/layers/dims/camera mutations) can't segfault Qt --
    # the real viewer is touched only on the Qt main thread. Internal subsystems
    # (helpers, tools, the Tensor Browser widget) keep the real viewer. See
    # docs/viewer-thread-safety.md. Headless has no Qt loop, so no proxy.
    viewer_handle = viewer
    if not headless:
        from ._helpers import patch_viewer_add_tensor
        from ._viewer_proxy import make_viewer_proxy

        patch_viewer_add_tensor(viewer, conn, compute_scheduler=compute_scheduler)
        viewer_handle = make_viewer_proxy(viewer)

    # 7. Namespace for execute_code.  client is refreshed per-job by the job
    #    runner (the connection service connects asynchronously).
    #    _dask_client was seeded to None at step 3 and is filled by the background
    #    attach thread; not set here so it stays the sole writer (a threads-mode
    #    attach can finish before this runs).
    #    _viewer_window_alive lets the tools detect a user-closed window (the
    #    Python `viewer` survives a window close, so mutations silently no-op).
    from ._helpers import resync_view_for_capture, viewer_window_alive

    ip.user_ns.update(
        {
            "viewer": viewer_handle,
            "np": np,
            "da": da,
            "client": None,
            "ops": ops,
            "_conn": conn,
            "_jobs": _jobs,
            "run_on_main": _jobs.run_on_main,
            "_viewer_window_alive": lambda: viewer_window_alive(viewer),
            "_resync_view": lambda: resync_view_for_capture(viewer),
        }
    )

    # 7b. User "bring your own tool" plugins (#92): load *.py files from
    #     ~/.config/biopb/kernel/ and biopb_mcp.namespace entry points into the
    #     namespace now that the built-in handles (viewer/client/np/da/ops) exist,
    #     so a plugin's code can reference them. Fail-open per plugin; the reserved
    #     handles are guarded against a shadowing plugin.
    _load_namespace_plugins(ip, config)

    # 8. Background source-catalog watcher (issue #44): a daemon thread that
    #    health-checks the server and re-lists sources when its source_count
    #    changes, so a catalog cached while the server was still indexing
    #    self-heals — for the agent (reads `_conn.sources` live) and, in a GUI
    #    session, the widget (which wires its own tree rebuild and also starts
    #    the watch; the call is idempotent). Thread-based, not a QTimer, so it
    #    runs even headless where there is no Qt loop.
    try:
        conn.start_source_watch(
            min_interval=get_setting(config, "tensor.health_poll_min_interval"),
            max_interval=get_setting(config, "tensor.health_poll_max_interval"),
        )
    except Exception:
        logger.exception("Failed to start source watcher")
