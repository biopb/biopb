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

from ._jobs import KERNEL_HANDLE_NAMES

logger = logging.getLogger(__name__)


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


def is_scratch_kernel():
    """Whether this kernel was spawned to verify a workflow (``_scratch``).

    A scratch kernel is a full bootstrap with the user-facing parts left out:
    nobody is watching it, and everything it builds is thrown away with the
    process a few seconds later. The one difference that is not cosmetic is the
    viewer -- ``napari.Viewer(show=False)`` maps no window, so the scratch kernel
    can take the session's own display, and its real GPU, without a window
    appearing in front of the user. Offscreen is not an alternative: it creates
    no GL context at all, so nothing renders and screenshots come back empty
    (docs/verification-scratch-kernel.md, "The display").

    The literal mirrors ``_kernel.ENV_SCRATCH``, which is where the launcher
    sets it; spelled out rather than imported because ``_kernel`` belongs to the
    session child and this module runs in the kernel.
    """
    return bool(os.environ.get("BIOPB_SCRATCH_KERNEL"))


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


def _start_update_check(viewer, config):
    """Kick off the kernel-start update reminder (issue #87), GUI branch only.

    Runs the network version check on a daemon thread so it can never delay
    window paint, then marshals a window-only reminder popup to the Qt main
    thread via ``run_on_main`` (which the popup returns from immediately — it
    ``.show()``s rather than ``.exec()``s). Fully best-effort and fail-open: the
    check itself swallows every error, and this wrapper swallows the rest, so it
    never disturbs a working session. The caller invokes this only when a real
    napari window exists.

    This is a *notify-only* reminder: it tells the user to run the install/
    upgrade script. biopb does not self-update (a graceful cross-platform apply
    needs a staging step we don't handle yet — see issue #87).
    """
    import threading

    def _worker():
        try:
            from ._jobs import run_on_main
            from ._update import check_for_update, handle_choice
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


# Load-bearing namespace names a user plugin (#92) must not shadow. A plugin now
# contributes exactly one binding -- its module -- so this is a single check per
# plugin rather than a sweep over everything it happened to define (#664).
# The list itself is `_jobs.KERNEL_HANDLE_NAMES`, which a scratch verification
# also seeds from: a handle added here and not there would be verified against a
# namespace that lacks it.
_RESERVED_NAMES = KERNEL_HANDLE_NAMES
# Plugin modules live under this prefix in ``sys.modules``, never under their bare
# stem: a user file named ``skimage.py`` must not be able to claim
# ``sys.modules["skimage"]`` for everything imported after it. The prefix is a key,
# not an importable package — nothing imports it, and by-value pickling (see
# _bind_by_value) means no unpickler ever resolves the name either.
_PLUGIN_MODULE_PREFIX = "biopb_kernel_plugins"


class _PluginLoader:
    """Hands back a plugin module that is already loaded. Nothing re-executes."""

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        """No-op: the file ran once, at bootstrap."""


class _PluginImportHook:
    """Make ``import <stem>`` reach a loaded kernel plugin.

    **Because ``import`` is what anyone writes.** A plugin is bound as a *name*
    in the namespace, which is the cheap part of #92 — but a name that exists
    while ``import <stem>`` raises `ModuleNotFoundError` is a design that reads
    as broken, and documenting the difference is a weaker fix than not having
    one. A benchmarked agent read `server_status`, saw `files: image_resolution`,
    wrote the import every Python programmer writes, got a traceback, and went
    looking for the file on disk (session 20260810-172816).

    **Appended to `sys.meta_path`, never prepended**, which is what keeps the
    guarantee the module prefix exists for. The standard finders run first, so a
    real installed package always wins and a user's `skimage.py` can never
    answer for `skimage`; this hook is consulted only once nothing else can
    resolve the name. `sys.modules[stem] = mod` would *not* be equivalent —
    imports short-circuit on `sys.modules` before any finder runs, so it would
    shadow a package imported later in the session, which is the exact hazard
    `_PLUGIN_MODULE_PREFIX` was introduced to close.

    Top-level names only (`path is None`): a plugin never answers for a
    submodule of a real package.
    """

    def __init__(self):
        self._modules: dict[str, object] = {}

    def register(self, stem: str, module) -> None:
        self._modules[stem] = module

    def unregister(self, stem: str) -> None:
        self._modules.pop(stem, None)

    def find_spec(self, fullname, path=None, target=None):
        if path is not None:
            return None
        module = self._modules.get(fullname)
        if module is None:
            return None
        import importlib.util

        return importlib.util.spec_from_loader(fullname, _PluginLoader(module))


#: One hook per process, installed on first use and left in place.
_PLUGIN_IMPORT_HOOK = _PluginImportHook()


def _install_plugin_import_hook() -> None:
    import sys

    if _PLUGIN_IMPORT_HOOK not in sys.meta_path:
        sys.meta_path.append(_PLUGIN_IMPORT_HOOK)


def _public_names(mapping: dict) -> dict:
    """The names a mapping plugin contributes: ``__all__`` if declared, else every
    public (non-``_``) name that is not itself an imported module (so a plugin's
    ``import numpy as np`` doesn't leak ``np`` into the namespace)."""
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


def _bind_one(ip, name: str, value, *, source: str) -> bool:
    """Bind a plugin's single contributed *name*, refusing a reserved one."""
    if name in _RESERVED_NAMES:
        logger.warning(
            "kernel plugin %s would shadow reserved name %r; skipped", source, name
        )
        return False
    ip.user_ns[name] = value
    logger.info("Loaded kernel plugin: %s (from %s)", name, source)
    return True


def _pickle_by_value(mod) -> None:
    """Make *mod*'s functions pickle by value, so they survive the trip to a dask
    worker.

    The old exec-into-the-namespace loader got this for free: a function defined in
    ``user_ns`` reports ``__module__ == "__main__"``, which cloudpickle always
    serializes by value. A function reached through an imported module pickles by
    *reference* instead -- a few bytes naming a module no worker can import, since
    the plugin dir is on no ``sys.path`` but this kernel's. ``dask.scheduler``
    defaults to distributed and the guides steer the agent toward dask, so without
    this a plugin function inside a ``da.map_blocks`` would fail at compute time,
    far from the load that caused it. Fail-open: in-process use still works.
    """
    try:
        import cloudpickle
    except ImportError:  # no distributed stack → nothing ships to a worker
        return
    try:
        cloudpickle.register_pickle_by_value(mod)
    except Exception:
        logger.warning(
            "kernel plugin %s: could not register for by-value pickling; its "
            "functions will not run on a dask worker",
            getattr(mod, "__name__", mod),
            exc_info=True,
        )


def _load_plugin_files(ip, plugin_dir) -> list[str]:
    """Import each ``*.py`` in *plugin_dir* as a module, bound under its stem.

    A plugin contributes exactly **one** name -- its module -- so its helpers and
    imports stay on the module instead of landing in the agent's namespace (#664).
    ``dir()`` then names the plugin rather than its parts, and
    ``inspect_object("<stem>")`` prints the module docstring plus every public
    callable with its signature.

    Loaded from the path, not by import: the kernel's interpreter need not be the
    tool env, so the plugin dir is reachable where installed-package metadata is
    not. Fail-open per file.

    Returns the stems that bound, for ``_requires.record_loaded_plugins`` -- the
    file being on disk is not the same fact, precisely because this is fail-open.
    """
    try:
        paths = sorted(plugin_dir.glob("*.py"))
    except OSError:
        return []
    import importlib.util
    import sys

    loaded = []
    for path in paths:
        if path.name.startswith("_"):
            continue
        module_name = f"{_PLUGIN_MODULE_PREFIX}.{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"no import machinery for {path}")
            mod = importlib.util.module_from_spec(spec)
            # Registered before exec so a plugin that re-imports itself (directly
            # or via pickle/dataclass machinery) resolves this same object rather
            # than executing the file a second time.
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(module_name, None)
            logger.exception("kernel plugin file %s failed to load", path.name)
            continue
        if _bind_one(ip, path.stem, mod, source=path.name):
            _pickle_by_value(mod)
            # So `import <stem>` finds it too, not only the bound name.
            _install_plugin_import_hook()
            _PLUGIN_IMPORT_HOOK.register(path.stem, mod)
            loaded.append(path.stem)
        else:
            sys.modules.pop(module_name, None)
            _PLUGIN_IMPORT_HOOK.unregister(path.stem)
    return loaded


def _load_entry_point_plugins(ip) -> list[str]:
    """Load ``biopb_mcp.namespace`` entry-point packages into the namespace.

    A module or mapping entry point binds **one** name -- the entry-point name --
    like a plugin file does (#664); a mapping is wrapped in a namespace object so
    its members are reached the same way. A ``register(namespace)`` callable stays
    the escape hatch for a plugin that must bind several names itself: it is called
    with a read-through snapshot of the namespace and only its new bindings are
    merged, each past the reserved-name guard. That snapshot is taken at load time,
    when ``client`` is still the ``None`` seeded at step 7 -- a hook wanting the
    live client must read it per call (see ``plugins/__init__.py``).

    Fail-open per entry point. Returns the names that loaded (see
    :func:`_load_plugin_files`).
    """
    from biopb._kernel_plugins import NAMESPACE_ENTRY_POINT_GROUP

    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return []
    try:
        eps = list(entry_points(group=NAMESPACE_ENTRY_POINT_GROUP))
    except Exception:
        logger.debug("kernel plugin: entry-point discovery failed", exc_info=True)
        return []

    import types
    from collections.abc import Mapping

    loaded = []
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:
            logger.exception("kernel plugin entry point %r failed to import", ep.name)
            continue
        try:
            if isinstance(obj, types.ModuleType):
                if not _bind_one(ip, ep.name, obj, source=ep.name):
                    continue
                _pickle_by_value(obj)
            elif isinstance(obj, Mapping):
                # Wrapped, not merged: a mapping is a namespace-like source, so it
                # binds one name like a module does. Filtered the same way too —
                # public names / honor __all__, drop the odd dunder. (A register()
                # hook, by contrast, writes literally.)
                holder = types.SimpleNamespace(**_public_names(dict(obj)))
                if not _bind_one(ip, ep.name, holder, source=ep.name):
                    continue
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
                logger.info(
                    "Loaded kernel plugin entry point: %s (register hook, %d name(s))",
                    ep.name,
                    len(writes),
                )
            else:
                logger.warning(
                    "kernel plugin entry point %r is not a register()/module/mapping;"
                    " ignored",
                    ep.name,
                )
                continue
            loaded.append(ep.name)
        except Exception:
            logger.exception("kernel plugin entry point %r failed to load", ep.name)
    return loaded


def _load_namespace_plugins(ip, config) -> None:
    """Load user "bring your own tool" plugins into the kernel namespace (#92).

    Two sources, both fail-open per unit so one bad plugin never breaks the
    bootstrap (the ``build_ops`` / skills precedent): ``*.py`` files under
    ``~/.config/biopb/kernel/`` and installed ``biopb_mcp.namespace`` entry points.
    Called after the built-in handles exist (step 7) so plugins can reference them.
    Gated by ``services.namespace_enabled``.

    What loaded is reported to :mod:`._requires` and printed by ``server_status``,
    so a skill's ``plugin:<name>`` is answered from the load's actual outcome
    instead of from the presence of a file this fail-open loader may have skipped.
    """
    from .._config import get_setting
    from . import _requires

    if not get_setting(config, "services.namespace_enabled", True):
        logger.info("kernel plugins disabled (services.namespace_enabled=false)")
        _requires.record_loaded_plugins(enabled=False)
        return
    from biopb._locations import mcp_plugin_dir

    files, entry_points = [], []
    try:
        files = _load_plugin_files(ip, mcp_plugin_dir())
    except Exception:
        logger.exception("kernel plugin: plugin-file load failed")
    try:
        entry_points = _load_entry_point_plugins(ip)
    except Exception:
        logger.exception("kernel plugin: entry-point load failed")
    _requires.record_loaded_plugins(files, entry_points)


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
    from IPython import get_ipython

    from .._config import get_setting, load_config

    ip = get_ipython()
    config = load_config()

    # 1. Qt integration must be enabled before the viewer is created so napari
    #    shares the kernel's integrated Qt event loop (programmatic %gui qt).
    #    Do it FIRST — before the heavy core imports below (dask.array, and on
    #    some platforms napari, get pulled in there) and long before the ~10 s
    #    napari.Viewer(). enable_gui("qt") is cheap (~0.1 s) and needs none of
    #    those deps, so popping the splash here covers the *whole* slow stretch;
    #    showing it after the imports (as before) left several seconds of blank
    #    screen the splash was meant to hide (issue #386). Best-effort: show_splash
    #    fails open to _NullSplash when Qt is unavailable.
    from ._splash import _NullSplash, show_splash

    # A scratch kernel is headless *by policy*, not by circumstance: the viewer
    # exists so an agent can show something to a person, and a verification has
    # no person in it. Skipping Qt entirely is what makes that policy free --
    # no event loop, no GL, no display, and none of napari's ~330 MiB.
    if is_scratch_kernel():
        splash = _NullSplash()
    else:
        ip.enable_gui("qt")
        splash = show_splash()

    # Heavy core imports, now covered by the splash. dask.array is the slow one
    # here; napari is pulled in transitively on some platforms, so this is the
    # phase the "Loading napari…" cue is for (the later `import napari` is then a
    # no-op — see step 4). numpy/da are bound for the execute_code namespace.
    splash.message("Loading napari…")  # a no-op in a scratch kernel
    import dask.array as da
    import numpy as np

    from .._connection import TensorConnection
    from . import _jobs
    from ._process_ops import build_ops_from_config

    # 2. Data-access service (dask-free), shared by the widget and the agent
    #    namespace. Created before dask so the viewer can come up without waiting
    #    on the distributed Client attach below.
    conn = TensorConnection()

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
    # final (url, token), which is what bounds the dask chunk cache -- the token is
    # only known post-connect.
    def _on_connect(url, token):
        with _dask_lock:
            _dask_state.update(url=url, token=token, connected=True)
            _register_cache_if_ready()

    conn.on_connect = _on_connect

    def _attach_dask():
        client = _configure_dask(config)
        with _dask_lock:
            _dask_state["client"] = client
            ip.user_ns["_dask_client"] = client
            ip.user_ns["_dask_attach_done"] = True
            _register_cache_if_ready()

    threading.Thread(target=_attach_dask, name="biopb-dask-attach", daemon=True).start()

    # 4. napari viewer + Tensor Browser -- unless this is a scratch kernel.
    #
    # **A scratch kernel is headless by policy.** The viewer is how an agent
    # shows something to a person; a verification has no person in it, so a
    # workflow that reaches for `viewer` is a workflow that will not run as the
    # document it is about to become -- the saved notebook has no viewer either
    # (`_notebook.WORKFLOW_BOOTSTRAP_SRC`). Failing here is the two ends
    # agreeing, which is the rule this whole feature rests on.
    #
    # It also costs nothing to enforce: no Qt, no GL, no display, and ~330 MiB
    # and ~1.7 s of napari.Viewer() that nobody was going to look at.
    viewer = None
    if not is_scratch_kernel():
        # 4. napari viewer + Tensor Browser (auto-connects on its own tick).
        #    compute_scheduler pins the viewer's serial slice reads to a
        #    single-process scheduler so they share the main-process chunk cache
        #    instead of scattering across the distributed cluster (issue #8).
        compute_scheduler = get_setting(config, "viewer.compute_scheduler")
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
            if is_scratch_kernel():
                # Hidden, and alone: no dock widget, no window-close hook (there is
                # no window to close and the pipe would report a teardown to the
                # launcher that means nothing), no update check (the user is not
                # here, and this process outlives nothing). A fresh empty viewer is
                # the point -- it is what makes a workflow that leans on a layer the
                # live session produced fail here, which is the defect verification
                # exists to catch.
                viewer = napari.Viewer(show=False)
                splash.close()
            else:
                viewer = napari.Viewer()
                tbw = TensorBrowserWidget(
                    viewer, connection=conn, compute_scheduler=compute_scheduler
                )
                viewer.window.add_dock_widget(tbw, name="Tensor Browser")
                # Hand the splash off to the viewer window (closes once it's shown).
                splash.finish(viewer)
                # Tear the kernel down to idle when the user closes the window:
                # signal the launcher's reader thread over the inherited pipe.
                _install_window_close_hook(viewer)

                # Kernel-start update reminder (issue #87): once a window exists,
                # check in the background whether a newer release-v* deployment is
                # available and, if so, remind the user to run the upgrade script.
                # Never blocks window paint.
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
    try:
        ops = build_ops_from_config(config, lambda: conn.client)
    except Exception:
        logger.exception("Failed to build ProcessImage ops")
        ops = {}

    # 6. Async job runner: execute_code runs in a background kernel thread so
    #    the main thread / Qt loop stays free for screenshot/status mid-job.
    #    install() stores the shell, installs the thread-aware stdout streams,
    #    and clears any prior job state.
    _jobs.install(ip)
    # 7. Namespace for execute_code.  client is refreshed per-job by the job
    #    runner (the connection service connects asynchronously).
    #    _dask_client was seeded to None at step 3 and is filled by the background
    #    attach thread; not set here so it stays the sole writer (a threads-mode
    #    attach can finish before this runs).
    #    _viewer_window_alive lets the tools detect a user-closed window (the
    #    Python `viewer` survives a window close, so mutations silently no-op).
    ns = {
        "np": np,
        "da": da,
        "client": None,
        "ops": ops,
        "_conn": conn,
        "_jobs": _jobs,
        # Safe to bind headless: with no QCoreApplication it runs `fn` inline
        # (see _jobs.run_on_main), so a plugin that marshals still works.
        "run_on_main": _jobs.run_on_main,
    }
    if viewer is not None:
        # The agent-facing `viewer` is a main-thread marshaling proxy so
        # arbitrary job-thread code (viewer/layers/dims/camera mutations) can't
        # segfault Qt -- the real viewer is touched only on the Qt main thread.
        # Internal subsystems (helpers, tools, the Tensor Browser widget) keep
        # the real viewer. See docs/viewer-thread-safety.md.
        from ._helpers import (
            patch_viewer_add_tensor,
            resync_view_for_capture,
            viewer_window_alive,
        )
        from ._viewer_proxy import make_viewer_proxy

        patch_viewer_add_tensor(viewer, conn, compute_scheduler=compute_scheduler)
        ns["viewer"] = make_viewer_proxy(viewer)
        ns["_viewer_window_alive"] = lambda: viewer_window_alive(viewer)
        ns["_resync_view"] = lambda: resync_view_for_capture(viewer)
    # `viewer` is simply absent in a scratch kernel -- a workflow that uses it
    # raises NameError, which is the verdict. The job-status snippet already
    # reads _viewer_window_alive with a default, so its absence is expected.
    ip.user_ns.update(ns)

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
    #    the watch; the call is idempotent). Thread-based, not a QTimer, so a
    #    busy Qt loop never starves the poll.
    try:
        conn.start_source_watch(
            min_interval=get_setting(config, "tensor.health_poll_min_interval"),
            max_interval=get_setting(config, "tensor.health_poll_max_interval"),
        )
    except Exception:
        logger.exception("Failed to start source watcher")
