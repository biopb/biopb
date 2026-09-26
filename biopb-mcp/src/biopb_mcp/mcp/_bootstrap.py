"""Bootstrap executed *inside* the MCP child kernel.

Injected via IPython ``exec_lines`` so it runs before the kernel services any
tool calls.  It populates the ``execute_code`` namespace: the data plane
(``client``), the algorithm plane (``ops`` and the user's plugin modules), and
-- when the session has a viewer -- enables the Qt event loop and opens a
napari viewer with the Tensor Browser widget. Dask is left as dask configures
itself: computes run in-process unless a cell builds a dask ``Client``.

A failure here does not abort the kernel (exec_lines errors are swallowed by
IPython), so ``bootstrap`` prints a ``BOOTSTRAP_ERROR`` sentinel that the
host's health probe detects via the absence of ``_jobs`` in the namespace.
"""

import logging
import os
import traceback

from ._jobs import KERNEL_HANDLE_NAMES

logger = logging.getLogger(__name__)


def is_scratch_kernel():
    """Whether this kernel was spawned to verify a workflow (``_scratch``).

    A scratch kernel is the bootstrap's machinery -- the job runner and the
    connection -- and **none of its workflow handles**: no
    ``viewer``, ``np``, ``da``, ``client`` or ``ops``, and no user plugins.
    Whoever opens the saved notebook gets a bare kernel, so the run that
    verifies it gets one too, and what the document needs it builds for itself
    (``biopb_mcp.workflow_env``). Anything bound here for free is something a
    workflow can pass on and then fail on.

    No Qt either, so it needs no display at all. A hidden
    ``napari.Viewer(show=False)`` is no substitute: its screenshots come back
    black, and under an offscreen Qt platform it renders nothing at all.

    The literal mirrors ``_kernel.ENV_SCRATCH``, which is where the launcher
    sets it; spelled out rather than imported because ``_kernel`` belongs to the
    session child and this module runs in the kernel.
    """
    return bool(os.environ.get("BIOPB_SCRATCH_KERNEL"))


def no_viewer_reason():
    """Why this kernel has no napari viewer, or None when it has one.

    A scratch kernel never has one: the viewer is how an agent shows something
    to a person, and a verification has no person in it. Any other kernel has
    none when the launcher says so -- it decides (config, whether napari is
    installed, whether there is a display) and hands the kernel its reason in
    ``BIOPB_NO_VIEWER``; the literal mirrors ``_kernel.ENV_NO_VIEWER``.
    """
    if is_scratch_kernel():
        return "a scratch kernel verifies a workflow and has no viewer"
    return os.environ.get("BIOPB_NO_VIEWER") or None


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
    the plugin dir is on no ``sys.path`` but this kernel's. So without this, a
    plugin function inside a ``da.map_blocks`` would fail at compute time --
    whenever a cluster is attached, far from the load that caused it. Fail-open:
    in-process use (the default since #970) still works.
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
    when ``client`` is still the ``None`` seeded at step 6 -- a hook wanting the
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
    Called after the built-in handles exist (step 6) so plugins can reference them.
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
        # swallowed by IPython, leaving the probe with only "_jobs absent".
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

    # A kernel without a viewer skips Qt entirely: no event loop, no GL, no
    # display, and none of napari's ~330 MiB.
    want_viewer = no_viewer_reason() is None
    if want_viewer:
        ip.enable_gui("qt")
        splash = show_splash()
    else:
        splash = _NullSplash()

    # Heavy core imports, now covered by the splash. dask.array is the slow one
    # here; napari is pulled in transitively on some platforms, so this is the
    # phase the "Loading napari…" cue is for (the later `import napari` is then a
    # no-op — see step 3). numpy/da are bound for the execute_code namespace.
    splash.message("Loading napari…")  # a no-op without a viewer
    import dask.array as da
    import numpy as np
    from biopb.tensor import Connection

    from . import _jobs
    from ._process_ops import build_ops_from_config

    # 2. The data-plane connection, shared by the widget and the agent
    #    namespace: the widget connects it, and each agent cell reads its client.
    conn = Connection()

    # 3. napari viewer + Tensor Browser -- when the kernel has one.
    viewer = None
    if want_viewer:
        #    The browser auto-connects on its own tick. compute_scheduler pins
        #    the viewer's serial slice reads to a single-process scheduler so
        #    they share the main-process chunk cache instead of scattering
        #    across a cluster a cell attached (issue #8).
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
            from biopb_napari_widget import TensorBrowserWidget

            splash.message("Opening viewer…")  # the slow step
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

    # 4. ProcessImage ops: thin Run() callables for each configured servicer.
    #    client_getter reads conn.client lazily so the async-connecting tensor
    #    client is picked up at call time.
    try:
        ops = build_ops_from_config(config, lambda: conn.client)
    except Exception:
        logger.exception("Failed to build ProcessImage ops")
        ops = {}

    # 5. The kernel's side of the jobs: cells held for Stop, run_async tasks.
    #    install() stores the shell and clears any prior job state.
    _jobs.install(ip)
    # 6. Namespace for execute_code.  client is refreshed before each agent
    #    cell (the connection service connects asynchronously).
    #    _viewer_window_alive lets the tools detect a user-closed window (the
    #    Python `viewer` survives a window close, so mutations silently no-op).
    ns = {
        "_conn": conn,
        "_jobs": _jobs,
    }
    if not is_scratch_kernel():
        # **A scratch kernel binds no workflow handles**, for the reason it
        # builds no viewer: what it verifies is a document someone will run
        # somewhere else, and the reader gets `np`, `da`, `client` and `ops`
        # only if the document builds them (`biopb_mcp.workflow_env`). Handing
        # them to the verification would prove a program that runs here and
        # nowhere else -- the exact defect this exists to catch, moved one level
        # up from variables to the environment. A document that forgets its
        # setup cell raises NameError, which is the verdict.
        ns.update(
            {
                "np": np,
                "da": da,
                "client": None,
                "ops": ops,
                # A long compute off the main thread; a workflow document
                # does not get it, since its reader has no such helper
                # either.
                "run_async": _jobs.run_async,
            }
        )
    if viewer is not None:
        # The agent-facing `viewer` is a main-thread marshaling proxy so
        # arbitrary job-thread code (viewer/layers/dims/camera mutations) can't
        # segfault Qt -- the real viewer is touched only on the Qt main thread.
        # Internal subsystems (helpers, tools, the Tensor Browser widget) keep
        # the real viewer.
        from ._helpers import (
            patch_viewer_tensor_methods,
            resync_view_for_capture,
            viewer_window_alive,
        )
        from ._viewer_proxy import make_viewer_proxy

        patch_viewer_tensor_methods(viewer, conn, compute_scheduler=compute_scheduler)
        ns["viewer"] = make_viewer_proxy(viewer)
        ns["_viewer_window_alive"] = lambda: viewer_window_alive(viewer)
        ns["_resync_view"] = lambda: resync_view_for_capture(viewer)
    # `viewer` is simply absent without one -- in a scratch kernel a workflow
    # that uses it raises NameError, which is the verdict. The job-status snippet already
    # reads _viewer_window_alive with a default, so its absence is expected.
    ip.user_ns.update(ns)

    # 7. User "bring your own tool" plugins (#92): load *.py files from
    #     ~/.config/biopb/kernel/ and biopb_mcp.namespace entry points into the
    #     namespace now that the built-in handles (viewer/client/np/da/ops) exist,
    #     so a plugin's code can reference them. Fail-open per plugin; the reserved
    #     handles are guarded against a shadowing plugin.
    if not is_scratch_kernel():
        # Not here either, and for the same reason: `workflow_env()` loads them
        # for the reader, so the verification has to reach them the same way.
        _load_namespace_plugins(ip, config)
