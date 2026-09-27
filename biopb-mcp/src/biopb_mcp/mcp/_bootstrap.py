"""Bootstrap executed *inside* the MCP child kernel.

Injected via IPython ``exec_lines`` so it runs before the kernel services any
tool calls.  It populates the ``execute_code`` namespace: the data plane
(``client``), the algorithm plane (``ops``), and -- when the session has a
viewer -- enables the Qt event loop and opens a napari viewer with the Tensor
Browser widget. Dask is left as dask configures itself: computes run
in-process unless a cell builds a dask ``Client``.

A failure here does not abort the kernel (exec_lines errors are swallowed by
IPython), so ``bootstrap`` prints a ``BOOTSTRAP_ERROR`` sentinel that the
host's health probe detects via the absence of ``_jobs`` in the namespace.
"""

import logging
import os
import traceback

logger = logging.getLogger(__name__)


def is_scratch_kernel():
    """Whether this kernel was spawned to verify a workflow (``_scratch``).

    A scratch kernel is the bootstrap's machinery -- the job runner and the
    connection -- and **none of its workflow handles**: no
    ``viewer``, ``np``, ``da``, ``client`` or ``ops``.
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
    from ._process_ops import Ops, build_ops_from_config

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

    # 4. The algorithm plane's ops, bound from the control's registry.
    #    client_getter reads conn.client lazily so the async-connecting tensor
    #    client is picked up at call time.
    try:
        ops = build_ops_from_config(config, lambda: conn.client)
    except Exception:
        logger.exception("Failed to build the ops")
        # Unbound rather than a bare {}, so callers (ops.status(), the kernel
        # banner) see one type of `ops` regardless of which branch ran.
        ops = Ops(lambda: conn.client)

    # 5. The kernel's side of the jobs: cells held for Stop, run_async tasks.
    #    install() stores the shell and clears any prior job state.
    _jobs.install(ip)
    # 6. Namespace for execute_code.
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

        # `client` tracks the connection, which connects asynchronously.
        # Refreshed from an event hook rather than a line prepended to the
        # cell, which would shift every traceback line number by one.
        def _refresh_client(_info):
            ip.user_ns["client"] = conn.client

        ip.events.register("pre_run_cell", _refresh_client)
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
