"""Launcher for the biopb-mcp MCP server.

This process *is* the MCP server: it owns a child Jupyter kernel that hosts a
visible napari viewer (requires ``$DISPLAY``).  Run it with::

    biopb-mcp        # console script
    python -m biopb_mcp.mcp

Install the optional dependencies first: ``pip install biopb-mcp[mcp]``.
"""

import atexit
import logging
import os
import shutil
import signal
import tempfile

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    from .._config import load_config
    from . import _server
    from ._kernel import KernelHost

    config = load_config()
    mcp_config = config.get("mcp", {})
    port = mcp_config.get("port", 8765)

    bootstrap_line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
    extra_arguments = [f"--IPKernelApp.exec_lines={bootstrap_line}"]

    # Pin BLAS/OpenMP to one thread in the kernel.  numpy's OpenBLAS parallel
    # LU path (dgetrf_parallel, reached via np.linalg.inv) allocates a large
    # working buffer on the *caller's* stack; napari's StatusChecker QThread —
    # which inverts the layer affine on every cursor move — has only a ~512 KB
    # stack, so that buffer overruns the guard page and segfaults the viewer
    # (observed on Intel macOS).  These matrices are tiny, so single-threaded
    # BLAS costs nothing here.  Must be set before numpy is imported in the
    # child; setdefault leaves any explicit user override intact.
    kernel_env = os.environ.copy()
    kernel_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    kernel_env.setdefault("OMP_NUM_THREADS", "1")

    # Prefer the gRPC socket over the tensor server's /dev/shm fast-path: the
    # shm path creates+writes+unlinks a fresh POSIX segment per chunk, measured
    # ~2.4-3x slower than the socket for large localhost chunks. Driven by
    # config so the installer can seed it (no-op on Windows, no POSIX shm).
    # setdefault leaves any explicit shell override intact. Inherited by the
    # kernel and the dask LocalCluster workers it spawns.
    if mcp_config.get("tensor_disable_shm"):
        kernel_env.setdefault("BIOPB_SHM_TRANSFER_DISABLED", "1")

    # Let the data-plane client cache chunks for a localhost tensor server
    # (otherwise skipped as redundant with the server's own cache). Bridges
    # interactive viewer responsiveness: repeated/overlapping plane reads
    # within a chunk hit the cache instead of re-fetching the whole chunk.
    # Inherited by the LocalCluster workers, where the bootstrap's cache plugin
    # bounds each worker at dask_cache_budget // n_workers.
    if mcp_config.get("tensor_cache_local"):
        kernel_env.setdefault("BIOPB_CACHE_LOCAL", "1")

    # Launcher-owned scratch dir for the dask LocalCluster's worker spill files.
    # The launcher rmtree's it on shutdown so a group-SIGKILL of the kernel
    # (which leaves workers no chance to clean up) doesn't leak spill dirs
    # (issue #13, secondary disk-leak note).
    dask_local_dir = tempfile.mkdtemp(prefix="biopb-mcp-dask-")
    kernel_env["BIOPB_DASK_LOCAL_DIR"] = dask_local_dir

    def _cleanup_dask_dir():
        shutil.rmtree(dask_local_dir, ignore_errors=True)

    # Register now (before host.start()) so the scratch dir is still removed on
    # interpreter exit if start() raises. rmtree(ignore_errors) makes this and
    # the explicit calls on the os._exit paths harmless if they both run.
    atexit.register(_cleanup_dask_dir)

    host = KernelHost(
        extra_arguments=extra_arguments,
        kernel_name=mcp_config.get("kernel_name", "python3"),
        startup_timeout=mcp_config.get("kernel_startup_timeout", 60.0),
        execute_timeout=mcp_config.get("execute_timeout", 120.0),
        busy_lock_timeout=mcp_config.get("busy_lock_timeout", 5.0),
        env=kernel_env,
        watchdog_interval=mcp_config.get("watchdog_interval", 5.0),
        watchdog_max_respawns=mcp_config.get("watchdog_max_respawns", 3),
        watchdog_respawn_window=mcp_config.get(
            "watchdog_respawn_window", 60.0
        ),
        parent_death_pipe=mcp_config.get("parent_death_pipe", True),
    )
    _server.set_kernel_host(host)
    _server.set_promote_after(mcp_config.get("promote_after", 10.0))

    logger.info("Starting napari kernel (a viewer window will appear)...")
    host.start()
    logger.info("Kernel ready.")

    atexit.register(host.shutdown)

    def _handle_signal(signum, frame):
        logger.info("Received signal %s, shutting down.", signum)
        host.shutdown()
        _cleanup_dask_dir()
        # Skip Python finalization: this process still has a live asyncio/epoll
        # event-loop thread and the numpy OpenBLAS worker pool running, and
        # tearing down the interpreter on top of them segfaults inside
        # Py_FinalizeEx (refcount write into a read-only static-type page).
        # The launcher's only remaining job is to exit, so exit immediately.
        os._exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _server.run(
        port,
        allowed_origins=mcp_config.get("allowed_origins", []),
        allowed_hosts=mcp_config.get("allowed_hosts", []),
    )

    # If the server loop returns on its own, exit the same way: shut down the
    # kernel and bypass Py_FinalizeEx (atexit handlers do not run after
    # os._exit, so shut down explicitly here).
    host.shutdown()
    _cleanup_dask_dir()
    os._exit(0)


if __name__ == "__main__":
    main()
