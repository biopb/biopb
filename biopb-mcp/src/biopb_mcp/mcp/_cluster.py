"""Daemon-owned dask cluster host.

The MCP daemon (``mcp/__main__.py::_serve_http``) owns the distributed
``LocalCluster`` and injects its scheduler address into each kernel it launches
(``BIOPB_DASK_ADDRESS``); the kernel attaches with a bare ``Client(address)``
instead of spinning its own.  That decouples the cluster's lifetime from the
kernel's: a ``restart_kernel`` / watchdog respawn / viewer-window close no longer
tears the cluster down and re-spins N workers — the dominant restart cost on
Windows, where each worker is a cold spawn (no fork).  Only a real daemon exit
(the ``_shutdown`` chokepoint) closes it.

Deliberately GUI-free and import-light: ``dask.distributed`` is imported inside
``ensure()`` (the cluster is spun lazily on the first kernel launch), so an idle
or never-started daemon pays nothing.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Env var carrying the daemon cluster's scheduler address into the kernel, read
# by _bootstrap._configure_dask ahead of the mcp.dask.address config value.
DASK_ADDRESS_ENV = "BIOPB_DASK_ADDRESS"


class DaskClusterHost:
    """Own a distributed ``LocalCluster`` on behalf of the MCP daemon.

    Lazily spins the cluster on the first :meth:`ensure` (i.e. the first kernel
    launch) and keeps it warm across kernel restarts.  :meth:`ensure` returns the
    scheduler address to inject into the kernel, or ``None`` when the daemon
    should not own a cluster (``owner != "daemon"``, a non-distributed scheduler,
    an external ``mcp.dask.address``, or a spin failure) — the kernel then falls
    back per its own config (see ``_bootstrap._configure_dask``).
    """

    def __init__(self, config, local_dir=None):
        self._config = config
        self._local_dir = local_dir
        self._cluster = None
        self._address = None
        # ensure() may be reached from the kernel launch path; a lock keeps a
        # spin and a concurrent close() from racing the _cluster/_address pair.
        self._lock = threading.Lock()

    def _should_own(self):
        """Whether the daemon should spin/own a cluster, per config.

        Only when it is explicitly the daemon's job, the scheduler is
        distributed, and no external scheduler was configured (the kernel
        attaches to an external ``mcp.dask.address`` directly, daemon owns
        nothing).
        """
        from .._config import get_setting

        return (
            get_setting(self._config, "mcp.dask.owner") == "daemon"
            and get_setting(self._config, "mcp.dask.scheduler") == "distributed"
            and not get_setting(self._config, "mcp.dask.address")
        )

    def ensure(self):
        """Return the scheduler address, spinning the cluster on first use.

        Idempotent: later calls return the cached address after a cheap liveness
        check, re-spinning if the cached cluster has died — this restores the
        self-healing that per-kernel-restart cluster creation gave for free (a
        dead cluster behind a live daemon would otherwise strand every later
        kernel on the threads scheduler).  Returns ``None`` (and injects no
        address) when the daemon should not own a cluster, or when spinning
        fails.
        """
        with self._lock:
            if not self._should_own():
                return None
            if self._cluster is not None:
                if self._is_alive(self._cluster):
                    return self._address
                logger.warning("Daemon dask cluster is not healthy; re-spinning.")
                self._close_locked()
            return self._spin_locked()

    def _spin_locked(self):
        from .._config import get_setting

        try:
            from dask.distributed import LocalCluster
        except Exception:
            # No distributed install: the kernel degrades to threads (owner
            # "daemon" + no injected address -> threads, per _configure_dask).
            logger.exception("distributed unavailable; kernel will degrade to threads")
            return None
        try:
            # 0 -> None so dask picks ~n_cores, matching the kernel-owned path
            # (mcp.dask.num_workers `or None`).
            num_workers = get_setting(self._config, "mcp.dask.num_workers") or None
            cluster = LocalCluster(
                n_workers=num_workers,
                processes=True,
                threads_per_worker=get_setting(
                    self._config, "mcp.dask.threads_per_worker"
                ),
                memory_limit=get_setting(self._config, "mcp.dask.memory_limit"),
                dashboard_address=get_setting(
                    self._config, "mcp.dask.dashboard_address"
                ),
                local_directory=self._local_dir or None,
            )
        except Exception:
            logger.exception(
                "Failed to spin daemon dask cluster; kernel will degrade to threads"
            )
            return None
        self._cluster = cluster
        self._address = cluster.scheduler_address
        logger.info(
            "Daemon dask cluster: %d worker(s) at %s",
            len(cluster.workers),
            self._address,
        )
        return self._address

    @staticmethod
    def _is_alive(cluster):
        """Cheap best-effort liveness: scheduler running with >=1 live worker.

        Reads the in-process ``LocalCluster`` scheduler's live worker map (no
        Client needed).  Any error -> treat as dead so ensure() re-spins.
        """
        try:
            status = getattr(cluster, "status", None)
            if getattr(status, "name", None) != "running":
                return False
            return len(cluster.scheduler.workers) > 0
        except Exception:
            return False

    def close(self):
        """Best-effort, idempotent teardown of the owned cluster.

        Called only on real daemon exit (the ``_shutdown`` chokepoint + atexit),
        never on a kernel restart/reap — that is what keeps the workers warm.
        """
        with self._lock:
            self._close_locked()

    def _close_locked(self):
        cluster = self._cluster
        self._cluster = None
        self._address = None
        if cluster is None:
            return
        try:
            cluster.close()
        except Exception:
            logger.debug("dask cluster close failed", exc_info=True)
