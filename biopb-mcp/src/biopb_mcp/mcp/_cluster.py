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
# by _bootstrap._configure_dask ahead of the dask.address config value.
DASK_ADDRESS_ENV = "BIOPB_DASK_ADDRESS"


class DaskClusterHost:
    """Own a distributed ``LocalCluster`` on behalf of the MCP daemon.

    Lazily spins the cluster on the first :meth:`ensure` (i.e. the first kernel
    launch) and keeps it warm across kernel restarts.  :meth:`ensure` returns the
    scheduler address to inject into the kernel, or ``None`` when the daemon
    should not own a cluster (``owner != "daemon"``, a non-distributed scheduler,
    an external ``dask.address``, or a spin failure) — the kernel then falls
    back per its own config (see ``_bootstrap._configure_dask``).
    """

    def __init__(self, config, local_dir=None):
        self._config = config
        self._local_dir = local_dir
        self._cluster = None
        self._address = None
        # Whether the *current* cluster has ever had >=1 worker register, gating
        # the liveness check: a running scheduler with 0 workers means "dead"
        # only after we've seen workers, not during the initial spawn window (see
        # _is_alive). Reset on each spin.
        self._saw_workers = False
        # ensure() may be reached from the kernel launch path; a lock keeps a
        # spin and a concurrent close() from racing the _cluster/_address pair.
        self._lock = threading.Lock()

    def _should_own(self):
        """Whether the daemon should spin/own a cluster, per config.

        Only when it is explicitly the daemon's job, the scheduler is
        distributed, and no external scheduler was configured (the kernel
        attaches to an external ``dask.address`` directly, daemon owns
        nothing).
        """
        from .._config import get_setting

        return (
            get_setting(self._config, "dask.owner") == "daemon"
            and get_setting(self._config, "dask.scheduler") == "distributed"
            and not get_setting(self._config, "dask.address")
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
            # (dask.num_workers `or None`).
            num_workers = get_setting(self._config, "dask.num_workers") or None
            cluster = LocalCluster(
                n_workers=num_workers,
                processes=True,
                threads_per_worker=get_setting(self._config, "dask.threads_per_worker"),
                memory_limit=get_setting(self._config, "dask.memory_limit"),
                dashboard_address=get_setting(self._config, "dask.dashboard_address"),
                local_directory=self._local_dir or None,
            )
        except Exception:
            logger.exception(
                "Failed to spin daemon dask cluster; kernel will degrade to threads"
            )
            return None
        self._cluster = cluster
        self._address = cluster.scheduler_address
        # Fresh cluster: workers have not registered *from the liveness check's
        # standpoint* yet, so a 0-worker reading is startup, not death, until the
        # first time we observe them (see _is_alive).
        self._saw_workers = False
        logger.info(
            "Daemon dask cluster: %d worker(s) at %s",
            len(cluster.workers),
            self._address,
        )
        return self._address

    def _is_alive(self, cluster):
        """Cheap best-effort liveness, read off the in-process scheduler.

        A stopped scheduler (``status != "running"``) is always dead. For a
        running one the live worker count decides — but a 0-worker reading is
        ambiguous: it is a *dead* cluster (all workers gone) only once we have
        seen workers register (``_saw_workers``); before that it is the initial
        spawn window still coming up. Treating that window as dead would re-spin
        (and close) a cluster that is merely slow to bring workers up — precisely
        the Windows cold-spawn case this host exists to avoid — so we hold it
        alive until the first worker appears, then flip to the strict >=1 check
        (which restores the self-heal a dead cluster needs). Any error -> dead so
        ensure() re-spins.
        """
        try:
            status = getattr(cluster, "status", None)
            if getattr(status, "name", None) != "running":
                return False
            n_workers = len(cluster.scheduler.workers)
        except Exception:
            return False
        if n_workers > 0:
            self._saw_workers = True
            return True
        return not self._saw_workers

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
