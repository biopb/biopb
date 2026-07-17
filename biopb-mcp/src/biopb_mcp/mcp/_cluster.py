"""Daemon-owned dask cluster host.

The MCP daemon (``mcp/__main__.py::_serve_http``) owns the distributed
``LocalCluster`` and injects its scheduler address into each kernel it launches
(``BIOPB_DASK_ADDRESS``); the kernel attaches with a bare ``Client(address)``
instead of spinning its own.  That decouples the cluster's lifetime from the
kernel's: a ``restart_kernel`` / watchdog respawn / viewer-window close no longer
tears the cluster down and re-spins N workers — the dominant restart cost on
Windows, where each worker is a cold spawn (no fork).  A real daemon exit (the
``_shutdown`` chokepoint) closes it, as does the idle reaper (``start_reaper``)
once the cluster has sat with *no kernel attached* for ``dask.idle_ttl`` — the
bound on that decoupling, so a session whose viewer is closed but whose agent is
still connected stops holding N idle workers indefinitely (biopb/biopb#409).

Deliberately GUI-free and import-light: ``dask.distributed`` is imported inside
``ensure()`` (the cluster is spun lazily on the first kernel launch), so an idle
or never-started daemon pays nothing.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Ceiling on the reaper's poll interval. The TTL is coarse (minutes), so polling
# is cheap at this rate; a shorter TTL polls at the TTL itself, which is what
# keeps the reaper testable without a sleep(ttl).
_REAP_POLL_MAX = 30.0

# Env var carrying the daemon cluster's scheduler address into the kernel, read
# by _bootstrap._configure_dask ahead of the dask.address config value.
DASK_ADDRESS_ENV = "BIOPB_DASK_ADDRESS"


class DaskClusterHost:
    """Own a distributed ``LocalCluster`` on behalf of the session child.

    Lazily spins the cluster on the first :meth:`ensure` (i.e. the first kernel
    launch) and keeps it warm across kernel restarts.  :meth:`ensure` returns the
    scheduler address to inject into the kernel, or ``None`` when the session
    child should not own a cluster (a non-distributed scheduler, an external
    ``dask.address``, or a spin failure) — the kernel then falls back per its
    own config (see ``_bootstrap._configure_dask``).
    """

    def __init__(self, config, local_dir=None, kernel_alive=None):
        self._config = config
        self._local_dir = local_dir
        self._cluster = None
        self._address = None
        # Predicate: is a kernel currently attached to our cluster? The reaper
        # only counts idle time while this is false, because a live kernel holds
        # a Client() on our scheduler address and nothing re-injects a new one --
        # closing under it would strand it (see _reap_loop). Set post-construction
        # via set_kernel_alive (the KernelHost is built after us, taking this host
        # as an argument). None -> assume a kernel is always alive, i.e. never
        # reap: an unknown answer must not be read as "safe to close".
        self._kernel_alive = kernel_alive
        # Monotonic timestamp from which the current no-kernel stretch is
        # measured; None when a kernel is alive (or was, on the last poll).
        self._idle_since = None
        self._reap_thread = None
        self._reap_stop = threading.Event()
        # Whether the *current* cluster has ever had >=1 worker register, gating
        # the liveness check: a running scheduler with 0 workers means "dead"
        # only after we've seen workers, not during the initial spawn window (see
        # _is_alive). Reset on each spin.
        self._saw_workers = False
        # ensure() may be reached from the kernel launch path; a lock keeps a
        # spin and a concurrent close() from racing the _cluster/_address pair.
        self._lock = threading.Lock()

    def _should_own(self):
        """Whether the session child should spin/own a cluster, per config.

        Only when the scheduler is distributed and no external scheduler was
        configured (with an external ``dask.address`` the kernel attaches to it
        directly and the session child owns nothing).
        """
        from .._config import get_setting

        return get_setting(
            self._config, "dask.scheduler"
        ) == "distributed" and not get_setting(self._config, "dask.address")

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
            # A kernel is launching: end any no-kernel stretch now rather than
            # waiting for the reaper's next poll to observe the live kernel.
            self._idle_since = None
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
            # No distributed install: the kernel degrades to threads (no
            # injected address -> threads, per _configure_dask).
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

    def set_kernel_alive(self, kernel_alive):
        """Install the "is a kernel attached?" predicate the reaper gates on.

        Separate from ``__init__`` only because the ``KernelHost`` that answers
        it is constructed *with* this host, so it cannot be passed in.
        """
        self._kernel_alive = kernel_alive

    def _idle_ttl(self):
        from .._config import get_setting

        return float(get_setting(self._config, "dask.idle_ttl"))

    def start_reaper(self):
        """Start the idle reaper, unless disabled or we own no cluster.

        Idempotent. Cheap to leave running: it polls in-process state and never
        touches dask until it actually closes something.
        """
        ttl = self._idle_ttl()
        if ttl <= 0 or not self._should_own() or self._reap_thread is not None:
            return
        self._reap_thread = threading.Thread(
            target=self._reap_loop, args=(ttl,), name="biopb-dask-reaper", daemon=True
        )
        self._reap_thread.start()
        logger.debug("Dask idle reaper started (ttl=%.0fs)", ttl)

    def _reap_loop(self, ttl):
        """Close the cluster once it has sat with no kernel attached for *ttl*.

        Reaping is gated on kernel liveness, not on dask activity: the kernel is
        handed our scheduler address once, at launch (``KernelHost._launch`` ->
        ``ensure()``), and holds that ``Client`` for its whole life with nothing
        to re-inject a new address. So closing under a live kernel would strand
        it on a dead scheduler, while closing with none attached costs only the
        next kernel launch a re-spin (``ensure()`` self-heals). The window this
        reclaims is real: closing the napari window tears the kernel down to idle
        while the session child lives on for as long as the agent stays
        connected, holding N idle workers the whole time (biopb/biopb#409).

        A kernel restart / watchdog respawn dips through "not alive" for seconds,
        far under a minutes-scale TTL, so the warm-across-restarts property the
        host exists for is preserved.
        """
        poll = min(_REAP_POLL_MAX, ttl)
        while not self._reap_stop.wait(poll):
            with self._lock:
                if self._cluster is None:
                    # Nothing to reap; the stretch restarts when one is spun.
                    self._idle_since = None
                    continue
                # No predicate -> we cannot prove no kernel is attached, so treat
                # it as attached and never reap.
                alive = True
                if self._kernel_alive is not None:
                    try:
                        alive = bool(self._kernel_alive())
                    except Exception:
                        logger.debug("kernel liveness probe failed", exc_info=True)
                if alive:
                    self._idle_since = None
                    continue
                now = time.monotonic()
                if self._idle_since is None:
                    self._idle_since = now
                elif now - self._idle_since >= ttl:
                    logger.info(
                        "Dask cluster idle (no kernel) for %.0fs; tearing it down. "
                        "The next kernel launch re-spins it.",
                        now - self._idle_since,
                    )
                    self._close_locked()
                    self._idle_since = None

    def close(self):
        """Best-effort, idempotent teardown of the owned cluster and its reaper.

        Called on real daemon exit (the ``_shutdown`` chokepoint + atexit). A
        kernel restart/reap never calls this — that is what keeps the workers
        warm; the reaper closes only the *cluster*, via ``_close_locked``, and
        leaves itself running to handle the next spin.
        """
        self._reap_stop.set()
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
