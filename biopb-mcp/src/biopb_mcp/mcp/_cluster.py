"""Daemon-owned dask cluster host.

When a cluster exists at all, the MCP daemon (``mcp/__main__.py::_serve_http``)
is what owns it: it spins the ``LocalCluster`` and hands out its scheduler
address, and the kernel attaches with a bare ``Client(address)`` instead of
spinning its own.  That decouples the cluster's lifetime from the kernel's: a
``restart_kernel`` / watchdog respawn / viewer-window close no longer tears the
cluster down and re-spins N workers — the dominant restart cost on Windows,
where each worker is a cold spawn (no fork).

**Nothing spins one by default.**  A cluster the session never asked for is a
cluster nobody watches, which is how biopb/biopb#970 happened: a host suspend
killed its workers and every later ``.compute()`` blocked forever on a scheduler
that was still listening.  So a cluster now comes from an explicit act — the
``attach_cluster`` tool (``ensure(on_demand=True)``) or a config that asks for
one (``dask.scheduler = "distributed"``, which keeps the old inject-at-launch
path) — and the kernel's default is the in-process scheduler (``_dask_ctl``).

A real daemon exit (the ``_shutdown`` chokepoint) closes the cluster, as does the
idle reaper (``start_reaper``) once it has sat with *no kernel attached* for
``dask.idle_ttl`` — the bound on that decoupling, so a session whose viewer is
closed but whose agent is still connected stops holding N idle workers
indefinitely (biopb/biopb#409).

Deliberately GUI-free and import-light: ``dask.distributed`` is imported inside
``ensure()``, so a daemon that never spins a cluster pays nothing.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Ceiling on the reaper's poll interval. The TTL is coarse (minutes), so polling
# is cheap at this rate; a shorter TTL polls at the TTL itself, which is what
# keeps the reaper testable without a sleep(ttl).
_REAP_POLL_MAX = 30.0


class DaskClusterHost:
    """Own a distributed ``LocalCluster`` on behalf of the session child.

    Lazily spins the cluster on the first :meth:`ensure` that asks for one — an
    ``attach_cluster`` call, or a kernel launch under a config that wants
    auto-attach — and keeps it warm across kernel restarts.  :meth:`ensure`
    returns the scheduler address to attach to, or ``None`` when there is to be
    no cluster (config does not ask for one and nobody requested it, an external
    ``dask.address`` is configured, or the spin failed) — the kernel then stays
    on its in-process scheduler (see ``_dask_ctl``).
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
        # Is anyone actually attached to our cluster? Kernel liveness alone
        # stopped answering that when attaching became explicit (#970): a kernel
        # that detached, or one launched without an address, is alive and holds
        # nothing. Set by note_attached() once an attach has landed -- not when
        # ensure() hands the address out, which the kernel can still refuse --
        # and cleared by note_detached().
        self._attached = False
        # Monotonic timestamp from which the current no-kernel stretch is
        # measured; None while the cluster is held.
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

    def ensure(self, on_demand=False):
        """Return the scheduler address, spinning the cluster on first use.

        Idempotent: later calls return the cached address after a cheap liveness
        check, re-spinning if the cached cluster has died.  That check is the
        health check biopb/biopb#970 asks for, and this is now exactly where it
        belongs — an attach is the moment someone is asking for a working
        cluster, where a re-spin costs a tool call's latency and nothing else.

        *on_demand* is that call (``attach_cluster``): it spins even though the
        config never asked for a cluster, which is the default now.  Without it
        this answers only for the config-driven path (kernel launch), and returns
        ``None`` unless ``dask.scheduler`` is ``"distributed"`` with no external
        address.  ``None`` also comes back when spinning fails; the kernel then
        stays in-process.
        """
        with self._lock:
            if not on_demand and not self._should_own():
                return None
            self._idle_since = None
            if self._cluster is not None and not self._is_alive(self._cluster):
                logger.warning("Daemon dask cluster is not healthy; re-spinning.")
                self._close_locked()
            # Handing the address out is not the attach -- the caller still has
            # to reach the kernel, which can refuse (a job is running) or fail
            # to connect. The ledger is updated by note_attached() once that has
            # actually happened, so a failed attach leaves the fresh cluster on
            # the reaper's clock instead of held by nobody's client.
            return self._address if self._cluster is not None else self._spin_locked()

    def _spin_locked(self):
        from .._config import get_setting

        try:
            from dask.distributed import LocalCluster
        except Exception:
            # No distributed install: no address to hand out, so the kernel
            # stays on its in-process scheduler.
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

    def note_attached(self, address):
        """Record the attach a kernel has just *completed*, to *address*.

        Only our own address counts: a kernel that attached to an external
        scheduler is not holding this cluster, so the same call that records one
        attach releases the other. Called after the kernel confirms
        (``attach_cluster``), and at launch for the config-driven attach, which
        has no confirmation channel -- the kernel attaches during its own
        bootstrap.
        """
        with self._lock:
            self._attached = address is not None and address == self._address

    def note_detached(self):
        """Nobody holds our cluster any more; start the reaper's idle clock.

        Called when a kernel that had the address goes away without a
        replacement taking it: ``detach_cluster``, and a kernel launch that
        injects no address (``KernelHost._launch``). The cluster is kept warm --
        the next ``ensure()`` reuses it -- but it is now on the clock.
        """
        with self._lock:
            self._attached = False

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
        """Start the idle reaper unless it is disabled or already running.

        Not gated on :meth:`_should_own`: with the config default no longer
        asking for a cluster, that gate would have left an ``attach_cluster``
        cluster un-reaped for the life of the session. The loop is cheap to
        leave running against a session that never spins one — it polls
        in-process state and never touches dask until it closes something.
        """
        ttl = self._idle_ttl()
        if ttl <= 0 or self._reap_thread is not None:
            return
        self._reap_thread = threading.Thread(
            target=self._reap_loop, args=(ttl,), name="biopb-dask-reaper", daemon=True
        )
        self._reap_thread.start()
        logger.debug("Dask idle reaper started (ttl=%.0fs)", ttl)

    def _reap_loop(self, ttl):
        """Close the cluster once it has sat with no kernel attached for *ttl*.

        Reaping is gated on kernel liveness, not on dask activity: a kernel that
        attached (at launch, or via ``attach_cluster``) holds that ``Client`` for
        its whole life with nothing to hand it a new address. So closing under a
        live kernel would strand it on a dead scheduler, while closing with none
        attached costs only the next ``ensure()`` a re-spin. The window this
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
                if alive and self._attached:
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
