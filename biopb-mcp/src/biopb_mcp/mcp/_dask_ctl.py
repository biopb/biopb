"""Kernel-side dask attachment: where this kernel's computes actually run.

Runs **in the kernel**. The default is the *in-process* scheduler — the same one
the napari viewer's slice reads are pinned to (biopb/biopb#8) — and the kernel
attaches to a distributed cluster only when asked: by config (``dask.scheduler =
"distributed"``, or a ``dask.address``), or at run time via the
``attach_cluster`` / ``detach_cluster`` tools.

**Why in-process is the default** (biopb/biopb#970). A cluster nobody opted into
is a cluster nobody watches: the session daemon used to spin one at start and
wire every kernel to it, so when a host suspend killed its workers the scheduler
stayed up, accepted work, and blocked every ``.compute()`` forever with no error.
Making the attach explicit removes the failure by construction — and there is
little to give up on the happy path, because a chunk read over loopback hands
back an mmap view of the server's own segment file that N worker processes would
share through the page cache anyway (``docs/localhost-fast-path.md``). Crossing a
process boundary to fetch it only adds a ``chunk_locate`` round trip and a hop
back per worker, plus N cold spawns at session start.

**What attaching still buys**, and why the tools exist: real CPU parallelism for
a compute that is not loopback-IO-bound, an external cluster's machines, and a
mid-compute cancel — ``interrupt_kernel`` stops a blocking ``.compute()`` by
cancelling in-flight futures, which needs a real ``Client``. On the in-process
scheduler that stop is best-effort: a ``KeyboardInterrupt`` that lands at the
next bytecode, so a chunk fetch inside a C call ends only when it returns.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Env var carrying a session-child-owned cluster's scheduler address into the
# kernel (set by _cluster/_kernel), read ahead of the dask.address config value.
DASK_ADDRESS_ENV = "BIOPB_DASK_ADDRESS"

# The one wording of biopb/biopb#970's advice. Every surface that has to say it
# -- the job runner's refusal, the attach_cluster reply, server_status -- prints
# this string, so the instruction cannot drift between them.
_DEAD_CLUSTER_ADVICE = (
    "The dask cluster attached to this kernel ({address}) has no workers left, "
    "so any .compute() here would block forever instead of failing. This is what "
    "a host suspend does to a cluster (biopb/biopb#970). Call attach_cluster() to "
    "re-spin it, or detach_cluster() to go back to the in-process scheduler the "
    "napari viewer uses. From the observe page: restart the kernel."
)

# Seconds a worker count stays good enough for the every-job liveness check.
# The client's own background refresh runs at this rate, so a shorter TTL buys
# no freshness -- only an RPC per submitted job.
_WORKER_COUNT_TTL = 2.0


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


def _register_cache_plugin(dask_client, url, token, config: dict, n_workers):
    """Split the data-plane chunk-cache budget across dask workers.

    Divides ``dask.cache_budget`` evenly across the workers and installs a
    worker-init plugin so each worker (current and future) caps its per-process
    *copy* cache at ``budget // n_workers`` -- bounding the aggregate cache that
    would otherwise be replicated per worker. Returns the per-worker budget in
    bytes (``None`` if nothing was registered), which is what ``attach_cluster``
    reports back.

    This budget now applies on localhost too. It bounds only the strong cache of
    chunks that cost real RAM (``do_get`` / over-budget copies); on localhost
    those are rare (the mmap fast path dominates), and its views are cached
    *weakly* -- shared with the OS page cache, so not replicated per worker and
    needing no budget. So the old "localhost clamps to 0" special-case is gone;
    each worker just applies this budget uniformly.

    Takes *n_workers* rather than counting them: the caller already knows the
    live count (:meth:`DaskAttachment._worker_count`), and asking the scheduler
    twice for one attach is a round trip for nothing. The split is per worker
    count, so it is recomputed on every attach and whenever that count changes.
    No-op without a distributed client. Best-effort: a failure here must not
    break the connect flow (or the attach) that invokes it.
    """
    if dask_client is None:
        return None
    try:
        from dask.utils import parse_bytes

        from .._config import get_setting

        n_workers = max(1, n_workers or 1)

        budget_cfg = get_setting(config, "dask.cache_budget")
        budget = (
            int(budget_cfg)
            if isinstance(budget_cfg, int | float)
            else parse_bytes(budget_cfg)
        )
        per_worker = max(0, budget // n_workers)

        plugin = _make_cache_plugin(url, token, per_worker)
        if plugin is None:
            return None
        dask_client.register_plugin(plugin)
        logger.info(
            "Chunk-cache plugin: %d B/worker x %d workers (%s)",
            per_worker,
            n_workers,
            url,
        )
        return per_worker
    except Exception:
        logger.exception("Failed to register chunk-cache budget plugin")
        return None


def _admission_refusal(writer):
    """Why *writer* may not change this kernel's scheduler right now, or ``None``.

    Both halves of the answer live in the kernel, where ``interrupt_kernel`` also
    asks them: the one-agent claim (``_jobs``' owner -- the scheduler is the whole
    namespace's, so a client that cannot run code here cannot move it either) and
    whether a job is running. Asking the server's mirrored claim instead would be
    a check-then-act across a round trip, and would need a second copy of the
    rule.
    """
    try:
        from . import _jobs

        refusal = _jobs.check_writer(writer)
        if refusal is not None:
            return refusal
        running = _jobs.running_job()
    except Exception:  # noqa: BLE001 - no job runner (a bare/test kernel)
        return None
    if running is None:
        return None
    return {
        "error": (
            f"job {running['job_id']} is running in this kernel; changing the "
            "scheduler under it would strand its work. Wait for it, or stop it."
        ),
        "busy": True,
    }


class DaskAttachment:
    """This kernel's dask scheduler: in-process by default, attached on request.

    One object owns everything that has to move together on an attach — the
    ``Client``, the ``_dask_client`` binding in the agent's namespace, dask's
    default scheduler, and the per-worker cache budget (recomputed per attach,
    since the split depends on the worker count). It is bound in the namespace
    as ``_dask_ctl``, which is how the ``attach_cluster`` / ``detach_cluster``
    tools reach it across the kernel hop.
    """

    def __init__(self, config: dict, ip=None):
        self._config = config
        self._ip = ip
        # Guards the client/address pair and the (url, token) the cache budget
        # needs. The connect hook and the startup attach genuinely race: either
        # can be the second to arrive, and the second one registers the plugin.
        # Never held across a network call -- dead_message() takes it on every
        # job start, and a stale client's close() can block for its comm timeout.
        self._lock = threading.RLock()
        self._client = None
        self._address = None
        self._url = None
        self._token = None
        self._per_worker_budget = None
        # Has the startup decision been made? The authority is here; the
        # namespace binding below is a mirror for the agent to read.
        self._settled = False
        # Last worker count and when it was read, so the every-job liveness
        # check can answer from a recent reading instead of an RPC (see
        # _worker_count).
        self._workers = None
        self._workers_at = 0.0
        self._publish(None)

    # -- namespace bindings ------------------------------------------------

    def _publish(self, client):
        """Bind (or clear) ``_dask_client`` in the agent namespace. Lock held."""
        self._client = client
        self._workers, self._workers_at = None, 0.0
        if self._ip is not None:
            self._ip.user_ns["_dask_client"] = client
            self._ip.user_ns["_dask_attach_done"] = self._settled

    def _mark_settled(self):
        """The arrangement is decided; mirror that for the agent.

        Set by whichever path decided it -- :meth:`start`, or an
        ``attach_cluster`` / ``detach_cluster`` that got there first -- so
        ``status()`` never reports "attaching" about a kernel that has settled.
        """
        self._settled = True
        if self._ip is not None:
            self._ip.user_ns["_dask_attach_done"] = True

    # -- startup -----------------------------------------------------------

    def auto_attach_address(self):
        """The address to attach at startup, or ``None`` to stay in-process.

        The config escape hatch, and it is entirely an address question.
        ``dask.scheduler = "distributed"`` reaches here as an *injected* address:
        the session child reads that setting, spins its ``LocalCluster`` and
        passes ``BIOPB_DASK_ADDRESS`` down at launch. A configured
        ``dask.address`` is the external case, and wins on its own whatever the
        scheduler setting says — a scheduler address that is never attached to
        would be a setting that silently does nothing.

        No address means no cluster to attach: the session child has none (the
        default) or its spin failed. Either way the kernel stays in-process; it
        never spins one of its own, which would die with every restart.
        """
        from .._config import get_setting

        # The session-child-injected address (its owned cluster) wins over the
        # configured external one.
        return (
            os.environ.get(DASK_ADDRESS_ENV)
            or get_setting(self._config, "dask.address")
            or None
        )

    def start(self):
        """Settle this kernel's scheduler. Runs on the bootstrap's attach thread.

        Off the bootstrap's own thread because even a bare ``Client(address)``
        connect costs a round trip and the viewer must not wait on it; until
        this settles, ``_dask_client`` is None and the tools guard for that.
        """
        try:
            address = self.auto_attach_address()
            if address:
                self.attach(address)
            else:
                with self._lock:
                    self._use_in_process()
        except Exception:
            logger.exception("Dask setup failed; kernel stays on in-process scheduler")
        finally:
            self._mark_settled()

    def _use_in_process(self):
        """Make dask's default the in-process scheduler. Lock held.

        The same scheduler the viewer's slice reads use, so viewer and agent
        share one process's chunk cache by construction rather than by the
        ``_viewer_compute`` pin.
        """
        import dask

        from .._config import get_setting

        scheduler = get_setting(self._config, "dask.scheduler")
        if scheduler == "distributed":
            # Asked for a cluster we have no address for: threads, not a
            # kernel-local cluster.
            scheduler = "threads"
        num_workers = get_setting(self._config, "dask.num_workers") or None
        dask.config.set(scheduler=scheduler, num_workers=num_workers)
        self._address = None
        self._per_worker_budget = None
        logger.info("Dask scheduler: %s, num_workers: %s", scheduler, num_workers)

    # -- attach / detach ---------------------------------------------------

    def attach(self, address: str, writer=None):
        """Attach to the scheduler at *address*; return :meth:`status`.

        The caller resolves *address* — the session child spins or reuses its own
        ``LocalCluster`` for a bare ``attach_cluster()`` — because only it can
        own a cluster; a kernel that spun its own would lose it on every restart.

        Refused for a client that does not hold this kernel, and while a job is
        running: the scheduler is the whole namespace's, and swapping it under an
        in-flight ``compute()`` would strand that job's futures. A failed connect
        leaves the current arrangement untouched and reports the error, so a bad
        address costs nothing.
        """
        refusal = _admission_refusal(writer)
        if refusal is not None:
            return refusal
        try:
            from dask.distributed import Client

            client = Client(address)
        except Exception as exc:
            logger.exception("Failed to attach to dask scheduler at %s", address)
            return {"error": f"{type(exc).__name__}: {exc}", "address": address}
        import dask

        with self._lock:
            previous = self._client
            self._publish(client)
            self._address = address
            # Say it in the config rather than relying on `Client.__init__`
            # registering itself as dask's global default: the in-process default
            # this kernel starts on is a config value ("threads"), and a config
            # value wins over the global client -- so an attach that did not
            # overwrite it would connect a Client that nothing computes on.
            dask.config.set(scheduler="distributed")
        self._close(previous, client)
        self._register_cache_if_ready()
        self._mark_settled()
        logger.info("Dask attached to distributed scheduler at %s", address)
        return self.status()

    def detach(self, writer=None):
        """Close the ``Client`` and go back to the in-process scheduler.

        The cluster itself is left alone: a session-child-owned one falls to the
        existing idle reaper (``dask.idle_ttl``), and an external one was never
        ours. Idempotent — detaching when nothing is attached just reasserts the
        in-process default. Gated like :meth:`attach`.
        """
        refusal = _admission_refusal(writer)
        if refusal is not None:
            return refusal
        with self._lock:
            client = self._client
            self._publish(None)
            self._use_in_process()
        self._close(client, None)
        self._mark_settled()
        return self.status()

    def _close(self, client, keep):
        """Close a superseded ``Client``, outside the lock and best-effort.

        Outside because this is the one call here that can block for a comm
        timeout -- and it blocks exactly when the scheduler has gone stale, which
        is the case this whole module is about. Holding the lock through it would
        stall :meth:`dead_message`, i.e. every job start.
        """
        if client is None or client is keep:
            return
        try:
            client.close()
        except Exception:
            logger.debug("dask client close failed", exc_info=True)

    # -- connection hook ---------------------------------------------------

    def on_connect(self, url, token):
        """Data plane connected: (re)install the cache budget on the workers.

        Fires in the kernel after every successful connect with the final
        ``(url, token)`` — the token is only known post-connect, and it is what
        bounds the workers' chunk cache.
        """
        with self._lock:
            self._url, self._token = url, token
        self._register_cache_if_ready()

    def _register_cache_if_ready(self):
        """Split the cache budget across the workers, if there is one to split.

        Called without the lock (it registers a plugin cluster-wide): it reads a
        consistent snapshot, does the round trip outside, and stores the result
        only if the client it computed for is still the one attached. No-op until
        both a ``Client`` and a live ``(url, token)`` exist -- the connect hook
        and the attach race, and whichever arrives second gets both.
        """
        with self._lock:
            client, url, token = self._client, self._url, self._token
        if client is None or url is None:
            return
        budget = _register_cache_plugin(
            client, url, token, self._config, self._worker_count()
        )
        with self._lock:
            if self._client is client:
                self._per_worker_budget = budget

    # -- reporting ---------------------------------------------------------

    def _worker_count(self, max_age=0.0):
        """Live worker count for the attached scheduler, or ``None`` if unknown.

        ``scheduler_info()`` is an RPC, not a local read, so *max_age* lets the
        every-job liveness check answer from a recent reading instead of asking
        again (the client refreshes the same identity in the background every 2 s
        anyway). A failed read keeps the last known count rather than reporting
        zero: this must not turn "the probe failed" into "the cluster is dead".
        """
        client = self._client
        if client is None:
            return None
        if (
            max_age
            and self._workers_at
            and time.monotonic() - self._workers_at <= max_age
        ):
            return self._workers
        try:
            # n_workers=-1 -> all of them; the default caps the reply at 5, which
            # would under-report the count *and* over-grant the per-worker cache
            # budget on any bigger cluster.
            info = client.scheduler_info(n_workers=-1)
        except TypeError:  # distributed without the n_workers argument
            info = client.scheduler_info()
        except Exception:
            return self._workers
        self._workers = len(info.get("workers", {}))
        self._workers_at = time.monotonic()
        return self._workers

    def status(self):
        """A JSON-able description of where this kernel's computes run.

        ``server_status`` renders it, and ``attach_cluster`` returns it. Reports
        the *mode* (in-process vs. attached, and to what) rather than only the
        worker count, so the arrangement — the thing biopb/biopb#970 made
        invisible — is legible in the one place an agent already looks.
        ``warning`` carries :meth:`dead_message` when it applies, so every
        surface prints one wording of that advice rather than its own.
        """
        with self._lock:
            client, address = self._client, self._address
            if client is None:
                import dask

                return {
                    "mode": "in-process" if self._settled else "attaching",
                    "scheduler": str(dask.config.get("scheduler", default="unknown")),
                    "workers": None,
                    "address": None,
                    "dashboard": None,
                    "cache_budget_per_worker": None,
                    "dead": False,
                    "warning": None,
                }
            budget = self._per_worker_budget
        workers = self._worker_count()
        try:
            dashboard = client.dashboard_link
        except Exception:
            dashboard = None
        return {
            "mode": "attached",
            "scheduler": "distributed",
            "workers": workers,
            "address": address,
            "dashboard": dashboard,
            "cache_budget_per_worker": budget,
            "dead": workers == 0,
            "warning": _DEAD_CLUSTER_ADVICE.format(address=address)
            if workers == 0
            else None,
        }

    def dead_message(self):
        """Why a compute would hang right now, or ``None``.

        The backstop biopb/biopb#970 asks for: an attached scheduler whose
        workers have all gone (a host suspend outliving its TTL is the way this
        happens) still accepts work and never runs it. The job runner raises this
        instead of letting the cell block forever, so it runs on every job start
        -- hence the cached count and no lock: a stale client's ``close()`` must
        not be able to stall a submit.
        """
        if self._client is None:
            return None
        if self._worker_count(max_age=_WORKER_COUNT_TTL) != 0:
            return None
        return _DEAD_CLUSTER_ADVICE.format(address=self._address)
