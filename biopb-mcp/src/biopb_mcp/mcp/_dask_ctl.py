"""Where this kernel's dask computes run: in-process, or on a cluster it spun.

Runs **in the kernel**, which owns the whole arrangement -- there is no
daemon-side cluster machinery. The default is the *in-process* scheduler the
napari viewer's slice reads are pinned to (biopb/biopb#8); a cluster arrives only
when something asks for one, and dies with the kernel that asked.

**Why in-process is the default** (biopb/biopb#970). A cluster nobody opted into
is a cluster nobody watches: the session daemon used to spin one at start and
wire every kernel to it, so when a host suspend killed its workers the scheduler
stayed up, accepted work, and blocked every ``.compute()`` forever with no error.
Making the attach explicit removes the failure by construction -- and there is
little to give up on the happy path, because a chunk read over loopback hands
back an mmap view of the server's own segment file that N worker processes would
share through the page cache anyway (``docs/localhost-fast-path.md``). Crossing a
process boundary to fetch it only adds a ``chunk_locate`` round trip and a hop
back per worker, plus N cold spawns at session start.

**What attaching still buys**: real CPU parallelism for a compute that is not
loopback-IO-bound, an external cluster's machines, and a mid-compute cancel --
``interrupt_kernel`` stops a blocking ``.compute()`` by cancelling in-flight
futures, which needs a real ``Client``. On the in-process scheduler that stop is
best-effort: a ``KeyboardInterrupt`` that lands at the next bytecode, so a chunk
fetch inside a C call ends only when it returns.

**Nothing here overrides dask's own precedence**, which is the reason this module
is small: with ``dask.config["scheduler"]`` left unset, a live ``Client``
registers itself as the global default and a closed one falls back to the
threaded scheduler, all by itself. So ``attach``/``detach`` are ordinary
``Client(...)`` / ``close()`` calls, and an agent that writes those in a cell
gets exactly what this object would have given it -- plus, here, the worker cache
budget and a cluster whose lifetime is managed.
"""

import logging
import shutil
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

# The one wording of biopb/biopb#970's advice. Every surface that has to say it
# -- the job runner's refusal, server_status -- prints this string, so the
# instruction cannot drift between them.
_DEAD_CLUSTER_ADVICE = (
    "The dask cluster attached to this kernel ({address}) has no workers left, "
    "so any .compute() here would block forever instead of failing. This is what "
    "a host suspend does to a cluster (biopb/biopb#970). Run "
    "`_dask_ctl.attach()` to get a fresh one, or `_dask_ctl.detach()` to go back "
    "to the in-process scheduler the napari viewer uses."
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
    bytes (``None`` if nothing was registered).

    This budget applies on localhost too. It bounds only the strong cache of
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


class DaskAttachment:
    """This kernel's dask scheduler: in-process by default, on a cluster on ask.

    Bound in the agent namespace as ``_dask_ctl``; :meth:`attach` and
    :meth:`detach` are called from a cell like any other handle, which is why
    there is no MCP tool for either. What it adds over writing ``Client(...)`` by
    hand is the parts that are easy to forget: a cluster sized from config whose
    lifetime is tied to this kernel, the per-worker chunk-cache budget, and the
    liveness the job runner reads.
    """

    def __init__(self, config: dict, ip=None):
        self._config = config
        self._ip = ip
        # Guards the client/cluster pair and the (url, token) the cache budget
        # needs. Never held across a network call -- dead_message() takes no lock
        # at all and runs on every job start, and a stale client's close() can
        # block for its comm timeout.
        self._lock = threading.RLock()
        self._client = None
        # The LocalCluster this kernel spun, if any; None when attached to
        # someone else's scheduler (which is not ours to close).
        self._cluster = None
        self._spill_dir = None
        self._address = None
        self._url = None
        self._token = None
        self._per_worker_budget = None
        # Has the startup decision been made? The authority is here; the
        # namespace binding is a mirror for the agent to read.
        self._settled = False
        # Last worker count and when it was read, so the every-job liveness
        # check can answer from a recent reading instead of an RPC.
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
        """The arrangement is decided; mirror that for the agent."""
        self._settled = True
        if self._ip is not None:
            self._ip.user_ns["_dask_attach_done"] = True

    # -- startup -----------------------------------------------------------

    def start(self):
        """Settle this kernel's scheduler. Runs on the bootstrap's attach thread.

        Off the bootstrap's own thread because spinning a cluster is seconds (N
        cold spawns on Windows) and even a bare ``Client(address)`` connect costs
        a round trip; the viewer must not wait on either. Until this settles,
        ``_dask_client`` is None and the tools guard for that.

        The config escape hatch, and the only thing that attaches without being
        asked: ``dask.address`` attaches to that external scheduler, and
        ``dask.scheduler = "distributed"`` spins a local cluster at kernel start
        the way every session did before biopb/biopb#970. Neither is the default.
        """
        try:
            from .._config import get_setting

            address = get_setting(self._config, "dask.address")
            if address:
                self.attach(address)
            elif get_setting(self._config, "dask.scheduler") == "distributed":
                self.attach()
        except Exception:
            logger.exception("Dask setup failed; kernel stays on in-process scheduler")
        finally:
            self._mark_settled()

    # -- attach / detach ---------------------------------------------------

    def attach(self, address=None):
        """Compute on a dask cluster; return :meth:`status`.

        Without *address*, spins a ``LocalCluster`` in this kernel sized from the
        ``dask.*`` config. With one, attaches to that scheduler -- but note its
        workers have to reach the data plane themselves, and off-loopback they
        lose the mmap fast path, so a read-heavy graph can be slower than
        in-process.

        Replaces whatever this kernel was on, closing a cluster it had spun. A
        failed connect leaves the current arrangement untouched and reports the
        error, so a bad address costs nothing.
        """
        cluster = None
        spill = None
        try:
            from dask.distributed import Client

            if address is None:
                cluster, spill = self._spin()
                address = cluster.scheduler_address
            client = Client(address)
        except Exception as exc:
            logger.exception("Failed to attach to a dask cluster at %s", address)
            self._discard(cluster, spill)
            return {"error": f"{type(exc).__name__}: {exc}", "address": address}
        with self._lock:
            previous, prev_cluster, prev_spill = (
                self._client,
                self._cluster,
                self._spill_dir,
            )
            self._publish(client)
            self._cluster, self._spill_dir = cluster, spill
            self._address = address
        # Outside the lock: closing a superseded client can block for its comm
        # timeout, and that is exactly what a stale scheduler does.
        self._close(previous, keep=client)
        self._discard(prev_cluster, prev_spill)
        self._register_cache_if_ready()
        self._mark_settled()
        logger.info("Dask attached to %s", address)
        return self.status()

    def detach(self):
        """Go back to the in-process scheduler; return :meth:`status`.

        Closes the client -- which is all it takes, since dask falls back to the
        threaded scheduler once no client is live -- and tears down the cluster
        if this kernel spun it. An external one was never ours. Idempotent.
        """
        with self._lock:
            client, cluster, spill = self._client, self._cluster, self._spill_dir
            self._publish(None)
            self._cluster, self._spill_dir, self._address = None, None, None
            self._per_worker_budget = None
        self._close(client, keep=None)
        self._discard(cluster, spill)
        self._mark_settled()
        return self.status()

    # Teardown for the kernel's own shutdown path (_kernel._DASK_RELEASE_SNIPPET),
    # which wants the workers stopped gracefully so they clean their spill files
    # rather than being reaped with the process group.
    shutdown = detach

    def _spin(self):
        """Start a ``LocalCluster`` in this kernel; return ``(cluster, spill_dir)``.

        Its workers are children of the kernel process, so they go with it --
        which is the whole reason the daemon no longer owns a cluster: nothing
        outlives the session that asked for it, and there is no idle reaper, no
        liveness ledger and no scheduler address to inject. The spill directory
        is ours to remove, since a group-kill gives the workers no chance to
        (biopb/biopb#13).
        """
        from dask.distributed import LocalCluster

        from .._config import get_setting

        spill = tempfile.mkdtemp(prefix="biopb-mcp-dask-")
        cluster = LocalCluster(
            # 0 -> None so dask picks ~n_cores.
            n_workers=get_setting(self._config, "dask.num_workers") or None,
            processes=True,
            threads_per_worker=get_setting(self._config, "dask.threads_per_worker"),
            memory_limit=get_setting(self._config, "dask.memory_limit"),
            dashboard_address=get_setting(self._config, "dask.dashboard_address"),
            local_directory=spill,
        )
        logger.info(
            "Spun a kernel-owned dask cluster: %d worker(s) at %s",
            len(cluster.workers),
            cluster.scheduler_address,
        )
        return cluster, spill

    def _close(self, client, keep):
        """Close a superseded ``Client``, best-effort. Call without the lock."""
        if client is None or client is keep:
            return
        try:
            client.close()
        except Exception:
            logger.debug("dask client close failed", exc_info=True)

    def _discard(self, cluster, spill_dir):
        """Tear down a cluster we spun and remove its spill dir. No lock."""
        if cluster is not None:
            try:
                cluster.close()
            except Exception:
                logger.debug("dask cluster close failed", exc_info=True)
        if spill_dir:
            shutil.rmtree(spill_dir, ignore_errors=True)

    # -- connection hook ---------------------------------------------------

    def on_connect(self, url, token):
        """Data plane connected: (re)install the cache budget on the workers.

        Fires in the kernel after every successful connect with the final
        ``(url, token)`` -- the token is only known post-connect, and it is what
        bounds the workers' chunk cache.
        """
        with self._lock:
            self._url, self._token = url, token
        self._register_cache_if_ready()

    def _register_cache_if_ready(self):
        """Split the cache budget across the workers, if there are any.

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

        ``server_status`` renders it. Reports the *mode* (in-process vs. attached,
        and to what) rather than only a worker count, so the arrangement -- the
        thing biopb/biopb#970 made invisible -- is legible in the one place an
        agent already looks. ``warning`` carries :meth:`dead_message` when it
        applies, so every surface prints one wording of that advice.
        """
        with self._lock:
            client, address, owned = self._client, self._address, self._cluster
            budget = self._per_worker_budget
        if client is None:
            return {
                "mode": "in-process" if self._settled else "attaching",
                "scheduler": "threads",
                "workers": None,
                "address": None,
                "dashboard": None,
                "owned": False,
                "cache_budget_per_worker": None,
                "dead": False,
                "warning": None,
            }
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
            "owned": owned is not None,
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
