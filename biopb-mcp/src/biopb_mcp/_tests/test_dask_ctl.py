"""Unit tests for the kernel's dask attachment -- in-process, or on a cluster.

No real cluster and no kernel: ``dask.distributed.Client`` is monkeypatched, and
a dict stands in for the IPython user namespace. Also covers
``_register_cache_plugin``, the cluster-wide chunk-cache budget split.
"""

import os
from unittest.mock import MagicMock, patch

import biopb.tensor.client as tclient
import pytest

from biopb_mcp.mcp import _dask_ctl
from biopb_mcp.mcp._dask_ctl import DaskAttachment

_MISSING = object()


@pytest.fixture(autouse=True)
def restore_dask_config():
    """Attaching sets dask's *global* default scheduler, which is the point --
    so put those two keys back, or the next test's compute() goes looking for a
    cluster. Only these keys: wholesale-restoring dask.config would drop the
    defaults `distributed` registers when it is imported."""
    import dask

    keys = ("scheduler", "num_workers")
    saved = {k: dask.config.config.get(k, _MISSING) for k in keys}
    yield
    for key, value in saved.items():
        if value is _MISSING:
            dask.config.config.pop(key, None)
        else:
            dask.config.config[key] = value


class _FakeIP:
    def __init__(self):
        self.user_ns = {"_dask_client": None, "_dask_attach_done": False}


class _FakeClient:
    def __init__(self, address, workers=2):
        self.address = address
        self.closed = False
        self.dashboard_link = "http://127.0.0.1:8787/status"
        self._workers = workers
        self.plugins = []
        self.info_calls = 0

    def scheduler_info(self, n_workers=5):
        # n_workers=-1 is what the attachment asks for: the default caps the
        # reply at 5, which would under-report a bigger cluster.
        self.info_calls += 1
        assert n_workers == -1
        return {"workers": {f"w{i}": {} for i in range(self._workers)}}

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    def close(self):
        self.closed = True


class _FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scheduler_address = "tcp://127.0.0.1:9999"
        self.workers = {"w0": object(), "w1": object()}
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    """Monkeypatch Client; yields the list of clients it constructs."""
    pytest.importorskip("dask.distributed")
    import dask.distributed as dd

    created = []

    def _factory(address, **kwargs):
        client = _FakeClient(address)
        created.append(client)
        return client

    monkeypatch.setattr(dd, "Client", _factory)
    return created


@pytest.fixture
def fake_cluster(monkeypatch):
    """Monkeypatch LocalCluster; yields the list of clusters it constructs."""
    pytest.importorskip("dask.distributed")
    import dask.distributed as dd

    created = []

    def _factory(**kwargs):
        cluster = _FakeCluster(**kwargs)
        created.append(cluster)
        return cluster

    monkeypatch.setattr(dd, "LocalCluster", _factory)
    return created


def _cfg(**over):
    dask = {"scheduler": "threads", "address": "", "num_workers": 0}
    dask.update(over)
    return {"dask": dask}


class TestStart:
    """The config escape hatch -- the only thing that attaches unasked."""

    def test_default_config_stays_in_process(self, fake_client, fake_cluster):
        import dask

        ip = _FakeIP()
        ctl = DaskAttachment(_cfg(), ip)
        ctl.start()
        assert fake_client == [] and fake_cluster == []
        assert ip.user_ns["_dask_client"] is None
        assert ip.user_ns["_dask_attach_done"] is True
        assert ctl.status()["mode"] == "in-process"
        # Nothing is written to dask's config: an unset "scheduler" key is what
        # lets a live Client win and a closed one fall back on its own -- and
        # what makes a hand-rolled Client() in a cell behave like attach().
        assert not dask.config.get("scheduler", default=None)

    def test_distributed_spins_a_cluster_at_start(self, fake_client, fake_cluster):
        ip = _FakeIP()
        DaskAttachment(_cfg(scheduler="distributed"), ip).start()
        assert len(fake_cluster) == 1
        assert [c.address for c in fake_client] == ["tcp://127.0.0.1:9999"]
        assert ip.user_ns["_dask_client"] is fake_client[0]

    def test_configured_address_attaches_without_spinning(
        self, fake_client, fake_cluster
    ):
        ctl = DaskAttachment(_cfg(address="tcp://ext:8786"), _FakeIP())
        ctl.start()
        assert [c.address for c in fake_client] == ["tcp://ext:8786"]
        assert fake_cluster == []  # not ours to spin
        assert ctl.status()["owned"] is False

    def test_address_wins_over_the_scheduler_setting(self, fake_client, fake_cluster):
        DaskAttachment(
            _cfg(scheduler="distributed", address="tcp://ext:8786"), _FakeIP()
        ).start()
        assert [c.address for c in fake_client] == ["tcp://ext:8786"]
        assert fake_cluster == []


class TestAttachDetach:
    def test_bare_attach_spins_a_kernel_owned_cluster(self, fake_client, fake_cluster):
        ip = _FakeIP()
        ctl = DaskAttachment(_cfg(num_workers=0, threads_per_worker=1), ip)
        status = ctl.attach()
        assert len(fake_cluster) == 1
        assert fake_cluster[0].kwargs["n_workers"] is None  # 0 -> dask picks
        assert fake_cluster[0].kwargs["local_directory"]  # its own spill dir
        assert status["mode"] == "attached" and status["owned"] is True
        assert status["address"] == "tcp://127.0.0.1:9999"
        assert ip.user_ns["_dask_client"] is fake_client[-1]

    def test_detach_closes_the_client_and_the_cluster_it_spun(
        self, fake_client, fake_cluster
    ):
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach()
        spill = fake_cluster[0].kwargs["local_directory"]
        status = ctl.detach()
        assert fake_client[-1].closed and fake_cluster[0].closed
        assert not os.path.exists(spill)  # the spill dir goes with it (#13)
        assert status["mode"] == "in-process"

    def test_detach_leaves_an_external_cluster_alone(self, fake_client, fake_cluster):
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach("tcp://ext:8786")
        ctl.detach()
        assert fake_client[-1].closed
        assert fake_cluster == []  # nothing of ours was involved

    def test_attaching_elsewhere_tears_down_the_cluster_we_spun(
        self, fake_client, fake_cluster
    ):
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach()
        ctl.attach("tcp://ext:8786")
        assert fake_cluster[0].closed
        assert fake_client[0].closed and fake_client[1].closed is False
        assert ctl.status()["owned"] is False

    def test_failed_attach_leaves_the_arrangement_alone(self, monkeypatch):
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        def _boom(address, **kwargs):
            raise OSError("unreachable")

        monkeypatch.setattr(dd, "Client", _boom)
        ip = _FakeIP()
        ctl = DaskAttachment(_cfg(), ip)
        ctl.start()
        result = ctl.attach("tcp://nowhere:8786")
        assert "unreachable" in result["error"]
        assert ip.user_ns["_dask_client"] is None
        assert ctl.status()["mode"] == "in-process"

    def test_a_failed_spin_does_not_leak_the_cluster(self, monkeypatch, fake_cluster):
        # The cluster comes up, the Client to it does not: close what we started
        # rather than leaving N workers behind.
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        def _boom(address, **kwargs):
            raise OSError("unreachable")

        monkeypatch.setattr(dd, "Client", _boom)
        ctl = DaskAttachment(_cfg(), _FakeIP())
        assert "unreachable" in ctl.attach()["error"]
        assert fake_cluster[0].closed is True


class TestCacheBudget:
    def test_budget_is_split_once_both_client_and_connection_exist(self, fake_client):
        ctl = DaskAttachment(_cfg(cache_budget="1G"), _FakeIP())
        ctl.attach("tcp://x:8786")
        # No connection yet -> nothing registered.
        assert fake_client[-1].plugins == []
        ctl.on_connect("grpc://127.0.0.1:8815", None)
        assert len(fake_client[-1].plugins) == 1
        # 1G (dask's parse_bytes: 10**9) across the fake client's 2 workers.
        assert ctl.status()["cache_budget_per_worker"] == 10**9 // 2

    def test_reattach_recomputes_the_split(self, fake_client):
        ctl = DaskAttachment(_cfg(cache_budget="1G"), _FakeIP())
        ctl.on_connect("grpc://127.0.0.1:8815", None)
        ctl.attach("tcp://x:8786")
        assert len(fake_client[-1].plugins) == 1


class TestDeadMessage:
    def test_none_when_in_process(self):
        assert DaskAttachment(_cfg(), _FakeIP()).dead_message() is None

    def test_none_while_workers_are_alive(self, fake_client):
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach("tcp://x:8786")
        assert ctl.dead_message() is None

    def test_names_the_fix_when_the_cluster_lost_its_workers(self, fake_client):
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach("tcp://x:8786")
        fake_client[-1]._workers = 0
        ctl._workers_at = 0.0  # expire the every-job count cache
        msg = ctl.dead_message()
        assert "no workers left" in msg
        assert "_dask_ctl.attach()" in msg and "_dask_ctl.detach()" in msg
        assert ctl.status()["dead"] is True
        assert ctl.status()["warning"] == msg  # one wording, every surface

    def test_the_every_job_check_reuses_a_recent_count(self, fake_client):
        # dead_message() runs on every job start; asking the scheduler each time
        # would add an RPC per submit (and the client refreshes at this rate
        # anyway). The cached reading is why it does not.
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach("tcp://x:8786")
        before = fake_client[-1].info_calls
        for _ in range(5):
            assert ctl.dead_message() is None
        assert fake_client[-1].info_calls == before


def _fake_dask_client(n_workers):
    dc = MagicMock()
    dc.scheduler_info.return_value = {
        "workers": {f"w{i}": {} for i in range(n_workers)}
    }
    return dc


class TestRegisterCachePlugin:
    def test_splits_budget_by_the_count_it_is_given(self):
        dc = _fake_dask_client(5)
        with patch.object(_dask_ctl, "_make_cache_plugin") as mk:
            mk.return_value = MagicMock(name="plugin")
            _dask_ctl._register_cache_plugin(
                dc, "grpc://remote:8815", "tok", {"dask": {"cache_budget": "1G"}}, 12
            )
        loc, tok, per_worker = mk.call_args.args
        assert loc == "grpc://remote:8815" and tok == "tok"
        assert per_worker == 1_000_000_000 // 12
        dc.register_plugin.assert_called_once_with(mk.return_value)
        # The caller already knows the count; asking the scheduler again would be
        # a second round trip for one attach.
        dc.scheduler_info.assert_not_called()

    def test_unknown_count_grants_one_worker_the_budget(self):
        # A count of None means the scheduler could not be read; bound the cache
        # as if there were a single worker rather than dividing by nothing.
        dc = _fake_dask_client(4)
        with patch.object(_dask_ctl, "_make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _dask_ctl._register_cache_plugin(
                dc, "grpc://remote:8815", None, {"dask": {"cache_budget": "1G"}}, None
            )
        assert mk.call_args.args[2] == 1_000_000_000

    def test_accepts_int_budget(self):
        dc = _fake_dask_client(2)
        with patch.object(_dask_ctl, "_make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _dask_ctl._register_cache_plugin(
                dc,
                "grpc://remote:8815",
                None,
                {"dask": {"cache_budget": 800_000_000}},
                2,
            )
        assert mk.call_args.args[2] == 400_000_000

    def test_localhost_is_not_special_cased(self):
        # The plugin splits the budget the same for localhost and remote URLs --
        # it never special-cases the host. (_resolve_cache_bytes no longer clamps
        # localhost to 0 either; localhost caches copies like any other host.)
        dc = _fake_dask_client(8)
        with patch.object(_dask_ctl, "_make_cache_plugin") as mk:
            mk.return_value = MagicMock()
            _dask_ctl._register_cache_plugin(
                dc, "grpc://localhost:8815", None, {"dask": {"cache_budget": "4G"}}, 8
            )
        assert mk.call_args.args[2] == 4_000_000_000 // 8
        dc.register_plugin.assert_called_once_with(mk.return_value)

    def test_noop_without_dask_client(self):
        # must not raise when there is no distributed client
        _dask_ctl._register_cache_plugin(None, "grpc://x:1", None, {}, 1)

    def test_noop_when_plugin_unavailable(self):
        dc = _fake_dask_client(3)
        with patch.object(_dask_ctl, "_make_cache_plugin", return_value=None) as mk:
            _dask_ctl._register_cache_plugin(dc, "grpc://remote:8815", None, {}, 3)
        mk.assert_called_once()
        dc.register_plugin.assert_not_called()


class TestMakeCachePlugin:
    """The dask WorkerPlugin factory, moved out of the tensor SDK into MCP."""

    def test_returns_none_or_named_plugin(self):
        plugin = _dask_ctl._make_cache_plugin("grpc://remote:8815", None, 1000)
        try:
            import distributed  # noqa: F401
        except Exception:
            assert plugin is None  # graceful no-op without distributed
            return
        assert plugin is not None
        assert plugin.name == "biopb-cache-config"

    def test_setup_pins_cache_via_sdk_configure_cache(self):
        pytest.importorskip("distributed")
        tclient._CACHE_POOL.clear()
        loc = "grpc://remote:8815"
        plugin = _dask_ctl._make_cache_plugin(loc, None, 777)
        try:
            plugin.setup(worker=None)  # what dask calls on each worker
            assert tclient._CACHE_POOL[(loc, None)].available_bytes == 777
        finally:
            tclient._CACHE_POOL.clear()
