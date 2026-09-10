"""Unit tests for the kernel's dask attachment -- in-process, or on a cluster.

No real cluster and no kernel: ``dask.distributed.Client`` is monkeypatched, and
a dict stands in for the IPython user namespace. Also covers
``_register_cache_plugin``, the cluster-wide chunk-cache budget split.
"""

from unittest.mock import MagicMock, patch

import biopb.tensor.client as tclient
import pytest

from biopb_mcp.mcp import _dask_ctl
from biopb_mcp.mcp._dask_ctl import DASK_ADDRESS_ENV, DaskAttachment

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


def _cfg(**over):
    dask = {"scheduler": "threads", "address": "", "num_workers": 0}
    dask.update(over)
    return {"dask": dask}


class TestAutoAttachAddress:
    def test_default_config_attaches_nothing(self, monkeypatch):
        monkeypatch.delenv(DASK_ADDRESS_ENV, raising=False)
        assert DaskAttachment(_cfg()).auto_attach_address() is None

    def test_distributed_without_address_attaches_nothing(self, monkeypatch):
        """The kernel never spins its own: no injected address -> in-process."""
        monkeypatch.delenv(DASK_ADDRESS_ENV, raising=False)
        ctl = DaskAttachment(_cfg(scheduler="distributed"))
        assert ctl.auto_attach_address() is None

    def test_external_address_wins_over_threads_default(self, monkeypatch):
        monkeypatch.delenv(DASK_ADDRESS_ENV, raising=False)
        ctl = DaskAttachment(_cfg(address="tcp://1.2.3.4:8786"))
        assert ctl.auto_attach_address() == "tcp://1.2.3.4:8786"

    def test_injected_address_wins_over_config(self, monkeypatch):
        monkeypatch.setenv(DASK_ADDRESS_ENV, "tcp://daemon:8786")
        ctl = DaskAttachment(_cfg(scheduler="distributed", address="tcp://cfg:1"))
        assert ctl.auto_attach_address() == "tcp://daemon:8786"


class TestStart:
    def test_default_start_stays_in_process(self, monkeypatch, fake_client):
        import dask

        monkeypatch.delenv(DASK_ADDRESS_ENV, raising=False)
        ip = _FakeIP()
        ctl = DaskAttachment(_cfg(), ip)
        with dask.config.set(scheduler="synchronous"):
            ctl.start()
            assert dask.config.get("scheduler") == "threads"
        assert fake_client == []
        assert ip.user_ns["_dask_client"] is None
        assert ip.user_ns["_dask_attach_done"] is True
        assert ctl.status()["mode"] == "in-process"

    def test_configured_address_attaches_at_start(self, monkeypatch, fake_client):
        monkeypatch.delenv(DASK_ADDRESS_ENV, raising=False)
        ip = _FakeIP()
        DaskAttachment(_cfg(address="tcp://ext:8786"), ip).start()
        assert [c.address for c in fake_client] == ["tcp://ext:8786"]
        assert ip.user_ns["_dask_client"] is fake_client[0]
        assert ip.user_ns["_dask_attach_done"] is True


class TestAttachDetach:
    def test_attach_publishes_client_and_default_scheduler(self, fake_client):
        import dask

        ip = _FakeIP()
        ctl = DaskAttachment(_cfg(), ip)
        ctl.start()
        with dask.config.set(scheduler="threads"):
            status = ctl.attach("tcp://127.0.0.1:8786")
            assert dask.config.get("scheduler") == "distributed"
        assert status["mode"] == "attached"
        assert status["address"] == "tcp://127.0.0.1:8786"
        assert status["workers"] == 2
        assert status["dead"] is False
        assert ip.user_ns["_dask_client"] is fake_client[-1]

    def test_detach_closes_client_and_restores_in_process(self, fake_client):
        import dask

        ip = _FakeIP()
        ctl = DaskAttachment(_cfg(), ip)
        ctl.attach("tcp://127.0.0.1:8786")
        client = fake_client[-1]
        with dask.config.set(scheduler="distributed"):
            status = ctl.detach()
            assert dask.config.get("scheduler") == "threads"
        assert client.closed
        assert status["mode"] == "in-process"
        assert ip.user_ns["_dask_client"] is None

    def test_reattach_closes_the_previous_client(self, fake_client):
        ctl = DaskAttachment(_cfg(), _FakeIP())
        ctl.attach("tcp://one:8786")
        ctl.attach("tcp://two:8786")
        assert fake_client[0].closed
        assert fake_client[1].closed is False

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

    def test_refuses_while_a_job_runs(self, fake_client, monkeypatch):
        from biopb_mcp.mcp import _jobs

        monkeypatch.setattr(_jobs, "running_job", lambda: {"job_id": "job-1"})
        monkeypatch.setattr(_jobs, "check_writer", lambda writer: None)
        ctl = DaskAttachment(_cfg(), _FakeIP())
        assert ctl.attach("tcp://x:8786").get("busy") is True
        assert ctl.detach().get("busy") is True
        assert fake_client == []


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


class TestAdmission:
    """attach/detach are kernel-state changes, gated where the claim lives."""

    def test_refused_for_a_client_that_does_not_hold_the_kernel(
        self, fake_client, monkeypatch
    ):
        from biopb_mcp.mcp import _jobs

        monkeypatch.setattr(
            _jobs,
            "check_writer",
            lambda writer: {"refused": "not_owner", "owner": "A", "owner_id": "a"},
        )
        ctl = DaskAttachment(_cfg(), _FakeIP())
        assert ctl.attach("tcp://x:8786", writer="b")["refused"] == "not_owner"
        assert ctl.detach(writer="b")["refused"] == "not_owner"
        assert fake_client == []


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
        assert "attach_cluster()" in msg and "detach_cluster()" in msg
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
