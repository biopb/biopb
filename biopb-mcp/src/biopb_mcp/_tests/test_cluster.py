"""Unit tests for DaskClusterHost (the daemon-owned dask cluster).

No real cluster is spun: dask.distributed.LocalCluster is monkeypatched with a
fake so these run fast and headless.
"""

import pytest

pytest.importorskip("dask.distributed")

from biopb_mcp.mcp._cluster import DaskClusterHost  # noqa: E402


class _FakeStatus:
    def __init__(self, name):
        self.name = name


class _FakeScheduler:
    def __init__(self, n_workers):
        self.workers = {f"w{i}": object() for i in range(n_workers)}


class _FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scheduler_address = "tcp://127.0.0.1:9999"
        self.workers = {"w0": object(), "w1": object()}
        self.worker_spec = {"w0": {}, "w1": {}}
        self.status = _FakeStatus("running")
        self.scheduler = _FakeScheduler(2)
        self.closed = False

    def close(self):
        self.closed = True


def _cfg(**over):
    dask = {"scheduler": "distributed", "address": "", "owner": "daemon"}
    dask.update(over)
    return {"mcp": {"dask": dask}}


@pytest.fixture
def fake_local_cluster(monkeypatch):
    """Monkeypatch LocalCluster; yields the list of clusters it constructs."""
    import dask.distributed as dd

    created = []

    def _factory(**kwargs):
        cluster = _FakeCluster(**kwargs)
        created.append(cluster)
        return cluster

    monkeypatch.setattr(dd, "LocalCluster", _factory)
    return created


class TestShouldOwn:
    def test_none_when_owner_kernel(self, fake_local_cluster):
        host = DaskClusterHost(_cfg(owner="kernel"))
        assert host.ensure() is None
        assert fake_local_cluster == []  # never spun

    def test_none_when_not_distributed(self, fake_local_cluster):
        host = DaskClusterHost(_cfg(scheduler="threads"))
        assert host.ensure() is None
        assert fake_local_cluster == []

    def test_none_when_external_address(self, fake_local_cluster):
        # An external scheduler is configured -> the kernel attaches to it
        # directly; the daemon owns nothing.
        host = DaskClusterHost(_cfg(address="tcp://1.2.3.4:8786"))
        assert host.ensure() is None
        assert fake_local_cluster == []


class TestEnsure:
    def test_spins_and_returns_address(self, fake_local_cluster):
        host = DaskClusterHost(_cfg(), local_dir="/tmp/spill")
        addr = host.ensure()
        assert addr == "tcp://127.0.0.1:9999"
        assert len(fake_local_cluster) == 1
        # local_dir is forwarded as the worker spill directory.
        assert fake_local_cluster[0].kwargs["local_directory"] == "/tmp/spill"

    def test_num_workers_zero_becomes_none(self, fake_local_cluster):
        # 0 -> None so dask picks ~n_cores (matches the kernel-owned path).
        host = DaskClusterHost(_cfg(num_workers=0))
        host.ensure()
        assert fake_local_cluster[0].kwargs["n_workers"] is None

    def test_caches_across_calls(self, fake_local_cluster):
        host = DaskClusterHost(_cfg())
        first = host.ensure()
        second = host.ensure()
        assert first == second
        assert len(fake_local_cluster) == 1  # spun once, cached after

    def test_respins_dead_cluster(self, fake_local_cluster):
        host = DaskClusterHost(_cfg())
        host.ensure()
        # A liveness check that observes the workers, so a later drop to 0 reads
        # as death rather than the not-yet-started spawn window.
        assert host.ensure() == "tcp://127.0.0.1:9999"
        assert len(fake_local_cluster) == 1  # still cached
        # Now the cached cluster dies (all workers gone) after having had some.
        fake_local_cluster[0].scheduler = _FakeScheduler(0)
        host.ensure()
        assert len(fake_local_cluster) == 2  # re-spun
        assert fake_local_cluster[0].closed is True  # dead one was closed

    def test_no_respin_during_worker_startup_window(self, fake_local_cluster):
        """A running cluster whose workers have not registered *yet* is coming
        up, not dead -- ensure() must keep it (else it would re-spin exactly the
        slow-to-spawn cluster it exists to keep warm, e.g. Windows cold-spawn)."""
        host = DaskClusterHost(_cfg())
        host.ensure()
        # Scheduler is running but no worker has registered so far.
        fake_local_cluster[0].scheduler = _FakeScheduler(0)
        host.ensure()
        assert len(fake_local_cluster) == 1  # kept, not re-spun
        assert fake_local_cluster[0].closed is False

    def test_respins_when_status_not_running(self, fake_local_cluster):
        host = DaskClusterHost(_cfg())
        host.ensure()
        fake_local_cluster[0].status = _FakeStatus("closed")
        host.ensure()
        assert len(fake_local_cluster) == 2

    def test_none_on_spin_failure(self, monkeypatch):
        import dask.distributed as dd

        def _boom(**kwargs):
            raise RuntimeError("no cluster for you")

        monkeypatch.setattr(dd, "LocalCluster", _boom)
        host = DaskClusterHost(_cfg())
        assert host.ensure() is None


class TestClose:
    def test_idempotent_when_never_spun(self):
        host = DaskClusterHost(_cfg())
        host.close()  # must not raise
        host.close()

    def test_closes_and_clears(self, fake_local_cluster):
        host = DaskClusterHost(_cfg())
        host.ensure()
        cluster = fake_local_cluster[0]
        host.close()
        assert cluster.closed is True
        # After close, ensure() spins a fresh one (state was cleared).
        host.ensure()
        assert len(fake_local_cluster) == 2
