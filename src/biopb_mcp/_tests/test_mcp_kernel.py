"""Tests for KernelHost (the child Jupyter kernel manager).

The unit tests start a *plain* python kernel (no napari bootstrap, no display)
and exercise execute/interrupt/restart/shutdown.  A separate, display-gated
test runs the real napari bootstrap end-to-end.
"""

import os
import sys
import time

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("jupyter_client")

from biopb_mcp.mcp._kernel import KernelHost  # noqa: E402


class TestConfigureDask:
    """Unit tests for _configure_dask (no kernel / no display needed)."""

    def test_in_process_scheduler_returns_no_client(self):
        """threads/synchronous schedulers yield no client and no cluster."""
        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask({"dask_scheduler": "threads"})
        assert client is None
        assert cluster is None

    def test_external_address_connects_without_cluster(self, monkeypatch):
        """distributed + an explicit address attaches a Client, no cluster."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        created = {}

        class _FakeClient:
            def __init__(self, address):
                created["address"] = address

        monkeypatch.setattr(dd, "Client", _FakeClient)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {
                "dask_scheduler": "distributed",
                "dask_distributed_address": "tcp://1.2.3.4:8786",
            }
        )
        assert isinstance(client, _FakeClient)
        assert created["address"] == "tcp://1.2.3.4:8786"
        assert cluster is None

    def test_local_cluster_failure_falls_back_to_threads(self, monkeypatch):
        """A LocalCluster spawn failure degrades to in-process, not a crash."""
        pytest.importorskip("dask.distributed")
        import dask.distributed as dd

        def _boom(*args, **kwargs):
            raise RuntimeError("no cluster for you")

        monkeypatch.setattr(dd, "LocalCluster", _boom)

        from biopb_mcp.mcp._bootstrap import _configure_dask

        client, cluster = _configure_dask(
            {"dask_scheduler": "distributed", "dask_distributed_address": ""}
        )
        assert client is None
        assert cluster is None


@pytest.fixture
def kernel():
    """A bare kernel with no bootstrap and no health probe."""
    host = KernelHost(health_probe_code=None, startup_timeout=60.0)
    host.start()
    yield host
    host.shutdown()


class TestKernelExecute:
    def test_stdout_captured(self, kernel):
        res = kernel.execute("print('hi')")
        assert res["status"] == "ok"
        assert "hi" in res["stdout"]

    def test_expression_result(self, kernel):
        res = kernel.execute("1 + 2")
        assert res["status"] == "ok"
        assert "3" in res["result_text"]

    def test_error_status_and_traceback(self, kernel):
        res = kernel.execute("1 / 0")
        assert res["status"] == "error"
        assert "ZeroDivisionError" in res["error_text"]
        # ANSI escape codes are stripped.
        assert "\x1b[" not in res["error_text"]

    def test_variables_persist(self, kernel):
        kernel.execute("my_var = 99")
        res = kernel.execute("print(my_var)")
        assert "99" in res["stdout"]

    def test_timeout_interrupts(self, kernel):
        res = kernel.execute("import time; time.sleep(10)", timeout=0.5)
        assert res["status"] == "timeout"
        # Kernel survives and accepts new work afterwards.
        res2 = kernel.execute("print('alive')", timeout=10.0)
        assert "alive" in res2["stdout"]


class TestKernelControl:
    def test_interrupt_frees_busy_loop(self, kernel):
        import threading

        results = {}

        def run():
            results["res"] = kernel.execute("while True: pass", timeout=30.0)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(1.0)
        kernel.interrupt()
        t.join(timeout=15.0)
        assert not t.is_alive()
        assert results["res"]["status"] in ("error", "ok")

    def test_restart_clears_namespace(self, kernel):
        kernel.execute("survivor = 1")
        assert "1" in kernel.execute("print(survivor)")["stdout"]
        kernel.restart()
        res = kernel.execute("print('survivor' in dir())")
        assert "False" in res["stdout"]

    def test_busy_returns_busy_status(self, kernel):
        import threading

        kernel._busy_lock_timeout = 0.2

        def run():
            kernel.execute("import time; time.sleep(3)", timeout=10.0)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.5)
        res = kernel.execute("print('x')")
        assert res["status"] == "busy"
        t.join(timeout=15.0)


class TestKernelLifecycle:
    def test_is_alive(self, kernel):
        assert kernel.is_alive()

    def test_shutdown_removes_connection_file(self):
        host = KernelHost(health_probe_code=None, startup_timeout=60.0)
        host.start()
        conn_file = host._km.connection_file
        assert os.path.exists(conn_file)
        host.shutdown()
        assert not host.is_alive()
        assert not os.path.exists(conn_file)

    def test_health_probe_failure_raises(self):
        # Probe expects a name that does not exist in a bare kernel.
        host = KernelHost(
            health_probe_code="print('viewer' in dir())",
            health_probe_expect="True",
            startup_timeout=60.0,
        )
        with pytest.raises(RuntimeError, match="health probe failed"):
            host.start()
        host.shutdown()


# ---------------------------------------------------------------------------
# Full napari bootstrap — only in a real desktop session.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("QT_QPA_PLATFORM") == "offscreen"
    or not os.getenv("DISPLAY")
    or (sys.platform == "darwin" and os.getenv("CI") == "true"),
    reason="napari bootstrap needs a real display",
)
class TestNapariBootstrap:
    @pytest.fixture
    def napari_kernel(self):
        line = "import biopb_mcp.mcp._bootstrap as _b; _b.bootstrap()"
        host = KernelHost(
            extra_arguments=[f"--IPKernelApp.exec_lines={line}"],
            startup_timeout=120.0,
        )
        host.start()
        yield host
        host.shutdown()

    def test_viewer_in_namespace(self, napari_kernel):
        res = napari_kernel.execute("print('viewer' in dir())")
        assert "True" in res["stdout"]

    def test_screenshot_round_trips(self, napari_kernel):
        snippet = (
            "import base64 as _b64, cv2 as _cv2\n"
            "_arr = viewer.screenshot(canvas_only=True)\n"
            "_bgra = _cv2.cvtColor(_arr, _cv2.COLOR_RGBA2BGRA)\n"
            "_ok, _buf = _cv2.imencode('.png', _bgra)\n"
            "print('<<PNG_B64>>' + _b64.b64encode(_buf.tobytes()).decode())\n"
        )
        res = napari_kernel.execute(snippet)
        assert "<<PNG_B64>>" in res["stdout"]
