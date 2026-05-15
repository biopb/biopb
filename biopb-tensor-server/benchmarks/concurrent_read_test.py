"""Client read-latency scalability benchmarks.

Benchmarks steady-state reads for increasing numbers of concurrent clients
on a single node. The active scenarios are:
- same_source: all clients read the same tensor window from one source
- random_source: each client reads the same tensor window from a different source
"""

from __future__ import annotations

import atexit
import concurrent.futures
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import threading
import time
import traceback
from typing import Dict, Iterable, List

import pytest

from benchmarks.conftest import _generate_and_get_path, _register_source_with_server


CLIENT_COUNTS = [1, 4, 16]
READ_EDGE = int(os.environ.get("BIOPB_BENCH_READ_EDGE", "2048"))
CHUNK_EDGE = int(os.environ.get("BIOPB_BENCH_CHUNK_EDGE", str(READ_EDGE)))
BASE_EDGE = int(os.environ.get("BIOPB_BENCH_BASE_EDGE", str(max(2048, CHUNK_EDGE))))
READ_WINDOW = (slice(0, READ_EDGE), slice(0, READ_EDGE))
SOURCE_POOL_SIZE = max(CLIENT_COUNTS) * 4
READ_BYTES = READ_EDGE * READ_EDGE * 2
SERVER_START_TIMEOUT_S = 15.0

_PROCESS_WORKER_CLIENT = None
_PROCESS_WAVE_BARRIER = None


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _latency_summary_ms(latencies_s: Iterable[float]) -> Dict[str, float]:
    latencies_ms = [latency * 1000 for latency in latencies_s]
    return {
        "min_ms": min(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 0.50),
        "p95_ms": _percentile(latencies_ms, 0.95),
        "max_ms": max(latencies_ms),
    }


def _run_server_process(
    temp_cache_dir: str,
    source_specs: List[Dict],
    port: int,
    ready_queue,
    stop_event,
) -> None:
    from biopb.tensor import TensorFlightClient
    from biopb_tensor_server.cache import CacheManager
    from biopb_tensor_server.config import CacheConfig
    from biopb_tensor_server.server import TensorFlightServer

    location = f"grpc://127.0.0.1:{port}"
    server = None
    server_thread = None

    try:
        CacheManager.initialize(
            CacheConfig(
                backend="file",
                file_cache_dir=Path(temp_cache_dir) / "server-cache",
                file_max_segment_bytes=256 * 1024 * 1024,
                file_max_total_bytes=64 * 1024 * 1024 * 1024,
            )
        )

        server = TensorFlightServer(location)
        for spec in source_specs:
            _register_source_with_server(spec, spec["path"], server)

        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()

        deadline = time.time() + SERVER_START_TIMEOUT_S
        last_error = None
        while time.time() < deadline:
            try:
                client = TensorFlightClient(location, cache_bytes=0)
                try:
                    health = client.health_check()
                    sources = client.list_sources()
                finally:
                    client.close()
                if health.get("status") and len(sources) >= len(source_specs):
                    ready_queue.put(None)
                    stop_event.wait()
                    return
            except Exception as exc:  # pragma: no cover - readiness race
                last_error = exc
                time.sleep(0.2)

        ready_queue.put(
            "Benchmark server failed to become ready at "
            f"{location}. last_error={last_error!r}"
        )
    except Exception:
        ready_queue.put(traceback.format_exc())
    finally:
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
        if server_thread is not None:
            server_thread.join(timeout=5)
        CacheManager.reset()


def _wait_for_server_ready(process, ready_queue, location: str) -> None:
    try:
        error = ready_queue.get(timeout=SERVER_START_TIMEOUT_S + 5)
    except queue.Empty as exc:
        raise RuntimeError(
            f"Timed out waiting for benchmark server process at {location}; exitcode={process.exitcode}"
        ) from exc

    if error is not None:
        raise RuntimeError(str(error))


def _stop_server_process(process, stop_event) -> None:
    stop_event.set()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)


@pytest.fixture
def concurrency_benchmark_env(temp_cache_dir):
    """Create synthetic sources and serve them from a dedicated child process."""
    source_specs = []

    for index in range(SOURCE_POOL_SIZE):
        source_id = f"synthetic-thread-scale-{index:02d}"
        spec = {
            "id": source_id,
            "type": "ome_zarr",
            "generator": "generate_multiresolution_zarr",
            "params": {
                "base_shape": [BASE_EDGE, BASE_EDGE],
                "chunks": [CHUNK_EDGE, CHUNK_EDGE],
                "n_levels": 1,
                "dtype": "uint16",
            },
            "expected_tensors": [source_id],
        }
        path = _generate_and_get_path(spec, temp_cache_dir)
        spec["path"] = path
        source_specs.append(spec)

    port = random.randint(8900, 8999)
    location = f"grpc://127.0.0.1:{port}"
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    stop_event = ctx.Event()
    process = ctx.Process(
        target=_run_server_process,
        args=(temp_cache_dir, source_specs, port, ready_queue, stop_event),
    )
    process.start()

    try:
        _wait_for_server_ready(process, ready_queue, location)
        yield {
            "location": location,
            "source_pool": source_specs,
        }
    finally:
        _stop_server_process(process, stop_event)


def _select_sources(source_pool: List[Dict], client_count: int, scenario: str) -> List[Dict]:
    if scenario == "same_source":
        return [source_pool[0]] * client_count

    rng = random.Random(client_count)
    return rng.sample(source_pool, k=client_count)


def _prime_sources(location: str, assignments: List[Dict]) -> None:
    from biopb.tensor import TensorFlightClient

    unique_assignments = {spec["id"]: spec for spec in assignments}

    for spec in unique_assignments.values():
        client = TensorFlightClient(location, cache_bytes=0)
        try:
            arr = client.get_tensor(
                spec["id"],
                spec["expected_tensors"][0],
                slice_hint=READ_WINDOW,
            )
            arr.compute()
        finally:
            client.close()


def _cleanup_worker_client():
    """Clean up worker's FlightClient connection on process exit."""
    global _PROCESS_WORKER_CLIENT
    if _PROCESS_WORKER_CLIENT is not None:
        try:
            _PROCESS_WORKER_CLIENT.close()
        except Exception:
            pass
        _PROCESS_WORKER_CLIENT = None


def _init_process_worker(location: str, wave_barrier) -> None:
    from biopb.tensor import TensorFlightClient

    global _PROCESS_WORKER_CLIENT
    global _PROCESS_WAVE_BARRIER
    _PROCESS_WORKER_CLIENT = TensorFlightClient(location, cache_bytes=0)
    _PROCESS_WORKER_CLIENT.list_sources()
    _PROCESS_WAVE_BARRIER = wave_barrier

    # Register cleanup to close client on worker process exit
    atexit.register(_cleanup_worker_client)


def _process_worker_pid(_: int) -> int:
    return os.getpid()


def _process_compute_once(arr) -> Dict[str, object]:
    """Compute a pre-created dask array in process worker."""
    _PROCESS_WAVE_BARRIER.wait(timeout=5)

    started = time.perf_counter()
    data = arr.compute()
    return {
        "shape": tuple(data.shape),
        "latency_s": time.perf_counter() - started,
        "pid": os.getpid(),
    }


def _create_prewarmed_clients(location: str, client_count: int):
    from biopb.tensor import TensorFlightClient

    clients = [
        TensorFlightClient(location, cache_bytes=0)
        for _ in range(client_count)
    ]

    for client in clients:
        client.list_sources()

    return clients


def _create_process_pool(location: str, client_count: int):
    wave_barrier = mp.get_context("spawn").Barrier(client_count + 1)
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=client_count,
        mp_context=mp.get_context("spawn"),
        initializer=_init_process_worker,
        initargs=(location, wave_barrier),
    )
    # Force worker startup before benchmarking so process creation is not timed.
    list(executor.map(_process_worker_pid, range(client_count)))
    return executor, wave_barrier


def _close_clients(clients) -> None:
    for client in clients:
        client.close()


def _create_prewarmed_arrays(clients, assignments: List[Dict]) -> List:
    """Pre-create dask arrays for each client (removes get_flight_info from benchmark)."""
    arrays = []
    for client, spec in zip(clients, assignments):
        arr = client.get_tensor(
            spec["id"],
            spec["expected_tensors"][0],
            slice_hint=READ_WINDOW,
        )
        arrays.append(arr)
    return arrays


def _run_concurrent_wave_compute_only(executor, arrays) -> Dict[str, object]:
    """Benchmark only .compute() on pre-created dask arrays."""
    barrier = threading.Barrier(len(arrays))

    def compute_once(arr) -> Dict[str, object]:
        barrier.wait(timeout=5)
        started = time.perf_counter()
        data = arr.compute()
        return {
            "shape": tuple(data.shape),
            "latency_s": time.perf_counter() - started,
        }

    burst_started = time.perf_counter()
    results = list(executor.map(compute_once, arrays))

    return {
        "burst_latency_s": time.perf_counter() - burst_started,
        "client_results": results,
    }


def _run_process_wave_compute_only(executor, wave_barrier, arrays: List) -> Dict[str, object]:
    """Benchmark only .compute() on pre-created dask arrays in process workers."""
    futures = [executor.submit(_process_compute_once, arr) for arr in arrays]
    burst_started = time.perf_counter()
    wave_barrier.wait(timeout=5)
    results = [future.result() for future in futures]

    return {
        "burst_latency_s": time.perf_counter() - burst_started,
        "client_results": results,
    }


def _assert_and_report(result: Dict[str, object], scenario: str, client_count: int) -> None:
    client_results = result["client_results"]
    assert len(client_results) == client_count
    assert all(client_result["shape"] == (READ_EDGE, READ_EDGE) for client_result in client_results)

    latencies_s = [client_result["latency_s"] for client_result in client_results]
    summary = _latency_summary_ms(latencies_s)
    burst_latency_ms = result["burst_latency_s"] * 1000
    # source_id may be absent when using pre-created arrays
    source_count = len({client_result.get("source_id", "?") for client_result in client_results})
    total_mb = (client_count * READ_BYTES) / (1024 * 1024)
    throughput_mb_s = total_mb / result["burst_latency_s"] if result["burst_latency_s"] else 0.0

    print(
        "\n"
        f"scenario={scenario} clients={client_count} unique_sources={source_count} "
        f"burst_ms={burst_latency_ms:.2f} throughput_mb_s={throughput_mb_s:.2f} "
        f"min_ms={summary['min_ms']:.2f} "
        f"p50_ms={summary['p50_ms']:.2f} p95_ms={summary['p95_ms']:.2f} "
        f"max_ms={summary['max_ms']:.2f}"
    )


class TestProcessClientReadLatencyScalability:
    """Benchmark process-based client read latency as concurrency increases."""

    @pytest.mark.parametrize("client_count", CLIENT_COUNTS)
    def test_same_source(self, benchmark, concurrency_benchmark_env, client_count):
        """All process clients read the same source concurrently."""
        assignments = _select_sources(
            concurrency_benchmark_env["source_pool"],
            client_count,
            "same_source",
        )
        _prime_sources(concurrency_benchmark_env["location"], assignments)

        # Pre-create dask arrays in main process (pickle-safe to pass to workers)
        dummy_clients = _create_prewarmed_clients(
            concurrency_benchmark_env["location"],
            client_count,
        )
        arrays = _create_prewarmed_arrays(
            dummy_clients,
            assignments,
        )
        _close_clients(dummy_clients)

        executor, wave_barrier = _create_process_pool(
            concurrency_benchmark_env["location"],
            client_count,
        )
        with executor:
            result = benchmark.pedantic(
                lambda: _run_process_wave_compute_only(executor, wave_barrier, arrays),
                rounds=4,
                iterations=1,
            )

        _assert_and_report(result, "same_source_process", client_count)

    @pytest.mark.parametrize("client_count", CLIENT_COUNTS)
    def test_random_source(self, benchmark, concurrency_benchmark_env, client_count):
        """Each process client reads a different randomly selected source concurrently."""
        assignments = _select_sources(
            concurrency_benchmark_env["source_pool"],
            client_count,
            "random_source",
        )
        _prime_sources(concurrency_benchmark_env["location"], assignments)

        # Pre-create dask arrays in main process (pickle-safe to pass to workers)
        dummy_clients = _create_prewarmed_clients(
            concurrency_benchmark_env["location"],
            client_count,
        )
        arrays = _create_prewarmed_arrays(
            dummy_clients,
            assignments,
        )
        _close_clients(dummy_clients)

        executor, wave_barrier = _create_process_pool(
            concurrency_benchmark_env["location"],
            client_count,
        )
        with executor:
            result = benchmark.pedantic(
                lambda: _run_process_wave_compute_only(executor, wave_barrier, arrays),
                rounds=4,
                iterations=1,
            )

        _assert_and_report(result, "random_source_process", client_count)


class TestClientReadLatencyScalability:
    """Benchmark client read latency as concurrency increases."""

    @pytest.mark.parametrize("client_count", CLIENT_COUNTS)
    def test_same_source(self, benchmark, concurrency_benchmark_env, client_count):
        """All clients read the same source concurrently."""
        assignments = _select_sources(
            concurrency_benchmark_env["source_pool"],
            client_count,
            "same_source",
        )
        _prime_sources(concurrency_benchmark_env["location"], assignments)
        clients = _create_prewarmed_clients(
            concurrency_benchmark_env["location"],
            client_count,
        )

        # Pre-create dask arrays (removes get_flight_info from benchmark)
        arrays = _create_prewarmed_arrays(
            clients,
            assignments,
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
                result = benchmark.pedantic(
                    lambda: _run_concurrent_wave_compute_only(executor, arrays),
                    rounds=4,
                    iterations=1,
                )
        finally:
            _close_clients(clients)

        _assert_and_report(result, "same_source", client_count)

    @pytest.mark.parametrize("client_count", CLIENT_COUNTS)
    def test_random_source(self, benchmark, concurrency_benchmark_env, client_count):
        """Each client reads a different randomly selected source concurrently."""
        assignments = _select_sources(
            concurrency_benchmark_env["source_pool"],
            client_count,
            "random_source",
        )
        _prime_sources(concurrency_benchmark_env["location"], assignments)
        clients = _create_prewarmed_clients(
            concurrency_benchmark_env["location"],
            client_count,
        )

        # Pre-create dask arrays (removes get_flight_info from benchmark)
        arrays = _create_prewarmed_arrays(
            clients,
            assignments,
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
                result = benchmark.pedantic(
                    lambda: _run_concurrent_wave_compute_only(executor, arrays),
                    rounds=4,
                    iterations=1,
                )
        finally:
            _close_clients(clients)

        _assert_and_report(result, "random_source", client_count)