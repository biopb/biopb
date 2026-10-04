"""What registering sources costs, per source type.

Registering a source opens and parses its file; on a large site that is hours.
This scans a tree with the real ``SourceManager`` and records, per source type,
how long each step took (the adapter's ``claim`` probes, ``create_from_config``,
``normalize_adapter``, the catalog row write) and how large the catalog row is,
so the cost of a start is read from numbers rather than guessed. The timers wrap
the production calls from outside; the server carries no instrumentation.

By default the tree is a handful of synthetic files. Point it at a real site::

    BIOPB_REG_COST_ROOT=/labs/site/data \\
        pytest benchmarks/registration_cost_test.py -s

``-s`` shows the table.
"""

import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import DiscoveryState
from biopb_tensor_server.serving.metadata_db import MetadataDatabase
from biopb_tensor_server.sources import reconciler as reconciler_module
from tests import catalog_server, make_manager

from benchmarks.utils import generate_synthetic_tiff, generate_synthetic_zarr

ROOT_ENV = "BIOPB_REG_COST_ROOT"


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class _Samples:
    """Seconds (or bytes) by (group, step), safe to append from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add(self, group: str, step: str, value: float) -> None:
        with self._lock:
            self._values[group][step].append(value)

    def groups(self) -> Dict[str, Dict[str, List[float]]]:
        with self._lock:
            return {
                g: {s: list(v) for s, v in d.items()} for g, d in self._values.items()
            }


def _summarize(values: Dict[str, List[float]], *, size: bool) -> str:
    parts = []
    for step, samples in values.items():
        if size:
            parts.append(
                f"{step} mean={sum(samples) / len(samples):.0f} max={max(samples):.0f}"
            )
        else:
            parts.append(
                f"{step} p50={_percentile(samples, 0.5) * 1000:.0f}ms "
                f"p95={_percentile(samples, 0.95) * 1000:.0f}ms "
                f"total={sum(samples):.1f}s"
            )
    return "; ".join(parts)


def _timed(samples: _Samples, group, step, fn):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            samples.add(group, step, time.perf_counter() - started)

    return wrapper


@pytest.fixture
def tree(tmp_path) -> Path:
    site = os.environ.get(ROOT_ENV)
    if site:
        return Path(site)
    for i in range(3):
        (tmp_path / f"t{i}").mkdir()
        generate_synthetic_tiff(str(tmp_path / f"t{i}"))
    for i in range(2):
        (tmp_path / f"z{i}").mkdir()
        generate_synthetic_zarr(str(tmp_path / f"z{i}"))
    return tmp_path


def _instrument(monkeypatch, registry, samples: _Samples) -> None:
    """Time each adapter's claim and create, and the normalize and row write."""
    for adapter_cls in registry._adapters:
        name = adapter_cls.__name__
        monkeypatch.setattr(
            adapter_cls,
            "claim",
            staticmethod(_timed(samples, f"claim {name}", "claim", adapter_cls.claim)),
        )
        monkeypatch.setattr(
            adapter_cls,
            "create_from_config",
            staticmethod(
                _timed(samples, name, "create", adapter_cls.create_from_config)
            ),
        )
    monkeypatch.setattr(
        reconciler_module,
        "normalize_adapter",
        _timed(samples, "all", "normalize", reconciler_module.normalize_adapter),
    )


def _size_rows(db: MetadataDatabase, samples: _Samples) -> None:
    rows = db.query(
        "SELECT source_type, coalesce(length(metadata_json), 0) AS metadata_bytes, "
        "length(CAST(tensors AS VARCHAR)) AS tensors_bytes FROM sources"
    ).to_pylist()
    for row in rows:
        samples.add(row["source_type"], "metadata_bytes", row["metadata_bytes"])
        samples.add(row["source_type"], "tensors_bytes", row["tensors_bytes"])


def test_registration_cost_by_source_type(tree, monkeypatch):
    registry = get_default_registry()
    samples = _Samples()
    _instrument(monkeypatch, registry, samples)
    server = catalog_server("localhost:0")
    db = server.metadata_db
    monkeypatch.setattr(
        db,
        "sync_source_added",
        _timed(samples, "all", "row_write", db.sync_source_added),
    )
    manager = make_manager(
        server=server,
        registry=registry,
        discovery_state=DiscoveryState(),
        metadata_db=db,
        monitored_dirs={tree},
        stability_window=0,
        registration_workers=0,
    )

    started = time.perf_counter()
    manager._handle_rescan()
    elapsed = time.perf_counter() - started

    _size_rows(db, samples)
    registered = len(server.sources)
    assert registered > 0, f"nothing was registered under {tree}"

    lines = []
    # Source types and the shared steps first, then the claim probes, slowest first.
    groups = samples.groups()
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            item[0].startswith("claim "),
            -sum(item[1].get("claim", [0])),
        ),
    )
    for group, values in ordered:
        size = "metadata_bytes" in values
        lines.append(f"  {group}: {_summarize(values, size=size)}")
    report = (
        f"Registration cost under {tree} ({registered} sources, {elapsed:.1f}s):\n"
        + "\n".join(lines)
    )
    print("\n" + report)
