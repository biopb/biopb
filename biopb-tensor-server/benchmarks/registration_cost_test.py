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
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pytest
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.discovery import DiscoveryState
from biopb_tensor_server.sources import reconciler as reconciler_module
from tests import catalog_server, make_manager

from benchmarks.utils import (
    generate_synthetic_tiff,
    generate_synthetic_zarr,
    percentile,
)

ROOT_ENV = "BIOPB_REG_COST_ROOT"

# step -> label (an adapter, a source type, or "all") -> samples
Samples = Dict[str, Dict[str, List[float]]]


def _timed(samples: Samples, step: str, label: str, fn):
    """``fn`` that records its wall time under ``samples[step][label]``."""

    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            samples[step][label].append(time.perf_counter() - started)

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


def _report(samples: Samples, sizes: Samples) -> str:
    lines = []
    for step, by_label in samples.items():
        # Slowest first, so a site's cost is at the top of each step.
        for label, values in sorted(by_label.items(), key=lambda i: -sum(i[1])):
            lines.append(
                f"  {step} {label}: n={len(values)} "
                f"p50={percentile(values, 0.5) * 1000:.0f}ms "
                f"p95={percentile(values, 0.95) * 1000:.0f}ms "
                f"total={sum(values):.1f}s"
            )
    for source_type, by_column in sorted(sizes.items()):
        columns = "; ".join(
            f"{column} mean={sum(v) / len(v):.0f} max={max(v):.0f}"
            for column, v in by_column.items()
        )
        lines.append(f"  row {source_type}: {columns}")
    return "\n".join(lines)


def test_registration_cost_by_source_type(tree, monkeypatch):
    registry = get_default_registry()
    samples: Samples = defaultdict(lambda: defaultdict(list))
    for adapter_cls in registry._adapters:
        for step, name in (("claim", "claim"), ("create", "create_from_config")):
            wrapped = _timed(
                samples, step, adapter_cls.__name__, getattr(adapter_cls, name)
            )
            monkeypatch.setattr(adapter_cls, name, staticmethod(wrapped))
    monkeypatch.setattr(
        reconciler_module,
        "normalize_adapter",
        _timed(samples, "normalize", "all", reconciler_module.normalize_adapter),
    )
    server = catalog_server("localhost:0")
    db = server.metadata_db
    monkeypatch.setattr(
        db,
        "sync_source_added",
        _timed(samples, "row_write", "all", db.sync_source_added),
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

    sizes: Samples = defaultdict(lambda: defaultdict(list))
    for row in db.query(
        "SELECT source_type, coalesce(length(metadata_json), 0) AS metadata_bytes, "
        "length(CAST(tensors AS VARCHAR)) AS tensors_bytes FROM sources"
    ).to_pylist():
        for column in ("metadata_bytes", "tensors_bytes"):
            sizes[row["source_type"]][column].append(row[column])

    # A step with no samples means a wrapper above no longer sits on the call path.
    assert len(server.sources) > 0, f"nothing was registered under {tree}"
    for step in ("claim", "create", "normalize", "row_write"):
        assert samples[step], f"no {step} was timed"
    print(
        f"\nRegistration cost under {tree} ({len(server.sources)} sources, "
        f"{elapsed:.1f}s):\n{_report(samples, sizes)}"
    )
