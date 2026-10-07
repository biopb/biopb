"""A source rebuilt from its row lists the uploaded tensors that are on disk now.

The row says what was listed at its last write; the registration that rebuilds the
source attaches whatever fields the disk holds, and relists the row when the two
differ.
"""

import shutil
import threading

import numpy as np
from biopb.tensor import TensorFlightClient
from biopb_tensor_server.adapters.fields import fields_root
from biopb_tensor_server.serving.metadata_db import MetadataDatabase

from tests import deferred_registration_test as drt
from tests.restore_test import _Run as _RestoreRun


class _Run(_RestoreRun):
    """A restore run on a writable server, listening, whose relists are recorded."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path, writable=True, write_dir=tmp_path / "w")
        self.server.mark_ready()
        threading.Thread(target=self.server.serve, daemon=True).start()
        self.relisted = []
        relist = self.db.relist_tensors

        def spy(source_id, adapter):
            changed = relist(source_id, adapter)
            self.relisted.append(changed)
            return changed

        self.db.relist_tensors = spy

    def listed(self, source_id):
        rows = self.db.query(
            "SELECT [t.array_id for t in tensors] AS ids FROM sources "
            f"WHERE source_id = '{source_id}'"
        ).to_pylist()
        assert len(rows) == 1
        return rows[0]["ids"]


def _first_run(tmp_path, fields=("raw",)):
    """Discover one zarr source and upload *fields* to it; return its id."""
    run = _Run(tmp_path)
    drt._make_zarr(run.monitored, "a.zarr")
    run.manager._handle_rescan()
    (sid,) = [s for s in drt._rows(run.server) if "scratch" not in s]
    client = TensorFlightClient(f"grpc://localhost:{run.server.port}")
    try:
        arr = np.full((4, 6), 7, dtype=np.uint16)
        for name in fields:
            desc = client.setup_array_upload(
                f"zarr://{sid}/@fields/{name}", arr, chunk_shape=(2, 3)
            )
            client.upload_array(desc, arr)
    finally:
        client.close()
    assert run.listed(sid) == [sid, *[f"{sid}/@fields/{n}" for n in fields]]
    run.stop()
    return sid


def _restart(tmp_path):
    run = _Run(tmp_path)
    run.restore()
    return run


class TestAHydratedSourceListsWhatIsOnDisk:
    def test_a_field_the_row_does_not_list_is_relisted(self, tmp_path):
        sid = _first_run(tmp_path)
        # The row as an earlier write (or a re-sync that failed) left it.
        db = MetadataDatabase(
            store_path=tmp_path / "catalog.duckdb", restore_sources=True
        )
        db.open()
        db._get_connection().execute(
            "UPDATE source_catalog SET tensors = [t for t in tensors "
            "if t.array_id NOT LIKE '%@fields%'] WHERE source_id = ?",
            [sid],
        )
        db.close()

        run = _restart(tmp_path)
        try:
            assert run.listed(sid) == [sid]  # restored as written
            run.server.sources.get_registered(sid)  # hydrates on a read
            assert run.listed(sid) == [sid, f"{sid}/@fields/raw"]
            assert run.relisted == [True]
        finally:
            run.stop()

    def test_a_field_removed_while_down_is_no_longer_listed(self, tmp_path):
        sid = _first_run(tmp_path)
        shutil.rmtree(fields_root(tmp_path / "w") / sid / "raw")

        run = _restart(tmp_path)
        try:
            assert run.listed(sid) == [sid, f"{sid}/@fields/raw"]  # restored as written
            adapter = run.server.sources.get_registered(sid)
            assert run.listed(sid) == [sid]
            assert [
                t.array_id for t in run.server.sources.catalog_tensors(sid, adapter)
            ] == [sid]
        finally:
            run.stop()

    def test_a_row_that_is_right_is_not_written(self, tmp_path):
        sid = _first_run(tmp_path, fields=("raw", "other"))

        run = _restart(tmp_path)
        try:
            run.server.sources.get_registered(sid)
            assert sorted(run.listed(sid)) == sorted(
                [sid, f"{sid}/@fields/other", f"{sid}/@fields/raw"]
            )
            assert run.relisted == [False]
        finally:
            run.stop()

    def test_a_failed_relist_does_not_cost_the_source(self, tmp_path, monkeypatch):
        sid = _first_run(tmp_path)
        run = _restart(tmp_path)
        try:

            def boom(source_id, adapter):
                raise RuntimeError("catalog write failed")

            run.db.relist_tensors = boom
            adapter = run.server.sources.get_registered(sid)
            assert adapter is not None
            assert not run.manager._reconciler.is_pending(sid)
        finally:
            run.stop()
