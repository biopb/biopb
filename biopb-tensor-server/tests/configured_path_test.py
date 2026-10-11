"""A configured local path (file, typed dataset, directory) is registered by the
manager's first tick, as a scan-once root -- the same way a drop would be."""

import logging

import numpy as np
import pytest
import tifffile
from biopb_tensor_server.adapters import get_default_registry
from biopb_tensor_server.core.config import SourceConfig
from biopb_tensor_server.sources.source_manager import create_source_manager

from tests import catalog_server


def _serve_config(*sources: SourceConfig):
    """The server and manager ``serve`` would build, after the first tick."""
    server = catalog_server("localhost:0")
    manager = create_source_manager(
        server=server,
        registry=get_default_registry(),
        sources=list(sources),
        metadata_db=server.metadata_db,
        stability_window=0.0,
    )
    manager._handle_rescan()
    return server


def _tiff(path):
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint16))
    return path


def _registered(server):
    return dict(server.sources.snapshot())


def test_a_single_file_registers_once(tmp_path):
    f = _tiff(tmp_path / "plain.tif")

    server = _serve_config(SourceConfig(url=str(f)))

    ((sid, adapter),) = _registered(server).items()
    assert type(adapter).__name__ == "TiffAdapter"
    assert adapter.catalog_url == f.as_uri()


def test_a_single_file_keeps_its_alias_as_the_tree_root(tmp_path):
    f = _tiff(tmp_path / "plain.tif")

    server = _serve_config(SourceConfig(url=str(f), alias="lab"))

    ((_, adapter),) = _registered(server).items()
    assert adapter.catalog_url == "lab"


def test_a_typed_directory_registers_as_one_source(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "arr.zarr"
    arr = zarr.open_array(str(path), mode="w", shape=(16, 16), dtype="uint16")
    arr[:] = 1

    server = _serve_config(SourceConfig(url=str(path), type="zarr"))

    ((_, adapter),) = _registered(server).items()
    assert type(adapter).__name__ == "ZarrAdapter"


def test_a_resident_cloud_file_resolves_on_the_first_tick(tmp_path):
    """Cloud-ness comes from residency: a file that is on disk is read now, a
    placeholder would register unresolved."""
    f = _tiff(tmp_path / "cloud.tif")

    server = _serve_config(SourceConfig(url=str(f), cloud=True))

    ((_, adapter),) = _registered(server).items()
    assert type(adapter).__name__ == "TiffAdapter"


def test_a_missing_path_is_warned_and_the_rest_still_register(tmp_path, caplog):
    good = _tiff(tmp_path / "good.tif")

    with caplog.at_level(logging.WARNING):
        server = _serve_config(
            SourceConfig(url=str(tmp_path / "typo.tif")),
            SourceConfig(url=str(good)),
        )

    assert len(_registered(server)) == 1
    assert "does not exist" in caplog.text
