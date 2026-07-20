"""Tests for discovery's system/cloud-directory and offline-placeholder skipping.

These guard against the WSL/OneDrive startup hang: pointing discovery at a broad
root (e.g. a Windows user profile) used to walk AppData and then stall forever on
an OneDrive "Files On-Demand" placeholder whose content recalls on read, leaving
the Flight server unbound. Discovery must instead prune those subtrees and skip
non-resident files without opening them.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from biopb_tensor_server.core import discovery
from biopb_tensor_server.core.discovery import (
    _is_offline_placeholder,
    _is_skippable_system_dir,
    walk_with_identity_tracking,
)

# Native Windows ``os.stat`` has no ``st_blocks``, so the POSIX zero-block
# placeholder heuristic — and the tests that exercise it — only apply where
# ``st_blocks`` exists (Linux, including WSL reading /mnt/c). On native Windows a
# real placeholder is caught by the ``st_file_attributes`` path instead.
requires_st_blocks = pytest.mark.skipif(
    not hasattr(os.stat_result, "st_blocks"),
    reason="os.stat has no st_blocks (native Windows); POSIX zero-block signal N/A",
)


class TestSkippableSystemDir:
    @pytest.mark.parametrize(
        "name",
        [
            "AppData",
            "appdata",
            "Windows",
            "Program Files",
            "Program Files (x86)",
            "ProgramData",
            "$Recycle.Bin",
            "node_modules",
            "OneDrive",
            "OneDrive - Contoso Ltd",
            "OneDrive-Personal",
        ],
    )
    def test_system_and_cloud_names_are_skipped(self, name):
        assert _is_skippable_system_dir(name) is True

    @pytest.mark.parametrize(
        "name",
        ["Microscopy", "data", "Data", "experiments", "onedrive_backup_2023", "images"],
    )
    def test_real_data_names_are_not_skipped(self, name):
        # "onedrive_backup_2023" must NOT match: only the OneDrive root naming
        # (OneDrive / "OneDrive - Org" / OneDrive-<x>) is a cloud root.
        assert _is_skippable_system_dir(name) is False


class TestOfflinePlaceholder:
    def test_normal_file_is_not_offline(self, tmp_path):
        f = tmp_path / "real.bin"
        f.write_bytes(b"actual resident bytes")
        assert _is_offline_placeholder(f) is False

    @requires_st_blocks
    def test_sparse_stub_is_treated_as_offline(self, tmp_path):
        # A file with a logical size but zero allocated blocks mirrors a cloud
        # placeholder on POSIX (st_size > 0, st_blocks == 0).
        f = tmp_path / "stub.bin"
        with open(f, "wb") as fh:
            fh.truncate(4 * 1024 * 1024)
        st = os.stat(f)
        if st.st_blocks != 0:
            pytest.skip("filesystem allocates blocks for sparse files; signal N/A")
        assert _is_offline_placeholder(f) is True

    @requires_st_blocks
    def test_empty_file_is_skipped(self, tmp_path):
        # An empty file has zero allocated blocks; with no size floor it is
        # skipped too. Harmless — there is no content to serve, and nothing to
        # recall, so it cannot be the source of a hang either way.
        f = tmp_path / "empty.bin"
        f.touch()
        assert _is_offline_placeholder(f) is True

    @requires_st_blocks
    def test_escape_hatch_disables_skip(self, tmp_path, monkeypatch):
        f = tmp_path / "stub.bin"
        with open(f, "wb") as fh:
            fh.truncate(4 * 1024 * 1024)
        if os.stat(f).st_blocks != 0:
            pytest.skip("filesystem allocates blocks for sparse files; signal N/A")
        monkeypatch.setattr(discovery, "_SKIP_OFFLINE", False)
        assert _is_offline_placeholder(f) is False


def _write_tiff(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint16))


class TestWalkPruning:
    def test_walk_prunes_system_dirs_and_offline_files(self, tmp_path):
        # A real data tree alongside system/cloud noise the walk must avoid.
        _write_tiff(tmp_path / "Microscopy" / "good.tif")
        _write_tiff(tmp_path / "AppData" / "Local" / "junk.tif")
        _write_tiff(tmp_path / "OneDrive - Lab" / "cloud.tif")

        # An offline-looking (sparse stub) file at the top level.
        stub = tmp_path / "placeholder.tif"
        with open(stub, "wb") as fh:
            fh.truncate(4 * 1024 * 1024)
        # getattr: native Windows os.stat has no st_blocks, so the offline
        # sub-check below is simply skipped there while name-pruning still runs.
        offline_supported = getattr(os.stat(stub), "st_blocks", None) == 0

        visited: set = set()
        walked = {str(p) for p in walk_with_identity_tracking(tmp_path, visited)}

        assert str(tmp_path / "Microscopy") in walked
        assert str(tmp_path / "Microscopy" / "good.tif") in walked

        # System/cloud subtrees pruned wholesale — neither the dir nor children.
        assert str(tmp_path / "AppData") not in walked
        assert str(tmp_path / "AppData" / "Local" / "junk.tif") not in walked
        assert str(tmp_path / "OneDrive - Lab") not in walked
        assert str(tmp_path / "OneDrive - Lab" / "cloud.tif") not in walked

        if offline_supported:
            assert str(stub) not in walked
