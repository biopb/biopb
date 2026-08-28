"""Tests for cache-dir storage classification (biopb/biopb#571 NFS/cloud gate).

The file cache mmaps its segments and assumes local-POSIX semantics, so the
launcher demotes to the memory backend when the configured cache dir is on a
network filesystem or a cloud-synced folder. These cover the platform-agnostic
classifiers plus the Linux mountinfo parse; the Windows/macOS syscalls are
smoke-checked for "never raises, positive-signal-only" behaviour.
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from biopb import _fs_detect as fs_detect
from biopb._fs_detect import (
    _classify_fstype,
    _cloud_path_hint,
    _is_unc,
    _linux_network_type,
    _nearest_existing,
    _path_under,
    _unescape_mountinfo,
    cloud_sync_hint,
    network_filesystem_type,
    unsafe_cache_dir_reason,
)


class TestClassifyFstype:
    @pytest.mark.parametrize(
        "fstype,expected",
        [
            ("nfs", "nfs"),
            ("nfs4", "nfs4"),
            ("NFS4", "nfs4"),  # case-insensitive
            ("cifs", "cifs"),
            ("smbfs", "smbfs"),
            ("fuse.sshfs", "fuse.sshfs"),
            ("fuse.rclone", "fuse.rclone"),
            ("fuse.blobfuse2", "fuse.blobfuse2"),
            # Local / non-network -> None
            ("ext4", None),
            ("xfs", None),
            ("btrfs", None),
            ("zfs", None),
            ("tmpfs", None),
            ("overlay", None),
            ("apfs", None),
            ("fuse.encfs", None),  # local FUSE
            ("fuse.gvfsd-fuse", None),
            ("", None),
            (None, None),
        ],
    )
    def test_classify(self, fstype, expected):
        assert _classify_fstype(fstype) == expected


class TestPathHelpers:
    def test_path_under(self):
        assert _path_under("/mnt/data/sub/x", "/mnt/data")
        assert _path_under("/mnt/data", "/mnt/data")
        assert _path_under("/mnt/data", "/mnt/data/")  # trailing slash tolerated
        assert _path_under("/anything/at/all", "/")  # everything is under root
        # Prefix must be a path boundary, not a string prefix.
        assert not _path_under("/mnt/database", "/mnt/data")
        assert not _path_under("/other", "/mnt/data")

    def test_unescape_mountinfo(self):
        assert _unescape_mountinfo("/mnt/plain") == "/mnt/plain"
        assert _unescape_mountinfo(r"/mnt/my\040share") == "/mnt/my share"
        assert _unescape_mountinfo(r"/a\011b") == "/a\tb"  # tab
        assert _unescape_mountinfo(r"/back\134slash") == "/back\\slash"

    def test_is_unc(self):
        assert _is_unc(r"\\server\share")
        assert _is_unc("//server/share")
        assert not _is_unc(r"C:\local")
        assert not _is_unc("/mnt/local")

    def test_nearest_existing_walks_up_to_real_ancestor(self):
        d = tempfile.mkdtemp()
        try:
            missing = Path(d) / "does" / "not" / "exist" / "yet"
            assert _nearest_existing(missing) == Path(d)
        finally:
            os.rmdir(d)


class TestLinuxNetworkType:
    """The /proc/self/mountinfo parse: pick the longest-prefix mount, classify."""

    # Two mounts: local root, and an NFS export mounted under it.
    _MOUNTINFO = (
        "23 28 0:21 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
        "44 23 0:41 / /mnt/lab rw,relatime shared:9 - nfs4 fs01:/lab rw,vers=4.2\n"
        "45 23 0:42 / /mnt/local rw,relatime shared:9 - xfs /dev/sdb1 rw\n"
    )

    def _classify(self, target: str, mountinfo: "str | None" = None):
        # _linux_network_type opens /proc/self/mountinfo and realpath()s target;
        # feed a fixed table and normalize realpath to forward slashes so this
        # Linux-only parser sees a POSIX path even when the test host is Windows
        # (str(Path("/mnt/lab")) yields backslashes there).
        with patch(
            "builtins.open",
            return_value=io.StringIO(
                self._MOUNTINFO if mountinfo is None else mountinfo
            ),
        ):
            with patch(
                "os.path.realpath", side_effect=lambda p: str(p).replace(os.sep, "/")
            ):
                return _linux_network_type(Path(target))

    def test_nfs_export_detected(self):
        assert self._classify("/mnt/lab/experiment/cache") == "nfs4"

    def test_local_root_is_none(self):
        assert self._classify("/home/user/cache") is None

    def test_local_xfs_mount_is_none(self):
        assert self._classify("/mnt/local/cache") is None

    def test_longest_prefix_wins_over_root(self):
        # /mnt/lab (nfs4) must beat / (ext4) even though both are prefixes.
        assert self._classify("/mnt/lab") == "nfs4"

    def test_nfs_shadowing_autofs_at_same_mountpoint(self):
        # The standard automount layout: an autofs entry with the real nfs
        # mounted on top at the *identical* mount point. mountinfo lists mounts
        # bottom-to-top, so the nfs (last) is the effective mount and must win --
        # a strict-'>' tie-break would keep the first-seen autofs and miss it,
        # silently leaving the mmap file cache on NFS (biopb/biopb#571).
        shadowed = (
            "23 28 0:21 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
            "44 23 0:41 / /home/FCAM rw,relatime shared:9 - autofs systemd-1 rw\n"
            "45 44 0:42 / /home/FCAM rw,relatime shared:9 - nfs fs01:/home rw,vers=4.2\n"
        )
        assert self._classify("/home/FCAM/jyu/cache", shadowed) == "nfs"


class TestCloudPathHint:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/home/u/OneDrive/cache", "OneDrive"),
            ("/home/u/OneDrive - Contoso/cache", "OneDrive"),
            ("/home/u/Dropbox/biopb", "Dropbox"),
            ("/Users/u/Library/Mobile Documents/com~apple~CloudDocs/c", "iCloud Drive"),
            (
                "/Users/u/Library/CloudStorage/OneDrive-Contoso/c",
                "macOS CloudStorage folder",
            ),
            ("/home/u/Nextcloud/c", "Nextcloud/ownCloud"),
            ("/tmp/biopb-cache-1000", None),
            ("/var/cache/biopb", None),
        ],
    )
    def test_hint(self, path, expected):
        assert _cloud_path_hint(Path(path)) == expected

    def test_hint_on_nonexistent_cloud_path(self):
        # cloud_sync_hint reads the *configured* path components, so a not-yet-
        # created dir under a cloud root is still flagged (POSIX; no stat needed).
        if os.name == "nt":
            pytest.skip("component heuristic exercised without Windows stat here")
        assert cloud_sync_hint("/home/u/OneDrive/never/created/biopb") == "OneDrive"


class TestNetworkFilesystemType:
    def test_real_local_tmpdir_is_not_network(self):
        # The dev/CI box's temp dir is local -> None (positive-signal-only).
        d = tempfile.mkdtemp()
        try:
            assert network_filesystem_type(d) is None
        finally:
            os.rmdir(d)

    def test_never_raises_on_garbage(self):
        # A bogus path must degrade to None, never raise.
        assert network_filesystem_type("\x00/not/a/real\x00/path") is None

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux dispatch")
    def test_dispatch_uses_linux_parser(self):
        with patch.object(fs_detect, "_linux_network_type", return_value="nfs4") as m:
            assert network_filesystem_type("/some/where") == "nfs4"
            assert m.called


class TestUnsafeCacheDirReason:
    def test_local_dir_is_safe(self):
        d = tempfile.mkdtemp()
        try:
            assert unsafe_cache_dir_reason(d) is None
        finally:
            os.rmdir(d)

    def test_network_reason(self):
        with patch.object(fs_detect, "network_filesystem_type", return_value="nfs4"):
            reason = unsafe_cache_dir_reason("/mnt/lab/cache")
            assert reason is not None and "nfs4" in reason and "network" in reason

    def test_cloud_reason_when_not_network(self):
        with patch.object(fs_detect, "network_filesystem_type", return_value=None):
            with patch.object(fs_detect, "cloud_sync_hint", return_value="OneDrive"):
                reason = unsafe_cache_dir_reason("/home/u/OneDrive/cache")
                assert reason is not None and "OneDrive" in reason

    def test_network_takes_precedence_over_cloud(self):
        with patch.object(fs_detect, "network_filesystem_type", return_value="cifs"):
            with patch.object(
                fs_detect, "cloud_sync_hint", return_value="OneDrive"
            ) as cloud:
                reason = unsafe_cache_dir_reason("/mnt/x")
                assert "cifs" in reason
                # Short-circuits: cloud probe not even consulted.
                assert not cloud.called


class TestMemoryFilesystem:
    """RAM-backed storage: mmap-safe, but the wrong place for a cache whose whole
    job is keeping bytes out of RAM (biopb/biopb docs/client-disk-cache.md)."""

    def test_tmpfs_is_reported(self):
        with patch.object(fs_detect, "_raw_fstype", return_value="tmpfs"):
            assert fs_detect.memory_filesystem_type("/dev/shm/x") == "tmpfs"

    @pytest.mark.parametrize("fstype", ["ramfs", "devtmpfs", "TMPFS"])
    def test_other_ram_backed_types(self, fstype):
        with patch.object(fs_detect, "_raw_fstype", return_value=fstype):
            assert fs_detect.memory_filesystem_type("/x") == fstype.lower()

    @pytest.mark.parametrize("fstype", ["ext4", "xfs", "btrfs", "apfs", None])
    def test_real_storage_is_not_memory(self, fstype):
        with patch.object(fs_detect, "_raw_fstype", return_value=fstype):
            assert fs_detect.memory_filesystem_type("/x") is None

    def test_never_raises(self):
        with patch.object(fs_detect, "_raw_fstype", side_effect=OSError("boom")):
            assert fs_detect.memory_filesystem_type("/x") is None


class TestRejectMemoryIsOptIn:
    """The two tenants disagree: the server's fallback for an unusable cache_dir
    IS memory, so demoting a tmpfs dir would trade RAM for the same RAM."""

    def test_default_ignores_a_ram_backed_dir(self):
        with patch.object(fs_detect, "memory_filesystem_type", return_value="tmpfs"):
            with patch.object(fs_detect, "network_filesystem_type", return_value=None):
                with patch.object(fs_detect, "cloud_sync_hint", return_value=None):
                    assert fs_detect.unsafe_cache_dir_reason("/dev/shm/x") is None

    def test_opt_in_rejects_it(self):
        with patch.object(fs_detect, "memory_filesystem_type", return_value="tmpfs"):
            with patch.object(fs_detect, "network_filesystem_type", return_value=None):
                with patch.object(fs_detect, "cloud_sync_hint", return_value=None):
                    reason = fs_detect.unsafe_cache_dir_reason(
                        "/dev/shm/x", reject_memory=True
                    )
        assert reason is not None and "tmpfs" in reason

    def test_network_still_wins_over_memory(self):
        with patch.object(fs_detect, "network_filesystem_type", return_value="nfs4"):
            with patch.object(
                fs_detect, "memory_filesystem_type", return_value="tmpfs"
            ):
                reason = fs_detect.unsafe_cache_dir_reason("/x", reject_memory=True)
        assert "nfs4" in reason


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mountinfo")
class TestRawFstypeOnThisMachine:
    def test_shm_is_ram_backed(self):
        if not Path("/dev/shm").exists():
            pytest.skip("no /dev/shm")
        assert fs_detect.memory_filesystem_type("/dev/shm") is not None

    def test_tmp_path_is_not_ram_backed_unless_it_really_is(self, tmp_path):
        # Only asserts the probe agrees with the mount table, not what /tmp is.
        raw = fs_detect._raw_fstype(tmp_path)
        expected = (
            raw.lower() if raw and raw.lower() in fs_detect._MEMORY_FSTYPES else None
        )
        assert fs_detect.memory_filesystem_type(tmp_path) == expected
