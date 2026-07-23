"""Best-effort classification of the cache directory's storage: plain local disk,
or network / cloud-synced storage the Arrow file cache can't safely use?

The file cache mmaps its segment files -- both the server (segment reads, boot
index) and the localhost client fast path (Option C, biopb/biopb#571) map them
and rely on local-POSIX semantics: an unlinked-but-mapped inode keeps its blocks
until the last close, and a mapped page never vanishes under the reader. Network
filesystems (NFS/CIFS/...) break that -- touching a mapping to a file the server
removed can SIGBUS/ESTALE -- and cloud "Files On-Demand" folders
(OneDrive/iCloud/Dropbox) dehydrate idle files, so a later mmap read stalls on a
recall. So the launcher classifies the configured ``cache_dir`` once at startup
and falls back to the in-memory backend when it isn't local disk (which also
disables the client fast path for free: a memory backend never locates a chunk).

Design rules for every probe here:

- **Metadata only.** Never opens or reads a file, so it can never trigger a
  cloud recall while trying to detect one.
- **Never raises.** Any failure -> ``None`` ("undeterminable").
- **Positive signals only.** We downgrade to memory solely on a *positive*
  network/cloud signal; an unknown or exotic-but-local filesystem returns
  ``None`` and keeps the file cache, so working deployments are never demoted by
  a detection gap.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Network filesystem type names (lowercased) as the OS mount table reports them,
# across Linux /proc/self/mountinfo and the macOS statfs f_fstypename.
_NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs3",
        "nfs4",
        "cifs",
        "smbfs",
        "smb",
        "smb2",
        "smb3",
        "ncpfs",
        "afs",
        "afpfs",
        "9p",
        "ceph",
        "lustre",
        "gpfs",
        "beegfs",
        "glusterfs",
        "ocfs2",
        "webdav",
        "davfs",
        "ftp",
    }
)

# FUSE subtypes (the token after "fuse.") that are network/cloud-backed. A plain
# or unknown "fuse.*" is left as local -- many FUSE mounts are local (encfs,
# gocryptfs, gvfs), and we only demote on a positive signal.
_NETWORK_FUSE_SUBTYPES = frozenset(
    {
        "sshfs",
        "s3fs",
        "rclone",
        "glusterfs",
        "davfs2",
        "cephfs",
        "curlftpfs",
        "gcsfuse",
        "blobfuse",
        "blobfuse2",
        "onedriver",
        "google-drive-ocamlfuse",
    }
)

# Windows file-attribute bits marking non-resident (cloud placeholder / HSM stub)
# content. Mirrors core.discovery's mask -- kept local so this module stays
# import-free of the discovery machinery; these are stable OS constants.
_FILE_ATTRIBUTE_OFFLINE = 0x00001000
_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_OFFLINE_ATTR_MASK = (
    _FILE_ATTRIBUTE_OFFLINE
    | _FILE_ATTRIBUTE_RECALL_ON_OPEN
    | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)


def unsafe_cache_dir_reason(path) -> Optional[str]:
    """One-line reason the mmap file cache should not use *path*, or None if it
    is local disk (or undeterminable, treated as local).

    Network storage is checked first, then cloud-sync; the returned string is
    meant to be dropped straight into a launcher warning.
    """
    net = network_filesystem_type(path)
    if net:
        return f"a network filesystem ({net})"
    cloud = cloud_sync_hint(path)
    if cloud:
        return f"cloud-synced storage ({cloud})"
    return None


def network_filesystem_type(path) -> Optional[str]:
    """Return the network filesystem type backing *path* (e.g. ``"nfs4"``), or
    None if it is local storage or cannot be determined. Never raises."""
    try:
        probe = _nearest_existing(Path(path))
        if os.name == "nt":
            return _windows_network_type(probe)
        if sys.platform.startswith("linux"):
            return _linux_network_type(probe)
        if sys.platform == "darwin":
            return _darwin_network_type(probe)
        return None
    except Exception:
        return None


def cloud_sync_hint(path) -> Optional[str]:
    """Best-effort label if *path* lives in a cloud-synced / Files-On-Demand
    folder (OneDrive/iCloud/Dropbox/...), else None. Metadata-only; never raises.
    """
    try:
        original = Path(path)
        # The Windows placeholder attribute needs a real path to stat; walk up to
        # the nearest existing ancestor (the cloud folder exists even if the leaf
        # cache dir doesn't yet).
        if os.name == "nt":
            attrs = getattr(_nearest_existing(original).stat(), "st_file_attributes", 0)
            if attrs and (attrs & _OFFLINE_ATTR_MASK):
                return "Windows cloud placeholder"
        # The path-component heuristic reads the *configured* path, so a not-yet-
        # created dir under a cloud root is still flagged (its marker component is
        # intact even though the leaf doesn't exist).
        return _cloud_path_hint(original)
    except Exception:
        return None


# ------------------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------------------


def _nearest_existing(path: Path) -> Path:
    """The nearest ancestor of *path* that exists (the cache dir may not yet)."""
    p = path
    while not p.exists():
        parent = p.parent
        if parent == p:  # reached the filesystem root
            break
        p = parent
    return p


def _classify_fstype(fstype: Optional[str]) -> Optional[str]:
    """Return the network fstype (lowercased) if *fstype* is a network one."""
    if not fstype:
        return None
    low = fstype.lower()
    if low in _NETWORK_FSTYPES:
        return low
    if low.startswith("fuse.") and low[len("fuse.") :] in _NETWORK_FUSE_SUBTYPES:
        return low
    return None


def _is_unc(path_str: str) -> bool:
    """A Windows UNC path (``\\\\server\\share`` or ``//server/share``) is network."""
    return path_str.startswith(("\\\\", "//"))


def _windows_network_type(path: Path) -> Optional[str]:
    import ctypes

    s = str(path)
    if _is_unc(s):
        return "unc"
    drive = os.path.splitdrive(os.path.abspath(s))[0]
    if not drive:
        return None
    _DRIVE_REMOTE = 4
    if ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == _DRIVE_REMOTE:
        return "remote drive"
    return None


def _linux_network_type(path: Path) -> Optional[str]:
    """Find *path*'s mount in /proc/self/mountinfo and classify its fstype.

    Picks the mount whose mount point is the longest prefix of the resolved
    path, so a network export mounted below a local root is detected.
    """
    target = os.path.realpath(path)
    best_len = -1
    best_type: Optional[str] = None
    with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.split(" ")
            # "<id> <pid> <maj:min> <root> <mount point> <opts> [<opt>...] - <fstype> ..."
            if len(fields) < 5:
                continue
            try:
                sep = fields.index("-")
            except ValueError:
                continue
            if sep + 1 >= len(fields):
                continue
            mount_point = _unescape_mountinfo(fields[4])
            fstype = fields[sep + 1]
            if _path_under(target, mount_point) and len(mount_point) > best_len:
                best_len = len(mount_point)
                best_type = fstype
    return _classify_fstype(best_type)


def _darwin_network_type(path: Path) -> Optional[str]:
    """macOS: read ``f_fstypename`` from ``statfs(2)`` via ctypes."""
    import ctypes

    _MFSTYPENAMELEN = 16
    _MAXPATHLEN = 1024

    class _Statfs(ctypes.Structure):
        # struct statfs (Darwin, _DARWIN_FEATURE_64_BIT_INODE): only the fixed
        # numeric header before f_fstypename has to line up byte-for-byte.
        _fields_ = [
            ("f_bsize", ctypes.c_uint32),
            ("f_iosize", ctypes.c_int32),
            ("f_blocks", ctypes.c_uint64),
            ("f_bfree", ctypes.c_uint64),
            ("f_bavail", ctypes.c_uint64),
            ("f_files", ctypes.c_uint64),
            ("f_ffree", ctypes.c_uint64),
            ("f_fsid", ctypes.c_int32 * 2),
            ("f_owner", ctypes.c_uint32),
            ("f_type", ctypes.c_uint32),
            ("f_flags", ctypes.c_uint32),
            ("f_fssubtype", ctypes.c_uint32),
            ("f_fstypename", ctypes.c_char * _MFSTYPENAMELEN),
            ("f_mntonname", ctypes.c_char * _MAXPATHLEN),
            ("f_mntfromname", ctypes.c_char * _MAXPATHLEN),
            ("f_flags_ext", ctypes.c_uint32),
            ("f_reserved", ctypes.c_uint32 * 7),
        ]

    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    buf = _Statfs()
    if libc.statfs(os.fsencode(str(path)), ctypes.byref(buf)) != 0:
        return None
    return _classify_fstype(buf.f_fstypename.decode("utf-8", "replace"))


def _path_under(target: str, mount_point: str) -> bool:
    """Whether absolute *target* lies within *mount_point*."""
    if mount_point == "/":
        return True
    mp = mount_point.rstrip("/")
    return target == mp or target.startswith(mp + "/")


def _unescape_mountinfo(field: str) -> str:
    """Decode mountinfo's octal escapes (space \\040, tab \\011, ..., backslash)."""
    if "\\" not in field:
        return field
    out = []
    i = 0
    n = len(field)
    while i < n:
        c = field[i]
        if c == "\\" and i + 3 < n + 1 and field[i + 1 : i + 4].isdigit():
            try:
                out.append(chr(int(field[i + 1 : i + 4], 8)))
                i += 4
                continue
            except ValueError:
                pass
        out.append(c)
        i += 1
    return "".join(out)


def _cloud_path_hint(path: Path) -> Optional[str]:
    """Path-component heuristic for known cloud-sync roots (all platforms)."""
    parts = [p.lower() for p in Path(os.path.abspath(path)).parts]
    joined = "/".join(parts)
    # macOS iCloud Drive and the modern CloudStorage container.
    if "com~apple~clouddocs" in joined or "mobile documents" in joined:
        return "iCloud Drive"
    if "library/cloudstorage" in joined:
        return "macOS CloudStorage folder"
    for part in parts:
        if part == "onedrive" or part.startswith(("onedrive -", "onedrive-")):
            return "OneDrive"
        if part == "dropbox":
            return "Dropbox"
        if part in ("google drive", "googledrive") or part.startswith("googledrive-"):
            return "Google Drive"
        if part in ("nextcloud", "owncloud"):
            return "Nextcloud/ownCloud"
        if part == "pclouddrive":
            return "pCloud"
    return None
