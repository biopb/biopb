"""Local data-plane credential handoff — shared, stdlib-only.

The control plane is the single source of truth for the data plane (#413), but
until now it handed clients the plane's *endpoint* without a *credential*: a
local client (the napari widget, biopb-mcp) rediscovered the token from its own
environment (``BIOPB_TENSOR_TOKEN``). That breaks the moment the control has a
token but the client never inherited it — exactly the mode #468 introduced,
where an agent spawns biopb-mcp over stdio with none of the control's env. And
the endpoint cannot simply *return* the token: it is loopback-reachable by every
uid on the box, so an unauthenticated reply would be a token oracle (#470).

This module moves credential distribution off the HTTP API and onto the
filesystem, the standard local-daemon handoff (Jupyter's runtime JSON, Docker):
the control writes the resolved data-plane token to a single file in the user's
own state dir, restricted to the owner, and local clients read it there. The
token never crosses the very channel it protects, and the boundary becomes
filesystem permissions — other uids cannot read a ``0600`` file, which is what
"defense in depth on a shared host" means. It does not defend against a
same-uid process, but neither does an env var (which additionally leaks via
``/proc/<pid>/environ``, ``ps e``, and every inherited child).

Deliberately stdlib-only (``os`` + ``pathlib`` + ``ctypes`` on Windows) so both
``biopb-control`` (writer) and ``biopb-mcp`` (reader) — which already depend on
core ``biopb`` but cannot import each other — bind to it without a new
dependency edge, alongside ``_locations`` / ``_endpoints``.

**The ``0600`` equivalent per platform.** On POSIX the file is created and
``chmod``-ed to ``0o600`` (owner read/write, nobody else). Windows has no POSIX
mode bits — ``os.chmod`` there only toggles the read-only attribute and cannot
express "owner only" — so :func:`_harden_windows` instead sets the file's DACL
to grant full access to *only* the current user's SID, with inheritance
disabled (``SE_DACL_PROTECTED``), the faithful analogue of ``0600``. That
explicit DACL is belt-and-suspenders over the state dir already living under
``%USERPROFILE%``, whose default ACL restricts it to the owner — but the issue
(#470) asked for an explicit check rather than trusting inheritance. All
hardening is best-effort and logged at debug: a failure degrades to the
inherited boundary rather than refusing to write the credential.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from ._locations import state_dir

logger = logging.getLogger(__name__)

# The data-plane credential the control writes and local clients read. State
# tree (per-machine, regenerable), beside the pid / sentinel files. The ``.token``
# suffix keeps it distinct from ``control.pid`` (whose own "token" field is an
# unrelated PID-reuse guard, not a credential).
_CREDENTIAL_NAME = "tensor-server.token"


def credential_file() -> Path:
    """Path to the local data-plane credential (``state/biopb/tensor-server.token``).

    Resolved at call time (not cached) so a test that repoints ``Path.home()`` /
    ``$XDG_STATE_HOME`` gets an isolated location.
    """
    return state_dir() / _CREDENTIAL_NAME


def _harden_posix(path: Path) -> None:
    """Restrict *path* to the owner (``0o600``)."""
    os.chmod(path, 0o600)


def _harden_windows(path: Path) -> bool:
    """Restrict *path*'s DACL to the current user, protected against inheritance.

    The Windows analogue of ``chmod 0600``: build a security descriptor whose
    DACL grants full access (``FA``) to only the current process user's SID and
    carries ``SE_DACL_PROTECTED`` (the ``P`` flag), so no ACE inherited from the
    parent directory can widen it. Returns True on success; a False return
    leaves the file protected only by the state dir's inherited ACL (owner-only
    by default under ``%USERPROFILE%``).

    Raw ``ctypes`` with explicit ``argtypes``/``restype`` on every call — on
    64-bit Windows an unannotated handle/pointer is truncated to ``c_int``,
    corrupting the SID and SD pointers — mirroring ``biopb._lifecycle`` proc/job.
    """
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.argtypes = [wintypes.LPVOID]
    kernel32.LocalFree.restype = wintypes.LPVOID

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,  # TOKEN_INFORMATION_CLASS
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID,  # PSID
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,  # StringSDRevision
        ctypes.POINTER(wintypes.LPVOID),  # PSECURITY_DESCRIPTOR*
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
        wintypes.BOOL
    )
    advapi32.SetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,  # SECURITY_INFORMATION
        wintypes.LPVOID,  # PSECURITY_DESCRIPTOR
    ]
    advapi32.SetFileSecurityW.restype = wintypes.BOOL

    _TOKEN_QUERY = 0x0008
    _TOKEN_USER = 1  # TOKEN_INFORMATION_CLASS.TokenUser
    _SDDL_REVISION_1 = 1
    _DACL_SECURITY_INFORMATION = 0x00000004

    # --- current user SID, as a string -------------------------------------
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
    ):
        return False
    try:
        size = wintypes.DWORD(0)
        # First call sizes the buffer (fails with ERROR_INSUFFICIENT_BUFFER).
        advapi32.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(size))
        if size.value == 0:
            return False
        buf = (ctypes.c_byte * size.value)()
        if not advapi32.GetTokenInformation(
            token, _TOKEN_USER, buf, size, ctypes.byref(size)
        ):
            return False
        # TOKEN_USER := SID_AND_ATTRIBUTES { PSID Sid; DWORD Attributes }, so the
        # first pointer-sized field of the buffer is the PSID (which points back
        # into the trailing bytes of the same buffer).
        sid = ctypes.cast(buf, ctypes.POINTER(wintypes.LPVOID))[0]
        str_sid = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(str_sid)):
            return False
        try:
            sid_text = str_sid.value
        finally:
            kernel32.LocalFree(str_sid)
    finally:
        kernel32.CloseHandle(token)

    if not sid_text:
        return False

    # --- SDDL -> security descriptor -> file DACL --------------------------
    # D:P(A;;FA;;;<sid>) — a protected (P: no inheritance) DACL with one ACE
    # granting File-All to the current user's SID. The owner/group are left
    # untouched (we set only DACL_SECURITY_INFORMATION).
    sddl = f"D:P(A;;FA;;;{sid_text})"
    sd = wintypes.LPVOID()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, _SDDL_REVISION_1, ctypes.byref(sd), None
    ):
        return False
    try:
        ok = advapi32.SetFileSecurityW(str(path), _DACL_SECURITY_INFORMATION, sd)
        return bool(ok)
    finally:
        kernel32.LocalFree(sd)


def _harden(path: Path) -> None:
    """Restrict *path* to the owner, cross-platform and best-effort.

    POSIX ``0o600`` / Windows owner-only protected DACL. Any failure is logged
    at debug and swallowed: on a shared host it weakens defense in depth but must
    never abort writing the credential (a missing credential is a worse failure
    than one protected by only the state dir's inherited ACL).
    """
    try:
        if sys.platform == "win32":
            if not _harden_windows(path):
                logger.debug(
                    "credential DACL hardening did not apply to %s "
                    "(falling back to the state dir's inherited ACL)",
                    path,
                )
        else:
            _harden_posix(path)
    except Exception as exc:  # noqa: BLE001 - hardening is best-effort (incl. a ctypes hiccup)
        logger.debug("credential hardening failed for %s: %s", path, exc)


def write_credential(token: str) -> Path:
    """Write *token* to the owner-only credential file; return its path.

    Atomic (sibling temp + ``os.replace`` on the same filesystem) so a concurrent
    reader never sees a half-written file, and hardened to owner-only *before* the
    replace so the token is never momentarily readable at its final name. On POSIX
    ``mkstemp`` already creates the temp file ``0600``; the explicit harden
    re-asserts it (and does the DACL work on Windows).
    """
    path = credential_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
        _harden(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def read_credential() -> str | None:
    """Read the data-plane token from the credential file, or ``None``.

    ``None`` on any of: the file is absent (the common tokenless-local case, or no
    control has written one), unreadable, or empty. Best-effort by design — the
    caller falls back to ``BIOPB_TENSOR_TOKEN`` / an actionable error, never
    raises.
    """
    path = credential_file()
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    return token or None


def remove_credential() -> None:
    """Remove the credential file if present (best-effort)."""
    path = credential_file()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("could not remove credential %s: %s", path, exc)
