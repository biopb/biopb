"""Local data-plane credential handoff via an owner-only file in the state dir.

The control writes the resolved token; local clients (napari widget, biopb-mcp)
read it. The HTTP API never returns the token (loopback is reachable by every uid
on the box), so the boundary is filesystem permissions. POSIX uses ``0o600``;
Windows sets a protected DACL granting only the current user's SID
(:func:`_harden_windows`). Hardening is best-effort and logged at debug.
Stdlib-only, so both ``biopb-control`` and ``biopb-mcp`` can use it.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from .._config.locations import state_dir

logger = logging.getLogger(__name__)

# The data-plane credential file, in the state tree beside the pid / sentinel files.
_CREDENTIAL_NAME = "tensor-server.token"


def credential_file(name: str = _CREDENTIAL_NAME) -> Path:
    """Path to a local credential file (default: the data plane's).

    Resolved at call time so tests can repoint the state dir. *name* selects
    another credential stored the same way (e.g. the chat client's provider key).
    """
    return state_dir() / name


def _harden_posix(path: Path) -> None:
    """Restrict *path* to the owner (``0o600``)."""
    os.chmod(path, 0o600)


def _harden_windows(path: Path) -> bool:
    """Restrict *path*'s DACL to the current user, protected against inheritance.

    The DACL grants full access only to the current user's SID and is protected
    from inheritance. Returns True on success; False leaves the state dir's
    inherited ACL in effect.

    Every ctypes call needs explicit ``argtypes``/``restype``: on 64-bit Windows
    unannotated handles/pointers are truncated to ``c_int``.
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
        # The first pointer-sized field of TOKEN_USER is the PSID.
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
    # Protected DACL (P) with one File-All ACE for the current user; owner/group untouched.
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

    Failures are logged at debug and swallowed: a missing credential is worse
    than one protected only by the state dir's ACL.
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


def write_credential(token: str, name: str = _CREDENTIAL_NAME) -> Path:
    """Write *token* to the owner-only credential file *name*; return its path.

    Atomic, and hardened *before* the replace so the token is never readable at
    its final name.
    """
    path = credential_file(name)
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


def read_credential(name: str = _CREDENTIAL_NAME) -> str | None:
    """Read a credential from the owner-only file *name*, or ``None``.

    ``None`` if the file is absent, unreadable, or empty; never raises.
    """
    path = credential_file(name)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    return token or None


def remove_credential(name: str = _CREDENTIAL_NAME) -> None:
    """Remove the credential file *name* if present (best-effort)."""
    path = credential_file(name)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("could not remove credential %s: %s", path, exc)
