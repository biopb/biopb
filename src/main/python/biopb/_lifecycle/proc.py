"""Low-level process liveness/identity primitives.

Two questions the daemon managers ask about a PID:

* **liveness** -- is the process still alive? (:func:`is_process_running`)
* **identity** -- is it the *same* process we launched, or an unrelated one that
  inherited a reused PID? (:func:`process_create_time`, a per-process create-time
  token: a recycled PID gets a different creation time.)

Kept dependency-free and shared by both ``biopb.cli`` and
``biopb_mcp.mcp.__main__`` so the PID file one writes is parsed/identified
identically by the other -- if the create-time computation drifted between the
two, identity checks would silently fail.
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

# Enough access to query liveness (exit code) and creation time.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


@contextlib.contextmanager
def _win_process_handle(pid: int):
    """Yield an OpenProcess handle for `pid`, closed on exit; None if it can't be
    opened (no such process, or access denied)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        yield None
        return
    try:
        yield handle
    finally:
        kernel32.CloseHandle(handle)


def is_process_running(pid: int) -> bool:
    """Whether the process with `pid` is alive."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # On Windows os.kill(pid, 0) is NOT a liveness probe: signal value 0 is
        # signal.CTRL_C_EVENT, so os.kill would call GenerateConsoleCtrlEvent and
        # deliver a real Ctrl+C to the console process group (killing the daemon
        # we just started, mid-import). Query the process handle instead.
        import ctypes
        from ctypes import wintypes

        with _win_process_handle(pid) as handle:
            if not handle:
                return False
            kernel32 = ctypes.windll.kernel32
            kernel32.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == _STILL_ACTIVE
    try:
        os.kill(pid, 0)  # Signal 0 just checks if process exists
        return True
    except OSError:
        return False


def process_create_time(pid: int) -> Optional[int]:
    """A per-process-unique creation-time token for `pid`, or None if it can't be
    determined.

    Comparing a token recorded at launch with the live value detects PID reuse:
    a recycled PID gets a different creation time. This is what makes the PID file
    an *identity* check, not just a liveness check -- crucial on Windows, which
    hard-kills the daemon at logout (so the PID file is never cleaned) and reuses
    PIDs aggressively, letting a stale PID name an unrelated process that `stop`
    would otherwise TerminateProcess.

    Returns None when unavailable -- the process is gone, or the platform has no
    cheap source (e.g. macOS) -- and callers then degrade to a liveness-only check
    (the pre-fix behavior), never a false "stopped".
    """
    if pid <= 0:
        return None
    if sys.platform == "win32":
        # GetProcessTimes -> creation FILETIME (100ns ticks since 1601), unique
        # enough per (PID, lifetime). PROCESS_QUERY_LIMITED_INFORMATION suffices.
        import ctypes
        from ctypes import wintypes

        class _FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        with _win_process_handle(pid) as handle:
            if not handle:
                return None
            kernel32 = ctypes.windll.kernel32
            kernel32.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_FILETIME),
                ctypes.POINTER(_FILETIME),
                ctypes.POINTER(_FILETIME),
                ctypes.POINTER(_FILETIME),
            ]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            creation, exit_t, kernel_t, user_t = (
                _FILETIME(),
                _FILETIME(),
                _FILETIME(),
                _FILETIME(),
            )
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_t),
                ctypes.byref(kernel_t),
                ctypes.byref(user_t),
            ):
                return None
            return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    if sys.platform.startswith("linux"):
        # /proc/<pid>/stat field 22 = starttime (clock ticks since boot). comm
        # (field 2) is parenthesized and may itself contain spaces/parens, so
        # parse the fields after the LAST ')': index 19 of those is starttime.
        try:
            data = Path(f"/proc/{pid}/stat").read_bytes()
            fields = data[data.rfind(b")") + 2 :].split()
            return int(fields[19])
        except (OSError, ValueError, IndexError):
            return None
    return None
