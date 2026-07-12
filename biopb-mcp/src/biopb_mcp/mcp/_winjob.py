"""Windows Job Object: tie the child kernel's lifetime to the daemon's.

POSIX ties the kernel to the daemon two ways -- a process group the launcher
group-kills (``KernelHost._shutdown_current``) and a parent-death pipe that
self-terminates the kernel if the launcher dies uncatchably (``_deathwatch``).
Both go through ``os.killpg``, which does not exist on Windows, so on Windows a
force-killed or crashed daemon can orphan the kernel (biopb/biopb#403).

A Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` closes that gap: the
daemon creates the job, assigns the kernel (and thus every process the kernel
spawns) to it, and holds the only handle. When the daemon exits for *any* reason
-- including a ``TerminateProcess`` from ``biopb mcp stop`` that runs no daemon
code, and regardless of whether the kernel's GIL is wedged -- the OS closes that
handle and kills the whole job. ``TerminateJobObject`` additionally gives a
from-outside tree-kill on the graceful teardown path, the Windows counterpart to
``os.killpg(pgid, SIGKILL)``.

The module also exposes the *inverse* Windows primitive -- ``open_for_wait`` /
``wait_for_process``, a blocking wait on some *other* process's exit -- so the
shim can watch its stdio **client** die and reap the session it owns (the
counterpart, in the other direction, to the kernel's parent-death pipe). A
process handle names the process object, not its (reusable) pid, so the wait is
immune to pid reuse.

Every function is Windows-only and best-effort: it returns ``None`` / ``False``
and logs at debug on any failure, so a ctypes/OS hiccup degrades to the
pre-#403 behavior instead of breaking kernel bring-up.
"""

import ctypes
import logging
import os

logger = logging.getLogger(__name__)

# JOBOBJECTINFOCLASS.JobObjectExtendedLimitInformation
_JOBOBJECTINFOCLASS_EXTENDED_LIMIT = 9
# JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
# OpenProcess access rights AssignProcessToJobObject requires
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
# OpenProcess access right WaitForSingleObject requires, plus the infinite wait
# and its "the object was signalled" (here: the process exited) return value.
_SYNCHRONIZE = 0x00100000
_INFINITE = 0xFFFFFFFF
_WAIT_OBJECT_0 = 0x0


if os.name == "nt":
    from ctypes import wintypes

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),  # ULONG_PTR
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _kernel32():
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        return k32


_kernel32 = _kernel32() if os.name == "nt" else None


def create_kill_on_close_job():
    """Create a kill-on-close Job Object; return an opaque handle or ``None``.

    While the daemon holds the returned handle, the job's member processes stay
    alive; when the last handle to the job closes (the daemon exits, however
    abruptly) the OS terminates every member. Kept open for the daemon's whole
    life and closed only on terminal shutdown.
    """
    if os.name != "nt":
        return None
    try:
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            handle,
            _JOBOBJECTINFOCLASS_EXTENDED_LIMIT,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            err = ctypes.get_last_error()
            _kernel32.CloseHandle(handle)
            raise ctypes.WinError(err)
        return handle
    except Exception:
        logger.debug("CreateJobObject failed; kernel not job-tied", exc_info=True)
        return None


def assign_process(job, pid):
    """Add process ``pid`` (and its future descendants) to ``job``.

    Best-effort: a failure (e.g. the process already died, or is in a
    non-nestable job) just leaves the kernel outside the job, i.e. the pre-#403
    reap behavior. On Windows 8+ a process already in a job is nested, so a
    daemon that is itself jobbed still works.
    """
    if os.name != "nt" or not job or not pid:
        return False
    try:
        proc = _kernel32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, int(pid)
        )
        if not proc:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not _kernel32.AssignProcessToJobObject(job, proc):
                raise ctypes.WinError(ctypes.get_last_error())
            return True
        finally:
            _kernel32.CloseHandle(proc)
    except Exception:
        logger.debug("AssignProcessToJobObject failed (pid=%s)", pid, exc_info=True)
        return False


def terminate_job(job, exit_code=1):
    """Kill every process in ``job`` now (the from-outside tree-kill).

    Leaves the job object usable, so a restart can reassign a fresh kernel to
    the same handle.
    """
    if os.name != "nt" or not job:
        return
    try:
        if not _kernel32.TerminateJobObject(job, exit_code):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        logger.debug("TerminateJobObject failed", exc_info=True)


def close_job(job):
    """Close the job handle (fires kill-on-close for any remaining members)."""
    if os.name != "nt" or not job:
        return
    try:
        if not _kernel32.CloseHandle(job):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        logger.debug("CloseHandle(job) failed", exc_info=True)


def open_for_wait(pid):
    """Open a SYNCHRONIZE-only handle to ``pid`` for :func:`wait_for_process`.

    Returns an opaque handle -- which names the *process object*, so the later
    wait is immune to pid reuse -- or ``None`` if the process cannot be opened
    (already gone, or access denied). The caller must pass the handle to
    :func:`wait_for_process` (which closes it) exactly once. Windows-only.
    """
    if os.name != "nt" or not pid or pid <= 0:
        return None
    try:
        handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, int(pid))
        return handle or None
    except Exception:
        logger.debug("OpenProcess(SYNCHRONIZE) failed (pid=%s)", pid, exc_info=True)
        return None


def wait_for_process(handle):
    """Block until the process behind ``handle`` exits; then close ``handle``.

    Returns ``True`` only when the exit was actually observed. Any wait error
    (or a non-signalled return) yields ``False`` so the caller never treats an
    uncertain result as a definite death. Windows-only.
    """
    if os.name != "nt" or not handle:
        return False
    try:
        return _kernel32.WaitForSingleObject(handle, _INFINITE) == _WAIT_OBJECT_0
    except Exception:
        logger.debug("WaitForSingleObject failed", exc_info=True)
        return False
    finally:
        try:
            _kernel32.CloseHandle(handle)
        except Exception:
            logger.debug("CloseHandle(wait handle) failed", exc_info=True)
