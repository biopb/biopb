"""Launcher-owned Xvfb: a virtual display for display-less Linux hosts (#90).

When no ``$DISPLAY``/``$WAYLAND_DISPLAY`` is present the launcher spawns one
``Xvfb`` as its own child and injects the allocated display into the kernel
env, so the kernel runs a *real* napari viewer (and ``take_screenshot`` works)
with no human-visible window. There is no compute-only fallback: if the binary
is missing the launcher fails fast with the install hint instead of silently
degrading to a viewer-less kernel.

Why not wrap the kernel command with ``xvfb-run``: that puts Xvfb inside the
kernel's process group, and ``interrupt_kernel`` signals that whole group
(jupyter_client prefers ``killpg``) — X servers exit on SIGINT, so every
interrupt would kill the display out from under the viewer and crash the
kernel. A launcher-owned sibling never sees kernel-group signals, and one
server survives any number of kernel restarts.

Lifetime: the launcher stops it via its normal ``_shutdown``/atexit paths; a
``PR_SET_PDEATHSIG`` set in the child is the backstop for an uncatchable
launcher death (SIGKILL/OOM) — Linux-only, which is exactly this module's
scope (macOS/Windows always have a window server; see ``_has_display``).

Known trade-off: the display runs without an xauth cookie (generating one
needs the ``xauth`` binary and adds a temp-file dance), so any process of any
uid on this host may connect to it — same-host-users trust, consistent with
the localhost security model. ``-nolisten tcp`` keeps it off the network.
"""

import logging
import os
import select
import shutil
import signal
import subprocess
import time

logger = logging.getLogger(__name__)

XVFB_BINARY = "Xvfb"

# Actionable message for the fail-fast path (also raised by a failed start).
INSTALL_HINT = (
    "no display is available and Xvfb was not found. Install it to run the "
    "napari viewer on a virtual display (Debian/Ubuntu: `sudo apt install "
    "xvfb`; Fedora: `sudo dnf install xorg-x11-server-Xvfb`), or start an "
    "X/Wayland session."
)

# 24-bit depth is required (napari/vispy GL); xvfb-run's 8-bit default is not.
_DEFAULT_SCREEN = "1280x1024x24"

# How long to wait for Xvfb to write its display number. Normally ~100 ms;
# generous because a loaded box paying a cold ELF/driver load should not be
# misread as a failure.
_READY_TIMEOUT = 30.0


def available() -> bool:
    """Whether the Xvfb binary is on PATH."""
    return shutil.which(XVFB_BINARY) is not None


def _set_pdeathsig():
    """(preexec_fn) Ask Linux to SIGTERM this process when its parent dies.

    Backstop only — the launcher's normal teardown paths call stop(). Survives
    the exec into Xvfb; best-effort because prctl is Linux-specific and this
    fallback never runs elsewhere.
    """
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL(None, use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0
        )
    except Exception:
        pass


def start(screen: str = _DEFAULT_SCREEN, timeout: float = _READY_TIMEOUT):
    """Spawn Xvfb on a free display and return ``(proc, display)``.

    ``-displayfd`` makes Xvfb pick a free display number itself and write it
    back over an inherited pipe *once the server is accepting connections* —
    so a successful return means both "no display-number race" and "ready to
    use". ``display`` is the ``":N"`` string for the kernel's ``DISPLAY``.

    Raises ``RuntimeError`` (with the install hint when the binary is absent)
    on any failure; the process is reaped before raising.
    """
    if not available():
        raise RuntimeError(INSTALL_HINT)

    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            [
                XVFB_BINARY,
                "-displayfd",
                str(write_fd),
                "-screen",
                "0",
                screen,
                "-nolisten",
                "tcp",
            ],
            pass_fds=(write_fd,),
            preexec_fn=_set_pdeathsig,
            # stdout/stderr inherit the launcher's fds (the session log), like
            # the kernel's — Xvfb is quiet unless something is actually wrong.
        )
    except OSError as exc:
        os.close(read_fd)
        raise RuntimeError(f"failed to spawn {XVFB_BINARY}: {exc}") from exc
    finally:
        # The child holds its own copy; close ours so its exit turns into EOF
        # on the read end instead of a hang.
        os.close(write_fd)

    try:
        display_num = _read_display_number(read_fd, proc, timeout)
    except Exception:
        stop(proc)
        raise
    finally:
        os.close(read_fd)

    display = f":{display_num}"
    logger.info("Xvfb serving virtual display %s (pid %d)", display, proc.pid)
    return proc, display


def _read_display_number(read_fd, proc, timeout: float) -> int:
    """Read the newline-terminated display number Xvfb writes to -displayfd."""
    deadline = time.monotonic() + timeout
    buf = b""
    while not buf.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"{XVFB_BINARY} did not report a display within {timeout:.0f}s"
            )
        ready, _, _ = select.select([read_fd], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(read_fd, 16)
        if not chunk:  # EOF: Xvfb exited without reporting
            code = proc.poll()
            raise RuntimeError(
                f"{XVFB_BINARY} exited during startup"
                + (f" (exit code {code})" if code is not None else "")
            )
        buf += chunk
    try:
        return int(buf.strip())
    except ValueError:
        raise RuntimeError(
            f"{XVFB_BINARY} reported an unparsable display: {buf!r}"
        ) from None


def stop(proc, timeout: float = 5.0):
    """Terminate the Xvfb child, escalating to SIGKILL. Idempotent."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    except OSError:
        logger.debug("stopping Xvfb failed", exc_info=True)
