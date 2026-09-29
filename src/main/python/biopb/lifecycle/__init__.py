"""Process-lifecycle primitives shared across biopb's subprocesses.

Public: biopb-control, biopb-mcp, biopb-tensor-server, and biopb-image-base
all depend on this directly, and biopb-control/biopb-mcp can't import each
other -- so this stays in the dependency-light core SDK where all four can
reach it, without hiding that dependency behind an underscore.

Two lifecycle patterns live here, plus the OS-level mechanics they share.

**Owned child** (the "Pattern O" toolkit) -- a subprocess a live parent spawns,
holds by its OS handle (no pid file -- the ``Popen`` object is the identity), and
must reap. Two owners share it: the tensor server (held by the control
supervisor) and the biopb-mcp session child (held by the stdio shim, which in
turn holds the kernel).

* :mod:`biopb.lifecycle.owned_child` -- the ``OwnedChild`` handle itself.
* :mod:`biopb.lifecycle.winjob` -- Windows Job Object: kill-on-close bind (the
  child dies with its parent for any reason) plus a from-outside tree-kill.
* :mod:`biopb.lifecycle.deathwatch` -- the child-side parent-death pipe watcher:
  self-terminate if the parent dies uncatchably (the POSIX counterpart to the
  Job Object).

**Detached daemon** -- a background process that *outlives* the command that
spawned it, is found again by a pidfile rather than a handle, and is signalled to
stop. The one such daemon is the control plane (``biopb control start``).

* :mod:`biopb.lifecycle.daemon` -- pidfile identity across a reused pid,
  graceful stop, and console-detach ``Popen`` kwargs.

Shared by both patterns:

* :mod:`biopb.lifecycle.proc` -- process liveness + create-time identity, the
  primitive that lets a pidfile owner tell its own child from an unrelated
  process that later inherited a reused PID.
* :mod:`biopb.lifecycle.file_lock` -- a cross-process advisory lock, in two
  scopes: ``file_lock`` for a block that must not race another owner (e.g.
  ``control start``), and ``ExclusiveFileLock`` for single-ownership held across
  a process lifetime (the tensor server's cache directory). Both put exclusion
  on an open descriptor, so a holder's death releases it with nothing left to
  reap -- which is why neither needs the pid-identity logic below.

The *keepalive* (restart-on-crash) loop is deliberately **not** here -- it is the
supervisor's concern, layered on top of an owned child, and only the tensor
server wants it.
"""

from . import daemon, deathwatch, file_lock, proc, winjob
from .owned_child import OwnedChild, open_child_log

__all__ = [
    "OwnedChild",
    "daemon",
    "deathwatch",
    "file_lock",
    "open_child_log",
    "proc",
    "winjob",
]
