"""Owned-child process-lifecycle primitives (the "Pattern O" toolkit).

An *owned child* is a subprocess a live parent spawns, holds by its OS handle
(no pid file -- the ``Popen`` object is the identity), and must reap. Two owners
share this pattern: the tensor server (held by the control supervisor) and the
biopb-mcp session child (held by the stdio shim, which in turn holds the kernel).
Because ``biopb-control`` and ``biopb-mcp`` cannot import each other, the OS-level
mechanics they share live here in the core SDK:

* :mod:`biopb._lifecycle.winjob` -- Windows Job Object: kill-on-close bind (the
  child dies with its parent for any reason) plus a from-outside tree-kill.
* :mod:`biopb._lifecycle.deathwatch` -- the child-side parent-death pipe watcher:
  self-terminate if the parent dies uncatchably (the POSIX counterpart to the
  Job Object).

The *keepalive* (restart-on-crash) loop is deliberately **not** here -- it is the
supervisor's concern, layered on top of an owned child, and only the tensor
server wants it.
"""

from . import deathwatch, winjob

__all__ = ["deathwatch", "winjob"]
