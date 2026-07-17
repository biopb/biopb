"""Inspect the kernel-namespace plugins — the "bring your own tool" surface (#92).

Two plugin sources feed the biopb-mcp agent kernel's namespace at bootstrap:

- ``*.py`` files in ``~/.config/biopb/kernel/`` (the low-friction path: drop a
  file; its top-level defs land in the namespace), and
- installed ``biopb_mcp.namespace`` entry-point packages (the distribution path).

This module is a **read-only inspector** for the control dashboard, mirroring
:mod:`biopb._algorithms`: it lists what is present **without ever importing or
executing it**. That static "what will load" view is deliberate — the *live* set
of names a running kernel actually bound depends on each file's top-level code and
each package's ``register()`` hook, which only executing it reveals, and the lean
control must not run user Python (invariant I2, and it would be a robustness /
security hole). A file's one-line summary comes from its module docstring parsed
with :mod:`ast` (parse, not exec); an entry point reports the distribution that
provides it, read from installed metadata.

Stdlib-only (``ast`` + ``importlib.metadata`` + ``pathlib``), like the other
shared core seams, so the control binds to it with no new dependency edge and the
bootstrap loader can name the same entry-point group without importing this side.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The entry-point group biopb-mcp loads into the kernel namespace. Defined here
# (not only in biopb-mcp) so the control's inspector and the kernel-side loader
# name the same group without importing each other.
NAMESPACE_ENTRY_POINT_GROUP = "biopb_mcp.namespace"


def _summary_from_source(text: str) -> str:
    """First line of a Python file's module docstring, or ``""``.

    Parsed with ``ast`` — never executed — so *listing* a plugin can't run it.
    """
    try:
        doc = ast.get_docstring(ast.parse(text))
    except (SyntaxError, ValueError):
        return ""
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def startup_files(plugin_dir: Optional[Path] = None) -> list[dict]:
    """List the ``*.py`` startup files, sorted by name.

    Each row is ``{name, summary}`` — the filename and its module-docstring
    one-liner (``""`` if none / unparseable). A missing dir reads as an empty list;
    ``_``-prefixed files (dunder / private helpers) are skipped. Never raises.
    """
    if plugin_dir is None:
        from biopb._locations import mcp_plugin_dir

        plugin_dir = mcp_plugin_dir()
    try:
        paths = sorted(plugin_dir.glob("*.py"))
    except OSError:
        return []
    rows: list[dict] = []
    for path in paths:
        if path.name.startswith("_"):
            continue
        try:
            summary = _summary_from_source(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = ""
        rows.append({"name": path.name, "summary": summary})
    return rows


def entry_point_plugins() -> list[dict]:
    """List installed ``biopb_mcp.namespace`` entry points, sorted by name.

    Each row is ``{name, dist}`` — the entry-point name and the distribution
    (``name version``) providing it, read from installed metadata **without
    importing the module**. Never raises.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return []
    try:
        eps = entry_points(group=NAMESPACE_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - metadata read, never break the dashboard
        logger.debug("kernel-plugin entry-point read failed", exc_info=True)
        return []
    rows: list[dict] = []
    for ep in eps:
        dist = ""
        d = getattr(ep, "dist", None)
        if d is not None:
            name = getattr(d, "name", "") or ""
            ver = getattr(d, "version", "") or ""
            dist = f"{name} {ver}".strip()
        rows.append({"name": ep.name, "dist": dist})
    return sorted(rows, key=lambda r: r["name"])


def summary(plugin_dir: Optional[Path] = None) -> dict:
    """The dashboard payload: ``{dir, files, entry_points}``.

    ``dir`` is the startup-file location as a string (for display); ``files`` and
    ``entry_points`` are the two lists above. Read-only, never raises.
    """
    if plugin_dir is None:
        from biopb._locations import mcp_plugin_dir

        plugin_dir = mcp_plugin_dir()
    return {
        "dir": str(plugin_dir),
        "files": startup_files(plugin_dir),
        "entry_points": entry_point_plugins(),
    }
