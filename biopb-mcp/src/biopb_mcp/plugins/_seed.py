"""Seed biopb-mcp's built-in example kernel plugins into the user's plugin dir (#92).

The installer runs this (``biopb-mcp-seed-plugins``) so the bundled example —
``rolling_ball.py`` plus the namespace ``__init__.py`` doc — lands in
``~/.config/biopb/kernel/``. Delivering the example as a **file there**, rather
than only as an installed module, makes it visible/editable to the user and loads
it through the startup-file path, which is robust to the kernel interpreter's
entry-point metadata view (the ``python3`` kernelspec need not be the biopb-mcp
tool env).

Seeding is **idempotent and never clobbers**: an existing file is left untouched
(the user may have edited it), mirroring how the installer preserves an existing
``mcp-config.json``. Stdlib-only (``shutil`` + ``pathlib``) and independent of the
heavy MCP/kernel stack, so it stays a cheap console entry.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Files bundled in this package that the installer seeds into the kernel dir.
# __init__.py documents the dir (the loader skips it — leading underscore);
# rolling_ball.py is the worked example plugin.
SEED_FILES = ("__init__.py", "rolling_ball.py")


def seed_kernel_plugins(dest: Path | str | None = None) -> list[tuple[str, str]]:
    """Copy the bundled example plugins into *dest* (default the kernel plugin dir).

    Returns ``[(filename, action)]`` where action is ``"created"`` (copied now) or
    ``"exists"`` (left as the user has it). Creates the directory if needed.
    """
    if dest is None:
        from biopb._locations import mcp_plugin_dir

        dest = mcp_plugin_dir()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    results: list[tuple[str, str]] = []
    for name in SEED_FILES:
        target = dest / name
        if target.exists():
            results.append((name, "exists"))
            continue
        shutil.copyfile(here / name, target)
        results.append((name, "created"))
    return results


def _cli(argv=None) -> int:
    """Console entry (``biopb-mcp-seed-plugins``): seed and report, never fail."""
    try:
        for name, action in seed_kernel_plugins():
            print(f"{action}\t{name}")
    except Exception as exc:  # noqa: BLE001 - best-effort installer step
        print(f"kernel-plugin seeding skipped: {exc}", file=sys.stderr)
        return 0
    return 0
