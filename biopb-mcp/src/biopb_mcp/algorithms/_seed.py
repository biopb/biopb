"""Seed biopb-mcp's bundled algorithm-plane ops into the algorithm registry.

The installer runs this (``biopb-mcp-seed-algorithms``) so the bundled ops land
in ``~/.config/biopb/algorithms/`` as script entries, alongside anything else the
registry holds (a ``cellpose.json`` url entry, a hand-written op file).

Seeding is **idempotent and never overwrites**: an existing file is left
untouched (the user may have edited it), mirroring how the installer preserves
an existing ``mcp-config.json``. Stdlib-only (``shutil`` + ``pathlib``) and
independent of the heavy MCP/kernel stack, so it stays a cheap console entry.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Files bundled in this package that the installer seeds into the algorithm
# registry. __init__.py documents the package (skipped -- leading underscore);
# segmentation_qc.py is IoU matching and F1 for instance segmentations;
# image_resolution.py is FRC and decorrelation analysis.
SEED_FILES = (
    "segmentation_qc.py",
    "image_resolution.py",
)


def seed_algorithms(dest: Path | str | None = None) -> list[tuple[str, str]]:
    """Copy the bundled ops into *dest* (default the algorithm registry dir).

    Returns ``[(filename, action)]`` where action is ``"created"`` (copied now) or
    ``"exists"`` (left as the user has it). Creates the directory if needed.
    """
    if dest is None:
        from biopb._config.locations import algorithms_dir

        dest = algorithms_dir()
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
    """Console entry (``biopb-mcp-seed-algorithms``): seed and report, never fail."""
    try:
        for name, action in seed_algorithms():
            print(f"{action}\t{name}")
    except Exception as exc:  # noqa: BLE001 - best-effort installer step
        print(f"algorithm seeding skipped: {exc}", file=sys.stderr)
        return 0
    return 0
