"""What the kernel-plugin loader actually loaded — the one fact `requires:` needs.

A skill declares what it touches (``viewer``, ``tensor``, ``dask``, ``ops:<kind>``,
``plugin:<name>``, ``pkg:<name>``). The agent resolves that list itself, against
``server_status`` and — for a ``pkg:`` token — an import: five of the six are
already in front of it, more accurately than a helper could bucket them (a closed
napari window, the scheduler behind ``da``, the real ImportError).

``plugin:<name>`` is the one that is **not** derivable, which is why this module
exists:

* The loader is **fail-open per unit**: a file that raises on ``exec``, or loses a
  name to the reserved-name guard, is on disk and *not* in the namespace. So
  scanning ``~/.config/biopb/kernel/`` answers "file present", not "plugin
  loaded" — and only the loader knows which of the two happened.
* The namespace can't answer either: a plugin *file* contributes its top-level
  function names, not its own name, so ``dir()`` never shows ``segmentation_qc``.

So the loader reports what survived here, and ``server_status`` prints the record.
Module state rather than a ``user_ns`` entry, so a plugin cannot clobber the record
of itself. One kernel, one record.
"""

from __future__ import annotations

_LOADED_FILES: list[str] = []
_LOADED_ENTRY_POINTS: list[str] = []
_PLUGINS_ENABLED = True


def record_loaded_plugins(files=(), entry_points=(), *, enabled: bool = True) -> None:
    """Record the kernel plugins that loaded, called once by the bootstrap.

    *files* are the ``~/.config/biopb/kernel/*.py`` stems and *entry_points* the
    ``biopb_mcp.namespace`` names that **survived** loading — kept apart because
    only the file half has a "then it failed on load" story to tell (an entry point
    that never installed simply isn't there to name). *enabled* is the
    ``services.namespace_enabled`` switch: off means no plugin could have loaded,
    and the report has to name the switch rather than suggest a seed that wouldn't
    help.
    """
    global _PLUGINS_ENABLED
    _PLUGINS_ENABLED = bool(enabled)
    _LOADED_FILES[:] = [str(n) for n in files if n]
    _LOADED_ENTRY_POINTS[:] = [str(n) for n in entry_points if n]


def plugin_status_lines() -> list[str]:
    """The body of the ``## Kernel plugins`` section of ``server_status``.

    Formatted here rather than in the status snippet so it is unit-testable
    without a kernel. A skill's ``plugin:<name>`` matches a name on either line;
    they are printed apart so the "not listed → it failed to load" reading is
    only offered where it holds.
    """
    if not _PLUGINS_ENABLED:
        return ["  (disabled — services.namespace_enabled)"]
    lines = []
    if _LOADED_FILES:
        lines.append("  files: " + ", ".join(sorted(_LOADED_FILES)))
    else:
        lines.append(
            "  files: (none — `biopb-mcp-seed-plugins` seeds the built-in example "
            "into ~/.config/biopb/kernel/, then the kernel must restart)"
        )
    lines.append(
        "    a *.py in that dir but missing above failed on load; "
        "the session log says why"
    )
    if _LOADED_ENTRY_POINTS:
        lines.append("  packages: " + ", ".join(sorted(_LOADED_ENTRY_POINTS)))
    return lines
