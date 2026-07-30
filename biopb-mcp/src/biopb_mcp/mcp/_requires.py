"""The two `requires:` facts the session can't show the agent by itself.

A skill declares what it touches (``viewer``, ``tensor``, ``dask``, ``ops:<kind>``,
``plugin:<name>``, ``pkg:<name>``). The agent resolves that list itself, against
``server_status`` and — for a ``pkg:`` token — an import: five of the six are
already in front of it, more accurately than a helper could bucket them (a closed
napari window, the scheduler behind ``da``, the real ImportError). What it cannot
see is *which plugin files loaded* and *how to add a package to this env*; this
module supplies both to the status report.

``plugin:<name>`` is not derivable, which is why the record exists:

* The loader is **fail-open per unit**: a file that raises on ``exec``, or loses a
  name to the reserved-name guard, is on disk and *not* in the namespace. So
  scanning ``~/.config/biopb/kernel/`` answers "file present", not "plugin
  loaded" — and only the loader knows which of the two happened.
* The namespace can't answer either: a plugin *file* contributes its top-level
  function names, not its own name, so ``dir()`` never shows ``segmentation_qc``.

So the loader reports what survived here, and ``server_status`` prints the record.
Module state rather than a ``user_ns`` entry, so a plugin cannot clobber the record
of itself. One kernel, one record.

The install target is the same shape of problem: the agent can read
``sys.executable``, but not that this env is uv-managed and therefore disposable —
see :func:`versions_status_lines`.
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


def versions_status_lines(
    *, prefix=None, executable=None, has_pip=None, version=None
) -> list[str]:
    """The body of ``## Versions``: this kernel's build, and how to add a package.

    The interpreter is named, not just its version, because a ``pkg:`` requirement
    is about *this* env while a bare ``pip install`` targets whatever env the user's
    shell has active — an install that succeeds and leaves the import here still
    failing. Which command is right depends on the env, so it is decided here
    rather than left to the reader to guess:

    * A **uv tool env** — biopb's installed deployment — is identified by the
      ``uv-receipt.toml`` uv writes at the env root, *not* by the absence of pip:
      the real deployment carries pip transitively, so a pip probe would take the
      ``-m pip`` branch and stay silent about the part that matters. The installer
      upgrades with ``uv tool install --force``, which rebuilds that env from the
      receipt's own requirement list, so a package added here is gone at the next
      upgrade. Saying so is the difference between the user planning for it and
      discovering it.
    * Anything else (a venv, conda, a source checkout) is the user's to keep:
      ``-m pip`` when pip is importable, ``uv pip install --python`` when it isn't.

    The keyword arguments are for tests, which can't run inside both kinds of env.
    """
    import os
    import sys

    if executable is None:
        executable = sys.executable
    if prefix is None:
        prefix = sys.prefix
    if version is None:
        try:
            from .. import __version__ as version
        except Exception:  # noqa: BLE001 - report the rest rather than nothing
            version = "unknown"
    if has_pip is None:
        from importlib.util import find_spec

        try:
            has_pip = find_spec("pip") is not None
        except Exception:  # noqa: BLE001 - a broken env is not an unknown answer
            has_pip = False

    managed = os.path.exists(os.path.join(str(prefix), "uv-receipt.toml"))
    lines = [
        f"  biopb-mcp: {version}",
        f"  python: {sys.version.split()[0]} at {executable}",
    ]
    if managed or not has_pip:
        lines.append(f"    add a package: uv pip install --python {executable} <pkg>")
    else:
        lines.append(f"    add a package: {executable} -m pip install <pkg>")
    if managed:
        lines.append(
            "    that env is uv-managed and a biopb upgrade rebuilds it, so also "
            "have the user add the requirement to ~/.config/biopb/extra-packages.txt "
            "(one per line) — the installer replays it on every upgrade"
        )
    return lines


def plugin_status_lines() -> list[str]:
    """The body of the ``## Kernel plugins`` section of ``server_status``.

    Formatted here rather than in the status snippet so it is unit-testable
    without a kernel. A skill's ``plugin:<name>`` matches a name on either line;
    they are printed apart so the "not listed → it failed to load" reading is
    only offered where it holds. Both lines print unconditionally, so a name
    absent from the report is absent because it did not load -- not because the
    line it would have been on was left out.
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
    else:
        lines.append(
            "  packages: (none — no installed package declares a "
            "`biopb_mcp.namespace` entry point)"
        )
    return lines
