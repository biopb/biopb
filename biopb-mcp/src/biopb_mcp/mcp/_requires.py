"""The Requirements fact the session can't show the agent by itself.

A procedure doc opens with a **Requirements** line naming what its steps touch: a
viewer, the data plane, dask, an ``ops`` kind, a third-party package. The agent
resolves that line itself, against ``server_status`` and — for a package — an
import: everything is already in front of it, more accurately than a helper could
bucket it (a closed napari window, the scheduler behind ``da``, the real
ImportError) — except *how to add a package to this env*, which this module
supplies to the status report. The ``requirements`` doc is the reader's half of
the same contract.

The install target is not derivable: the agent can read ``sys.executable``, but
not that this env is uv-managed and therefore disposable — see
:func:`versions_status_lines`.
"""

from __future__ import annotations


def versions_status_lines(
    *, prefix=None, executable=None, has_pip=None, version=None
) -> list[str]:
    """The body of ``## Versions``: this kernel's build, and how to add a package.

    The interpreter is named, not just its version, because a package requirement
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
