"""Resolve a skill's ``requires:`` tokens against this session, for find_skills.

A skill declares what it touches (``viewer``, ``tensor``, ``dask``, ``ops:<kind>``,
``plugin:<name>``, ``pkg:<name>[>=version]``). Until now that was inert metadata
passed straight through to the agent, so a skill naming a kernel plugin the
install doesn't have read as available and dead-ended partway through its own
steps. This turns it into a discovery-time signal.

**Only definite failures are reported.** ``find_skills`` runs in the MCP server
process and is called *before* ``start_kernel`` in the normal flow, so the kernel
namespace, the tensor connection, the dask cluster and ``ops`` simply are not
knowable at that point -- and the kernel may not even be the same interpreter as
this process (the ``python3`` kernelspec need not be the tool env). A token this
module cannot decide is left alone rather than guessed at: a wrong "missing" is
worse than silence, because it argues the agent out of a skill that would have
worked. Which tokens are decidable here:

===================== ========== =========================================
token                 decidable  source
===================== ========== =========================================
``viewer``            yes        the launcher's headless flag
``plugin:<name>``     yes        static scan of the kernel plugin dir +
                                 ``biopb_mcp.namespace`` entry points,
                                 never importing either
``pkg:biopb-mcp``     yes        installed metadata -- this *is* biopb-mcp
``tensor``/``dask``   no         kernel/connection state
``ops:<kind>``        no         built in the kernel from configured servers
``pkg:<other>``       no         the kernel's interpreter may differ, and a
                                 package-tier skill carries its own import
                                 check and degraded path anyway
===================== ========== =========================================

Every check is wrapped: this decorates a fail-open documentation lookup, so a
failure here must cost an annotation, never the skill.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Prefixes we understand. Anything else is left undecided -- the vocabulary is
# curated in biopb-site and grows without this file needing an edit, which is the
# same reason the tag vocabulary stopped being a constant.
_SEEN_PREFIXES = ("ops:", "plugin:", "pkg:")

_SEED_HINT = "run biopb-mcp-seed-plugins, then restart the kernel (ask first)"


def _is_headless() -> bool:
    from . import _server

    return bool(_server.is_headless())


def _namespace_enabled() -> bool:
    # No default restated here: get_setting falls back to DEFAULT_CONFIG, so the
    # value is declared once in _config.py (the drift _skills.py records).
    from .._config import CONFIG, get_setting

    return bool(get_setting(CONFIG.as_dict(), "services.namespace_enabled"))


def _available_plugins() -> set:
    """Plugin names that *will* load into the kernel namespace, read statically.

    Names come from ``<name>.py`` in the kernel plugin dir and from installed
    ``biopb_mcp.namespace`` entry points. Static on purpose -- listing a plugin
    must not execute it (:mod:`biopb._kernel_plugins`).
    """
    from biopb import _kernel_plugins
    from biopb._locations import mcp_plugin_dir

    names = set()
    for row in _kernel_plugins.startup_files(mcp_plugin_dir()):
        name = row.get("name", "")
        names.add(name[:-3] if name.endswith(".py") else name)
    for row in _kernel_plugins.entry_point_plugins():
        names.add(row.get("name", ""))
    names.discard("")
    return names


def _split_spec(rest: str):
    """``"biopb-mcp>=0.12"`` -> ``("biopb-mcp", ">=", "0.12")``; no spec -> op None."""
    for op in (">=", "=="):
        head, sep, tail = rest.partition(op)
        if sep:
            return head.strip(), op, tail.strip()
    return rest.strip(), None, ""


def _release(version: str):
    from packaging.version import Version

    return Version(version).release


def _version_ok(installed: str, op: str, required: str):
    """Compare on the *release* tuple; ``None`` when undecidable.

    Release-tuple comparison, not full PEP 440 ordering, so a pre-release of the
    required version counts as meeting it: ``0.12.0rc8.dev32+g9268773`` satisfies
    ``>=0.12.0``. A strict compare ranks every rc *below* its own final release
    and would tell anyone on a dev build to upgrade to the thing they are running.
    """
    try:
        have, want = _release(installed), _release(required)
    except Exception:  # noqa: BLE001 - unparseable version, stay silent
        logger.debug("version compare failed: %r %s %r", installed, op, required)
        return None
    width = max(len(have), len(want))
    have += (0,) * (width - len(have))
    want += (0,) * (width - len(want))
    if op == ">=":
        return have >= want
    if op == "==":
        return have[: len(want)] == want
    return None


def _check_pkg(rest: str):
    """Reason string if a ``pkg:`` token is definitely unmet, else ``None``."""
    name, op, required = _split_spec(rest)
    # Only our own distribution: for anything else this process's environment is
    # not authoritative for the kernel's.
    if name.replace("_", "-").lower() != "biopb-mcp":
        return None

    from importlib.metadata import version as _dist_version

    installed = _dist_version("biopb-mcp")
    if op is None:
        return None  # installed by definition — we are running from it
    ok = _version_ok(installed, op, required)
    if ok is False:
        return f"needs biopb-mcp {op} {required}, this session is {installed}"
    return None


def _check_plugin(name: str):
    """Reason string if a ``plugin:`` token is definitely unmet, else ``None``."""
    if not _namespace_enabled():
        return (
            "kernel plugins are disabled for this session (services.namespace_enabled)"
        )
    if name in _available_plugins():
        return None
    return f"no kernel plugin {name!r} in this install; {_SEED_HINT}"


def _reason(token: str):
    """Why *token* is unmet, or ``None`` when met or undecidable."""
    token = token.strip()
    if not token:
        return None
    if token == "viewer":
        if _is_headless():
            return (
                "this session is headless (no viewer); use the numeric fallback "
                "in each visual check"
            )
        return None
    if token.startswith("plugin:"):
        return _check_plugin(token[len("plugin:") :].strip())
    if token.startswith("pkg:"):
        return _check_pkg(token[len("pkg:") :].strip())
    # tensor / dask / ops:* / anything unrecognised: not decidable here.
    return None


def unmet(requires) -> list[str]:
    """Requirements this session definitely does not meet, as ``"token — why"``.

    Empty when everything is met *or* undecidable. Never raises: a broken check
    degrades to "no annotation", because the caller is a fail-open lookup whose
    value is agent context.
    """
    if not isinstance(requires, list):
        return []
    out = []
    for token in requires:
        if not isinstance(token, str):
            continue
        try:
            reason = _reason(token)
        except Exception:  # noqa: BLE001 - annotation must never break discovery
            logger.debug("requires check failed for %r", token, exc_info=True)
            continue
        if reason:
            out.append(f"{token} — {reason}")
    return out
