"""Resolve a skill's ``requires:`` tokens against the live session, in the kernel.

A skill declares what it touches (``viewer``, ``tensor``, ``dask``, ``ops:<kind>``,
``plugin:<name>``, ``pkg:<name>[>=version]``). That was inert metadata: emitted by
``find_skills``, checked by nobody, so a skill naming a kernel plugin the install
doesn't have read as available and dead-ended partway through its own steps. Each
skill body compensated with its own ad-hoc prose check (a ``dir()`` dance here, a
``find_spec`` there) — N hand-written variants of one question, drifting apart.

This answers it once, as ``check_skill_requirements()`` in the agent's namespace.

**The check belongs in the kernel, not in the MCP server process.** The kernel is
the only place every token is decidable *and* accurate:

===================== ==========================================================
token                 how it is decided here
===================== ==========================================================
``viewer``            the handle itself (a headless kernel binds a stand-in)
``tensor``            the live connection's client
``dask``              the ``da`` handle -- see ``_check_dask`` on why a
                      *distributed* cluster is not part of this question
``ops:<kind>``        membership in the built ``ops`` dict, and the reason names
                      what the servers do offer
``plugin:<name>``     the loader's record of what actually loaded
``pkg:<name>``        this interpreter's own metadata / import machinery
===================== ==========================================================

A server-side version of this check could only reach a subset, and would be
*wrong* on the part it did reach: it would scan the kernel plugin dir for files,
but the loader is fail-open per file, so a plugin that raised on ``exec`` (or lost
a name to the reserved-name guard) is on disk and not in the namespace. Hence
:func:`record_loaded_plugins` — the loader reports what survived, and this module
answers from that rather than from the filesystem.

Per-token failures are contained, but not silenced: a check that blows up is
reported as ``unknown`` with the error, because the caller is an agent reading a
report, not a gate. The one thing this must never do is claim a requirement is met
when it isn't.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# What the loader actually loaded, recorded rather than re-derived: a plugin *file*
# contributes its top-level function names, not its own name, so namespace
# membership cannot answer "did segmentation_qc load?". Module state, not a
# namespace entry, so a plugin cannot clobber the record of itself.
_LOADED_PLUGINS: set[str] = set()
_PLUGINS_ENABLED = True

_SEED_HINT = (
    "biopb-mcp ships some plugins as seedable files: `biopb-mcp-seed-plugins` "
    "installs them and the kernel must then restart. Both need the user's OK."
)


def record_loaded_plugins(names, *, enabled: bool = True) -> None:
    """Record the kernel plugins that loaded, called once by the bootstrap.

    *names* are the ones that **survived** loading (fail-open skips the rest), so
    ``plugin:<name>`` answers from the namespace's real contents. *enabled* is the
    ``services.namespace_enabled`` switch: off means no file could have loaded, and
    the reason string has to name the switch instead of suggesting a seed.
    """
    global _PLUGINS_ENABLED
    _PLUGINS_ENABLED = bool(enabled)
    _LOADED_PLUGINS.clear()
    _LOADED_PLUGINS.update(str(n) for n in names if n)


# --------------------------------------------------------------------------- #
# Per-token checks: each returns (met | unmet | unknown, reason)
# --------------------------------------------------------------------------- #
_MET = "met"
_UNMET = "unmet"
_UNKNOWN = "unknown"


def _check_viewer(ns):
    from ._bootstrap import _HeadlessViewer

    viewer = ns.get("viewer")
    # isinstance, not truthiness: the stand-in is falsy on purpose, but a real
    # napari viewer's truthiness is not ours to depend on.
    if isinstance(viewer, _HeadlessViewer):
        return _UNMET, (
            "this kernel started headless (no display): no viewer window and no "
            "screenshot. Use the numeric fallback in each of the skill's visual "
            "checks, and report results as numbers"
        )
    if viewer is None:
        return _UNMET, "no viewer in the namespace (the bootstrap failed?)"
    return _MET, ""


def _check_tensor(ns):
    # The connection is the truth; `client` is a per-job copy of it, so reading
    # _conn is right even if this runs outside a job.
    conn = ns.get("_conn")
    client = getattr(conn, "client", None) if conn is not None else ns.get("client")
    if client is not None:
        return _MET, ""
    detail = ""
    if conn is not None:
        detail = str(getattr(conn, "last_message", "") or "")
    return _UNMET, (
        "no tensor-server connection, so there is no data to read"
        + (f" ({detail})" if detail else "")
        + ". Check `biopb control status` with the user"
    )


def _check_dask(ns):
    # Deliberately *not* "is there a distributed cluster". `da` is always usable;
    # the scheduler behind it (distributed cluster / in-process threads) is a
    # performance property, reported by server_status, and a skill that says
    # `dask` means "this works on lazy arrays" -- which never stops being true.
    if ns.get("da") is None:
        return _UNMET, "dask.array is not in the namespace (the bootstrap failed?)"
    return _MET, ""


def _check_ops(ns, kind: str):
    ops = ns.get("ops")
    if not isinstance(ops, dict):
        return _UNMET, "no `ops` in the namespace (the bootstrap failed?)"
    if kind in ops:
        return _MET, ""
    if not ops:
        return _UNMET, (
            "no ProcessImage ops at all: none are configured "
            "(services.process_image_servers) or the servers are unreachable"
        )
    return _UNMET, (
        f"no {kind!r} op on the configured servers; they offer: "
        f"{', '.join(sorted(ops))}"
    )


def _check_plugin(name: str):
    if not _PLUGINS_ENABLED:
        return _UNMET, (
            "kernel plugins are switched off for this session "
            "(services.namespace_enabled), so no plugin can be loaded"
        )
    if name in _LOADED_PLUGINS:
        return _MET, ""
    if _LOADED_PLUGINS:
        loaded = ", ".join(sorted(_LOADED_PLUGINS))
        return _UNMET, (
            f"kernel plugin {name!r} did not load (these did: {loaded}). "
            f"If the file is present it failed on load -- the session log says why. "
            f"{_SEED_HINT}"
        )
    return _UNMET, f"no kernel plugins loaded in this session. {_SEED_HINT}"


def _split_spec(rest: str):
    """``"biopb-mcp>=0.12"`` -> ``("biopb-mcp", ">=", "0.12")``; no spec -> op None."""
    for op in (">=", "=="):
        head, sep, tail = rest.partition(op)
        if sep:
            return head.strip(), op, tail.strip()
    return rest.strip(), None, ""


def _version_ok(installed: str, op: str, required: str):
    """Compare on the *release* tuple; ``None`` when undecidable.

    Release tuples rather than full PEP 440 ordering, so a pre-release of the
    required version counts as meeting it: ``0.12.0rc8.dev32+g9268773`` satisfies
    ``>=0.12.0``. A strict compare ranks every rc *below* its own final release and
    would tell anyone on a dev build to upgrade to the thing they are running.
    """
    from packaging.version import Version

    try:
        have, want = Version(installed).release, Version(required).release
    except Exception:  # noqa: BLE001 - unparseable version, say so rather than guess
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


def _installed_version(name: str):
    """Version of *name* if it is an installed distribution, else ``None``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _importable(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name.replace("-", "_")) is not None
    except Exception:  # noqa: BLE001 - a broken parent package raises here
        logger.debug("find_spec(%r) failed", name, exc_info=True)
        return False


def _check_pkg(rest: str):
    """A ``pkg:`` token, resolved in *this* interpreter -- the one that will run it."""
    name, op, required = _split_spec(rest)
    if not name:
        return _UNKNOWN, "no package name in the token"
    # An operator we don't parse would otherwise ride along *inside* the name and
    # be reported as a package nobody has ("biopb-mcp~=0.1 is not installed").
    if op is None and any(c in name for c in "<>!~="):
        return _UNKNOWN, (
            f"cannot read the version constraint in {name!r} "
            f"(only >= and == are understood)"
        )
    installed = _installed_version(name)
    if installed is None:
        # A distribution name need not be the import name (scikit-image/skimage),
        # so absent metadata is not absent code.
        if not _importable(name):
            return _UNMET, (
                f"{name} is not installed in this kernel. Use the skill's degraded "
                f"path if it names one; installing needs the user's OK"
            )
        if op is None:
            return _MET, ""
        return _UNKNOWN, (
            f"{name} imports but has no version metadata, so {op}{required} "
            f"cannot be checked"
        )
    if op is None:
        return _MET, ""
    ok = _version_ok(installed, op, required)
    if ok is None:
        return _UNKNOWN, (
            f"cannot compare {installed!r} against {op}{required} "
            f"(only >= and == are understood)"
        )
    if ok:
        return _MET, ""
    return _UNMET, f"needs {name} {op} {required}, this kernel has {installed}"


def _check(token: str, ns):
    if token == "viewer":
        return _check_viewer(ns)
    if token == "tensor":
        return _check_tensor(ns)
    if token == "dask":
        return _check_dask(ns)
    if token.startswith("ops:"):
        return _check_ops(ns, token[len("ops:") :].strip())
    if token.startswith("plugin:"):
        return _check_plugin(token[len("plugin:") :].strip())
    if token.startswith("pkg:"):
        return _check_pkg(token[len("pkg:") :].strip())
    # The vocabulary is curated in biopb-site and grows without this file needing
    # an edit, so an unrecognised token is reported, not silently passed: the agent
    # can read it and use judgement, which is more than this function can do.
    return _UNKNOWN, "not a requirement this kernel knows how to check"


def check(requires, ns) -> dict:
    """Resolve *requires* against the namespace *ns*. See :func:`make_checker`."""
    if isinstance(requires, dict):  # a find_skills entry, passed whole
        requires = requires.get("requires", [])
    if isinstance(requires, str):  # a single token, unwrapped
        requires = [requires]
    if requires is None:
        requires = []
    try:
        tokens = [str(t).strip() for t in requires if str(t).strip()]
    except TypeError:
        return {"ok": True, "met": [], "unmet": [], "unknown": ["<not a list>"]}

    met, unmet, unknown = [], [], []
    for token in tokens:
        try:
            status, reason = _check(token, ns)
        except Exception as exc:  # noqa: BLE001 - one bad token, still report the rest
            logger.debug("requires check failed for %r", token, exc_info=True)
            status, reason = _UNKNOWN, f"the check itself failed: {exc!r}"
        if status == _MET:
            met.append(token)
        elif status == _UNMET:
            unmet.append(f"{token} — {reason}")
        else:
            unknown.append(f"{token} — {reason}")
    return {"ok": not unmet, "met": met, "unmet": unmet, "unknown": unknown}


def make_checker(namespace):
    """Build the namespace-bound ``check_skill_requirements`` for the bootstrap.

    Closes over the live ``user_ns`` dict, so every call reads the handles as they
    are *now* (a connection that came up after the kernel started, a plugin
    reloaded by hand) rather than as they were at bootstrap.
    """

    def check_skill_requirements(requires):
        """Check a skill's requirements against this session before you start it.

        Pass the ``requires`` list from ``find_skills`` (or the whole entry, or one
        token). Every token is resolved here, in the kernel that will run the
        skill — nothing is assumed::

            >>> check_skill_requirements(["viewer", "plugin:segmentation_qc"])
            {'ok': False,
             'met': ['viewer'],
             'unmet': ["plugin:segmentation_qc — kernel plugin 'segmentation_qc' …"],
             'unknown': []}

        ``ok`` is False when something is definitely missing. **Tell the user what
        is missing and let them decide** — installing a package, seeding a plugin
        and restarting the kernel all need their consent, and a skill worth reading
        is still worth reading with a gap named up front. ``unknown`` holds tokens
        this version cannot check (a newer vocabulary, an unreadable version); use
        your own judgement on those.

        Recognised: ``viewer``, ``tensor``, ``dask``, ``ops:<name>``,
        ``plugin:<name>``, ``pkg:<name>`` with an optional ``>=``/``==`` version.
        """
        return check(requires, namespace)

    return check_skill_requirements
