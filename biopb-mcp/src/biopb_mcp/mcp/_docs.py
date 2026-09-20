"""The knowledge store: flat markdown docs in two tiers, and an agent-edited index.

See ``docs/knowledge.md``. A doc is a markdown file; its **id is its path under
the tier directory, minus ``.md``**, and may contain ``/``. Two tiers, both read
fresh on every access:

* **shipped** — ``_docs_data/`` inside the wheel, read-only. Every file in it
  ships and reads back by id. A ``_``-prefixed name is **banked**: the release
  does not list it, so it stays out of the reconciliation tail, but
  :func:`read_doc` still returns it and one hand-written index line promotes it.
* **local** — ``~/.config/biopb/docs/`` (``services.docs_local_dir`` moves it),
  written by :func:`write_doc`. A local doc shadows a shipped one of the same id.

The **index** is itself a doc (``index`` in the local tier), edited like any
other and seeded from the package on first run. It is not derived from the
files: a derived listing cannot carry the agent's own hooks, grouping and
"don't use X for Y" notes. What the loader knows and the file cannot -- an
entry with no file, a shadowed shipped doc, a shipped doc the index has never
mentioned -- is added at render time (:func:`render_index`).

``services.docs_enabled`` is the benchmark's ablation switch. Off, it withholds
``kind: procedure`` docs -- the curated workflows -- from the index and from
:func:`read_doc`; reference docs stay, so the ablated arm loses the procedures
and not the API documentation.

**Fail-open.** A malformed or unreadable file is skipped and debug-logged. The
write path is the exception: a refusal there is returned to the agent as text
saying why, since a write that silently does nothing is worse than one that
explains itself.
"""

from __future__ import annotations

import difflib
import logging
import re
from datetime import date
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

# The shipped seed, as package data (see pyproject [tool.setuptools.package-data]).
_DATA_PKG = "biopb_mcp.mcp"
_DATA_DIR = "_docs_data"

_SUFFIX = ".md"

#: The index is a doc, under this reserved id.
INDEX_ID = "index"

#: ``kind:``, on shipped docs only. Procedures are what the ablation withholds.
KIND_REFERENCE = "reference"
KIND_PROCEDURE = "procedure"

#: Index entries the write tool will accept. At the cap the rendered index is
#: roughly 25 KB, paid once per session in the handshake; past it the agent has
#: to condense rather than keep appending.
MAX_INDEX_ENTRIES = 200

#: A doc is read whole into context, so a long one is a doc that wants splitting.
MAX_BODY_LINES = 300

_ORIGIN_SHIPPED = "shipped"
_ORIGIN_LOCAL = "local"


# --------------------------------------------------------------------------- #
# Ids
# --------------------------------------------------------------------------- #
# An id becomes a path under a tier directory, so it is validated rather than
# sanitised: anything outside this charset is refused by name instead of being
# rewritten into something the agent did not ask for.
# A leading `_` is allowed: that is what a banked doc's id starts with, and
# `read_doc` has to resolve one. `.` and `-` are not, which is what keeps `..`
# and a flag-shaped name out.
_SEGMENT = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9._-]*\Z")


def valid_id(doc_id: str) -> bool:
    """True if *doc_id* names a doc: slash-separated segments, no traversal.

    A trailing ``/`` is not an id. It is reserved for the collections of
    ``docs/knowledge.md`` §9 and is accepted only where that section says it is
    -- verbatim in an index line -- so a seed written for a later release parses
    here and means something there.
    """
    if not doc_id or doc_id != doc_id.strip():
        return False
    return all(_SEGMENT.match(part) for part in doc_id.split("/"))


def _relative_path(doc_id: str) -> str:
    return doc_id + _SUFFIX


# --------------------------------------------------------------------------- #
# Frontmatter
# --------------------------------------------------------------------------- #
# Deliberately weak: scalars and inline `[a, b]` lists, no YAML dependency, and
# anything it cannot parse is ignored rather than fatal. Every field has a
# fallback, so a bare markdown file with no frontmatter at all still loads --
# which is what lets a user drop a file in the local dir and have it work.
_FM_BLOCK = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")


def strip_frontmatter(text: str) -> str:
    """Drop a leading ``--- … ---`` block; the agent wants the prose."""
    return _FM_BLOCK.sub("", text, count=1).lstrip()


def parse_frontmatter(text: str) -> dict:
    """Read a leading ``--- … ---`` block into a flat dict of strings/lists.

    Understands ``key: value`` and ``key: [a, b]``; quotes are stripped. Lines it
    does not understand (nesting, block lists, folded scalars) are skipped.
    """
    match = _FM_BLOCK.match(text)
    if not match:
        return {}
    out: dict = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace():
            continue  # indented: part of a nested block this reader doesn't do
        fields = _FM_LINE.match(line.strip())
        if not fields:
            continue
        key, value = fields.group(1).lower(), fields.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            out[key] = [
                item.strip().strip("\"'")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        else:
            out[key] = value.strip("\"'")
    return out


def _first_h1(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _first_prose(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", ">", "-", "*", "|", "`")):
            return line[:300]
    return ""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_UNSET = object()


def _setting(path: str, default=_UNSET):
    """One config read. Defaults live in ``_config.py``, never restated here."""
    from .._config import CONFIG, get_setting

    if default is _UNSET:
        return get_setting(CONFIG.as_dict(), path)
    return get_setting(CONFIG.as_dict(), path, default)


def procedures_enabled() -> bool:
    """Whether ``kind: procedure`` docs are served (``services.docs_enabled``)."""
    try:
        return bool(_setting("services.docs_enabled"))
    except Exception:  # pragma: no cover - config always loadable in practice
        logger.debug("docs: docs_enabled unreadable, assuming on", exc_info=True)
        return True


def local_dir() -> Path | None:
    """The local tier's directory. Resolved per call, never created here."""
    configured = (_setting("services.docs_local_dir", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        from biopb._locations import mcp_docs_dir

        return mcp_docs_dir()
    except Exception:  # pragma: no cover - core SDK always present in practice
        logger.debug("docs: no local dir resolvable", exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# The shipped tier
# --------------------------------------------------------------------------- #
def _shipped_root():
    """The packaged ``_docs_data`` directory, as a Traversable.

    ``joinpath`` does not touch the filesystem, so a missing directory surfaces
    at the walk below rather than here. Kept a function because tests redirect it.
    """
    return resources.files(_DATA_PKG).joinpath(_DATA_DIR)


def _walk_shipped(node, prefix: str = ""):
    for child in sorted(node.iterdir(), key=lambda p: p.name):
        name = child.name
        if name.startswith(("_", ".")):
            continue  # banked: not listed by the release (see shipped_ids)
        if child.is_dir():
            yield from _walk_shipped(child, f"{prefix}{name}/")
        elif name.endswith(_SUFFIX):
            yield prefix + name[: -len(_SUFFIX)]


def shipped_ids() -> list[str]:
    """The shipped docs **the release lists**, index excluded.

    Read by one caller, :func:`render_index`, for the *New shipped docs* tail.
    Reading a doc does not come through here, so a banked (``_``-prefixed) one
    is absent from this list and still returns from :func:`read_doc` — which is
    the whole of what banked means now.

    Two actors, two mechanisms, and neither can express the other's decision:
    this is the release's, and the index's ``ignored:`` line is the agent's.
    """
    try:
        return [i for i in _walk_shipped(_shipped_root()) if i != INDEX_ID]
    except (FileNotFoundError, NotADirectoryError, OSError):
        logger.warning(
            "docs: the shipped set is missing or unreadable (%s). This is an "
            "install or packaging problem, not a configuration one.",
            _shipped_root(),
        )
        return []


def _shipped_text(doc_id: str) -> str | None:
    if not valid_id(doc_id):
        return None
    try:
        node = _shipped_root()
        for part in _relative_path(doc_id).split("/"):
            node = node.joinpath(part)
        return node.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, OSError, UnicodeError):
        # UnicodeError (a ValueError, not an OSError) is a corrupt file: it has
        # to fail open here too, not crash the read.
        return None


# --------------------------------------------------------------------------- #
# The local tier
# --------------------------------------------------------------------------- #
def _local_path(doc_id: str) -> Path | None:
    root = local_dir()
    if root is None or not valid_id(doc_id):
        return None
    return root / _relative_path(doc_id)


def _local_text(doc_id: str) -> str | None:
    path = _local_path(doc_id)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def local_ids() -> list[str]:
    """Every local ``*.md``, index excluded. Unreadable dir is the normal case."""
    root = local_dir()
    if root is None:
        return []
    try:
        paths = sorted(p for p in root.rglob("*") if p.is_file())
    except OSError:
        logger.debug("docs: local dir unreadable (fail-open)", exc_info=True)
        return []
    out = []
    for path in paths:
        if not path.name.lower().endswith(_SUFFIX):
            continue
        doc_id = path.relative_to(root).as_posix()[: -len(_SUFFIX)]
        if doc_id != INDEX_ID and valid_id(doc_id):
            out.append(doc_id)
    return out


def local_dir_status() -> str:
    """The ``## Docs`` body of ``server_status``.

    The local dir is a default, not a constant, and this is the only place the
    agent can read where a doc it writes lands.
    """
    root = local_dir()
    if root is None:
        return "  local_dir: (unresolvable — biopb core SDK missing)"
    line = f"  local_dir: {root}"
    try:
        if not root.is_dir():
            return line + " (not created yet — write_doc creates it)"
        n = len(local_ids())
    except OSError:
        return line + " (unreadable)"
    line += f" ({n} local doc{'' if n == 1 else 's'})"
    legacy = root.parent / "skills"
    try:
        # The pre-redesign local tier. Nothing reads it any more, so a user who
        # had files there would otherwise find them silently gone.
        if legacy.is_dir() and any(legacy.glob("*.md")):
            line += f"\n  legacy skills dir (no longer read): {legacy}"
    except OSError:
        pass
    return line


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _updated(text: str, path: Path | None) -> str:
    """``updated:`` if the author wrote one, else the local file's mtime.

    Derived rather than stamped by :func:`write_doc`: an ``old``/``new`` write
    that also rewrote the frontmatter would be a replace that changed something
    the agent did not name, and mtime answers the same question for free.
    """
    stamped = str(parse_frontmatter(text).get("updated") or "").strip()
    if stamped:
        return stamped
    if path is not None:
        try:
            return date.fromtimestamp(path.stat().st_mtime).isoformat()
        except OSError:
            pass
    return ""


def describe(doc_id: str) -> dict | None:
    """Resolve *doc_id* to its text and what the loader knows about it.

    Keys: ``id``, ``text``, ``body``, ``title``, ``description``, ``kind``,
    ``packages``, ``updated``, ``origin``, ``shadows_shipped``. ``None`` when
    no tier holds it.
    """
    if not valid_id(doc_id):
        return None
    local = _local_text(doc_id)
    shipped = _shipped_text(doc_id)
    if local is None and shipped is None:
        return None

    text = local if local is not None else shipped
    origin = _ORIGIN_LOCAL if local is not None else _ORIGIN_SHIPPED
    front = parse_frontmatter(text)
    body = strip_frontmatter(text)
    kind = str(front.get("kind") or "").strip().lower()
    if origin == _ORIGIN_LOCAL or kind not in (KIND_REFERENCE, KIND_PROCEDURE):
        # `kind` is a shipped-doc key. A local doc is a procedure, which is what
        # keeps the ablation honest: it withholds every curated workflow, not
        # only the ones that happen to ship.
        kind = KIND_PROCEDURE
    packages = front.get("packages")
    if isinstance(packages, str):
        packages = [p.strip() for p in packages.split(",") if p.strip()]
    elif not isinstance(packages, list):
        packages = []

    return {
        "id": doc_id,
        "text": text,
        "body": body,
        "title": str(front.get("title") or _first_h1(body) or doc_id).strip(),
        "description": str(
            front.get("description") or _first_prose(body) or doc_id
        ).strip(),
        "kind": kind,
        "packages": [str(p) for p in packages],
        "updated": _updated(text, _local_path(doc_id) if local is not None else None),
        "origin": origin,
        "shadows_shipped": local is not None and shipped is not None,
    }


def _visible(meta: dict, procedures: bool) -> bool:
    return procedures or meta["kind"] == KIND_REFERENCE


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
# Two line shapes are recognised; everything else -- headings, prose, ordering --
# is the agent's and passes through untouched.
_ENTRY = re.compile(r"\A(\s*[-*]\s+)([A-Za-z0-9][A-Za-z0-9._/-]*/?)\s*:\s*(.*)\Z")
_IGNORED = re.compile(r"\A\s*ignored\s*:\s*(.*)\Z", re.IGNORECASE)

_UNFILED = "## Unfiled"


def _entry_id(line: str) -> str | None:
    """The id an index line names, or ``None`` if it is not an entry."""
    match = _ENTRY.match(line)
    return match.group(2) if match else None


def _ignored_ids(text: str) -> set[str]:
    out: set[str] = set()
    for line in text.splitlines():
        match = _IGNORED.match(line)
        if match:
            out.update(i.strip() for i in match.group(1).split(",") if i.strip())
    return out


def index_entry_count(text: str) -> int:
    """Index entries in *text*. A collection line counts as one."""
    return sum(1 for line in text.splitlines() if _entry_id(line))


def _seed_index_text() -> str:
    return _shipped_text(INDEX_ID) or "# biopb docs\n"


def index_text() -> str:
    """The local index, seeded from the package on first run.

    Seeding writes, which is the one thing a read does to the config tree: a
    fresh install is meant to start from the curated ordering, and an index the
    agent cannot edit is not the design's index. A directory that cannot be
    created falls back to serving the shipped text unwritten.
    """
    existing = _local_text(INDEX_ID)
    if existing is not None:
        return existing
    seed = _seed_index_text()
    path = _local_path(INDEX_ID)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(seed, encoding="utf-8")
        except OSError:
            logger.debug("docs: could not seed the local index", exc_info=True)
    return seed


def render_index(text: str | None = None) -> str:
    """The index file plus what the loader knows and the file cannot.

    Entry lines gain ``(missing)`` where no tier holds the id and ``(local
    copy)`` where a local doc shadows a shipped one; a trailing *New shipped
    docs* line names the shipped docs this index neither lists nor ignores. The
    tail is not truncated -- it is bounded by what one release adds, and
    truncating it would hide the upgrade it exists to report.
    """
    if text is None:
        text = index_text()
    procedures = procedures_enabled()

    named: set[str] = set(_ignored_ids(text))
    out: list[str] = []
    for line in text.splitlines():
        doc_id = _entry_id(line)
        if doc_id is None:
            out.append(line)
            continue
        named.add(doc_id)
        if doc_id.endswith("/"):
            out.append(line)  # a collection line: kept verbatim (knowledge.md §9)
            continue
        meta = describe(doc_id)
        if meta is None:
            out.append(f"{line} (missing)")
            continue
        if not _visible(meta, procedures):
            continue
        out.append(f"{line} (local copy)" if meta["shadows_shipped"] else line)

    new = [
        i
        for i in shipped_ids()
        if i not in named and _visible(describe(i) or {"kind": ""}, procedures)
    ]
    if new:
        out.append("")
        out.append(f"New shipped docs: {', '.join(new)}")
    return "\n".join(out).rstrip() + "\n"


def _append_entry(text: str, doc_id: str, hook: str) -> str:
    """Add ``- <id>: <hook>`` under a trailing ``## Unfiled``, adding the heading.

    Unfiled rather than guessed-at placement: where a doc belongs is the agent's
    judgement, and it moves the line when it next edits the index.
    """
    line = f"- {doc_id}: {hook}".rstrip()
    lines = text.splitlines()
    for i, existing in enumerate(lines):
        if existing.strip() == _UNFILED:
            end = i + 1
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("#"):
                    break
                if lines[j].strip():
                    end = j + 1
            lines.insert(end, line)
            return "\n".join(lines) + "\n"
    body = "\n".join(lines).rstrip()
    return f"{body}\n\n{_UNFILED}\n\n{line}\n"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _header(meta: dict) -> str:
    bits = [meta["origin"]]
    if meta["shadows_shipped"]:
        bits.append("shadows shipped")
    if meta["origin"] == _ORIGIN_SHIPPED:
        bits.append(meta["kind"])
    if meta["updated"]:
        bits.append(f"updated {meta['updated']}")
    return f"{meta['id']} — {', '.join(bits)}"


def read_doc(doc_id: str) -> str:
    """The doc's body under a one-line header, or the index rendered.

    Fail-open: an unknown id returns a sentence saying so, never an error --
    the value is agent context, not executed code.
    """
    doc_id = (doc_id or "").strip()
    if doc_id == INDEX_ID:
        return render_index()
    meta = describe(doc_id)
    if meta is None:
        return (
            f"No doc '{doc_id}'. Read the index with read_doc('index') for what "
            "there is."
        )
    if not _visible(meta, procedures_enabled()):
        return (
            f"Doc '{doc_id}' is a procedure, and procedures are switched off on "
            "this server (services.docs_enabled)."
        )
    return f"{_header(meta)}\n\n{meta['body']}"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _diff(before: str, after: str, doc_id: str) -> str:
    """What the write actually changed.

    Returned whichever form was used, so a replace that landed somewhere
    unexpected shows in the result rather than in the next read.
    """
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{doc_id} (before)",
            tofile=f"{doc_id} (after)",
            lineterm="",
            n=2,
        )
    )
    return "\n".join(lines) if lines else "(no change)"


def write_doc(
    doc_id: str,
    body: str | None = None,
    old: str | None = None,
    new: str | None = None,
) -> str:
    """Create or edit a local doc. Returns a diff, or a sentence saying why not.

    ``body`` replaces the whole doc; ``old``/``new`` replaces one exact
    occurrence. A write to a shipped id is copy-on-write: it creates a local doc
    shadowing it, and the shipped file is never touched, so an upgrade can still
    replace it.
    """
    doc_id = (doc_id or "").strip()
    if doc_id.endswith("/"):
        return (
            f"'{doc_id}' names a collection, not a doc. Collections are not writable."
        )
    if not valid_id(doc_id):
        return (
            f"'{doc_id}' is not a doc id. Ids are letters, digits, '.', '-', '_' "
            "and '/' as a separator."
        )
    if body is not None and (old is not None or new is not None):
        return "Pass either body, or the old/new pair — not both."
    if body is None and (old is None or new is None):
        return "Pass body to write the whole doc, or both old and new to edit it."

    path = _local_path(doc_id)
    if path is None:
        return "No local docs directory is resolvable, so nothing can be written."

    before_local = _local_text(doc_id)
    base = before_local if before_local is not None else _shipped_text(doc_id)

    if body is not None:
        after = body if body.endswith("\n") else body + "\n"
    else:
        if base is None:
            return f"No doc '{doc_id}' to edit. Pass body to create it."
        found = base.count(old)
        if found == 0:
            return f"The old text is not in '{doc_id}', so nothing was changed."
        if found > 1:
            return (
                f"The old text appears {found} times in '{doc_id}'. Include "
                "enough surrounding lines to name one of them."
            )
        after = base.replace(old, new, 1)

    line_count = len(after.splitlines())
    if line_count > MAX_BODY_LINES:
        return (
            f"That body is {line_count} lines; the cap is {MAX_BODY_LINES}. A doc "
            "is read whole into context — split it, or cut it down."
        )
    if doc_id == INDEX_ID:
        entries = index_entry_count(after)
        if entries > MAX_INDEX_ENTRIES:
            return (
                f"That index has {entries} entries; the cap is "
                f"{MAX_INDEX_ENTRIES}. Condense or retire entries instead."
            )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after, encoding="utf-8")
    except OSError as exc:
        return f"Could not write '{doc_id}': {exc}"

    result = _diff(base or "", after, doc_id)
    if doc_id != INDEX_ID and before_local is None:
        result += _file_index_entry(doc_id)
    return result


def _file_index_entry(doc_id: str) -> str:
    """Give a freshly created doc an index line, unless the index has one.

    A doc the index does not name does not exist to the agent, so creation and
    listing are one call. An id already named -- the common case when a shipped
    doc is shadowed -- is left alone rather than listed twice.
    """
    text = index_text()
    if doc_id in _ignored_ids(text) or any(
        _entry_id(line) == doc_id for line in text.splitlines()
    ):
        return ""
    meta = describe(doc_id)
    hook = meta["description"] if meta else doc_id
    after = _append_entry(text, doc_id, hook)
    path = _local_path(INDEX_ID)
    if path is None:
        return ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after, encoding="utf-8")
    except OSError:
        logger.debug("docs: could not file %s in the index", doc_id, exc_info=True)
        return ""
    return f"\n\nFiled in the index under '{_UNFILED}':\n- {doc_id}: {hook}"
