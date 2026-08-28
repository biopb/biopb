"""Curated skills catalog — discovery + retrieval (docs/skills.md).

Two entry points are wired into ``_server.py``:

* :func:`list_skills` — the ``list_skills`` tool: query the catalog metadata and
  return a tailored subset (each carrying its ``skill://<id>`` URI).
* :func:`get_skill_body` — the ``skill://{id}`` resource read: the skill's full
  markdown body, frontmatter stripped.

**Skills ship with the package** (``_skills_data/*.md``) and are never fetched.
A skill is documentation about a specific runtime version — it quotes an API,
assumes a namespace handle, depends on packages resolving a particular way — so
it rides that version's upgrade cycle rather than being served to every release
in the field at once. The cost is that a skill fix needs a release; the local
dir below is the escape hatch. See ``docs/skills.md`` §1.

There is no catalog index. The frontmatter *is* the metadata, and this module
parses it, so there is no generated file to disagree with the bodies.

**Three sources.** User-authored skills in ``~/.config/biopb/skills/*.md``
**merge** with the shipped set (local wins a shared id, and every entry carries
``origin``). Both are re-read on every call: a skill the user is editing must
show up without a restart, and the shipped set is a handful of small files.

The third is the **kernel plugins**, described by their own module docstrings
and marked ``kind: "plugin"``. They are not skills and are not presented as
some — no body, no ``skill://`` uri, just the handle they are bound under — but
discovery is a catalog's job and this is where the agent looks. See
:func:`_scan_plugins` for why a bare name in ``server_status`` was not enough.

**Fail-open.** A malformed or unreadable file is skipped and debug-logged, never
fatal — one bad skill must not sink discovery, and nothing here raises into a
tool call or the bootstrap.
"""

import logging
import re
from collections.abc import Sequence
from datetime import date
from importlib import resources
from pathlib import Path

from ._skills_layout import is_skill_file

logger = logging.getLogger(__name__)

# The shipped skills, as package data (see pyproject [tool.setuptools.package-data]).
_DATA_PKG = "biopb_mcp.mcp"
_DATA_DIR = "_skills_data"

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

# Origin travels with every entry so the agent can tell a personal draft from a
# reviewed one.
_ORIGIN_LOCAL = "local"
_ORIGIN_CATALOG = "catalog"
_ORIGIN_PLUGIN_FILE = "plugin-file"
_ORIGIN_PLUGIN_PACKAGE = "plugin-package"

# What a row *is*. A skill is a procedure to follow and has a body to read; a
# plugin is an object already in the namespace, and the only thing to "read" is
# its signature. Conflating them sends an agent to `skill://rolling_ball`, which
# does not exist, so the kind is on every row rather than inferred from which
# other keys happen to be present.
KIND_SKILL = "skill"
KIND_PLUGIN = "plugin"

# Frontmatter reader. Deliberately weak: scalars and inline `[a, b]` lists, no
# YAML dependency in this stdlib-only module, and anything it can't parse is
# ignored rather than fatal. A personal skill must never need a validator the
# user doesn't have -- id and description are inferred when absent, so a bare
# markdown file with no frontmatter at all still loads. The authoring gate in
# `_tests/skills/` is the strict reader; this one only has to not crash.
_FM_BLOCK = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")


# --------------------------------------------------------------------------- #
# Markdown / frontmatter readers
# --------------------------------------------------------------------------- #
def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``--- … ---`` YAML frontmatter block; the agent context
    wants the workflow prose, not the metadata already carried in the entry."""
    return _FRONTMATTER.sub("", text, count=1).lstrip()


def _parse_frontmatter(text: str) -> dict:
    """Read a leading ``--- … ---`` block into a flat dict of strings/lists.

    Understands ``key: value`` and ``key: [a, b]``; quotes are stripped. Lines it
    doesn't understand (nesting, block lists, folded scalars) are skipped, since
    every field has a fallback and half a parse still beats none.
    """
    m = _FM_BLOCK.match(text)
    if not m:
        return {}
    out: dict = {}
    for line in m.group(1).splitlines():
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
    """First non-heading, non-empty line — the description fallback."""
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", ">", "-", "*", "|", "`")):
            return line[:300]
    return ""


def _entry(text: str, *, stem: str, origin: str, updated: str = "") -> dict | None:
    """One skill file → a catalog-shaped entry, or ``None`` if unusable.

    Shared by both sources: the shipped set and a user's own files are the same
    kind of thing (a markdown file with frontmatter), and reading them two ways
    would be two things to keep agreeing.
    """
    fm = _parse_frontmatter(text)
    body = _strip_frontmatter(text)

    skill_id = str(fm.get("id") or stem).strip()
    if not skill_id:
        return None
    title = str(fm.get("title") or _first_h1(body) or skill_id).strip()
    description = str(fm.get("description") or _first_prose(body) or title).strip()
    if not description:
        return None

    tags = fm.get("tags")
    tags = [str(t) for t in tags] if isinstance(tags, list) else []
    # `requires:` was this key's name until the rename; a local skill in the
    # user's own directory may still say it, and dropping their list silently is
    # worse than reading it.
    checklist = fm.get("checklist", fm.get("requires"))
    checklist = [str(c) for c in checklist] if isinstance(checklist, list) else []

    return {
        "id": skill_id,
        "title": title,
        "description": description,
        "tags": tags,
        "version": str(fm.get("version") or ""),
        "checklist": checklist,
        "updated": str(fm.get("updated") or updated or ""),
        "origin": origin,
        "kind": KIND_SKILL,
    }


# --------------------------------------------------------------------------- #
# Source 1: the shipped skills
# --------------------------------------------------------------------------- #
def _data_dir():
    """The packaged ``_skills_data`` directory, as a Traversable.

    Not guarded: this asks about the package this module lives in, so a failure
    here means ``import biopb_mcp.mcp._skills`` already failed. ``joinpath`` does
    not touch the filesystem either — a directory that is not there surfaces at
    :func:`_scan_shipped`, which is where it can be reported usefully. Kept as a
    function because the tests redirect it.
    """
    return resources.files(_DATA_PKG).joinpath(_DATA_DIR)


def _shipped_text(name: str) -> str | None:
    """Read one packaged skill file, or ``None`` if absent/unreadable."""
    try:
        return _data_dir().joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        # UnicodeError: a corrupt / non-UTF8 file (a ValueError subclass, not an
        # OSError) must also fail open, not crash the read.
        return None


# One warning per process, not per call: load_catalog() runs on every
# list_skills, and a broken install would otherwise fill the session log.
_warned_empty = False


def _warn_empty_once(detail: str) -> None:
    """An empty shipped set is always a bug — say so, once, and carry on.

    This used to be an ordinary state: the catalog came over the network, so
    "nothing resolved" meant offline, and quietly returning [] was right. Skills
    ship with the package now, so nothing legitimate produces zero of them. The
    realistic cause is a packaging regression -- the wheel's contents come from a
    `[tool.setuptools.package-data]` glob that is independent of this code, so
    the .md files can stop shipping while the .py files still do. Everything
    imports, every test that reads the checkout passes, and the agent is simply
    never told skills exist.

    Still not raised: this is on the agent's path, and a missing skill must not
    take down a tool call. `_tests/skills/test_packaging.py` is what actually
    prevents it; this is the breadcrumb for when something slips past.
    """
    global _warned_empty
    if _warned_empty:
        return
    _warned_empty = True
    logger.warning(
        "skills: no skills found in the package (%s). This is an install or "
        "packaging problem, not a configuration one -- list_skills will only "
        "return skills from %s.",
        detail,
        _local_dir() or "the local skills dir",
    )


def _scan_shipped() -> list[dict]:
    """Every readable packaged ``*.md``. Fail-open per file."""
    directory = _data_dir()
    try:
        names = sorted(p.name for p in directory.iterdir() if is_skill_file(p.name))
    except (FileNotFoundError, NotADirectoryError, OSError):
        _warn_empty_once(f"{directory} is missing or unreadable")
        return []
    if not names:
        _warn_empty_once(f"{directory} contains no *.md")
        return []

    out = []
    for name in names:
        text = _shipped_text(name)
        if text is None:
            continue
        try:
            entry = _entry(text, stem=name[:-3], origin=_ORIGIN_CATALOG)
        except Exception:
            logger.debug("skills: skipping shipped %s", name, exc_info=True)
            continue
        if entry is not None:
            entry["_file"] = name
            out.append(entry)
    if not out:
        # Files present but none usable: a different cause, same severity.
        _warn_empty_once(f"{len(names)} file(s) in {directory}, none usable")
    return out


# --------------------------------------------------------------------------- #
# Source 2: local skills (user-authored, ~/.config/biopb/skills)
# --------------------------------------------------------------------------- #
def _local_dir() -> Path | None:
    """Configured local-skills dir, else the standard one. Never created here."""
    configured = (_setting("services.skills_local_dir", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        from biopb._locations import mcp_skill_dir

        return mcp_skill_dir()
    except Exception:  # pragma: no cover - core SDK always present in practice
        logger.debug("skills: no local dir resolvable", exc_info=True)
        return None


def local_dir_status() -> str:
    """The body of the ``## Skills`` section of ``server_status``.

    The local dir is a default, not a constant -- ``services.skills_local_dir``
    and the config-tree env vars both move it -- and this is the only place an
    agent can read where a skill it writes has to land. Formatted here rather
    than in the status assembly so it is unit-testable.
    """
    directory = _local_dir()
    if directory is None:
        return "  local_dir: (unresolvable — biopb core SDK missing)"
    line = f"  local_dir: {directory}"
    try:
        if not directory.is_dir():
            # Not created on access anywhere in biopb, so an agent writing the
            # first local skill has to mkdir it -- say so rather than let a
            # write_text fail on a path the report just showed as the right one.
            return line + " (does not exist yet — mkdir it to write the first skill)"
        n = sum(
            1 for p in directory.glob("*.md") if p.is_file() and is_skill_file(p.name)
        )
    except OSError:
        return line + " (unreadable)"
    return line + f" ({n} skill{'' if n == 1 else 's'})"


def _local_entry(path: Path) -> dict | None:
    """One local file → a catalog-shaped entry, or ``None`` if unusable."""
    try:
        updated = date.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        updated = ""
    entry = _entry(
        path.read_text(encoding="utf-8"),
        stem=path.stem,
        origin=_ORIGIN_LOCAL,
        updated=updated,
    )
    if entry is not None:
        entry["_path"] = str(path)
    return entry


def _scan_local() -> list[dict]:
    """All readable ``*.md`` in the local dir. Fail-open per file (one bad skill
    never sinks discovery) and per directory (missing dir is the normal case)."""
    directory = _local_dir()
    if directory is None:
        return []
    try:
        paths = sorted(p for p in directory.glob("*.md") if p.is_file())
    except OSError:
        logger.debug("skills: local dir unreadable (fail-open)", exc_info=True)
        return []

    out = []
    for path in paths:
        if not is_skill_file(path.name):
            continue
        try:
            entry = _local_entry(path)
        except Exception:
            logger.debug("skills: skipping local %s", path.name, exc_info=True)
            continue
        if entry is not None:
            out.append(entry)
    return out


def _merge_local(shipped: list[dict]) -> list[dict]:
    """Union the shipped skills with local ones; local wins a shared id.

    A user iterating on their own version of a shipped skill expects to get
    theirs. Since skills now arrive only with a release, this is also the sole
    way a skill reaches a machine out of band.
    """
    local = _scan_local()
    if not local:
        return shipped
    merged = {entry["id"]: entry for entry in shipped}
    for entry in local:
        if entry["id"] in merged:
            logger.debug("skills: local %s shadows the shipped entry", entry["id"])
        merged[entry["id"]] = entry
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Source 3: kernel plugins (#92's "bring your own tool" surface)
# --------------------------------------------------------------------------- #
#
# A plugin is not a skill and this does not pretend otherwise -- rows carry
# `kind: "plugin"` and a namespace `handle` instead of a `skill://` uri. What it
# fixes is discovery. `server_status` already reports which plugins loaded, but
# only as bare names, and a *name* says nothing about capability: across eight
# benchmark arms, five were shown `## Kernel plugins  files: <name>` and not one
# followed it up, while all three that reached the same plugin through a skill
# body called it. So the thing that has to reach `list_skills` is the sentence
# the module already carries.
#
# The docstring is that sentence, and reading it here keeps the rule the plugin
# system was built on -- *the docstring is the doc, so code and doc cannot
# drift*. Nothing is imported: `biopb._kernel_plugins` parses with `ast`, which
# is also what lets this answer before the kernel exists, which is when the agent
# actually asks.
#
# Discovery is static, verification stays dynamic. This lists what *will* load;
# the loader is fail-open, so a `checklist: plugin:<name>` resolved against
# `server_status` remains the only thing that says a plugin is really there.


def _plugin_entry(
    name: str,
    *,
    summary: str = "",
    blurb: str = "",
    origin: str,
    version: str = "",
) -> dict | None:
    """One plugin row → a catalog-shaped entry, or ``None`` if unusable."""
    handle = name[:-3] if name.endswith(".py") else name
    handle = handle.strip()
    if not handle:
        return None
    # An undocumented plugin is still listed. Excluding it would hide a working
    # third-party tool completely, which is worse than a row that says "look at
    # this yourself" -- and the placeholder is honest about which it is.
    described = bool(summary)
    return {
        "id": handle,
        "title": summary or f"Kernel plugin `{handle}`",
        "description": (
            summary
            or f"Kernel plugin bound in the namespace as `{handle}`. It carries no "
            "module docstring, so call inspect_object to see what it provides."
        ),
        "tags": [],
        "version": version,
        "checklist": [],
        "updated": "",
        "origin": origin,
        "kind": KIND_PLUGIN,
        # How to reach it: it is already in the namespace under this name.
        "handle": handle,
        # Search text only. The opening paragraph, so a two-word query can match
        # a term the one-line summary had no room for.
        "_blurb": blurb if described else "",
    }


_entry_point_rows: list[dict] | None = None


def _entry_point_plugins(_kernel_plugins) -> list[dict]:
    """`entry_point_plugins()`, read once per process.

    The uncached call is the expensive half of a catalog read by an order of
    magnitude -- it stats and parses the metadata of *every* installed
    distribution -- and unlike the plugin *files* beside it, its answer cannot
    change while this interpreter runs: an entry point arrives with an install,
    which the running kernel would not pick up either. So the catalog stays
    live where liveness is real (a dropped-in ``.py``) and stops re-deriving
    where it is not.
    """
    global _entry_point_rows
    if _entry_point_rows is None:
        _entry_point_rows = _kernel_plugins.entry_point_plugins()
    return _entry_point_rows


def _scan_plugins() -> list[dict]:
    """Kernel plugins as catalog entries. Fail-open, like every other source.

    Gated by ``namespace_enabled`` as well as its own switch: if the plugins are
    not going to be loaded into the namespace, advertising them would send the
    agent after a handle that will not be bound.
    """
    if not _setting("services.skills_index_plugins", True):
        return []
    if not _setting("services.namespace_enabled", True):
        return []
    try:
        from biopb import _kernel_plugins
    except Exception:  # pragma: no cover - core SDK always present in practice
        logger.debug("skills: no kernel-plugin reader available", exc_info=True)
        return []

    out: list[dict] = []
    try:
        for row in _kernel_plugins.startup_files():
            entry = _plugin_entry(
                str(row.get("name") or ""),
                summary=str(row.get("summary") or ""),
                blurb=str(row.get("blurb") or ""),
                origin=_ORIGIN_PLUGIN_FILE,
            )
            if entry is not None:
                out.append(entry)
    except Exception:
        logger.debug("skills: plugin-file scan failed (fail-open)", exc_info=True)

    try:
        for row in _entry_point_plugins(_kernel_plugins):
            # No docstring here by construction: reading one would mean importing
            # the module, which this side does not do.
            entry = _plugin_entry(
                str(row.get("name") or ""),
                origin=_ORIGIN_PLUGIN_PACKAGE,
                version=str(row.get("dist") or ""),
            )
            if entry is not None:
                out.append(entry)
    except Exception:
        logger.debug("skills: plugin entry-point scan failed", exc_info=True)
    return out


def _merge_plugins(skills: list[dict], plugins: list[dict]) -> list[dict]:
    """Append the plugins whose handle no curated skill has already claimed.

    A skill wins the id: it is the reviewed artifact, it has a body to read, and
    two rows under one id would make `skill://<id>` ambiguous.
    """
    taken = {entry["id"] for entry in skills}
    keep = []
    for entry in plugins:
        if entry["id"] in taken:
            logger.debug(
                "skills: skill %s shadows the plugin of that name", entry["id"]
            )
            continue
        taken.add(entry["id"])
        keep.append(entry)
    return skills + keep


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _config():
    from .._config import CONFIG

    return CONFIG.as_dict()


# Sentinel: "no default supplied" -> defer to get_setting's DEFAULT_CONFIG
# fallback, so a skills default is declared once (in _config.py) and never
# restated here (a restated literal is how skills_enabled's default silently
# diverged once already).
_UNSET = object()


def _setting(path: str, default=_UNSET):
    from .._config import get_setting

    if default is _UNSET:
        return get_setting(_config(), path)
    return get_setting(_config(), path, default)


def load_catalog() -> list[dict]:
    """Return the resolved skills list (metadata only), fail-open.

    The shipped set merged with the user's local dir, both read fresh, plus the
    kernel plugins. Returns ``[]`` when the feature is disabled or every file is
    unusable. Not cached: the reads are a handful of small local files, and
    re-reading is what makes a local edit live immediately — the authoring loop
    would be unusable if a draft needed a restart to appear.

    **``skills_enabled`` gates the plugins too**, and that is deliberate rather
    than incidental. It is the switch the benchmark's ablation flips
    (``--bench-skills=false``), and the run then asserts the catalog came back
    empty; a plugin row surviving it would both fail that assertion and quietly
    give the ablated arm back a form of the discovery it is meant to be without.
    """
    if not _setting("services.skills_enabled"):
        return []
    return _merge_plugins(_merge_local(_scan_shipped()), _scan_plugins())


# --------------------------------------------------------------------------- #
# Public: discovery tool + resource read
# --------------------------------------------------------------------------- #
def _search_text(s: dict) -> str:
    """The haystack one skill is matched against.

    The ``id`` is in it, with hyphens opened out to spaces, because a user who
    names the skill ("flatfield") is making the most specific request there is;
    matching only prose would miss it. Everything else is what the ``list_skills``
    docstring advertises: title, description, tags.

    A plugin row adds ``_blurb`` — its docstring's opening paragraph — and needs
    it. A skill's ``description`` is written to be retrieved; a module's first
    line is written for someone already reading the file, so it carries the
    subject and rarely the verb: ``["background", "subtract"]`` misses
    "Rolling-ball background subtraction (Sternberg 1983), the fast ImageJ port"
    on the second term. ``_blurb`` is empty for skills, so their matching is
    unchanged.
    """
    return " ".join(
        (
            s["id"].replace("-", " "),
            s["title"],
            s["description"],
            " ".join(s["tags"]),
            s.get("_blurb", ""),
        )
    ).lower()


def search_terms(keywords: Sequence[str] | str) -> list[str]:
    """The terms *keywords* actually searches for. Each element is split.

    **A caller that passes one string gets it split, not matched whole.** The
    parameter is a list to say "keywords, not a sentence" in the signature
    itself, but a model handed a list will still sometimes put a phrase in one
    element — ``["stitch tiles"]``, or the whole task in ``["how do I stitch
    these tiles"]`` — and matching that as a single substring would fail against
    a skill whose title says "stitch" and whose description says "tiles". A bare
    string is accepted for the same reason and split the same way, which is also
    what keeps this module's callers (and their tests) working unchanged.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    return [term for k in keywords for term in str(k).lower().split()]


def _matches(haystack: str, terms: Sequence[str]) -> bool:
    """Every term present in *haystack* — the narrowing filter list_skills documents."""
    return all(t in haystack for t in terms)


def list_skills(keywords: Sequence[str] | str = ()) -> list[dict]:
    """Filter the catalog by *keywords* over id/title/description/tags.

    Empty returns everything. **Every keyword must appear** somewhere in that
    text; order and adjacency do not matter. That is a narrowing filter, so each
    keyword added can only ever remove results — which is why the tool asks for
    a few, and why a caller that gets nothing back should retry with fewer
    rather than conclude the catalog is empty.

    Substring terms, not tokens: "measure" is meant to find "measurements". The
    cost is that a term also matches mid-word, which at catalog scale is a
    trade worth making.

    Returns a list of metadata dicts. Every row carries ``kind``: a
    ``"skill"`` has a ``uri`` (``skill://<id>``) to read for the full workflow,
    a ``"plugin"`` has a ``handle`` — the name it is already bound under in the
    kernel namespace — and is inspected with ``inspect_object`` instead.

    Skills sort before plugins, each group by title: a curated procedure is the
    better answer when both match, so it should not depend on where the
    alphabet happens to put it.
    """
    skills = load_catalog()
    terms = search_terms(keywords)
    if terms:
        # The haystack once per skill, not once per (skill, term): `search_terms`
        # splits phrases, so the term count is not the caller's list length.
        skills = [s for s in skills if _matches(_search_text(s), terms)]
    out = []
    for s in sorted(
        skills, key=lambda s: (s.get("kind") != KIND_SKILL, s["title"].lower())
    ):
        row = {
            "id": s["id"],
            "title": s["title"],
            "description": s["description"],
            "tags": s["tags"],
            "version": s["version"],
            "updated": s["updated"],
            "checklist": s["checklist"],
            # "local" = the user's own file, unreviewed. The agent should be
            # able to say so rather than let a draft pass as curated.
            "origin": s.get("origin", _ORIGIN_CATALOG),
            "kind": s.get("kind", KIND_SKILL),
        }
        if row["kind"] == KIND_PLUGIN:
            row["handle"] = s["handle"]
        else:
            row["uri"] = f"skill://{s['id']}"
        out.append(row)
    return out


def get_skill_body(skill_id: str) -> str:
    """Return the full markdown body for *skill_id*, fail-open.

    Frontmatter is stripped (the metadata is already in the entry). On an unknown
    id or an unreadable file, returns a short human-readable string rather than
    raising — the value is agent context, not executed code.
    """
    entry = next((s for s in load_catalog() if s["id"] == skill_id), None)
    if entry is None:
        return (
            f"No skill '{skill_id}' in the catalog. "
            "Call list_skills to list available skills."
        )

    if entry.get("kind") == KIND_PLUGIN:
        # Reachable: list_skills returns plugins too, and a `skill://<id>` read
        # is the habit that row is sitting next to. Say what it is instead of
        # reporting it missing -- it exists, it is just not a document.
        handle = entry.get("handle", skill_id)
        return (
            f"'{skill_id}' is a kernel plugin, not a skill, so it has no workflow "
            f"body. It is already bound in the kernel namespace as `{handle}`; "
            f"call inspect_object('{handle}') for its callables and their "
            "signatures — the docstring is the documentation. Confirm it actually "
            "loaded in server_status under '## Kernel plugins' before relying on it."
        )

    local_path = entry.get("_path")
    if local_path:
        try:
            return _strip_frontmatter(Path(local_path).read_text(encoding="utf-8"))
        except OSError:
            logger.debug("skills: local body unreadable %s", local_path, exc_info=True)
            return f"Local skill '{skill_id}' could not be read from {local_path}."

    text = _shipped_text(entry.get("_file") or f"{skill_id}.md")
    if text is not None:
        return _strip_frontmatter(text)

    return f"Could not read the body for skill '{skill_id}'."
