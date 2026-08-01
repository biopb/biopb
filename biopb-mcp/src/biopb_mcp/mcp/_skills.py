"""Curated skills catalog — discovery + retrieval (skill-interface design, P2).

Two entry points are wired into ``_server.py``:

* :func:`find_skills` — the ``find_skills`` tool: query the catalog metadata and
  return a tailored subset (each carrying its ``skill://<id>`` URI).
* :func:`get_skill_body` — the ``skill://{id}`` resource read: the skill's full
  markdown body, frontmatter stripped.

**Skills ship with the package** (``_skills_data/*.md``) and are never fetched.
A skill is documentation about a specific runtime version — it quotes an API,
assumes a namespace handle, depends on packages resolving a particular way — so
it rides that version's upgrade cycle rather than being served to every release
in the field at once. The cost is that a skill fix needs a release; the local
dir below is the escape hatch. See ``docs/skill-testing.md`` §9.

There is no catalog index. The frontmatter *is* the metadata, and this module
parses it, so there is no generated file to disagree with the bodies.

**Two sources.** User-authored skills in ``~/.config/biopb/skills/*.md`` **merge**
with the shipped set (local wins a shared id, and every entry carries
``origin``). Both are re-read on every call: a skill the user is editing must
show up without a restart, and the shipped set is a handful of small files.

**Fail-open.** A malformed or unreadable file is skipped and debug-logged, never
fatal — one bad skill must not sink discovery, and nothing here raises into a
tool call or the bootstrap.
"""

import logging
import re
from datetime import date
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

# The shipped skills, as package data (see pyproject [tool.setuptools.package-data]).
_DATA_PKG = "biopb_mcp.mcp"
_DATA_DIR = "_skills_data"

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

# Origin travels with every entry so the agent can tell a personal draft from a
# reviewed one.
_ORIGIN_LOCAL = "local"
_ORIGIN_CATALOG = "catalog"

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
    requires = fm.get("requires")
    requires = [str(r) for r in requires] if isinstance(requires, list) else []

    return {
        "id": skill_id,
        "title": title,
        "description": description,
        "tags": tags,
        "version": str(fm.get("version") or ""),
        "requires": requires,
        "updated": str(fm.get("updated") or updated or ""),
        "origin": origin,
    }


# --------------------------------------------------------------------------- #
# Source 1: the shipped skills
# --------------------------------------------------------------------------- #
def _data_dir():
    """The packaged ``_skills_data`` directory as a Traversable, or ``None``."""
    try:
        return resources.files(_DATA_PKG).joinpath(_DATA_DIR)
    except (ModuleNotFoundError, OSError):  # pragma: no cover - package is present
        logger.debug("skills: package data dir unresolvable", exc_info=True)
        return None


def _shipped_text(name: str) -> str | None:
    """Read one packaged skill file, or ``None`` if absent/unreadable."""
    directory = _data_dir()
    if directory is None:
        return None
    try:
        return directory.joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        # UnicodeError: a corrupt / non-UTF8 file (a ValueError subclass, not an
        # OSError) must also fail open, not crash the read.
        return None


def _scan_shipped() -> list[dict]:
    """Every readable packaged ``*.md``. Fail-open per file."""
    directory = _data_dir()
    if directory is None:
        return []
    try:
        names = sorted(
            p.name
            for p in directory.iterdir()
            if p.name.endswith(".md") and not p.name.startswith("_")
        )
    except (FileNotFoundError, OSError):
        logger.debug("skills: package data dir unreadable (fail-open)", exc_info=True)
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
        if path.stem.startswith("_"):  # private, like the kernel-plugin loader
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

    The shipped set merged with the user's local dir, both read fresh. Returns
    ``[]`` when the feature is disabled or every file is unusable. Not cached:
    the reads are a handful of small local files, and re-reading is what makes a
    local edit live immediately — the authoring loop would be unusable if a
    draft needed a restart to appear.
    """
    if not _setting("services.skills_enabled"):
        return []
    return _merge_local(_scan_shipped())


# --------------------------------------------------------------------------- #
# Public: discovery tool + resource read
# --------------------------------------------------------------------------- #
def _search_text(s: dict) -> str:
    """The haystack one skill is matched against.

    The ``id`` is in it, with hyphens opened out to spaces, because a user who
    names the skill ("flatfield") is making the most specific request there is;
    matching only prose would miss it. Everything else is what the ``find_skills``
    docstring advertises: title, description, tags.
    """
    return " ".join(
        (
            s["id"].replace("-", " "),
            s["title"],
            s["description"],
            " ".join(s["tags"]),
        )
    ).lower()


def find_skills(query: str = "") -> list[dict]:
    """Filter the catalog by *query* over id/title/description/tags (empty = all).

    Every whitespace-separated term must appear somewhere in that text; order and
    adjacency do not matter. The tool docstring offers the agent multi-word
    queries ("segment nuclei", "measure labels"), and matching the query as one
    substring could not serve them — "stitch tiles" missed a skill whose title
    says "stitch" and whose description says "tiles". Since a whole-query
    substring hit implies every term hits, this only ever widens the result set.

    Substring terms, not tokens: "measure" is meant to find "measurements". The
    cost is that a term also matches mid-word, which at catalog scale is a
    trade worth making. Natural-language sentences are still out of scope --
    "how do I stitch tiles?" carries terms no description contains.

    Returns a list of metadata dicts, each with a ``uri`` (``skill://<id>``) the
    caller reads for the full workflow. Sorted by title.
    """
    skills = load_catalog()
    terms = (query or "").lower().split()
    if terms:
        skills = [s for s in skills if all(t in _search_text(s) for t in terms)]
    out = []
    for s in sorted(skills, key=lambda s: s["title"].lower()):
        out.append(
            {
                "id": s["id"],
                "title": s["title"],
                "description": s["description"],
                "tags": s["tags"],
                "version": s["version"],
                "updated": s["updated"],
                "requires": s["requires"],
                # "local" = the user's own file, unreviewed. The agent should be
                # able to say so rather than let a draft pass as curated.
                "origin": s.get("origin", _ORIGIN_CATALOG),
                "uri": f"skill://{s['id']}",
            }
        )
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
            "Call find_skills to list available skills."
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
