"""Curated skills catalog — discovery + retrieval (skill-interface design, P2).

The agent's counterpart to the published skills catalog (``docs/skill-interface.md``).
Two entry points are wired into ``_server.py``:

* :func:`find_skills` — the ``find_skills`` tool: query the catalog metadata and
  return a tailored subset (each carrying its ``skill://<id>`` URI).
* :func:`get_skill_body` — the ``skill://{id}`` resource read: lazily fetch a
  skill's full markdown body, integrity-check it against the catalog ``sha256``,
  and cache it.

**Fail-open, like** :mod:`._update`. Every network/parse error degrades quietly:
the catalog resolves *network → on-disk cache → bundled snapshot*, and a body
resolves *sha-keyed cache → network → bundled snapshot*. Nothing here raises into
a tool call or the bootstrap; failures are debug-logged and surface to the agent
as an empty result or a short explanatory string.

The bundled snapshot under ``_skills_bundle/`` is a *point-in-time copy* of the
biopb-site ``skills/`` tree (``catalog.json`` + bodies), regenerated from that
repo. It is the last-resort floor so discovery and retrieval still work fully
offline / before the catalog is first published; the network copy supersedes it
whenever reachable.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.request
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

# Highest catalog.json schema this consumer understands. A network catalog
# declaring a higher version is treated as unknown-future and rejected (fall
# back to last-good / bundle) rather than mis-parsed — mirrors the site's
# "conservative in what it publishes" contract (skill-interface.md §5.3).
CATALOG_VERSION = 1

# Short on purpose: catalog discovery must never delay a tool call (§3c). Body
# fetches reuse it. Not a config knob — it is a fail-open safety bound, not a
# tuning parameter.
_FETCH_TIMEOUT = 8.0

_BUNDLE_PKG = "biopb_mcp.mcp"
_BUNDLE_DIR = "_skills_bundle"
_USER_AGENT = "biopb-mcp-skills"

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

# Module cache for the resolved catalog (skills list). Guarded by _lock; TTL from
# config. Bodies are cached on disk (sha-keyed), not here — they are larger and
# read one at a time.
_lock = threading.Lock()
_cache = {"at": 0.0, "skills": None}


# --------------------------------------------------------------------------- #
# Paths / IO helpers
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
    """``~/.cache/biopb-mcp/skills`` (created). Runtime cache, not user config —
    so the XDG *cache* tree, distinct from ~/.config/biopb-mcp and the log dir."""
    d = Path.home() / ".cache" / "biopb-mcp" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically: a uniquely-named temp file in the same
    directory, then ``os.replace``.

    The skills cache is shared by every concurrent biopb-mcp session — since
    de-daemonization each MCP client owns its own *process*, so a plain
    ``write_bytes`` would let one session read a half-written file another is
    mid-write (and two writers interleave). ``os.replace`` is atomic on POSIX and
    Windows for same-filesystem paths, so a reader always sees either the old
    complete file or the new one, with no cross-process lock: bodies are
    content-addressed (racing writers write identical bytes) and the catalog is a
    cache (last-writer-wins is fine). ``mkstemp`` gives each writer its own temp,
    so writers never clobber each other's partial file either. Raises ``OSError``
    on failure (caller logs and degrades); the temp is cleaned up first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _http_get(url: str, timeout: float) -> bytes:
    """GET *url*, returning raw bytes. Raises on any network/HTTP error (the
    caller fails open)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured https catalog
        return resp.read()


def _bundle_text(name: str) -> str | None:
    """Read a bundled snapshot file, or ``None`` if absent/unreadable."""
    try:
        return (
            resources.files(_BUNDLE_PKG)
            .joinpath(_BUNDLE_DIR, name)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError, UnicodeError):
        # UnicodeError: a corrupt / non-UTF8 bundle file (a ValueError subclass,
        # not an OSError) must also fail open, not crash the read.
        return None


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``--- … ---`` YAML frontmatter block; the agent context
    wants the workflow prose, not the metadata already carried in the catalog."""
    return _FRONTMATTER.sub("", text, count=1).lstrip()


# --------------------------------------------------------------------------- #
# Catalog parsing (tolerant reader — §5.4)
# --------------------------------------------------------------------------- #
def _accept_catalog(raw: bytes) -> list[dict]:
    """Parse catalog bytes into a normalized skills list.

    Raises on a structurally unusable payload (not a dict, or an unknown-future
    ``catalog_version``) so the caller falls back to cache/bundle. A single
    malformed *entry* is skipped, never fatal (§5.4).
    """
    cat = json.loads(raw)
    if not isinstance(cat, dict):
        raise ValueError("catalog is not a JSON object")
    cv = cat.get("catalog_version")
    if isinstance(cv, int) and cv > CATALOG_VERSION:
        raise ValueError(f"catalog_version {cv} newer than supported {CATALOG_VERSION}")
    out = []
    for entry in cat.get("skills") or []:
        norm = _normalize_entry(entry)
        if norm is not None:
            out.append(norm)
    return out


def _normalize_entry(entry) -> dict | None:
    """Coerce one catalog entry to the fields the agent needs, or ``None`` if it
    lacks the two load-bearing fields (``id``, ``description``). Unknown keys are
    ignored; optionals are defaulted."""
    if not isinstance(entry, dict):
        return None
    skill_id = entry.get("id")
    description = entry.get("description")
    if not isinstance(skill_id, str) or not isinstance(description, str):
        return None
    if not skill_id.strip() or not description.strip():
        return None
    tags = entry.get("tags")
    tags = [str(t) for t in tags] if isinstance(tags, list) else []
    requires = entry.get("requires")
    requires = [str(r) for r in requires] if isinstance(requires, list) else []
    return {
        "id": skill_id,
        "title": str(entry.get("title") or skill_id),
        "description": description,
        "tags": tags,
        "version": str(entry.get("version") or ""),
        "requires": requires,
        "updated": str(entry.get("updated") or ""),
        "url": str(entry.get("url") or ""),
        "sha256": str(entry.get("sha256") or ""),
    }


# --------------------------------------------------------------------------- #
# Catalog resolution: network -> disk cache -> bundle
# --------------------------------------------------------------------------- #
def _config():
    from .._config import CONFIG

    return CONFIG.as_dict()


def _setting(path: str, default=None):
    from .._config import get_setting

    return get_setting(_config(), path, default)


def load_catalog(*, force: bool = False) -> list[dict]:
    """Return the resolved skills list (metadata only), fail-open.

    Honors the in-memory TTL cache first (``mcp.services.skills.cache_ttl``),
    then tries the network, then the on-disk cache (regardless of age — a stale
    cache beats nothing), then the bundled snapshot. Returns ``[]`` when the
    feature is disabled or every source fails.
    """
    if not _setting("mcp.services.skills.enabled", True):
        return []
    ttl = _setting("mcp.services.skills.cache_ttl", 3600)
    url = _setting("mcp.services.skills.catalog_url", "")

    with _lock:
        if (
            not force
            and _cache["skills"] is not None
            and (time.time() - _cache["at"]) < ttl
        ):
            return _cache["skills"]

        skills = _resolve_catalog(url)
        _cache["skills"] = skills
        _cache["at"] = time.time()
        return skills


def _resolve_catalog(url: str) -> list[dict]:
    cache_file = _cache_dir() / "catalog.json"

    if url:
        try:
            raw = _http_get(url, _FETCH_TIMEOUT)
            skills = _accept_catalog(raw)
            try:
                _atomic_write(cache_file, raw)
            except OSError:
                logger.debug("skills: could not write catalog cache", exc_info=True)
            return skills
        except Exception:
            logger.debug(
                "skills: network catalog fetch failed (fail-open)", exc_info=True
            )

    try:
        return _accept_catalog(cache_file.read_bytes())
    except Exception:
        logger.debug("skills: no usable on-disk catalog cache", exc_info=True)

    bundled = _bundle_text("catalog.json")
    if bundled is not None:
        try:
            return _accept_catalog(bundled.encode("utf-8"))
        except Exception:
            logger.debug("skills: bundled catalog unusable", exc_info=True)
    return []


# --------------------------------------------------------------------------- #
# Public: discovery tool + resource read
# --------------------------------------------------------------------------- #
def find_skills(query: str = "") -> list[dict]:
    """Filter the catalog by *query* over title/description/tags (empty = all).

    Returns a list of metadata dicts, each with a ``uri`` (``skill://<id>``) the
    caller reads for the full workflow. Sorted by title.
    """
    skills = load_catalog()
    q = (query or "").strip().lower()
    if q:
        skills = [
            s
            for s in skills
            if q
            in (s["title"] + " " + s["description"] + " " + " ".join(s["tags"])).lower()
        ]
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
                "uri": f"skill://{s['id']}",
            }
        )
    return out


def get_skill_body(skill_id: str) -> str:
    """Return the full markdown body for *skill_id*, fail-open.

    Resolution: sha-keyed disk cache → network (verify sha, cache) → bundled
    snapshot. Frontmatter is stripped (metadata lives in the catalog). On an
    unknown id or total failure, returns a short human-readable string rather
    than raising — the value is agent context, not executed code.
    """
    entry = next((s for s in load_catalog() if s["id"] == skill_id), None)
    if entry is None:
        return (
            f"No skill '{skill_id}' in the catalog. "
            "Call find_skills to list available skills."
        )

    sha = entry.get("sha256") or ""
    body_dir = _cache_dir() / "bodies"

    # 1. sha-keyed cache. Re-verify the sha on read (the sha is both integrity
    #    check and cache key, §1): a truncated/corrupt file — e.g. a crash
    #    mid-write, or another concurrent session — is treated as a miss and
    #    refetched, never returned as content. Atomic writes below make this rare,
    #    but the check costs a hash of a small file and closes the hole entirely.
    if sha:
        cached = body_dir / f"{sha}.md"
        try:
            raw = cached.read_bytes()
        except OSError:
            raw = None
        if raw is not None and hashlib.sha256(raw).hexdigest() == sha:
            return _strip_frontmatter(raw.decode("utf-8"))

    # 2. network, verified against the catalog sha.
    url = entry.get("url") or ""
    if url:
        try:
            raw = _http_get(url, _FETCH_TIMEOUT)
            got = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8")
            if sha and got != sha:
                # Drift between catalog and body: don't cache under the expected
                # key, but still hand back the content (context, not code).
                logger.debug(
                    "skills: sha mismatch for %s (want %s got %s)", skill_id, sha, got
                )
            elif sha:
                try:
                    # Store the exact fetched bytes so the sha-verified read above
                    # matches; atomic so a concurrent reader never sees a partial.
                    _atomic_write(body_dir / f"{sha}.md", raw)
                except OSError:
                    logger.debug("skills: could not cache body", exc_info=True)
            return _strip_frontmatter(text)
        except Exception:
            logger.debug(
                "skills: body fetch failed for %s (fail-open)", skill_id, exc_info=True
            )

    # 3. bundled snapshot.
    bundled = _bundle_text(f"{skill_id}.md")
    if bundled is not None:
        return _strip_frontmatter(bundled)

    return (
        f"Could not retrieve the body for skill '{skill_id}' "
        "(offline, and no cached or bundled copy is available)."
    )
