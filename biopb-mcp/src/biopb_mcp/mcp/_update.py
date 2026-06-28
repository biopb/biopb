"""Kernel-start auto-updater — version check (issue #87).

Pure, **fail-open** check for whether a newer biopb ``release-v*`` *deployment*
is available than the one installed. Every error path — disabled, a dev/editable
build, a missing marker, offline/DNS/TLS/HTTP/rate-limit, or a parse failure —
degrades to "no update" and is logged at debug. Nothing here raises into the
bootstrap, and nothing here applies an update or pops UI; this module only
answers *"is there a newer deployment than the one installed?"*.

Design notes that pin this to the current release model (``docs/release-model.md``):

* The product ships on the ``biopb/biopb`` ``release-v*`` line, NOT on
  ``/releases/latest`` (repo-wide; would surface a ``v*``/``mcp-v*`` PyPI library
  tag). We list releases and select the highest matching ``release-vX.Y.Z``,
  skipping prereleases unless the user opts into the ``prerelease`` channel —
  mirroring the installer's selection exactly.
* The comparison baseline is the installer-written marker at
  ``~/.config/biopb/release.version`` (the deployment's ``versions.json``
  ``release`` field). ``biopb_mcp.__version__`` is a decoupled *library* version
  (its own ``mcp-v*`` cadence) and is deliberately NOT the comparison basis; it
  is only consulted to suppress checks on a dev/editable build.
* The remote candidate version is taken from the chosen release's **tag** (strip
  ``release-v``), which equals that release's ``versions.json`` ``release`` by
  construction (``release.yaml`` derives the release version from the tag) — so
  no second per-release asset fetch is needed in the viewer-start hot path.
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The deployment release line. Fixed (not configurable): the installer and the
# release pipeline both hard-code this prefix (docs/release-model.md).
RELEASE_TAG_PREFIX = "release-v"

_GITHUB_API = "https://api.github.com"


@dataclass(frozen=True)
class UpdateInfo:
    """A newer deployment is available.

    ``current``/``latest`` are clean PEP 440 versions; ``tag`` is the GitHub tag
    (``release-v<latest>``); ``html_url`` is the release page (for a notify-only
    branch or a "view release" action).
    """

    current: str
    latest: str
    tag: str
    html_url: str


def marker_path() -> Path:
    """Path of the installer-written installed-release marker.

    The biopb *umbrella* config dir (``~/.config/biopb``), NOT
    ``~/.config/biopb-mcp`` — the marker is a whole-deployment fact written by
    ``install/install.sh`` / ``install/biopb-engine.ps1``. Keep this in sync with
    them.
    """
    return Path.home() / ".config" / "biopb" / "release.version"


def read_installed_version(path: Path | None = None) -> str | None:
    """The recorded installed deployment version, or ``None`` if unavailable.

    ``None`` covers a missing marker (e.g. a from-source dev checkout that never
    ran the installer) and any read error — both of which suppress the check.
    """
    p = path or marker_path()
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def is_dev_version(version: str) -> bool:
    """Whether ``version`` is a setuptools_scm dev/editable build.

    ``0.7.3.dev6+g1a2b3c4`` and ``0.7.3+g1a2b3c4`` are a developer's own checkout
    between releases — never nag those. A clean ``X.Y.Z`` (or a ``…rcN``
    prerelease) is not a dev build.
    """
    return ".dev" in version or "+" in version


def tag_to_version(tag: str, prefix: str = RELEASE_TAG_PREFIX) -> str:
    """``release-v0.6.6`` -> ``0.6.6``; leave a non-matching tag untouched."""
    return tag[len(prefix):] if tag.startswith(prefix) else tag


def select_latest(releases, *, prefix=RELEASE_TAG_PREFIX, allow_prerelease=False):
    """Pure: pick the highest-version release on the deployment line.

    ``releases`` is the GitHub ``GET /releases`` payload (a list of dicts).
    Mirrors the installer: consider only tags ``release-vX.Y.Z``; skip drafts;
    skip prereleases (both the PEP 440 ``…a/b/rc`` form and GitHub's
    ``prerelease`` flag) unless ``allow_prerelease``. Returns the chosen release
    dict, or ``None`` when nothing qualifies. Tolerant of malformed entries
    (missing/unparseable tags are skipped, not fatal).
    """
    from packaging.version import InvalidVersion, Version

    best = None
    best_ver = None
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        tag = rel.get("tag_name") or ""
        if not tag.startswith(prefix):
            continue
        try:
            ver = Version(tag_to_version(tag, prefix))
        except InvalidVersion:
            continue
        if not allow_prerelease and (ver.is_prerelease or rel.get("prerelease")):
            continue
        if best_ver is None or ver > best_ver:
            best_ver, best = ver, rel
    return best


def _http_json(url: str, timeout: float):
    """GET ``url`` and parse JSON. Raises on any network/parse error (the caller
    fails open)."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "biopb-mcp-update-check",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https GitHub API
        return json.loads(resp.read().decode("utf-8"))


def _running_library_version() -> str:
    """``biopb_mcp.__version__`` read as an attribute (so tests can monkeypatch
    it), empty string if unavailable."""
    try:
        import biopb_mcp

        return getattr(biopb_mcp, "__version__", "") or ""
    except Exception:
        return ""


def check_for_update(config: dict, *, installed: str | None = None) -> UpdateInfo | None:
    """Return an :class:`UpdateInfo` if a newer deployment is available, else
    ``None``. Never raises — every failure mode yields ``None`` (debug-logged).

    ``installed`` overrides the marker read (for tests / callers that already
    know the baseline).
    """
    try:
        from packaging.version import InvalidVersion, Version

        from .._config import get_setting

        if not get_setting(config, "mcp.update.enabled"):
            return None

        # Suppress on a dev/editable build of the running code — a developer
        # working from source manages their own versions.
        running = _running_library_version()
        if running and is_dev_version(running):
            logger.debug("update check: dev/editable build (%s); skipping", running)
            return None

        current = installed if installed is not None else read_installed_version()
        if not current:
            logger.debug("update check: no installed-release marker; skipping")
            return None

        repo = get_setting(config, "mcp.update.repo")
        channel = get_setting(config, "mcp.update.channel")
        timeout = get_setting(config, "mcp.update.timeout")
        skipped = get_setting(config, "mcp.update.skipped_version") or ""
        allow_prerelease = channel == "prerelease"

        releases = _http_json(
            f"{_GITHUB_API}/repos/{repo}/releases?per_page=100", timeout
        )
        if not isinstance(releases, list):
            # An error response (rate-limit, 404) is a JSON object, not a list.
            logger.debug("update check: unexpected releases payload; skipping")
            return None

        rel = select_latest(releases, allow_prerelease=allow_prerelease)
        if rel is None:
            return None

        latest = tag_to_version(rel.get("tag_name") or "")
        try:
            if Version(latest) <= Version(current):
                return None
        except InvalidVersion:
            logger.debug(
                "update check: unparseable version(s) latest=%r current=%r; skipping",
                latest,
                current,
            )
            return None

        if skipped and latest == skipped:
            logger.debug("update check: %s is the skipped version; skipping", latest)
            return None

        return UpdateInfo(
            current=current,
            latest=latest,
            tag=rel.get("tag_name") or "",
            html_url=rel.get("html_url") or "",
        )
    except Exception:
        # Fail open on EVERYTHING (offline/DNS/TLS/HTTP/rate-limit/parse/etc.):
        # an update check must never disturb kernel bring-up.
        logger.debug("update check failed (fail-open)", exc_info=True)
        return None
