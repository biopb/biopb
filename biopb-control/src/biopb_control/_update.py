"""Whether a newer biopb release exists than the one installed.

The baseline is ``release.version`` in the config dir, written by the installer
(a from-source or conda install has none, and is never checked). The candidate is
the highest ``release-vX.Y.Z`` on the product repo's releases, which is the line
the installer follows too; a release candidate counts only when the installed
version is itself one. Every failure (offline, rate limit, bad payload) leaves "no
update" and is logged at debug: the check must never get in the control's way.
``BIOPB_UPDATE_CHECK=0`` turns it off.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_UPDATE_CHECK = "BIOPB_UPDATE_CHECK"

# The product ships on this repo's `release-v*` tags, not on /releases/latest
# (which would surface a `v*` SDK tag).
REPO = "biopb/biopb"
TAG_PREFIX = "release-v"

# GitHub's unauthenticated limit is 60 requests an hour per address.
CHECK_INTERVAL = 6 * 3600
TIMEOUT = 5.0


# The installer is fetched from here every time, never from a URL that came in
# with a request. It is pinned to the release published with it, so re-running it
# is the upgrade.
INSTALL_SH_URL = "https://biopb.org/install.sh"
INSTALL_PS1_URL = "https://biopb.org/install.ps1"

ENV_RESULT = "BIOPB_UPDATE_RESULT"

# An installer that has not finished after this long is taken to be dead.
INSTALL_STALE = 30 * 60


@dataclass(frozen=True)
class UpdateInfo:
    """A newer release exists: versions are clean PEP 440, ``url`` is its page."""

    current: str
    latest: str
    url: str


def enabled() -> bool:
    return os.environ.get(ENV_UPDATE_CHECK, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def installed_version() -> str | None:
    """The installer's recorded release, or ``None`` when there is no marker."""
    from biopb._config import locations

    try:
        text = (locations.config_dir() / "release.version").read_text(encoding="utf-8")
    except OSError:
        return None
    return text.strip() or None


def is_dev_version(version: str) -> bool:
    """A setuptools_scm dev or editable build (``0.7.3.dev6+g1a2b3c4``): a
    developer's own checkout, never nagged."""
    return ".dev" in version or "+" in version


def select_latest(releases, *, allow_prerelease: bool = False) -> dict | None:
    """The highest ``release-vX.Y.Z`` in a GitHub ``/releases`` payload, skipping
    drafts and (unless allowed) prereleases, as the installer does."""
    from packaging.version import InvalidVersion, Version

    best, best_ver = None, None
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        tag = rel.get("tag_name") or ""
        if not tag.startswith(TAG_PREFIX):
            continue
        try:
            ver = Version(tag[len(TAG_PREFIX) :])
        except InvalidVersion:
            continue
        if not allow_prerelease and (ver.is_prerelease or rel.get("prerelease")):
            continue
        if best_ver is None or ver > best_ver:
            best, best_ver = rel, ver
    return best


def _fetch_releases() -> object:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases?per_page=100",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "biopb-control-update-check",
        },
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read().decode("utf-8"))


def check(running: str, *, installed: str | None = None) -> UpdateInfo | None:
    """The newer release, or ``None``. ``running`` is this control's own version,
    consulted only to skip a dev build. Never raises."""
    try:
        from packaging.version import Version

        if running and is_dev_version(running):
            return None
        current = installed if installed is not None else installed_version()
        if not current:
            return None
        releases = _fetch_releases()
        if not isinstance(releases, list):  # an error body is an object
            return None
        rel = select_latest(releases, allow_prerelease=Version(current).is_prerelease)
        if rel is None:
            return None
        latest = (rel.get("tag_name") or "")[len(TAG_PREFIX) :]
        if Version(latest) <= Version(current):
            return None
        return UpdateInfo(current, latest, rel.get("html_url") or "")
    except Exception:  # noqa: BLE001 - fail open on everything
        logger.debug("update check failed", exc_info=True)
        return None


def result_file() -> Path:
    """Where the installer records how its last run ended (``update-result.json``:
    ``status`` running / ok / failed, ``version``, ``time``)."""
    from biopb._config import locations

    return locations.state_dir() / "update-result.json"


def last_result() -> dict | None:
    """The installer's last record, or ``None``."""
    try:
        data = json.loads(result_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class InstallRunning(Exception):
    """An installer started earlier has not finished."""


def _installer_argv() -> list[str]:
    # A failed fetch runs no installer to record the failure, so the shell does.
    failed = '{"status": "failed", "version": "", "time": ""}'
    if sys.platform == "win32":
        script = (
            f"try {{ irm {INSTALL_PS1_URL} | iex }} catch {{ "
            f"Set-Content -Path $env:{ENV_RESULT} -Value '{failed}' }}"
        )
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ]
    script = (
        f"set -o pipefail; curl -fsSL {INSTALL_SH_URL} | bash"
        f" || printf '%s\\n' '{failed}' > \"${ENV_RESULT}\""
    )
    return ["bash", "-c", script]


def start_install(info: UpdateInfo) -> str | None:
    """Hand the update to the installer, detached from this control.

    The installer stops this control (and, on Windows, everything running from
    the tool env), swaps the install and starts a control again with the saved
    options, so after this returns the caller does nothing more. Returns the log
    path. Raises :class:`InstallRunning` while an earlier run is in progress.
    """
    from biopb._lifecycle import daemon
    from biopb._security import credentials
    from packaging.version import Version

    from . import _logs

    path = result_file()
    prior = last_result()
    if prior and prior.get("status") == "running":
        try:
            fresh = time.time() - path.stat().st_mtime < INSTALL_STALE
        except OSError:
            fresh = False
        if fresh:
            raise InstallRunning("an update is already running")

    env = credentials.scrub_control_secrets(os.environ.copy())
    # An unattended run never asks (a fresh install would otherwise answer its
    # consent prompts with yes).
    env["BIOPB_NONINTERACTIVE"] = "1"
    env[ENV_RESULT] = str(path)
    if Version(info.current).is_prerelease:
        env["BIOPB_INSTALL_RC"] = "1"
    else:
        env.pop("BIOPB_INSTALL_RC", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "running", "version": info.latest}), encoding="utf-8"
    )
    log, log_path = _logs.open_log("update")
    out = log if log else subprocess.DEVNULL
    try:
        argv = _installer_argv()
        kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": out,
            "stderr": out,
            "env": env,
            "close_fds": True,
        }
        try:
            subprocess.Popen(argv, **kwargs, **daemon.detach_kwargs(breakaway=True))
        except OSError:
            if sys.platform != "win32":
                raise
            # A job that forbids breakaway refuses the flag; detach without it.
            subprocess.Popen(argv, **kwargs, **daemon.detach_kwargs())
    finally:
        if log:
            log.close()
    return str(log_path) if log_path else None


class UpdateChecker:
    """Checks in a background thread and keeps the last answer in memory."""

    def __init__(self, running: str, interval: float = CHECK_INTERVAL) -> None:
        self._running = running
        self._interval = interval
        self._info: UpdateInfo | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="update-check", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._info = check(self._running)
            if self._info:
                logger.info(
                    "biopb update available: %s -> %s",
                    self._info.current,
                    self._info.latest,
                )
            self._stop.wait(self._interval)

    def found(self) -> UpdateInfo | None:
        return self._info

    def info(self) -> dict | None:
        return asdict(self._info) if self._info else None
