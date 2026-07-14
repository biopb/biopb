"""User-choice handling for the update **nagger** (issue #87).

This is intentionally a *notify-only reminder*, not a self-applying updater. A
fully automatic cross-platform apply needs a staging step we don't handle
gracefully yet (Windows can't reinstall into a tool dir held open by the running
daemon), so for now the popup just tells the user a newer release exists and to
run the install/upgrade script themselves. The user can opt out per-version
(``skip``) or entirely (``disable`` -> ``update.enabled = false``).

``handle_choice`` is pure control flow over the config seams so it is
unit-testable; it runs on the Qt main thread (the popup's button signal delivers
there). The clipboard copy of the upgrade command is handled in the popup (it has
Qt); :func:`upgrade_command` is the single source of that one-liner.
"""

import logging
import os

logger = logging.getLogger(__name__)

# The published install/upgrade one-liners (docs/release-model.md). Re-running
# the script is how a user updates; the nagger just surfaces the command.
_INSTALL_SH = "curl -fsSL https://biopb.org/install.sh | bash"
_INSTALL_PS1 = "irm https://biopb.org/install.ps1 | iex"


def _is_windows() -> bool:
    """Whether the running platform is Windows.

    A seam so tests can select the platform without mutating the global
    ``os.name`` — patching that corrupts ``pathlib`` on Python < 3.12 (it picks
    ``WindowsPath``/``PosixPath`` off ``os.name``, and instantiating the
    non-native one raises ``NotImplementedError``), which crashes pytest's own
    location reporting mid-test.
    """
    return os.name == "nt"


def upgrade_command() -> str:
    """The platform-appropriate install/upgrade command to show the user."""
    return _INSTALL_PS1 if _is_windows() else _INSTALL_SH


def handle_choice(action: str, info, config: dict) -> None:
    """Dispatch a nagger choice (``skip`` / ``disable`` / ``later``).

    * ``skip``    -> persist ``update.skipped_version`` so *this* version
                     never prompts again.
    * ``disable`` -> turn the check off entirely (``update.enabled = false``).
    * ``later``   -> do nothing; the check re-prompts on the next kernel start.

    (The ``copy`` button is handled in the popup itself.) Fail-open: a handler
    error is logged, never raised (this runs in a Qt slot).
    """
    try:
        if action == "skip":
            _persist_skip(info.latest)
        elif action == "disable":
            _disable_checks()
        elif action == "later":
            logger.debug("update %s deferred (Later)", info.latest)
        else:
            logger.debug("update popup: unknown action %r", action)
    except Exception:
        logger.exception("update choice handler failed for action %r", action)


def _persist_skip(version: str) -> None:
    """Record the user's "Skip vX.Y.Z" so :func:`_update.check_for_update`
    suppresses exactly that version on future starts."""
    try:
        from .._config import CONFIG

        CONFIG.set("update.skipped_version", version)
        logger.info("update %s skipped on user request", version)
    except Exception:
        logger.exception("failed to persist skipped update version %s", version)


def _disable_checks() -> None:
    """Opt out of update reminders entirely (``update.enabled = false``)."""
    try:
        from .._config import CONFIG

        CONFIG.set("update.enabled", False)
        logger.info("update reminders disabled on user request")
    except Exception:
        logger.exception("failed to disable update reminders")
