"""The algorithm plane, as the control answers for it.

Each entry is a row ``{name, kind, url, state, ops, error, ...}``: ``kind`` is
``"script"`` (a server file the control runs) or ``"url"`` (a server someone
else runs), ``ops`` the ``OpInfo`` messages as JSON. A running script entry's
row carries its loopback ``url`` and the ``token`` it checks.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Optional

from ._client import _request


def _listing(method: str, path: str, timeout: float) -> Optional[list[dict]]:
    try:
        answer = _request(method, path, {}, timeout)
    except OSError:
        return None
    return answer.get("servers", []) if isinstance(answer, dict) else []


def algorithms(timeout: float = 10.0) -> Optional[list[dict]]:
    """Every registry entry with its state and cached ops, or ``None`` when no
    control answers. A url entry is probed, so this can take a probe's time."""
    return _listing("GET", "/api/algorithms", timeout)


def refresh_algorithms(timeout: float = 10.0) -> Optional[list[dict]]:
    """Have the control install and describe new or edited script entries;
    answers the rows at once, with those entries ``installing``."""
    return _listing("POST", "/api/algorithms/refresh", timeout)


def _verb(method: str, verb: str, name: str, timeout: float, **params) -> dict:
    params = {"name": name, **params}
    try:
        return _request(method, f"/api/algorithms/{verb}", params, timeout)
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read().decode()).get("error") or str(exc)
        except (ValueError, AttributeError, OSError):
            message = str(exc)
        if exc.code == 404:
            raise LookupError(message) from exc
        if exc.code == 400:
            raise ValueError(message) from exc
        raise RuntimeError(message) from exc
    except OSError as exc:
        raise RuntimeError(f"no control answered: {exc}") from exc


def ensure_algorithm(name: str, timeout: float = 600.0) -> dict:
    """Bring a script entry up, installing it first if its file changed, and
    answer its row; a url entry is probed. Waits under *timeout*: a row still
    ``installing`` or ``starting`` means ask again.

    Raises ``LookupError`` for an unknown name.
    """
    return _verb("POST", "ensure", name, timeout, client_timeout=timeout)["server"]


def stop_algorithm(name: str, timeout: float = 30.0) -> dict:
    """Stop a script entry's server. ``ValueError`` for a url entry."""
    return _verb("POST", "stop", name, timeout)["server"]


def restart_algorithm(name: str, timeout: float = 600.0) -> dict:
    """Stop a script entry and ensure it again. ``ValueError`` for a url entry."""
    return _verb("POST", "restart", name, timeout, client_timeout=timeout)["server"]


def algorithm_logs(name: str, lines: int = 200, timeout: float = 10.0) -> list[str]:
    """The tail of a script entry's log: its installs and its server's output.
    ``ValueError`` for a url entry."""
    return _verb("GET", "logs", name, timeout, lines=lines)["lines"]
