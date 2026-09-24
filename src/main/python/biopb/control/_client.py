"""The two calls a client makes to the control's HTTP API."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional
from urllib.parse import urlencode

from . import _data_plane
from ._endpoints import control_base_url

logger = logging.getLogger(__name__)


def base_url() -> str:
    """Where the control listens, e.g. ``http://127.0.0.1:8813``."""
    return control_base_url()


def _answer(url: Optional[str]) -> Optional[dict]:
    """``{"url", "token"}`` for a plane the control named, else ``None``.

    The credential file is read here and only here: it is the control's
    credential for its own plane, so it goes only to an address the control
    gave.
    """
    if not url:
        return None
    return {"url": url, "token": _data_plane.resolve_token()}


def data_plane(timeout: float = 1.0) -> Optional[dict]:
    """The plane the control names, ``{"url", "token"}``, or ``None``.

    A plain read of ``GET /health``: ``None`` when no control answers or it
    names no plane. It does not start the plane; :func:`ensure_data_plane`
    does.
    """
    return _answer(_data_plane.control_grpc_url(timeout=timeout))


def ensure_data_plane(timeout: float = 60.0) -> Optional[dict]:
    """Have the control bring its plane up; ``{"url", "token"}`` or ``None``.

    ``POST /api/data_plane/ensure``, idempotent on the control's side.
    *timeout* is both this call's HTTP timeout and the ``client_timeout`` the
    control keeps its own wait under, so a slow start comes back as a verdict
    rather than as a timeout that looks like no control at all. ``None`` when
    no control answers or it could not bring the plane up.
    """
    token = _data_plane.resolve_token()
    req = urllib.request.Request(
        f"{base_url()}/api/data_plane/ensure?{urlencode({'client_timeout': timeout})}",
        data=b"",
        method="POST",
        # The token also clears the control's CSRF gate on this POST. Without one
        # (a tokenless local control) the gate falls back to a loopback Host.
        headers={"X-Biopb-Token": token} if token else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - no answer is None, not an error
        logger.info("control ensure_data_plane failed: %s", exc)
        return None
    snapshot = payload.get("data_plane") if isinstance(payload, dict) else None
    url = snapshot.get("grpc_url") if isinstance(snapshot, dict) else None
    if not url:
        logger.warning("control answered ensure without a data-plane url")
    return _answer(url)
