"""The two calls a client makes to the control's HTTP API."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlencode

from . import _data_plane
from ._endpoints import control_base_url

logger = logging.getLogger(__name__)


def base_url() -> str:
    """Where the control listens, e.g. ``http://127.0.0.1:8813``."""
    return control_base_url()


def _request(
    method: str, path: str, params: dict, timeout: float, body: Optional[dict] = None
) -> dict:
    """The control's JSON answer to *method* ``path`` with *params* as its query
    string and, if given, *body* as a JSON body; raises ``OSError`` when no control answers and
    ``urllib.error.HTTPError`` when it refuses."""
    token = _data_plane.resolve_data_plane_token()
    query = f"?{urlencode(params)}" if params else ""
    req = urllib.request.Request(
        f"{base_url()}{path}{query}",
        data=json.dumps(body).encode()
        if body is not None
        else b""
        if method == "POST"
        else None,
        method=method,
        # The token also clears the control's CSRF gate on a POST. Without one
        # (a tokenless local control) the gate falls back to a loopback Host.
        headers={
            **({"X-Biopb-Token": token} if token else {}),
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _answer(url: Optional[str]) -> Optional[dict]:
    """``{"url", "token"}`` for a plane the control named, else ``None``.

    The credential file is read here and only here: it is the control's
    credential for its own plane, so it goes only to an address the control
    gave.
    """
    if not url:
        return None
    return {"url": url, "token": _data_plane.resolve_data_plane_token()}


def find_data_plane(timeout: float = 1.0) -> Optional[dict]:
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
    try:
        payload = _request(
            "POST", "/api/data_plane/ensure", {"client_timeout": timeout}, timeout
        )
    except Exception as exc:  # noqa: BLE001 - no answer is None, not an error
        logger.info("control ensure_data_plane failed: %s", exc)
        return None
    snapshot = payload.get("data_plane") if isinstance(payload, dict) else None
    url = snapshot.get("grpc_url") if isinstance(snapshot, dict) else None
    if not url:
        logger.warning("control answered ensure without a data-plane url")
    return _answer(url)


class CaptureError(Exception):
    """The control could not have a viewer page draw the view; the message says why."""


def capture_view(view: str, max_edge: int = 1024, timeout: float = 30.0) -> dict:
    """Have an open viewer page draw *view*: ``{"png", "partial", "notes"}``.

    *view* is the viewer's query string. ``png`` is base64. Raises
    :class:`CaptureError` with the control's reason when no visible page can
    answer, and :class:`OSError` when no control does.
    """
    try:
        # The control answers within *timeout*; the margin keeps its verdict from
        # arriving as a socket timeout.
        got = _request(
            "POST",
            "/api/viewer/capture",
            {},
            timeout + 5,
            {"view": view, "max_edge": max_edge, "timeout": timeout},
        )
        if not isinstance(got, dict) or not got.get("png"):
            raise CaptureError("the control answered without an image")
        return got
    except ValueError as exc:  # a body that is not JSON
        raise CaptureError("the control did not answer in JSON") from exc
    except urllib.error.HTTPError as exc:
        try:
            reason = json.loads(exc.read().decode()).get("error")
        except Exception:  # noqa: BLE001 - the status alone will do
            reason = None
        raise CaptureError(reason or f"the control answered HTTP {exc.code}") from exc
