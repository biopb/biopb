"""Client for the biopb control (control plane) control API — stdlib only.

Since Layer 2 of the de-daemonization migration
(docs/mcp-dedaemonization-migration.md), ``_connection`` is a *pure client*: it
never shells out ``biopb server start`` to bring the data plane up. When the
plane is down it asks the control to ensure it, via this thin urllib client. The
control is the durable root that owns the data plane; ``_connection`` only uses it.

The endpoint is resolved from ``biopb._config_control`` (the shared, stdlib-only
core-SDK module), the same location the control itself binds — biopb-mcp cannot
import ``biopb-control`` any more than it can import ``biopb-tensor-server``, so the
one shared fact (where the control listens) lives in core ``biopb``.
"""

from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def _base_url() -> str:
    from biopb._config_control import control_base_url

    return control_base_url()


def control_reachable(timeout: float = 1.0) -> bool:
    """Whether the control's control API answers ``GET /health``."""
    try:
        with urllib.request.urlopen(f"{_base_url()}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_data_plane(timeout: float = 60.0) -> dict | None:
    """Ask the control to ensure the data plane is up; return its snapshot.

    POSTs ``/api/data_plane/ensure`` — idempotent on the control side (spawn the
    plane it owns, then wait until listening). Returns the ``data_plane`` snapshot dict
    on success, or ``None`` if the control is unreachable (not running) or errored,
    so the caller can fall back to surfacing "start the control" rather than
    raising.

    ``timeout`` is BOTH our HTTP timeout and the hint we pass the server as
    ``?client_timeout``: the server caps its own ensure wait below this so it
    always returns a verdict before our ``urlopen`` times out — otherwise a
    slow-but-working control plane would look unreachable and we'd wrongly report
    "no control plane".
    """
    from urllib.parse import urlencode

    url = (
        f"{_base_url()}/api/data_plane/ensure?{urlencode({'client_timeout': timeout})}"
    )
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("data_plane")
    except Exception as exc:  # noqa: BLE001 - best-effort; caller handles None
        logger.info("control ensure_data_plane failed: %s", exc)
        return None
