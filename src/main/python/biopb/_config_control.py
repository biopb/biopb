"""Control / control-plane endpoint location — shared, stdlib-only.

The control (control plane) exposes a small loopback HTTP control API. Two
independent processes need to agree on where it listens:

- the control itself (``biopb-control``, a separate workspace package), and
- ``biopb-mcp``'s ``_connection``, which asks the control to ensure the data
  plane is up instead of shelling out ``biopb server start`` itself.

Neither can import the other (``biopb-mcp`` cannot import ``biopb-control`` any
more than it can import ``biopb-tensor-server`` — see the
"shared config lives in core biopb SDK" rationale), so the one thing they must
share — the endpoint — lives here in the dependency-light core ``biopb`` SDK,
next to ``_config_location`` / ``_config_constraints``. Kept stdlib-only so
importing it never drags in the heavy server/mcp stacks.

This is the seed of the Layer-3 single-origin web front (see
``biopb-mcp/docs/mcp-dedaemonization-migration.md``): for now the control serves
only a control API here; later the same origin fronts the dataviewer + observe
routes.
"""

import os

# Loopback control API. Distinct from the other biopb ports so all four can run
# at once on one host: tensor-server web 8814 / gRPC 8815, MCP /mcp 8765.
CONTROL_DEFAULT_HOST = "127.0.0.1"
CONTROL_DEFAULT_PORT = 8813


def control_host() -> str:
    """The control-API bind/connect host (``BIOPB_CONTROL_HOST`` override)."""
    return os.environ.get("BIOPB_CONTROL_HOST") or CONTROL_DEFAULT_HOST


def control_port() -> int:
    """The control-API port (``BIOPB_CONTROL_PORT`` override, else 8813).

    A malformed override falls back to the default rather than raising, so a
    stray env value can never wedge a client that only wants to probe the control.
    """
    raw = os.environ.get("BIOPB_CONTROL_PORT")
    if not raw:
        return CONTROL_DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        return CONTROL_DEFAULT_PORT


def control_base_url() -> str:
    """The control-API base URL, e.g. ``http://127.0.0.1:8813``."""
    return f"http://{control_host()}:{control_port()}"
