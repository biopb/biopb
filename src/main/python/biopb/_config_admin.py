"""Admin / control-plane endpoint location — shared, stdlib-only.

The admin (control plane) exposes a small loopback HTTP control API. Two
independent processes need to agree on where it listens:

- the admin itself (``biopb-admin``, a separate workspace package), and
- ``biopb-mcp``'s ``_connection``, which asks the admin to ensure the data
  plane is up instead of shelling out ``biopb server start`` itself.

Neither can import the other (``biopb-mcp`` cannot import ``biopb-admin`` any
more than it can import ``biopb-tensor-server`` — see the
"shared config lives in core biopb SDK" rationale), so the one thing they must
share — the endpoint — lives here in the dependency-light core ``biopb`` SDK,
next to ``_config_location`` / ``_config_constraints``. Kept stdlib-only so
importing it never drags in the heavy server/mcp stacks.

This is the seed of the Layer-3 single-origin web front (see
``biopb-mcp/docs/mcp-dedaemonization-migration.md``): for now the admin serves
only a control API here; later the same origin fronts the dataviewer + observe
routes.
"""

import os

# Loopback control API. Distinct from the other biopb ports so all four can run
# at once on one host: tensor-server web 8814 / gRPC 8815, MCP /mcp 8765.
ADMIN_DEFAULT_HOST = "127.0.0.1"
ADMIN_DEFAULT_PORT = 8813


def admin_host() -> str:
    """The admin control-API bind/connect host (``BIOPB_ADMIN_HOST`` override)."""
    return os.environ.get("BIOPB_ADMIN_HOST") or ADMIN_DEFAULT_HOST


def admin_port() -> int:
    """The admin control-API port (``BIOPB_ADMIN_PORT`` override, else 8813).

    A malformed override falls back to the default rather than raising, so a
    stray env value can never wedge a client that only wants to probe the admin.
    """
    raw = os.environ.get("BIOPB_ADMIN_PORT")
    if not raw:
        return ADMIN_DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        return ADMIN_DEFAULT_PORT


def admin_base_url() -> str:
    """The admin control-API base URL, e.g. ``http://127.0.0.1:8813``."""
    return f"http://{admin_host()}:{admin_port()}"
