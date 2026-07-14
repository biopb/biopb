"""Single source of truth for the biopb server config file location & format.

Resolves *where* the tensor-server config lives and *which* format wins when
both exist. Imported by all three consumers -- ``biopb-tensor-server``
(``config.find_config``), the umbrella ``biopb`` CLI, and ``biopb-mcp`` -- each
of which depends on this core ``biopb`` package, so the logic lives here once
instead of being mirrored.

Deliberately stdlib-only (``pathlib`` + ``logging``) so importing it is cheap on
every CLI invocation; it pulls in none of the heavy adapter/discovery machinery
that ``biopb_tensor_server.core.config`` does.

JSON is the *canonical* on-disk format going forward: the config is
machine-generated (the installer / a future generator write it), and once nobody
hand-edits it, TOML's hand-editing ergonomics (comments, ``[[sources]]``) stop
paying for its one wart -- no stdlib *writer*. JSON has a stdlib writer on both
ends (Python ``json.dumps``, PowerShell ``ConvertTo-Json``), unifies the format
with biopb-mcp's ``config.json``, and pairs with JSON Schema for validation.
TOML stays *readable* through a deprecation window so no existing ``biopb.toml``
breaks on upgrade. See biopb/biopb#34.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "biopb"
CANONICAL_CONFIG_NAME = "biopb.json"
LEGACY_CONFIG_NAME = "biopb.toml"

# biopb-mcp's own settings file, co-located in the same dir. Distinct from the
# installer's client-definition ``mcp.json`` (which registers biopb-mcp with MCP
# clients). Defined here so the three consumers that touch it -- biopb-mcp
# (its config module) and the lean control plane + ``biopb._algorithms`` (which
# read it WITHOUT importing biopb_mcp, invariant I2) -- agree on one location
# and cannot drift. See biopb/biopb#34.
MCP_CONFIG_NAME = "mcp-config.json"


def mcp_config_path() -> Path:
    """The biopb-mcp settings file (``~/.config/biopb/mcp-config.json``).

    Computed from ``Path.home()`` at call time (not the import-time
    ``DEFAULT_CONFIG_DIR`` constant) so a test that repoints ``Path.home()`` gets
    an isolated location.
    """
    return Path.home() / ".config" / "biopb" / MCP_CONFIG_NAME


def find_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Path:
    """Resolve the config file in *config_dir*, preferring JSON over TOML.

    Returns the first of ``biopb.json`` / ``biopb.toml`` that exists. When
    neither exists, returns the canonical JSON path so callers seed / print the
    forward-looking name. Callers that need a guaranteed-existing file should
    still check ``.exists()`` on the result.

    When *both* files exist the legacy TOML is silently shadowed, so this logs a
    warning naming the file that is being ignored.
    """
    json_path = config_dir / CANONICAL_CONFIG_NAME
    toml_path = config_dir / LEGACY_CONFIG_NAME
    if json_path.exists():
        if toml_path.exists():
            logger.warning(
                "Both %s and %s exist in %s; using %s and ignoring the legacy "
                "%s. Remove the TOML file to silence this. See biopb/biopb#34.",
                CANONICAL_CONFIG_NAME,
                LEGACY_CONFIG_NAME,
                config_dir,
                CANONICAL_CONFIG_NAME,
                LEGACY_CONFIG_NAME,
            )
        return json_path
    if toml_path.exists():
        return toml_path
    return json_path
