# biopb installers

Cross-platform bootstrap installers for the biopb stack (the `biopb` SDK, the
`biopb-tensor-server` data plane, and optionally `biopb-mcp` + napari). They set
up `uv`, an appropriate Python, the packages, the data-browser webapp, a default
config, and wire biopb-mcp into any detected AI agent.

| Script | Platform | One-liner |
|--------|----------|-----------|
| `install.sh`  | Linux, macOS, WSL | `curl -fsSL https://biopb.org/install.sh \| bash` |
| `install.ps1` | Windows (PowerShell 5.1+) | `irm https://biopb.org/install.ps1 \| iex` |

Both are **idempotent** — rerun to upgrade or to add components you skipped.

## Install modes

### Release (default)

Downloads the prebuilt `biopb-mcp`, `biopb`, and `biopb-tensor-server` wheels
from the **latest `biopb/biopb-mcp` GitHub release** — a single release that
carries all three as a mutually-paired set — and installs them into one shared
`uv` tool environment. Because the trio ships from one build, the server always
runs against the exact `biopb` it was tested with and `biopb-mcp` against the
exact pair it expects (the server wheel may use proto fields newer than any
`biopb` on PyPI), with no PyPI-vs-release version skew. Only `napari` comes from
PyPI.

Requirements: `curl`/`tar` (POSIX) or built-in PowerShell tooling (Windows), plus
`uv` (installed automatically). **No git, buf, or compiler needed** — the release
wheels ship the generated protobuf/Flight stubs.

### Source (opt-in)

Set `BIOPB_INSTALL_FROM_SOURCE=1` to build the bleeding-edge `main` from a git
checkout instead. This is the fast path for development/testing.

```sh
# Linux / macOS / WSL
BIOPB_INSTALL_FROM_SOURCE=1 curl -fsSL https://biopb.org/install.sh | bash
```

```powershell
# Windows PowerShell
$env:BIOPB_INSTALL_FROM_SOURCE = "1"; irm https://biopb.org/install.ps1 | iex
```

Source mode additionally requires **git** and installs **buf** (to generate the
proto stubs at build time); on macOS it also needs the Xcode Command Line Tools.

## What gets installed

- **biopb-mcp + biopb + biopb-tensor-server** — the matched triple from the one
  biopb-mcp release, into a single `uv` tool environment so the components can
  import and drive each other (`biopb server start`, the napari viewer, etc.).
- **napari** — from PyPI, into the same environment.
- **Data browser** (optional) — `webapp.tar.gz` from the same biopb-mcp release,
  unpacked to `~/.local/share/biopb/webapp`.
- The installer also registers the biopb MCP server with any detected agent
  (Claude Code/Desktop, Cursor, opencode, Hermes) and can install opencode if
  none is found. biopb-mcp speaks MCP over **stdio**, so the agent spawns
  `biopb-mcp --transport stdio` itself (which opens the napari window and brings
  up the data plane) — there is no separate server to start by hand.

## Config & data locations

- Data-server config: `~/.config/biopb/biopb.toml` (preserved on rerun)
- biopb-mcp config: `~/.config/biopb-mcp/config.json`
- MCP client definition: `~/.config/biopb/mcp.json`
- Webapp: `~/.local/share/biopb/webapp`

Set `BIOPB_DATA_DIR` to skip the interactive data-directory prompt.

## Notes

- Release assets are read from the `biopb/biopb-mcp` GitHub Releases API; the
  latest `v*` release carries all three wheels (`biopb-mcp`, `biopb`,
  `biopb-tensor-server`) plus the webapp tarball. biopb-mcp's release CI bakes
  in the `biopb` + `biopb-tensor-server` wheels (pinned in `pyproject.toml`) so
  the trio is built and shipped together.
- Migration rationale (why the installer and all three wheels now live in one
  repo/release) lives in `../docs/installer-migration.md`.
- Distribution rationale (why the server ships via Docker/GitHub release rather
  than PyPI) lives in the architecture overview, `../development.md`.
