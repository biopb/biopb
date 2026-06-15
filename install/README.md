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

## Release channels

Both installers download the prebuilt `biopb-mcp`, `biopb`, and
`biopb-tensor-server` wheels from a single **`biopb/biopb` `release-v*` GitHub
deployment** — one release that carries all three as a mutually-paired set — and
install them into one shared `uv` tool environment. Because the trio ships from
one build, the server always runs against the exact `biopb` it was tested with
and `biopb-mcp` against the exact pair it expects (the server wheel may use proto
fields newer than any `biopb` on PyPI), with no PyPI-vs-release version skew.
Only `napari` comes from PyPI.

Requirements: `curl`/`tar` (POSIX) or built-in PowerShell tooling (Windows), plus
`uv` (installed automatically). **No git, buf, or compiler needed** — the release
wheels ship the generated protobuf/Flight stubs.

### Stable (default)

Tracks the latest **stable** release — a clean `release-vX.Y.Z` tag. Prerelease
tags are skipped.

### Release candidate (opt-in)

Set `BIOPB_INSTALL_RC=1` to also admit the latest **release candidate** — a PEP
440 prerelease (`…rc1`/`…a1`/`…b1`, typically cut off the `dev` branch to
validate a deployment before it lands on `main`). The newest matching release
wins, so if an RC is newer than the current stable you get the RC. This replaces
the old build-from-source path as the fast way to test an upcoming release — it
still installs prebuilt wheels, so no git/buf/compiler is needed.

```sh
# Linux / macOS / WSL
BIOPB_INSTALL_RC=1 curl -fsSL https://biopb.org/install.sh | bash
```

```powershell
# Windows PowerShell
$env:BIOPB_INSTALL_RC = "1"; irm https://biopb.org/install.ps1 | iex
```

See `../docs/release-model.md` for how `release-v*` tags and release candidates
are produced.

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

- Release assets are read from the `biopb/biopb` GitHub Releases API; the latest
  `release-v*` deployment carries all three wheels (`biopb-mcp`, `biopb`,
  `biopb-tensor-server`) plus the webapp tarball. The `release.yaml` CI builds
  the trio from the tagged commit so they are shipped together as one matched
  set (see `../docs/release-model.md`).
- Migration rationale (why the installer and all three wheels now live in one
  repo/release) lives in `../docs/installer-migration.md`.
- Distribution rationale (why the server ships via Docker/GitHub release rather
  than PyPI) lives in the architecture overview, `../development.md`.
