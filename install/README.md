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

Both installers download the prebuilt `biopb-mcp`, `biopb`,
`biopb-tensor-server`, and `biopb-control` wheels from a single **`biopb/biopb`
`release-v*` GitHub deployment** — one release that carries all four as a
mutually-paired set — and install them into one shared `uv` tool environment.
Because the set ships from one build, the server always runs against the exact
`biopb` it was tested with and `biopb-mcp` against the exact peers it expects (the
server wheel may use proto fields newer than any `biopb` on PyPI), with no
PyPI-vs-release version skew. Only `napari` comes from PyPI.

Requirements: `curl`/`tar` (POSIX) or built-in PowerShell tooling (Windows), plus
`uv` (installed automatically). **No git, buf, or compiler needed** — the release
wheels ship the generated protobuf/Flight stubs.

**Minimum macOS:** Apple Silicon (arm64) needs **macOS 12 Monterey or newer**;
Intel (x86_64) needs **macOS 10.15 Catalina or newer**. napari's optional `numba`
accelerator pulls `llvmlite`, and the current prebuilt `llvmlite` wheels target
those OS floors (arm64 → `macosx_12_0`, x86_64 → `macosx_10_15`). On an older
macOS there is no matching wheel, so the install falls back to building `llvmlite`
from source — which needs an LLVM toolchain and fails on a stock machine. If you
hit that, upgrade macOS (there is no build workaround the installer can apply).

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
curl -fsSL https://biopb.org/install.sh | BIOPB_INSTALL_RC=1 bash
```

```powershell
# Windows PowerShell
$env:BIOPB_INSTALL_RC = "1"; irm https://biopb.org/install.ps1 | iex
```

See `../docs/release-model.md` for how `release-v*` tags and release candidates
are produced.

## What gets installed

- **biopb-mcp + biopb + biopb-tensor-server + biopb-control** — the matched wheel
  set from the one release, into a single `uv` tool environment so the components
  can import and drive each other (`biopb control start`, the napari viewer, etc.).
- **napari** — from PyPI, into the same environment.
- **Web interface** — `webapp.tar.gz` from the same release, unpacked to
  `~/.local/share/biopb/webapp`. Carries the image viewer and the server admin
  page, and is the dashboard `biopb dashboard` opens, so it is **always
  installed** (not optional).
- **A "biopb Dashboard" Desktop shortcut** that runs `biopb dashboard` — start
  the control plane (if it isn't already up) and open the dashboard in your
  default browser. Set `BIOPB_INSTALL_SHORTCUT=0` to skip creating it. You can
  run `biopb dashboard` from a terminal at any time instead.
- The installer also registers the biopb MCP server with any detected agent
  (Claude Code/Desktop, Cursor, opencode) and can install opencode if
  none is found. biopb-mcp speaks MCP over **stdio**, so the agent spawns
  `biopb-mcp --transport stdio` itself — there is no separate server to start by
  hand. The napari viewer does not open with it: it comes up on the first
  `start_kernel` call, so prompt the agent (e.g. "start biopb and report
  status") to bring up the viewer and the data plane.

## Config & data locations

- Data-server config: `~/.config/biopb/biopb.json` (preserved on rerun; a
  legacy `biopb.toml` is no longer read and must be migrated to JSON — via
  `biopb-tensor-server migrate-config`, or automatically when you pick a new data
  folder)
- biopb-mcp config: `~/.config/biopb/mcp-config.json`
- MCP client definition: `~/.config/biopb/mcp.json`
- Agent kernel plugins: `~/.config/biopb/kernel/` (drop a `*.py` here to add tools
  to the agent's namespace; the installer seeds a `rolling_ball.py` example there,
  never clobbering your edits)
- Extra Python packages: `~/.config/biopb/extra-packages.txt` (see below)
- Webapp: `~/.local/share/biopb/webapp`

### Extra Python packages

Everything installs into **one** `uv` tool environment, and that environment is
what the agent's napari kernel runs in — so an optional dependency a workflow
needs (`basicpy`, `m2stitch`, a newer `scikit-image`) has to live there too.

Every rerun of the installer rebuilds that environment from the release's own
requirement list, which means a package you add by hand is **dropped at the next
upgrade** — usually noticed much later, as an import that used to work. To keep
one, name it in `~/.config/biopb/extra-packages.txt`, one
[PEP 508](https://peps.python.org/pep-0508/) requirement per line (`#` comments
and blank lines ignored):

```
basicpy
m2stitch==0.9.0
```

The installer replays that list on every run, so the packages are reinstalled
with everything else. If one of them can't be resolved alongside the release's
own pins, the install **does not fail**: it retries without your extras, says
which were dropped, and points back at this file — biopb still upgrades, and you
fix the offending line at your leisure.

To install one for the current session only, use the exact command
`server_status` prints under `## Versions` (it names this environment's
interpreter); a bare `pip install` targets whatever environment your shell has
active, which is usually not this one.

### Unattended / unmanned upgrades

Set `BIOPB_NONINTERACTIVE=1` to suppress **every** prompt, so the installer can
run from cron / Task Scheduler / CI to upgrade in place. The common case — a
rerun with an existing config — keeps that config untouched and asks nothing.

It is an **upgrade** feature, not a fresh-install one: with no existing config
the installer won't guess a data directory, so a *fresh* unattended install
**errors out unless `BIOPB_DATA_DIR` is set** (which lets you provision a new box
unattended on purpose). The remote algorithm plugins stay **off** unless
`BIOPB_REMOTE_PLUGINS=1` (consent can't be asked unattended, so the off-site
IP-logging servers are never enabled silently). Both console front-ends
(`install.sh`, `install.ps1`) honor these; the env vars apply to the `curl|bash`
and `irm|iex` paths alike.

## Uninstall

`install.sh` takes an `--uninstall` flag (mirrors the Windows installer's
Add/Remove Programs entry). It stops the data and MCP servers, unregisters biopb
from any detected agent (Claude Code/Desktop, Cursor, opencode), and removes the
shared `uv` tool environment. Add `--purge` to also delete config and cached
data — your **image data is never touched**.

```sh
# Remove the stack, keep config + cached data
curl -fsSL https://biopb.org/install.sh | bash -s -- --uninstall

# Remove everything biopb owns, including config and cache
curl -fsSL https://biopb.org/install.sh | bash -s -- --uninstall --purge
```

`--purge` deletes `~/.config/biopb`, `~/.local/state/biopb` (logs, session
registry, pids), and `~/.local/share/biopb` (webapp, samples). `uv` and any AI
agent (e.g. opencode) are left installed.
On Windows, uninstall through Add/Remove Programs instead.

## Notes

- Release assets are read from the `biopb/biopb` GitHub Releases API; the latest
  `release-v*` deployment carries all four wheels (`biopb-mcp`, `biopb`,
  `biopb-tensor-server`, `biopb-control`) plus the webapp tarball. The
  `release.yaml` CI builds the set from the tagged commit so they are shipped
  together as one matched set (see `../docs/release-model.md`).
