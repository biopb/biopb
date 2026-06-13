# Migrating `biopb-mcp` into the `biopb` monorepo

`biopb-mcp` was a sibling repository (`github.com/biopb/biopb-mcp`). It is now a
subdirectory of the `biopb` monorepo (`biopb/biopb-mcp/`), alongside
`biopb-tensor-server/` and `biopb-image-runtime/`. This note records *why*, *how*
the move was done, the decisions that shaped it, and what still needs validating
on live CI (which cannot be exercised from a local checkout).

## Why

`biopb-mcp` is tightly coupled to the rest of the stack: it depends on
`biopb[tensor]` (the Arrow Flight client) and ships, version-paired, with
`biopb-tensor-server` (which is **never published to PyPI**). Keeping it in its
own repo forced a brittle cross-repo release dance: `biopb-mcp`'s dependency
groups pinned **direct-URL wheels** off a specific `biopb` `server-v*` GitHub
release, and its bundle workflow `curl`ed those wheels + the webapp back to
assemble the installer. Every server release meant bumping pinned URLs/SHAs and
re-locking. A monorepo collapses that seam: the packages resolve each other from
the tree and are automatically paired with the checkout.

## Decisions

1. **History preserved** via `git subtree add --prefix=biopb-mcp` (not a squash
   copy). Pre-merge commits remain in the graph.
2. **`biopb-mcp` stays an independently-versioned, independently-published
   package** for now — its own PyPI release and napari-hub plugin — under a new
   **`mcp-v*`** git-tag prefix. This avoids colliding with the client's `v*` and
   the server's `server-v*` tags.
3. **Interim dual-track, unified-release future.** The `mcp-v*` and `server-v*`
   release lines continue for now. The intended end state is a **single unified
   release** that builds and bundles the whole stack (client + tensor-server +
   mcp + webapp + installers) under one tag, retiring both `mcp-v*` and
   `server-v*`. See "Toward a unified release" below.

## What changed

### Versioning (`biopb-mcp/pyproject.toml`)
Mirrors `biopb-tensor-server`'s subdir setup:
```toml
[tool.setuptools_scm]
root = ".."
tag_regex = "^mcp-v(?P<version>[0-9]+\\.[0-9]+\\.[0-9]+)$"
[tool.setuptools_scm.scm.git]
describe_command = "git describe --tags --match 'mcp-v*'"
```
The matched `git describe` is mandatory: without `--match`, the version would
pick up whichever `v*`/`server-v*` tag is most recent. Until the first `mcp-v*`
tag is cut, `setuptools_scm` falls back to a `0.1.dev…` version — expected.

### Dependency wiring (uv workspace)
The root `pyproject.toml` gained a uv workspace:
```toml
[tool.uv.workspace]
members = ["biopb-image-runtime", "biopb-tensor-server", "biopb-mcp"]
[tool.uv.sources]
biopb = { workspace = true }
biopb-image-base = { workspace = true }
biopb-tensor-server = { workspace = true }
biopb-mcp = { workspace = true }
```
`biopb-mcp`'s `integration`/`testing` dependency groups dropped the direct-URL
wheels for plain names (`biopb[tensor]`, `biopb-tensor-server[web]`), which now
resolve to the in-tree members. The per-package `biopb-mcp/uv.lock` was
collapsed into a single root `uv.lock`. `[tool.uv]` is ignored by pip/setuptools,
so the legacy `pip install -e` flow into the shared root `.venv` still works;
`uv sync` is the canonical setup.

**Consequence for CI:** tox+pip is *not* workspace-aware and cannot resolve
`biopb-tensor-server` (not on PyPI), so MCP CI moved to `uv`.

### CI (`.github/workflows/`)
The imported `biopb-mcp/.github/*` were inert (GitHub only runs root workflows).
Replaced with:
- **`mcp-ci.yaml`** — uv-based test matrix (path-filtered to `biopb-mcp/**` +
  shared proto/server), PyPI publish on `mcp-v*`, then triggers the bundle.
- **`mcp-release.yaml`** — builds all three wheels + the webapp from the tagged
  tree, then the cross-platform PyInstaller bundle (run from `biopb-mcp/`), and
  assembles one GitHub release.

### Installer (`biopb-mcp/install/install.sh`, `install.ps1`)
- `RELEASE_REPO` → `biopb/biopb`.
- Release fetch now **filters by the `mcp-v` tag prefix** — the monorepo's
  `/releases/latest` is repo-wide across `mcp-v*`/`server-v*`, so it lists
  releases and takes the newest matching the prefix.
- Source-mode builds `biopb-mcp` from the monorepo with
  `#subdirectory=biopb-mcp`.

## Toward a unified release (future work)

The interim keeps `mcp-v*` and `server-v*` separate. The planned end state is a
single `unified-release.yaml` triggered by one tag (e.g. `release-vX.Y.Z`) that:
- builds the client, tensor-server, and mcp wheels + the webapp once,
- builds the PyInstaller bundles,
- and publishes **one** GitHub release that is the single source of truth for the
  installer and auto-updater.

At that point `mcp-release.yaml` and `tensor-server-ci.yaml`'s publish job (and
the `mcp-v*`/`server-v*` tag lines) are retired. The installer's
`RELEASE_TAG_PREFIX` becomes the unified prefix.

## Needs live-CI validation (cannot be checked from a local checkout)

- `uv sync --package biopb-mcp --extra mcp --group testing` resolving + building
  the full Qt/napari stack on all matrix platforms.
- The PyInstaller bundle on linux/windows/macos-intel/macos-arm from the
  `biopb-mcp/` working directory (the spec uses paths relative to its own dir).
- `uv publish` to PyPI on an `mcp-v*` tag (token: `PYPI_API_TOKEN`).
- The end-to-end installer against a real `mcp-v*` monorepo release (prefix
  filtering, asset names, source-mode subdirectory installs).
- First real `mcp-v*` tag produces a clean (non-`0.1.dev`) version.
