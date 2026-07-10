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
   > **Superseded:** biopb-mcp's PyPI publishing (and the napari-hub listing it
   > fed) has since been retired — it ships only inside the `release-v*` wheel
   > triple. The `mcp-v*` tag is kept purely as its setuptools_scm version
   > marker. See `docs/release-model.md`.
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
- **`mcp-ci.yaml`** — uv-based test matrix, path-filtered to `biopb-mcp/**` +
  shared proto/server. Test-only.

biopb-mcp is **no longer published to PyPI** (see the "Superseded" note on
decision #2 above and `docs/release-model.md`); mcp-ci runs tests only, and the
`mcp-v*` tag is now just biopb-mcp's setuptools_scm version marker. Product
deployment (the GitHub release + Docker that the installer/operators consume) is
the unified `release.yaml` on `release-v*` tags. The cross-platform PyInstaller
"frozen app" bundles are disabled (everything installs via uv); the spec/hooks
(`biopb-mcp/biopb-mcp.spec`, `biopb-mcp/hooks/`) remain in the tree.

### Installer (`install/install.sh`, `install.ps1`)
Originally staged under `biopb-mcp/install/`; promoted to the repo-root `install/`
after the first `release-v*` (the transitional copy was removed).
- `RELEASE_REPO` → `biopb/biopb`; release fetch filters to the deployment line
  (`release-v*`, clean `X.Y.Z` only, so prereleases are skipped).
- Source-mode builds `biopb-mcp` from the monorepo with `#subdirectory=biopb-mcp`.

## Unified release

The interim per-component releases (`mcp-release.yaml`, `tensor-server-ci`'s
publish job) have been **superseded** by a single `release.yaml` on `release-v*`
tags — see **`docs/release-model.md`** (repo root) for the full model: PyPI stays
per-package (`v*`/`mcp-v*`); `release-v*` produces the GitHub release (wheel
triple + webapp + installers) and the Docker images (tensor-server, image-base),
with no PyPI.

## Live-CI validation (all confirmed)

Everything below has now been exercised on real runs — the migration is fully
validated end to end:

- `uv sync --package biopb-mcp --extra mcp --group testing` resolves + builds the
  full Qt/napari stack on all matrix platforms — green on `mcp-ci`.
- `uv publish` to PyPI on `mcp-v*` tags (`PYPI_API_TOKEN`) — published
  `biopb-mcp` 0.7.1a0 and 0.7.2.
- The end-to-end installer against a real monorepo release — validated by a manual
  docker install test against `release-v0.8.0` (the deployment line; the installer
  now consumes `release-v*`, not `mcp-v*`).
- A real tag produces a clean (non-`0.1.dev`) version — `mcp-v0.7.2` → `0.7.2`.
