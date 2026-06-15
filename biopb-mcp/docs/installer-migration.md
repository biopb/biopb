# Migration note: biopb installer → biopb-mcp

> **Historical / partly superseded.** This note predates the monorepo merge and
> still describes a two-repo world (e.g. `RELEASE_REPO = biopb/biopb-mcp`). After
> the merge the installer points at `biopb/biopb` and selects releases by the
> `mcp-v*` tag prefix — see `monorepo-migration.md` for the current state.

**Status:** **In progress.** Phase 0 + Phase 1 landed in this change; Phases 2–4
are tracked in the linked issue. This note records the plan and the decisions
behind it.

## Why move the installer here

The bootstrap installer (`install.sh` / `install.ps1`, one-liner
`curl -fsSL https://biopb.org/install.sh | bash`) historically lived in
`biopb/biopb` and pulled the stack from **two independent sources**:

- `biopb` + `biopb-tensor-server` — a matched wheel **pair** from the latest
  `biopb/biopb` (`server-v*`) GitHub release, and
- `biopb-mcp` + `napari` — from **PyPI**.

Two sources means two chances for a version mismatch, and no single place an
auto-updater can point at. biopb-mcp turned out **too tightly coupled to biopb**
to publish independently, yet it never lived in the monorepo — so the natural fix
is to make biopb-mcp the **distribution hub**: one repo owns the release that
carries the whole matched set, the installer that consumes it, and the
auto-updater that re-applies it (see `autoupdater.md`).

The precondition was bundling all three wheels into one biopb-mcp release
(`release_bundle.yml`); this migration builds on it.

## Decisions (locked)

1. **biopb-mcp is the sole owner.** `biopb/biopb` deletes its `install/` and
   stops attaching `install.sh` to its release; biopb-mcp is the single source.
   (Phase 3 — cross-repo.)
2. **The webapp ships from the biopb-mcp release too.** `release_bundle.yml`
   downloads `webapp.tar.gz` from the pinned biopb server release and re-attaches
   it, so the installer hits **only** the biopb-mcp release.
3. **`biopb.org/install.{sh,ps1}` stays the canonical URL**, repointed to serve
   biopb-mcp's copy (an infra step on the biopb.org host — Phase 3).
4. **Scope of the first change:** Phase 0 + Phase 1 (below).

## What the installer does (unchanged behavior)

Installs the stack into **one `uv` tool environment** so the components can import
and drive each other (`biopb server start`, the napari viewer, `ops`), unpacks
the data-browser webapp, writes default config
(`~/.config/biopb/biopb.toml`, `~/.config/biopb-mcp/config.json`,
`~/.config/biopb/mcp.json`), wires biopb-mcp into any detected AI agent
(Claude Code/Desktop, Cursor, opencode, Hermes) as a **stdio** MCP server, and
starts the data server. Idempotent; rerun to upgrade. By default it tracks the
latest stable `release-v*`; `BIOPB_INSTALL_RC=1` tracks the latest release
candidate (a/b/rc prerelease) instead — both install prebuilt release wheels.

## The sourcing change (the heart of Phases 0–1)

**Before:** `biopb` (primary) + `biopb-tensor-server` from the biopb release as
file:// wheels; `biopb-mcp[mcp]>=0.6.0` + `napari[all]` from PyPI.

**After:** all three — `biopb-mcp[mcp]`, `biopb[tensor]`,
`biopb-tensor-server` — pinned to **file:// wheels from the one biopb-mcp
release**; only `napari[all]` still comes from PyPI (it is decoupled and
published normally). One download is one mutually-paired set, so the
PyPI-vs-release skew the pairing exists to prevent is gone by construction.

Concretely, in both scripts:

- `RELEASE_REPO` → `biopb/biopb-mcp`; the release asset match adds a third regex
  `biopb_mcp-*.whl` (which does **not** collide with `biopb-*.whl` — wheel names
  normalize the dash to `biopb_mcp`, so `^biopb-` only matches the SDK).
- Source mode splits the single `REPO` into **two**: biopb-mcp builds from
  `github.com/biopb/biopb-mcp`, while `biopb` + `biopb-tensor-server` still build
  from the `biopb` monorepo (`#subdirectory=biopb-tensor-server`).
- `biopb-mcp[mcp]` is now a file:// wheel ref with the extra applied, not a PyPI
  spec; the old `>=0.6.0` floor (which existed to get stdio-default + drop a
  stray `grpcio-tools` pin) is moot — the release wheel is always a recent build.

**Release CI (Phase 0, `release_bundle.yml`).** The release job already builds
the biopb-mcp wheel and pulls the paired biopb/tensor-server wheels from the URLs
pinned in `pyproject.toml`. This change additionally derives the server release's
asset base from those same pinned URLs and downloads `webapp.tar.gz`, and attaches
`install.sh` / `install.ps1` to the release. Net: one biopb-mcp release now
carries the three wheels + webapp + both installers.

## Phases

- **Phase 0 — release self-sufficiency** *(done)*: `release_bundle.yml` bundles
  `webapp.tar.gz` and attaches the install scripts.
- **Phase 1 — copy + adapt** *(done)*: `install/` (both scripts, README, the
  `test/` Dockerfiles) moved here; sourcing repointed to the biopb-mcp release;
  per-package source mode; matched-pair → matched-triple.
- **Phase 2 — CI/tests**: add a Linux install-smoke workflow that runs
  `install.sh` against a real release in a clean container (the `test/`
  Dockerfiles `COPY ../install.sh`, so they already exercise the local copy).
- **Phase 3 — cutover (cross-repo)**: repoint `biopb.org/install.{sh,ps1}` to
  biopb-mcp's copy; delete `biopb/biopb`'s `install/` and stop attaching
  `install.sh` in its `tensor-server-ci.yaml`; fix biopb-site links if the URL
  shape changes.
- **Phase 4 — auto-updater apply step** (`autoupdater.md` / its tracking issue):
  the installer's launch wrapper grows the "apply a staged update on clean start"
  path, so the updater and the installer share one install routine.

## Verification notes / caveats

- Both scripts are byte-verified to parse (`bash -n`, PowerShell tokenizer).
- The **webapp fetch degrades gracefully**: if a release lacks `webapp.tar.gz`
  the installer logs a warning and the server runs API-only — so Phase 1 does not
  hard-depend on Phase 0 having shipped in any given release.
- Until Phase 3, **`biopb.org/install.sh` still serves biopb's copy**; this
  migrated copy is reachable via the raw GitHub URL or the release asset. Don't
  advertise the one-liner as "migrated" until the host is repointed.
- The installer assumes the `biopb/biopb` and `biopb/biopb-mcp` release assets
  are **public** (unauthenticated `curl`), matching today's behavior.
