# biopb release model

How the monorepo is released. The guiding split: **library distribution (PyPI)**
and **product deployment (GitHub release + Docker)** are different things with
different audiences and cadences, and are driven by different tags.

## Two kinds of release

| | Audience | Mechanism | Tag(s) |
|---|---|---|---|
| **Library distribution** | developers / integrators (`pip install`, napari-hub) | **PyPI** | `v*` (biopb), `mcp-v*` (biopb-mcp) |
| **Product deployment** | end users (`install.sh`), operators (Docker) | **GitHub release + Docker** | `release-v*` |

PyPI is deliberately **excluded** from the `release-v*` deployment. `biopb` and
`biopb-mcp` publish to PyPI on their own per-package tags, on their own cadence.
(`biopb-tensor-server` and `biopb-image-base` are never on PyPI.)

## A release is a coordinated set of tags on one commit

A release **bumps every (changed) subproject** — each on its own version line
(independent numbers, fully traceable). Cut these tags **manually** on the
release commit:

| Tag | Drives | Notes |
|---|---|---|
| `v<A>` | `python-ci` → PyPI: `biopb` | also the client's version marker |
| `mcp-v<C>` | `mcp-ci` → PyPI: `biopb-mcp` | also mcp's version marker |
| `server-v<B>` | (nothing on its own) | version marker only — tensor-server isn't on PyPI |
| `release-v<R>` | `release.yaml` | the all-in-one deployment (below) |

Because the per-package tags sit on the **same commit**, `release.yaml`'s
`setuptools_scm` build produces **clean** wheel versions (`biopb 0.6.6`,
`biopb_tensor_server 0.4.6`, `biopb_mcp 0.7.2`) — not `.devN+gSHA`. The wheel
filename still pins the exact provenance.

`biopb-image-base` is the exception: it is the **most stable** component, carries
a **static** version in `biopb-image-runtime/pyproject.toml`, and is bumped by
hand only when it actually changes (see idempotent publish below). It has no tag.

## What `release-v*` produces (`release.yaml`)

One pipeline, built from the tagged commit:

1. **Wheels + webapp** — build `biopb`, `biopb-tensor-server`, `biopb-mcp` wheels
   (+ mcp sdist) and the data-browser `webapp.tar.gz`.
2. **GitHub release `release-v<R>`** — attaches the wheel triple + sdist +
   `webapp.tar.gz` + `install.sh`/`install.ps1`. **This is the installer's single
   source of truth** (it `file://`-installs the wheels; nothing from PyPI except
   `napari[all]`).
3. **Docker publish** (ghcr.io + Docker Hub `jiyuuchc/`), each image tagged with
   **its own package version** (not `R`) + `:latest`:
   - `biopb-tensor-server:<B>`
   - `biopb-image-base:<image-base static version>`

### Idempotent Docker publish (the "did the version change?" check)

Each image is **published only if that version tag is not already in the
registry**:

```bash
if docker manifest inspect ghcr.io/biopb/biopb-image-base:$VER >/dev/null 2>&1; then
  echo "$VER already published — skip"
else
  # build + push :$VER (+ move :latest)
fi
```

Most releases don't touch `image-base`, so its static version is unchanged, the
tag already exists, and CI **skips** it. Bump its static version and it
publishes. The same guard applies to `tensor-server` (usually a fresh `server-v*`
version, so it publishes; re-runs stay clean). This is why `image-base` keeps a
**static** version — a `setuptools_scm` dev version changes every commit via
`+gSHA` and would defeat the check.

## Installer

The user-facing installer is `install.sh` / `install.ps1`, fetched (by end users,
via `biopb.org`) from the **latest `release-v*` GitHub release** of `biopb/biopb`:
`RELEASE_TAG_PREFIX = "release-v"`, and **prereleases are skipped** so an
`a`/`b`/`rc` test release never becomes the default download. Asset regexes and
the `file://` install are unchanged.

**Canonical location: the repo-root `install/`** — this is the copy users track.

**Transition (in progress).** The new full-stack installer design currently lives
in `biopb-mcp/install/` (it replaces the old root installer — see
`biopb-mcp/docs/installer-migration.md`). Once this release model is settled and
the **first `release-v*` is cut**, the new design is promoted into the root
`install/`, `biopb-mcp/install/` is removed, and the root `install/` is the single
copy. Until then both exist; the root one is the destination of record.

## Per-tag workflow summary

| Tag | Workflow | Publishes |
|---|---|---|
| `v*` | `python-ci` | PyPI: biopb |
| `mcp-v*` | `mcp-ci` | PyPI: biopb-mcp |
| `server-v*` | — | (version marker only) |
| `release-v*` | `release.yaml` | GitHub release + Docker (tensor-server, image-base) |

`tensor-server-ci` and `image-runtime-ci` keep their PR test/build jobs but **do
not publish** — publishing is consolidated in `release.yaml`. The old
`mcp-release.yaml` and `tensor-server-ci`'s `publish` job are retired.

## Open items

- **Installer promotion (post-first-release):** move `biopb-mcp/install/` → root
  `install/` (the new design replaces the old root copy), delete
  `biopb-mcp/install/`, and confirm `biopb.org/install.sh` serves the root copy.
- Prerelease test tags (`release-v…a`, etc.) publish harmlessly; the installer
  skips them.
