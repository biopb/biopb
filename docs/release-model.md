# biopb release model

How the monorepo is released. The guiding split: **library distribution (PyPI)**
and **product deployment (GitHub release + Docker)** are different things with
different audiences and cadences, and are driven by different tags.

## Two kinds of release

| | Audience | Mechanism | Tag(s) |
|---|---|---|---|
| **Library distribution** | developers / integrators (`pip install`) | **PyPI** | `v*` (biopb) |
| **Product deployment** | end users (`install.sh`), operators (Docker) | **GitHub release + Docker** | `release-v*` |

PyPI is deliberately **excluded** from the `release-v*` deployment. `biopb`
publishes to PyPI on its own `v*` tag, on its own cadence.

**biopb-mcp is no longer published to PyPI.** It only ever reached end users as
part of the `release-v*` wheel triple the installer `file://`-installs (a plain
`pip install biopb-mcp` could never pull the tensor server or the full-stack
dependency groups anyway), so its PyPI upload carried no product weight and was
retired. Its `mcp-v*` tag is kept purely as a **version marker** (below).
(`biopb-tensor-server` and `biopb-image-base` were never on PyPI either.)

## A release is a coordinated set of tags on one commit

A release **bumps every (changed) subproject** — each on its own version line
(independent numbers, fully traceable). Cut these tags **manually** on the
release commit:

| Tag | Drives | Notes |
|---|---|---|
| `v<A>` | `python-ci` → PyPI: `biopb` | also the client's version marker |
| `mcp-v<C>` | (nothing on its own) | version marker only — mcp isn't on PyPI |
| `server-v<B>` | (nothing on its own) | version marker only — tensor-server isn't on PyPI |
| `release-v<R>` | `release.yaml` | the all-in-one deployment (below) |

Because the per-package tags sit on the **same commit**, `release.yaml`'s
`setuptools_scm` build produces **clean** wheel versions (`biopb 0.6.6`,
`biopb_tensor_server 0.4.6`, `biopb_mcp 0.7.2`) — not `.devN+gSHA`. The wheel
filename still pins the exact provenance.

`biopb-image-base` is the exception: it is the **most stable** component, carries
a **static** version in `biopb-image-runtime/pyproject.toml`, and is bumped by
hand only when it actually changes (see idempotent publish below). It has no tag.

### Release candidates (prereleases, e.g. off `dev`)

A `release-v*` tag whose version is a **PEP 440 prerelease** (`…rc1`, `…a1`,
`…b1`) is treated as a candidate, not a stable release. Tags are branch-agnostic,
so an RC is typically cut on a `dev` commit to validate the full deployment
before it lands on `main`. On a prerelease tag, `release.yaml`:

- marks the **GitHub release** `prerelease: true` (the installer skips
  prereleases — see below);
- pushes each Docker image's **version-pinned** tag (e.g.
  `biopb-tensor-server:0.5.0rc1`) but **does not move `:latest`** — `:latest`
  only tracks a clean `X.Y.Z` release, so an RC never becomes the default pull.

The `v*` PyPI tag (biopb) follows PyPI's own prerelease rules: a `…rc1` version
uploads as a prerelease, which `pip` ignores unless `--pre`.

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
The full-stack installer design (formerly staged in `biopb-mcp/install/`) was
promoted into the root `install/` after the first `release-v*`; the transitional
`biopb-mcp/install/` copy has been removed, so the root `install/` is now the
single source of truth.

## Per-tag workflow summary

| Tag | Workflow | Publishes |
|---|---|---|
| `v*` | `python-ci` | PyPI: biopb |
| `mcp-v*` | — | (version marker only) |
| `server-v*` | — | (version marker only) |
| `release-v*` | `release.yaml` | GitHub release + Docker (tensor-server, image-base) |

`mcp-ci`, `tensor-server-ci`, and `image-runtime-ci` keep their PR test/build
jobs but **do not publish** — the only PyPI publish left is `python-ci` (biopb),
and product publishing is consolidated in `release.yaml`. The old
`mcp-release.yaml`, `mcp-ci`'s `deploy` job, and `tensor-server-ci`'s `publish`
job are all retired.

## Open items

- Confirm `biopb.org/install.sh` serves the repo-root `install/install.sh` (the
  installer was promoted to root post-first-release; the host-side copy/redirect
  should point at it).
- Prerelease test tags (`release-v…rc1`, etc.) publish harmlessly: the installer
  skips the GitHub release and the Docker `:latest` tag is left on the last
  stable image (see "Release candidates" above).
