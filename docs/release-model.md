# biopb release model

How the monorepo is released. The guiding split: **library distribution (PyPI)**
and **product deployment (GitHub release + Docker)** are different things with
different audiences and cadences, and are driven by different tags.

## Two kinds of release, two tags

The whole scheme is **two version lines, two tags**:

| | Audience | Mechanism | Tag | Members |
|---|---|---|---|---|
| **SDK / library** | developers / integrators | **PyPI + Maven Central** (+ a Docker base image) | `v*` | `biopb` (Python → PyPI, Java → Maven Central) and `biopb-image-base` (Docker base image only) |
| **Product / deployment** | end users (`install.sh`), operators (Docker) | **GitHub release + Docker** | `release-v*` | `biopb-tensor-server`, `biopb-mcp`, `biopb-control`, and the `web/` bundle |

Every product component reads the **single `release-v*` tag** (setuptools_scm /
`sync-version.js`), so the tensor server, mcp, control, and the web bundle always
share one product version — there are no per-package `server-v*` / `mcp-v*` /
`control-v*` tags anymore. Everything on the SDK side reads `v*`.

PyPI is deliberately **excluded** from the `release-v*` deployment: `biopb`
publishes to PyPI (and Maven Central) on its own `v*` tag, on its own cadence.
`biopb-mcp`, `biopb-tensor-server`, and `biopb-control` are **not** on PyPI — they
only reach end users as the `release-v*` wheel bundle the installer
`file://`-installs. `biopb-image-base` is not on PyPI either; it ships **only** as
a Docker base image, versioned off the SDK `v*` line (it's the foundation others
build compute servers on).

## Cutting a release: at most two tags

At most **two tags** on the release commit — no per-package markers to keep in
sync:

| Tag | Cut it when | Drives |
|---|---|---|
| `v<A>` | the SDK changed (or you want image-base rebuilt) | `python-ci` → PyPI (`biopb`), `java-ci` → Maven Central, and image-base's Docker version |
| `release-v<R>` | the product changed | `release.yaml` — the all-in-one product deployment (below) |

Because each package reads its tag from the **same release commit**,
`release.yaml`'s `setuptools_scm` build produces **clean** wheel versions
(`biopb_tensor_server 0.11.0`, `biopb_mcp 0.11.0`, `biopb_control 0.11.0` — all
`R`) — not `.devN+gSHA`. The wheel filename still pins the exact provenance.

A product-only change needs only `release-v<R>`. An SDK/protocol change needs
`v<A>` (which also rebuilds `image-base`, since it tracks the SDK line — see
idempotent publish below); if that same commit also ships product changes, cut
`release-v<R>` too. A commit that is not on a clean tag (a dry run) just yields a
`.devN+gSHA` version, which is fine for testing.

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
   its version + `:latest`:
   - `biopb-tensor-server:<R>` — the product/release version.
   - `biopb-image-base:<A>` — the SDK `v*` version (image-base tracks the SDK
     line, not the product), read from the nearest `v*` tag name.

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

A **product-only** release doesn't cut a new `v*`, so image-base's version (the
nearest `v*` tag) is unchanged, its tag already exists, and CI **skips** it — the
same "rebuild only when it changes" property it had as a static version, now
driven by the `v*` tag instead of a hand-bump. `tensor-server` gets a fresh `R`
each release, so it publishes; re-runs stay clean. The check needs a **clean**
version with no `+gSHA` local segment (a Docker reference forbids `+`): image-base
uses the nearest `v*` **tag name** (always clean), and tensor-server's dev/local
version is sanitized to a Docker-safe form on a dry run.

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
| `v*` | `python-ci`, `java-ci` | PyPI (`biopb`) + Maven Central (`biopb` Java); image-base's Docker version tracks this tag |
| `release-v*` | `release.yaml` | GitHub release + Docker (`tensor-server:R`, `image-base:A`) |

`mcp-ci`, `tensor-server-ci`, `control-ci`, and `image-runtime-ci` keep their PR
test/build jobs but **do not publish** — the only PyPI/Maven publish is on `v*`
(`python-ci` / `java-ci`), and product publishing is consolidated in
`release.yaml`. There are no `server-v*` / `mcp-v*` / `control-v*` tags anymore.

## Open items

- Confirm `biopb.org/install.sh` serves the repo-root `install/install.sh` (the
  installer was promoted to root post-first-release; the host-side copy/redirect
  should point at it).
- Prerelease test tags (`release-v…rc1`, etc.) publish harmlessly: the installer
  skips the GitHub release and the Docker `:latest` tag is left on the last
  stable image (see "Release candidates" above).
