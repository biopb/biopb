# biopb release model

How the monorepo is released. The guiding split: **library distribution (PyPI)**,
**the tensor-server image**, and **product deployment (GitHub release)** are
different things with different audiences and cadences, and are driven by
different tags.

## Three release lines, three tags

The scheme is **three version lines, three tags**:

| | Audience | Mechanism | Tag | Members |
|---|---|---|---|---|
| **SDK / library** | developers / integrators | **PyPI + Maven Central** + a Docker base image | `v*` | `biopb` (Python → PyPI, Java → Maven Central) and `biopb-image-base` (Docker base image) |
| **Tensor server** | operators | **Docker** | `server-v*` | `biopb-tensor-server` (Docker image) |
| **Product / deployment** | end users (`install.sh`), operators | **GitHub release** | `release-v*` | `biopb-tensor-server`, `biopb-mcp`, `biopb-control`, and the `web/` bundle (wheels + webapp) |

Each package reads exactly one tag prefix (setuptools_scm `tag_regex` +
`git describe --match`, or `sync-version.js`):

- **`biopb`** and **`biopb-image-base`** read `v*`.
- **`biopb-tensor-server`** reads `server-v*` — its own version line.
- **`biopb-mcp`**, **`biopb-control`**, and the **`web/` bundle** read `release-v*`,
  so those three always share one product version.

Note that **the tensor server appears in two lines**: its **version** and its
**Docker image** are keyed to `server-v*`, but its **wheel** is still bundled into
the `release-v*` GitHub release (so the installer `file://`-installs it alongside
mcp/control). That wheel therefore carries its own `server-v*`-derived version,
which need not equal the `release-v*` version of the bundle it ships in.

PyPI is deliberately **excluded** from the `release-v*` deployment: `biopb`
publishes to PyPI (and Maven Central) on its own `v*` tag, on its own cadence.
`biopb-mcp`, `biopb-tensor-server`, and `biopb-control` are **not** on PyPI — they
only reach end users as the `release-v*` wheel bundle the installer
`file://`-installs. `biopb-image-base` is not on PyPI either; it ships **only** as
a Docker base image, versioned off the SDK `v*` line (it's the foundation others
build compute servers on).

## Cutting a release: up to three tags

Put whichever of the three tags apply on the release commit:

| Tag | Cut it when | Drives |
|---|---|---|
| `v<A>` | the SDK changed (or you want image-base rebuilt) | `python-ci` → PyPI (`biopb`), `java-ci` → Maven Central, **`image-runtime-ci` → `biopb-image-base:<A>` Docker** |
| `server-v<S>` | the tensor server changed and you want a new image | **`tensor-server-ci` → `biopb-tensor-server:<S>` Docker** |
| `release-v<R>` | the product bundle changed | `release.yaml` → the GitHub release bundle (below) |

The tags are independent: an SDK-only change is just `v<A>`; a tensor-server
image bump is just `server-v<S>`; a product bundle is just `release-v<R>`. Cut
several on one commit when one commit changes several lines. A commit that is not
on a clean tag (a dry run) just yields a `.devN+gSHA` version, which is fine for
testing.

Because each product-bundle package reads `release-v*` from the **same release
commit**, `release.yaml`'s `setuptools_scm` build produces **clean** wheel
versions for mcp/control (`biopb_mcp 0.11.0`, `biopb_control 0.11.0`) — not
`.devN+gSHA`. The tensor-server wheel in that bundle is versioned off the nearest
`server-v*` tag instead, so at a bundle cut it is that tag's version (`+ .devN` if
`release-v*` is ahead of the last `server-v*`).

### Release candidates (prereleases, e.g. off `dev`)

A tag whose version is a **PEP 440 prerelease** (`…rc1`, `…a1`, `…b1`) is treated
as a candidate. Tags are branch-agnostic, so an RC is typically cut on a `dev`
commit to validate before it lands on `main`.

- **`release-v…rc1`** marks the **GitHub release** `prerelease: true` (the
  installer skips prereleases — see below).
- **`server-v…rc1`** / **`v…rc1`** push the image's **version-pinned** tag (e.g.
  `biopb-tensor-server:0.5.0rc1`) but **do not move `:latest`** — `:latest` only
  tracks a clean `X.Y.Z` release, so an RC never becomes the default pull.

The `v*` PyPI tag (biopb) follows PyPI's own prerelease rules: a `…rc1` version
uploads as a prerelease, which `pip` ignores unless `--pre`.

## What each tag produces

### `release-v*` → the GitHub release bundle (`release.yaml`)

One pipeline, built from the tagged commit — **no Docker**:

1. **Wheels + webapp** — build `biopb`, `biopb-tensor-server`, `biopb-mcp`,
   `biopb-control` wheels (+ mcp sdist) and the data-browser `webapp.tar.gz`, plus
   the curated `biopb-samples.tar.gz`.
2. **GitHub release `release-v<R>`** — attaches the wheel set + sdist +
   `webapp.tar.gz` + `biopb-samples.tar.gz` + `versions.json` + `SHA256SUMS` +
   `install.sh`/`install.ps1` + the Windows GUI installer. **This is the
   installer's single source of truth** (it `file://`-installs the wheels; nothing
   from PyPI except `napari[all]`).

`versions.json` carries `release`, `tensor_server` (the shipped wheel's version),
and `napari` (the pinned Qt binding). Docker versions are **not** in it — the
images ship on their own tags.

### `server-v*` → the tensor-server image (`tensor-server-ci`)

On a `server-v*` tag, `tensor-server-ci`'s `publish` job builds the
`biopb-tensor-server` image (biopb + tensor-server + control wheels + the webapp)
and pushes it to **ghcr.io + Docker Hub `jiyuuchc/`**, tagged with the version
(and `:latest` for a clean `X.Y.Z`). Otherwise this workflow only runs
tests/verify builds on branch pushes & PRs.

### `v*` → PyPI/Maven + the image-base image (`image-runtime-ci`)

On a `v*` tag, `python-ci`/`java-ci` publish `biopb` to PyPI/Maven, and
`image-runtime-ci`'s `publish` job builds the `biopb-image-base` image (biopb +
tensor-server wheels) and pushes it to **ghcr.io + Docker Hub `jiyuuchc/`**,
tagged with the SDK version (and `:latest` for a clean `X.Y.Z`).

### Idempotent Docker publish (the "did the version change?" check)

Each image `publish` job (tensor-server-ci, image-runtime-ci) is **idempotent**:
it publishes only if that version tag is missing from **at least one** of the two
registries (ghcr.io, Docker Hub), so re-running a tag with both already present is
a noop, while a partial prior publish (one registry pushed, the other failed)
re-runs to fill the gap:

```bash
if docker manifest inspect ghcr.io/biopb/biopb-image-base:$VER >/dev/null 2>&1 \
   && docker manifest inspect docker.io/jiyuuchc/biopb-image-base:$VER >/dev/null 2>&1; then
  echo "$VER already in both registries — skip"
else
  # build + push :$VER to both (+ move :latest for a clean X.Y.Z)
fi
```

The check needs a **clean** version with no `+gSHA` local segment (a Docker
reference forbids `+`): both jobs derive `$VER` straight from the tag name
(`${GITHUB_REF#refs/tags/…}`), which is always clean, so no sanitization is
needed.

## Installer

The user-facing installer is `install.sh` / `install.ps1`, fetched (by end users,
via `biopb.org`) from a `release-v*` GitHub release of `biopb/biopb`:
`RELEASE_TAG_PREFIX = "release-v"`. Asset regexes and the `file://` install are
unchanged.

**Every shipped installer is pinned to its paired release.** The scripts are
published *with* each release (from the tagged commit), and `release.yaml`'s **Pin
installers to this release** step stamps the exact tag into `BIOPB_PINNED_RELEASE`
(`install.sh`) / `$script:BiopbPinnedRelease` (`install.ps1`, `biopb-engine.ps1`)
*before* they ship — so this holds for **all** the copies that leave a release:
the GitHub-release **assets** (`install.sh`/`install.ps1`), the **biopb.org**
publish, and the Windows **`.exe`** engine (stamped in the `windows-installer`
job). Stamping runs for **every** `release-v*` tag, stable or rc, so a copy
downloaded from an rc release installs that rc — only the *biopb.org rsync* is
gated to stable (that canonical URL tracks the latest stable release). The result:
any installer you download from a given release installs the *exact* release it
came from, not "whatever is newest at run time"; re-fetching is how you move
forward. A **raw / git-checkout** copy has an empty pin and tracks the **latest
stable** release (prereleases skipped).

`install.ps1` is a thin bootstrapper that loads the install *engine*
(`biopb-engine.ps1`) and drives it. When it is pinned (or `BIOPB_INSTALL_VERSION`
is set) it fetches the engine **from that release's GitHub assets** — a versioned
copy that matches the wheels — instead of the unversioned biopb.org one, falling
back to biopb.org only if the release predates the engine-as-asset. So a lone
`install.ps1` downloaded from a release is self-contained: one script resolves a
release and pulls the engine **and** wheels from it, exactly like `install.sh`
(no separate engine download — which is why `biopb-engine.ps1` is also a release
asset). A sibling `biopb-engine.ps1` on disk (a checkout) still wins. Overrides
(all paths): `BIOPB_INSTALL_VERSION=X.Y.Z` installs/downgrades to an exact
release; `BIOPB_INSTALL_RC=1` tracks the latest candidate (ignores the pin, since
rc builds are not published to `biopb.org`).

**Canonical location: the repo-root `install/`** — this is the copy users track.
The full-stack installer design (formerly staged in `biopb-mcp/install/`) was
promoted into the root `install/` after the first `release-v*`; the transitional
`biopb-mcp/install/` copy has been removed, so the root `install/` is now the
single source of truth.

## Per-tag workflow summary

| Tag | Workflow | Publishes |
|---|---|---|
| `v*` | `python-ci`, `java-ci`, `image-runtime-ci` | PyPI (`biopb`) + Maven Central (`biopb` Java) + Docker `biopb-image-base:A` |
| `server-v*` | `tensor-server-ci` | Docker `biopb-tensor-server:S` |
| `release-v*` | `release.yaml` | GitHub release (wheel set + sdist + webapp + samples + installers) — **and**, for a stable tag, the canonical `biopb.org/{install.sh,install.ps1,biopb-engine.ps1}` |

The canonical install scripts are published by a **step inside `release.yaml`**,
after the GitHub release is created (formerly a standalone push-to-main
`install-scripts.yaml`). Folding it in means the live installer publishes *only*
if the release succeeds — a failed release can never leave a `biopb.org`
installer newer than any release it can install. The step is gated to a **stable**
tag: the `guard` job already blocks an off-main final tag, and prereleases
(`…rc/a/b`) are skipped so the canonical URL keeps tracking the latest stable
release (`install.sh` defaults to stable; `BIOPB_INSTALL_RC=1` opts into
candidates on demand).

`mcp-ci` and `control-ci` keep their PR test/build jobs but **do not publish**.
`tensor-server-ci` and `image-runtime-ci` keep their PR test/verify-build jobs and
additionally **publish their Docker image on their tag** (`server-v*` / `v*`).
There are no `mcp-v*` / `control-v*` tags — mcp/control/web ship together on
`release-v*`.

## Open items

- Confirm `biopb.org/install.sh` serves the repo-root `install/install.sh` (the
  installer was promoted to root post-first-release; the host-side copy/redirect
  should point at it).
- Prerelease test tags (`release-v…rc1`, `server-v…rc1`, `v…rc1`) publish
  harmlessly: the installer skips the GitHub release and the Docker `:latest` tag
  is left on the last stable image (see "Release candidates" above).
