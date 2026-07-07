# Migrating vendor-format reads from aicsimageio to bioio

Status: **in progress** (this doc drives the change on branch `deps/aicsimageio-to-bioio`).

## 1. Why

The tensor server pins `numpy < 2.0` (and, downstream, `scipy < 1.16`), and the
root SDK's `[tensor]` extra pins `numcodecs < 0.16`. None of these are wanted on
their own — they are all *consequences* of one dependency: **aicsimageio**.

The constraint chain:

```
aicsimageio 4.14.0  ──requires──►  tifffile <2023.3.15,>=2021.8.30
        │                                   │
        │                    old tifffile calls ndarray.newbyteorder()
        │                    which numpy 2.0 removed
        │                                   ▼
        │                          forces  numpy < 2.0
        │                                   │
        │                          scipy < 1.16 pinned to *stay* on numpy<2.0
        │                          (scipy>=1.16 requires numpy>=2.0; pulled in
        │                           transitively via dask[array])
        │
        └──requires──►  zarr <2.16  (forces zarr 2.15.x)
                                │
                   zarr 2.15 imports numcodecs.blosc.cbuffer_sizes, removed in
                   numcodecs 0.16 → numcodecs < 0.16 pinned as a workaround
```

So a single abandoned package holds numpy, tifffile, scipy, zarr, and numcodecs
back across the whole workspace. aicsimageio is in **maintenance-only** mode; its
successor from the same group (Allen Institute for Cell Science) is **bioio**.

biopb's own code is already numpy-2.0-clean — the only `newbyteorder` call sites
in the tree are on `numpy.dtype` (which survives in numpy 2.0), not on `ndarray`
(the removed method). The break lives entirely in the old tifffile that
aicsimageio drags in.

### What we get

- `numpy` unpinned → resolves to 2.x
- `scipy` unpinned
- `tifffile` modernized (no `<2023.3.15` ceiling)
- `numcodecs < 0.16` workaround removed
- one deliberate new pin, `zarr < 3` (see §4)

## 2. Why bioio is a safe swap

bioio is the official successor to aicsimageio: same lineage, near drop-in API
(`BioImage` replaces `AICSImage`; `.scenes`, `.set_scene`, `.dask_data`,
`.dims.order`, `.ome_metadata`, `fs_kwargs` all carry over). The one
architectural change that matters here: aicsimageio bundles every format reader
(and the union of their dependency caps) into one package, whereas bioio splits
each reader into its **own** pip package (`bioio-czi`, `bioio-nd2`, …), each with
its own modern, uncapped dependencies. No single package can hold the whole tree
back.

### Coverage (Bio-Formats excluded)

Native (pure-Python) reader parity for everything the `[aics]` extra actually
uses:

| Format (ext)              | biopb adapter    | aicsimageio native | bioio plugin       |
|---------------------------|------------------|--------------------|--------------------|
| Zeiss CZI (`.czi`)        | `ZeissAdapter`   | aicspylibczi       | `bioio-czi`        |
| Zeiss LSM (`.lsm`)        | `ZeissAdapter`   | tifffile           | `bioio-tifffile`   |
| Leica LIF (`.lif`)        | `LeicaAdapter`   | readlif            | `bioio-lif`        |
| Nikon ND2 (`.nd2`)        | `NikonAdapter`   | nd2                | `bioio-nd2`        |
| DeltaVision DV (`.dv`)    | `DvAdapter`      | mrc                | `bioio-dv`         |
| plain TIFF (`.tif`)       | fallback         | tifffile           | `bioio-tifffile`   |
| generic raster/video      | fallback (opt-in)| imageio            | `bioio-imageio`    |
| Olympus OIF/OIB, Imaris,  | `OlympusAdapter`,| Bio-Formats only   | `bioio-bioformats` |
| KLB, FITS, NRRD, ICS, …   | fallback         | (never native)     | (never native)     |

The "Bio-Formats only" rows are already Bio-Formats-only under aicsimageio
(routed through its internal `BioformatsReader`), so nothing native is lost.
bioio also *adds* one native reader aicsimageio lacked: 3i SlideBook
(`bioio-sldy`) — not adopted here (no biopb users of it), noted for completeness.

**`bioio-ome-zarr` is deliberately excluded** from `[aics]`. bioio *can* read
OME-Zarr, but (1) biopb already reads OME-Zarr/Zarr through its own
`OmeZarrAdapter`/`ZarrAdapter`, which win in the registry ahead of the generic
bioio fallback — a `.zarr` store never reaches a bioio reader — and (2)
`bioio-ome-zarr` requires **zarr ≥ 3**, colliding with the `zarr < 3` pin (§4).
Adopting it is the same trigger as the deferred zarr 2→3 port. The transitive
closure of the eight `[aics]` plugins reaches neither `bioio-ome-zarr` nor
`ome-zarr`, and the whole lock stays on zarr 2.18.x.

Note: OME-TIFF is **not** in this table — biopb reads OME-TIFF through its own
pure-tifffile `OmeTiffAdapter` (biopb/biopb#213), which never touched
aicsimageio. The aics/bioio adapter only sees a remote/exotic `.tif` that
`OmeTiffAdapter` declined.

## 3. Thread safety (unchanged)

aicsimageio is not thread-safe; neither is bioio — for the same reason. The
non-safety lives in the per-format reader libraries (`aicspylibczi`, `nd2`,
`readlif`, `mrc`, `tifffile`), which are the *same* libraries under both. bioio
publishes no thread-safety guarantee and adds no internal lock; it hands back a
dask array exactly as aicsimageio did.

biopb already mitigates this: every read is serialized through a per-source
`threading.Lock` (`_io_lock`). **That lock is kept verbatim** — the migration is
thread-safety-neutral. Granularity stays per-source, so different sources still
read in parallel; only concurrent reads of one source serialize (correct for all
the non-zarr readers).

## 4. The `zarr < 3` refinement

Removing aicsimageio's `zarr <2.16` cap lets the resolver float zarr to **3.x on
Python ≥3.11** (Python 3.10 can't use zarr 3, so it would stay on 2.18 — an
*inconsistent* resolution across our `>=3.10,<3.13` range). biopb's own
`ZarrAdapter` / `OmeZarrAdapter` are written against the **zarr 2.x** API, so a
silent jump to zarr 3 would be a separate, larger port.

Fix: pin **`zarr < 3`** explicitly. The Phase 0 spike (§6) proved this resolves
cleanly with numpy 2.5.1 + modern tifffile + unpinned scipy while keeping zarr on
the 2.18.x line biopb targets. A zarr 2→3 migration is thereby an independent
future decision, not a rider on this one.

### The tifffile ↔ zarr coupling (discovered during implementation)

The `zarr < 3` pin has a **second, non-obvious consequence** that the install-time
spike could not see: `OmeTiffAdapter`'s pure-tifffile read path (biopb/biopb#213)
opens each scene as `series.aszarr(level=0, chunkmode="page")` — a tifffile Zarr
store — and reads pixels through it. tifffile's Zarr store is version-coupled:

- **tifffile 2025.5.10** — last release whose `aszarr` store speaks **Zarr 2**
  ("Raise ValueError when using Zarr 3", #296).
- **tifffile 2025.5.21** — "Require Zarr 3 for Zarr stores and remove support for
  Zarr 2 (breaking)". Every release since (incl. 2026.x) `import`s the Zarr-3-only
  `zarr.abc.store` API and raises `ValueError: zarr < 3 is not supported` at read.

So under `zarr < 3`, tifffile must also stay `< 2025.5.21`, or every OME-TIFF read
raises. This is a *runtime* coupling (tifffile declares Zarr only as an optional
extra, so nothing surfaces it at resolve time) — it only appeared when the
OME-TIFF read tests ran. The pin is therefore
**`tifffile >= 2024.8.10, < 2025.5.21`**: the lower bound is numpy-2 safe (older
tifffile called the removed `ndarray.newbyteorder`), the upper bound keeps the
Zarr-2 `aszarr` store. 2025.5.10 (May 2025) is still a large jump forward from the
aicsimageio-era 2023.2.28, and — like `zarr < 3` — it lifts automatically once
biopb's Zarr adapters move to Zarr 3.

## 5. Dependency mapping

### `biopb-tensor-server/pyproject.toml`

Core `dependencies`:
- **remove** `numpy < 2.0`
- **remove** `scipy < 1.16`
- **change** `zarr >= 2.0.0` → `zarr >= 2.0.0, <3`
- **change** `tifffile >= 2021.8.30` → `tifffile >= 2024.8.10, < 2025.5.21`
  (numpy-2 safe **and** Zarr-2 `aszarr` store — see §4)

`[aics]` extra — replace the monolith with per-format plugins:

| Before                    | After          |
|---------------------------|----------------|
| `aicsimageio >= 4.0.0`    | `bioio >= 3.0` |
| `aicspylibczi >= 3.1.1`   | `bioio-czi`    |
| `readlif >= 0.6.4`        | `bioio-lif`    |
| `aicsimageio[nd2]`        | `bioio-nd2`    |
| `aicsimageio[dv]`         | `bioio-dv`     |
| (plain TIFF / Zeiss LSM)  | `bioio-tifffile` |
| (generic imageio)         | `bioio-imageio`|
| (plain/remote OME-TIFF)   | `bioio-ome-tiff` |

`bioio-tifffile` is **required, not optional**: without it `BioImage` cannot read
plain non-OME `.tif` or `.lsm` (`bioio-ome-tiff` only claims OME-TIFF), so
`ZeissAdapter`'s `.lsm` path and the generic fallback's plain-`.tif` path raise
`UnsupportedFileFormatError`. Under aicsimageio these went through its bundled
tifffile reader; bioio splits that into its own plugin.

`[bioformats]` extra — Bio-Formats moves to its own bioio plugin:

| Before             | After               |
|--------------------|---------------------|
| `bioformats_jar`   | `bioio-bioformats`  |
| `scyjava >= 1.9.0` | `scyjava >= 1.9.0`  |
| `cjdk >= 0.5.0`    | `cjdk >= 0.5.0`     |

### Root `pyproject.toml` — `[tensor]` extra

The SDK's own code does not import aicsimageio (only a comment mentions it), so
its `[tensor]` aics dependency is vestigial:
- **remove** `aicsimageio >= 4.14.0`
- **remove** `numcodecs < 0.16` (and its explanatory comment)

## 6. Phase 0 resolvability spike (done — PASS)

`uv pip compile` of the replacement set against `numpy>=2` across the supported
Python range:

| Python | numpy | scipy | tifffile   | zarr (with `<3`) | readers |
|--------|-------|-------|------------|------------------|---------|
| 3.10   | 2.2.6 | 1.15.3| 2025.5.10  | 2.18.3           | aicspylibczi 3.3.1, nd2 0.11.3, readlif 0.6.6, mrc 0.4.0 |
| 3.12   | 2.5.1 | 1.18.0| 2026.6.1   | 2.18.7           | (same) |

Nothing in the bioio plugin tree re-imposes `numpy<2`, `tifffile<2023.3.15`, or
`scipy<1.16`. Verdict: green.

## 7. Change surface

- `biopb-tensor-server/pyproject.toml` — deps (§5)
- `pyproject.toml` (root) — `[tensor]` (§5)
- `biopb_tensor_server/adapters/aicsimageio.py` → renamed `bioio.py`
  - `AICSImage` → `BioImage`, imports, attribute/docstring renames
  - **delete** `_install_ome_parse_dedup()` and its call (patched aicsimageio's
    `OmeTiffReader._get_ome` for the #192 triple-parse; that reader is gone, and
    biopb's OME-TIFF path is pure-tifffile — see §8)
  - keep `_io_lock`, all `claim()` logic, `SOURCE_TYPE` strings (catalog
    back-compat), extension sets
- `biopb_tensor_server/adapters/__init__.py`, `cli.py`, `__init__.py` — import
  path + optional-import guard updates (guard now keys on `bioio`)
- tests: `adapter_unit_test`, `adapter_integration_test`, `multifield_test`,
  `ometiff_descriptor_test`, `cloud_phase2_test`, `tensor_extended_test`,
  `list_flights_test` — swap `importorskip("aicsimageio")` → `"bioio"`; replace
  the aicsimageio reference-oracle in `ometiff_descriptor_test`; delete the
  `TestOmeParseDedup` class (the monkeypatch it covers is gone)
- docs/CI: `deploy.md` numpy note, both `CLAUDE.md` adapter tables,
  `ARCHITECTURE.md`, `benchmarks/biopb-bench.def`
- regenerate `uv.lock`

## 8. Accepted risks / notes

- **OME-TIFF triple-parse (#192).** The deleted dedup optimized aicsimageio's
  `OmeTiffReader`. It does not apply to bioio-ome-tiff, and it was only ever hit
  for OME-TIFFs read through the *fallback* adapter (remote/exotic `.tif` that
  `OmeTiffAdapter` declined) — the common OME-TIFF path is pure-tifffile and
  unaffected. If a large remote OME-TIFF read via the fallback proves slow, that
  is a separate, contained optimization, not a blocker.
- **bioio API drift.** `dims.order`, scene enumeration, and the `ome_metadata`
  model are intended to match aicsimageio but should be confirmed by the
  per-format read verification (§9).
- **Bio-Formats path.** Swapping `bioformats_jar` → `bioio-bioformats` is the
  riskiest rename (Java bridge via a different plugin). It is all-or-nothing with
  the numpy unpin: leaving `[bioformats]` on aicsimageio would re-pin numpy, so
  both move together.

- **Installer.** The `[aics]` extra name is unchanged, so `install/install.sh`,
  `install/biopb-engine.ps1`, the `Dockerfile`, and CI need no edits for the
  rename. Two follow-ups:
  - The installers' Python-3.12 cap was justified by aicsimageio's `lxml<5`
    (no 3.13 wheel). That reason is gone; the comments/warnings in `install.sh`
    and `biopb-engine.ps1` were updated to the real reason — the packages'
    `requires-python <3.13` plus **no cp313 wheel for the CZI reader**
    (`pylibczirw` / `aicspylibczi`). The cap itself stays.
  - **Wheel coverage (deferred to the planned installer-wheel-coverage CI):**
    the new native dep `pylibczirw` (via `bioio-czi`) ships wheels for
    cp310/311/312 × {manylinux x86_64+aarch64, macOS arm64, win_amd64} — good for
    the primary install targets — but **no macOS x86_64 (Intel Mac) wheel** and
    **no cp313**. When the wheel-coverage check lands, add `pylibczirw` (and
    `aicspylibczi`) to its guarded set so an upstream wheel regression or a Python
    bump is caught before release.

## 9. Verification (done)

Resolved versions in the workspace lock: **numpy 2.2.6 / 2.3.5**, scipy
1.15.3–1.18.0, **tifffile 2025.5.10**, **zarr 2.18.x**, bioio 3.4.0 (+ czi/lif/
nd2/dv/ome-tiff/imageio/bioformats plugins). aicsimageio and bioformats-jar are
gone from the lock.

- **Full `biopb-tensor-server/tests/` suite: 1084 passed, 1 skipped, 0 failed.**
  This exercises the OME-TIFF-via-bioio generic-adapter read path, the
  pure-tifffile `aszarr` read path end-to-end (server → client → dask compute),
  the descriptor fast-path parity (now oracled against `bioio.BioImage`), and the
  concurrent-read `_io_lock` path.
- The three physical-scale unit tests (which mock the adapter's reader) pass
  against the renamed `_bio_image` attribute.

**Not covered by the suite (no fixtures):** live CZI / LIF / ND2 / DV reads —
these have no committed sample data, so their bioio plugins are import-verified
but not read-exercised here. Registering one real file of each format and pulling
it through the client is the recommended pre-merge smoke test; `dims.order` and
plugin auto-selection are the likely divergence points if bioio differs from
aicsimageio.
