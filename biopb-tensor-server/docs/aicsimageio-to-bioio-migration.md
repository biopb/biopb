# Vendor-format reads via bioio

Status: **complete** — the tensor server reads vendor microscopy formats through bioio (`adapters/bioio.py`), the maintained successor to aicsimageio. This is the *why we're on bioio + what stays pinned* reference.

## Why: aicsimageio held the whole workspace back

`aicsimageio` is in maintenance-only mode, and a single transitive constraint chain from it pinned numpy, tifffile, scipy, zarr, and numcodecs across the tree:

```
aicsimageio 4.14.0 ──► tifffile <2023.3.15   (old tifffile calls ndarray.newbyteorder(),
        │                                      removed in numpy 2.0)
        │              ⇒ forces numpy < 2.0
        │              ⇒ scipy < 1.16 pinned to stay on numpy<2.0
        │                (scipy>=1.16 needs numpy>=2.0; pulled in via dask[array])
        └──► zarr <2.16 ⇒ zarr 2.15.x imports numcodecs.blosc.cbuffer_sizes
                          (removed in numcodecs 0.16) ⇒ numcodecs < 0.16 pinned
```

biopb's own code was already numpy-2 clean — its only `newbyteorder` call sites are on `numpy.dtype` (survives in numpy 2), not `ndarray`. The break lived entirely in the old tifffile aicsimageio dragged in. Migrating to **bioio** unpins numpy (→2.x), scipy, tifffile, and drops the `numcodecs<0.16` workaround — in exchange for one deliberate pin, `zarr<3` (below).

## Why bioio is a safe swap

bioio is the official successor from the same group (Allen Institute for Cell Science), near drop-in: `BioImage` replaces `AICSImage`; `.scenes`, `.set_scene`, `.dask_data`, `.dims.order`, `.ome_metadata`, `fs_kwargs` all carry over. The architectural win: aicsimageio bundled every reader (and the union of their dependency caps) into one package; bioio splits each reader into its own `bioio-*` pip package with modern, uncapped deps, so no single package can hold the tree back — and a slim install pulls only the formats it needs.

## Adapter structure

`adapters/bioio.py` — `_BioioAdapterBase` (dual-role: source-level lists scenes, scene-level does data access; multi-scene files expose each scene as a separate tensor) plus format-specific subclasses that set a meaningful `SOURCE_TYPE`:

| Subclass | `SOURCE_TYPE` | Claims | bioio plugin |
|---|---|---|---|
| `ZeissAdapter` | `zeiss` | `.czi`, `.lsm` | `bioio-czi`, `bioio-tifffile` |
| `LeicaAdapter` | `leica` | `.lif` | `bioio-lif` |
| `NikonAdapter` | `nikon` | `.nd2` | `bioio-nd2` |
| `DvAdapter` | `dv` | `.dv` | `bioio-dv` |
| `OlympusAdapter` | `olympus` | `.oif`, `.oib` | `bioio-bioformats` |
| `BioformatsAdapter` | `bioformats` | `.zvi`/`.lei`/`.vsi` (gated on `bioio_bioformats` importable) | `bioio-bioformats` |
| `AicsImageIoAdapter` (fallback) | `aics` | plain `.tif`, microscopy/scientific exts, remote/exotic `.tif` OmeTiffAdapter declined | `bioio-tifffile`, `bioio-imageio` |

`SOURCE_TYPE` strings are kept verbatim for catalog back-compat. bioio *adds* a native SlideBook reader (`bioio-sldy`) aicsimageio lacked — not adopted (no users). OME-TIFF is **not** here: biopb reads it through its own pure-tifffile `OmeTiffAdapter` (biopb/biopb#213); the bioio fallback only sees a remote/exotic `.tif` that adapter declined.

## The pins that remain, and why

`biopb-tensor-server/pyproject.toml` core deps:

- **`zarr >= 2.0.0, <3`** — biopb's own `ZarrAdapter`/`OmeZarrAdapter` are written against the zarr 2.x API. Removing aicsimageio's cap would let the resolver float zarr to 3.x on Python ≥3.11 only (3.10 can't use zarr 3) — an *inconsistent* resolution across the `>=3.10,<3.13` range, and a silent jump would be a separate, larger port. Pinning `<3` keeps the lock on the 2.18.x line and makes zarr 2→3 an independent future decision.
- **`tifffile >= 2024.8.10, < 2025.5.21`** — the non-obvious consequence of `zarr<3`. `OmeTiffAdapter`'s pure-tifffile read path opens each scene as `series.aszarr(...)`, a tifffile Zarr store, and that store is version-coupled:
  - **tifffile 2025.5.10** — last release whose `aszarr` store speaks **Zarr 2**.
  - **tifffile 2025.5.21** — requires Zarr 3, removes Zarr 2 (breaking); every later release raises `ValueError: zarr < 3 is not supported` at read.

  So under `zarr<3`, tifffile must stay `<2025.5.21` or every OME-TIFF read raises. This is a **runtime** coupling (tifffile declares Zarr only as an optional extra, invisible at resolve time — it surfaced only when OME-TIFF read tests ran). Lower bound is numpy-2 safe. Both this ceiling and `zarr<3` lift automatically once biopb's zarr adapters move to Zarr 3.

### Extra layout (`[aics]`, `[czi]`, `[bioformats]`)

- **`[aics]`** = `bioio>=3.0` + `bioio-lif`/`bioio-nd2`/`bioio-dv`/`bioio-tifffile`/`bioio-ome-tiff`/`bioio-imageio`. `bioio-tifffile` is **required, not optional** — without it `BioImage` cannot read plain non-OME `.tif` or `.lsm` (`bioio-ome-tiff` only claims OME-TIFF), so `ZeissAdapter`'s `.lsm` path and the generic fallback raise `UnsupportedFileFormatError`.
- **`bioio-ome-zarr` deliberately excluded** — (1) redundant: biopb reads OME-Zarr/Zarr via its own `OmeZarrAdapter`/`ZarrAdapter`, which win in the registry ahead of the generic bioio fallback (a `.zarr` store never reaches a bioio reader); (2) it requires **zarr ≥ 3**, colliding with the `zarr<3` pin. The transitive closure of the `[aics]` plugins reaches neither it nor `ome-zarr`, so the lock stays on zarr 2.18.x.
- **`[czi]`** = `bioio-czi`, carved out of `[aics]`. Its `pylibczirw` dep publishes arm64-macOS/Linux/Windows wheels but **no Intel-macOS wheel** (any version), so bundling it in the default set would source-build libCZI (cmake + compiler) and fail on every Intel Mac. Installers add `[czi]` on every platform except Intel macOS; a direct install wanting CZI everywhere uses `biopb-tensor-server[aics,czi]`. Also **no cp313 wheel** — one of the reasons for the `requires-python <3.13` cap.
- **`[bioformats]`** = `bioio-bioformats` + `scyjava>=1.9.0` + `cjdk>=0.5.0`. Java Bio-Formats fallback for legacy formats with no pure-Python reader (ZVI the headline case); a JDK is downloaded lazily by scyjava/cjdk on first read, not a build/system dep.
- Root `pyproject.toml` `[tensor]` extra: aicsimageio was vestigial there (only a comment referenced it) — removed, along with the `numcodecs<0.16` workaround.

Resolved lock: numpy 2.2.6/2.3.5, scipy 1.15.3–1.18.0, tifffile 2025.5.10, zarr 2.18.x, bioio 3.4.0 (+ czi/lif/nd2/dv/ome-tiff/imageio/bioformats plugins). aicsimageio and bioformats-jar are gone.

## Gotchas

- **Thread safety is unchanged — and unsafe.** Neither aicsimageio nor bioio is thread-safe; the non-safety lives in the per-format reader libs (`aicspylibczi`, `nd2`, `readlif`, `mrc`, `tifffile`), the *same* under both. biopb serializes every read through a per-source `threading.Lock` (`_io_lock`), kept verbatim. Granularity stays per-source: different sources read in parallel, concurrent reads of one source serialize.
- **The RGB/samples OME descriptor trap.** `list_tensor_descriptors` fast-paths shapes from OME metadata (no scene switching) *only* when `dims.order == "TCZYX"`. An RGB/samples source reports `dims.order` `"TCZYXS"` (bioio folds interleaved samples into a trailing S axis; its dask shape carries C=1,S=3 where OME reports C=3,no-S), so the canonical 5-D OME shape would disagree with the 6 labels and yield a malformed descriptor that fails to open. Those are deferred to the authoritative scene-switching fallback.
- **Scene-index resolution avoids re-parsing OME-XML.** `_scene_index_for_field` prefers the cached descriptor order (position = scene index) so a read does not re-enumerate `BioImage.scenes`, which would trigger the OME-XML parse the fast path avoided at registration (biopb/biopb#168).
- **OME-TIFF triple-parse dedup is gone.** The old `_install_ome_parse_dedup()` monkeypatched aicsimageio's `OmeTiffReader._get_ome` (biopb/biopb#192); that reader no longer exists and biopb's common OME-TIFF path is pure-tifffile, so it was deleted. A slow large *remote* OME-TIFF read via the bioio fallback would be a separate, contained optimization, not a blocker.
- **CZI/LIF/ND2/DV reads are import-verified, not read-exercised** — no committed sample fixtures. `dims.order` and plugin auto-selection are the likely divergence points if bioio ever differs from aicsimageio; a real file of each format through the client is the recommended smoke test.
