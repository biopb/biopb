# TODO - biopb/tensor Improvements

## Current Status (v0.2)

Completed:
- [x] ZarrAdapter
- [x] Hdf5Adapter
- [x] OmeTiffAdapter (single-file)
- [x] MultiFileOmeTiffAdapter (basic)
- [x] OmeZarrAdapter with precomputed pyramid support
- [x] AicsImageIoAdapter for vendor formats (CZI, LIF, ND2, DV)
- [x] Client-side chunk caching with cachey
- [x] Config support for all adapter types
- [x] CLI with --cache-size option
- [x] Virtual scaling (runtime downsampling)
- [x] Precomputed multi-scale pyramid routing
- [x] Metadata transmission protocol (metadata_json)
- [x] Modular adapter architecture
- [x] Synthetic test fixtures (no external data dependency)
- [x] Integration tests for server + client + dask compute
- [x] Server-side nested array_id handling for level adapters

---

## TODO: Multi-file OME-TIFF Improvements

### Known Issue

- **Multi-file OME-TIFF fixture creates companion file instead of embedded metadata**
  - Fixture: `create_multifile_ome_dataset()` creates `_metadata.txt` with OME-XML
  - Result: tifffile treats it as single-file dataset (shape=[64,64] instead of [3,64,64])
  - Workaround: Use `create_multifile_micromanager_dataset()` which works correctly
  - Test: `TestMultiFileOmeTiffIntegration::test_multifile_read` (currently skipped)

### Future

- [ ] Handle incomplete multi-file datasets gracefully
- [ ] Cache file handles (risk: file descriptor limits)
- [ ] Support Bio-Formats companion format

---

## TODO: OME-Zarr Improvements

### High Priority

- [ ] Support remote OME-Zarr (S3, HTTP)
  - Current: Only local filesystem paths
  - Needed: fsspec integration for s3://, https:// URLs

### Medium Priority

- [ ] Add ome-zarr package as optional dependency for full metadata validation

---

## TODO: Performance

- [ ] Add cache statistics to server (hit/miss rates per adapter)
- [ ] Prefetch adjacent chunks for sequential access patterns
- [ ] Async chunk loading with asyncio
- [ ] Optional LZ4/ZSTD compression for network transfer

---

## TODO: Multi-Field / Multi-Position Dataset Handling

### Current Issues

1. **Micro-Manager multi-file**: `detect_source_type()` doesn't recognize directories with `metadata.txt` + `img_channel*_*.tif` as `ome-tiff-multifile`
2. **Vendor formats with positions**: aicsimageio scene expansion handles multi-scene, but position metadata may not be properly exposed
3. **Parent directory discovery**: Scanning `FOV_1/`, `FOV_2/` subdirectories produces separate tensors without relationship metadata

### High Priority

- [ ] Fix Micro-Manager multi-file detection (update glob patterns, check for `metadata.txt`)
- [ ] Research multi-field support in vendor formats (ND2, CZI, LIF)

### Future

- [ ] Design position metadata protocol
- [ ] Implement position-aware directory scanning
- [ ] OME-Zarr HCS plate support

---

## TODO: Metadata Transmission Protocol

### Future: Schema Discovery Protocol

- [ ] Add GetSchema RPC for leaf organization discovery
- [ ] Specialized client implementations (OmeZarrPlateClient, MultiResolutionClient)

---

## Key Files

| File | Purpose |
|------|---------|
| `biopb-tensor-server/biopb_tensor_server/base.py` | BackendAdapter interface + utilities |
| `biopb-tensor-server/biopb_tensor_server/server.py` | Flight server |
| `biopb-tensor-server/biopb_tensor_server/fixtures.py` | Synthetic test fixture factories |
| `biopb-tensor-server/tests/conftest.py` | pytest fixtures |
| `biopb-tensor-server/tests/adapter_integration_test.py` | Integration tests |
| `src/main/python/biopb/tensor/client.py` | Python client |
| `src/test/python/tensor_test.py` | Adapter tests |
| `src/test/python/tensor_extended_test.py` | Extended tests (now uses synthetic fixtures) |

---

## Dependencies

Core: pyarrow, grpcio, tifffile, zarr, dask, cachey
Optional: ome-zarr (metadata validation), h5py (HDF5), aicsimageio (vendor formats)