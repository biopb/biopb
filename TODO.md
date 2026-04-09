# TODO - biopb/tensor Improvements

## Current Status

Implemented (v0.1):
- [x] ZarrAdapter with LRU caching
- [x] Hdf5Adapter with LRU caching
- [x] OmeTiffAdapter (single-file) with LRU caching
- [x] MultiFileOmeTiffAdapter (basic)
- [x] OmeZarrAdapter with precomputed pyramid support
- [x] Client-side chunk caching with cachey
- [x] Config support for all adapter types
- [x] CLI with --cache-size option
- [x] Virtual scaling (runtime downsampling)
- [x] Precomputed multi-scale pyramid routing
- [x] Metadata transmission protocol (metadata_json)
- [x] Modular adapter architecture (base.py, zarr.py, hdf5.py, tiff.py, ome_zarr.py)

---

## TODO: Multi-file OME-TIFF Improvements

### High Priority

- [ ] **Parse `_metadata.txt` for complete OME-XML**
  - Current: Only reads OME-XML embedded in TIFF tags
  - Needed: Parse companion metadata file for full dataset description
  - File: `src/main/python/biopb/tensor/tiff.py` (MultiFileOmeTiffAdapter)
  - Benefit: Correct channel/Z/T ordering for Micro-Manager datasets

- [ ] **Handle incomplete multi-file datasets**
  - Current: Opens first file, relies on tifffile's auto-discovery
  - Issue: tifffile warns "MMStack series is missing files" for partial datasets
  - Needed: Explicit file list from `_metadata.txt` OME-XML

- [ ] **Add test for true multi-file dataset**
  - Current test uses single `.ome.tif` file
  - Needed: Micro-Manager style `img_0.ome.tif`, `img_1.ome.tif`, ...

### Medium Priority

- [ ] **Cache file handles**
  - Current: Opens/closes TiffFile for each chunk read
  - Improvement: Keep file handles open for frequently accessed files
  - Risk: File descriptor limits

- [ ] **Support Bio-Formats companion format**
  - Binary-only OME-TIFF + companion XML
  - Different from Micro-Manager format

---

## TODO: OME-Zarr Improvements

### High Priority

- [ ] **Support remote OME-Zarr (S3, HTTP)**
  - Current: Only local filesystem paths
  - Needed: fsspec integration for s3://, https:// URLs
  - Files: `config.py`, `cli.py`, `OmeZarrAdapter`

### Completed

- [x] **Multi-resolution pyramid support**
  - Implemented: `reduction_method="precompute"` uses precomputed levels
  - Routing: Exact scale match for precompute, virtual scaling for other methods
  - Slice hints: Base coordinates, server converts to level coords
  - Files: `ome_zarr.py`, `base.py`, `server.py`, `client.py`

### Medium Priority

- [ ] **Add ome-zarr package as optional dependency**
  - Current: Lightweight .zattrs parsing only
  - Enhancement: Use ome-zarr for full metadata validation
  - File: `pyproject.toml` (already added as optional)

---

## TODO: Performance

### High Priority

- [ ] **Add cache statistics to server**
  - Track hit/miss rates per adapter
  - Expose via metrics endpoint or logs

- [ ] **Prefetch adjacent chunks**
  - For sequential access patterns (e.g., Z-stack traversal)
  - Prefetch next chunk while current is being processed

### Medium Priority

- [ ] **Async chunk loading**
  - Current: Synchronous reads
  - Enhancement: asyncio for non-blocking I/O

- [ ] **Compression options**
  - Current: Raw Arrow serialization
  - Enhancement: Optional LZ4/ZSTD for network transfer

---

## TODO: aicsimageio Integration

### Low Priority (deferred)

- [ ] **AicsImageIoAdapter**
  - Wrap aicsimageio for vendor format support (CZI, LIF, ND2)
  - Make it optional dependency (~50-100MB)
  - Use fsspec for network transparency where supported

---

## Testing

- [ ] **Create synthetic test fixtures**
  - Multi-file Micro-Manager dataset (complete + incomplete)
  - OME-Zarr with multiple resolution levels
  - Tiled OME-TIFF (currently only tested non-tiled)

- [ ] **Integration tests**
  - End-to-end: server + client + dask compute
  - Performance benchmarks

- [ ] **CI/CD setup**
  - Download test fixtures from release assets
  - Skip tests gracefully when data unavailable (implemented)

---

## Documentation

- [ ] **Usage examples**
  - Config file examples for each source type
  - Python client usage with dask

- [ ] **Architecture diagram**
  - Adapter pattern
  - Flight protocol flow
  - Caching layers

---

## TODO: Metadata Transmission Protocol

### Completed

- [x] **Add `metadata_json` field to TensorDescriptor**
  - File: `proto/biopb/tensor/descriptor.proto` (field 8)
  - Design: JSON format compatible with OME-NGFF (.zattrs schema)

- [x] **Implement `get_metadata()` in all adapters**
  - OmeZarrAdapter: Returns parsed .zattrs (multiscales, axes, omero)
  - Other adapters: Return empty dict
  - Files: `base.py`, `ome_zarr.py`, `zarr.py`, `hdf5.py`, `tiff.py`

- [x] **Include metadata in FlightInfo**
  - Server populates `metadata_json` in TensorDescriptor
  - File: `server.py`

- [x] **Parse metadata on client side**
  - Added `get_metadata()` method to TensorFlightClient
  - File: `client.py`

### Future: Schema Discovery Protocol

- [ ] **Add GetSchema RPC to proto**
  - Protocol: Generic tensor store, no assumptions about data organization
  - Server provides schema describing leaf organization (plates, wells, resolutions)
  - Client validates expectations and discovers available arrays
  - File: `proto/biopb/tensor/descriptor.proto`, `server.py`

- [ ] **Specialized client implementations**
  - Base `TensorFlightClient` for generic array access
  - `OmeZarrPlateClient` for plate datasets (navigates wells)
  - `MultiResolutionClient` for pyramids (navigates resolution levels)
  - Design: Client-side logic for leaf discovery, not protocol-level

### Design Decisions

- Protocol is a **generic tensor store** - each array_id is independent
- Leaf discovery (plates, wells, resolutions) is **client-side responsibility**
- Metadata format: **JSON** (compatible with OME-NGFF schema)
- Typed protobuf metadata (biopb.image.Pixel) deferred - JSON is more flexible

---

## Notes

### Test Data
- Location: `/home/jiyu/hpc/datasets/mm_test_data`
- Contents: `10 um_MMImages.ome.tif`, `10 um_MMImages_metadata.txt`
- Size: ~10MB
- Not tracked in git (use BIOPB_TEST_DATA_DIR env var)

### Key Files
- `src/main/python/biopb/tensor/base.py` - BackendAdapter interface + utilities
- `src/main/python/biopb/tensor/zarr.py` - ZarrAdapter
- `src/main/python/biopb/tensor/hdf5.py` - Hdf5Adapter
- `src/main/python/biopb/tensor/tiff.py` - OmeTiffAdapter, MultiFileOmeTiffAdapter
- `src/main/python/biopb/tensor/ome_zarr.py` - OmeZarrAdapter (precompute support)
- `src/main/python/biopb/tensor/config.py` - Config parsing
- `src/main/python/biopb/tensor/cli.py` - CLI entrypoint
- `src/main/python/biopb/tensor/client.py` - Python client
- `src/main/python/biopb/tensor/server.py` - Flight server
- `src/test/python/tensor_test.py` - Adapter tests

### Dependencies
- Core: pyarrow, grpcio, tifffile, zarr, dask, cachey
- Optional: ome-zarr (for OME-Zarr metadata), h5py (for HDF5)
