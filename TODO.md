# TODO - biopb/tensor Improvements

## Current Status

Implemented (v0.1):
- [x] ZarrAdapter with LRU caching
- [x] Hdf5Adapter with LRU caching
- [x] OmeTiffAdapter (single-file) with LRU caching
- [x] MultiFileOmeTiffAdapter (basic)
- [x] OmeZarrAdapter with precomputed pyramid support
- [x] AicsImageIoAdapter for vendor formats (CZI, LIF, ND2, DV)
  - Multi-scene file support (1 file → multiple tensors)
  - Scene discovery with auto-expansion
  - LRU caching at adapter level
- [x] Client-side chunk caching with cachey
- [x] Config support for all adapter types
- [x] CLI with --cache-size option
- [x] Virtual scaling (runtime downsampling)
- [x] Precomputed multi-scale pyramid routing
- [x] Metadata transmission protocol (metadata_json)
- [x] Modular adapter architecture (base.py, zarr.py, hdf5.py, tiff.py, ome_zarr.py, aicsimageio.py)

---

## TODO: Multi-file OME-TIFF Improvements

### High Priority

- [x] **Parse `_metadata.txt` for complete OME-XML**
  - Current: Reads companion _metadata.txt (Micro-Manager JSON format) and embedded OME-XML
  - File: `src/main/python/biopb/tensor/tiff.py` (MultiFileOmeTiffAdapter)
  - Implemented: `_parse_metadata_txt()` handles raw OME-XML and Micro-Manager JSON formats

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
  - OmeTiffAdapter: Returns embedded OME-XML as JSON-serializable dict
  - MultiFileOmeTiffAdapter: Returns companion _metadata.txt (Micro-Manager JSON) or embedded OME-XML
  - ZarrAdapter/Hdf5Adapter: Return empty dict (no OME metadata)
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
- `src/main/python/biopb/tensor/tiff.py` - OmeTiffAdapter, MultiFileOmeTiffAdapter (with metadata support)
- `src/main/python/biopb/tensor/ome_zarr.py` - OmeZarrAdapter (precompute support)
- `biopb-tensor-server/biopb_tensor_server/adapters/aicsimageio.py` - AicsImageIoAdapter (vendor formats)
- `biopb-tensor-server/biopb_tensor_server/config.py` - Config parsing, source discovery
- `biopb-tensor-server/biopb_tensor_server/cli.py` - CLI entrypoint
- `src/main/python/biopb/tensor/client.py` - Python client
- `src/main/python/biopb/tensor/server.py` - Flight server
- `src/test/python/tensor_test.py` - Adapter tests
- `src/test/python/tensor_extended_test.py` - Extended tests (requires test data)

### Dependencies
- Core: pyarrow, grpcio, tifffile, zarr, dask, cachey
- Optional: ome-zarr (for OME-Zarr metadata), h5py (for HDF5), aicsimageio (for vendor formats)

---

## TODO: Multi-Field / Multi-Position Dataset Handling

### Problem Statement

Many microscopy acquisition formats support multiple fields of view (positions/FOVs) in a single dataset:

| Format | Multi-field support | Current handling |
|--------|--------------------|------------------|
| Micro-Manager multi-file | `FOV_1/`, `FOV_2/` directories | Each FOV directory → separate source discovery (broken) |
| Nikon ND2 | Single file with multiple positions | Single tensor (no position separation) |
| Zeiss CZI | Scenes within file | Handled via aicsimageio scene expansion ✓ |
| Leica LIF | Scenes within file | Handled via aicsimageio scene expansion ✓ |
| OME-Zarr HCS plate | Wells/positions | Not handled (plate-specific client needed) |

### Current Issues

1. **Micro-Manager multi-file**: `detect_source_type()` doesn't recognize directories with `metadata.txt` + `img_channel*_*.tif` as `ome-tiff-multifile`. Each individual TIFF gets discovered as separate source.

2. **Vendor formats with positions**: `aicsimageio` scene expansion handles multi-scene files, but position metadata may not be properly exposed.

3. **Parent directory discovery**: Scanning `FOV_1/`, `FOV_2/` subdirectories produces separate tensors without relationship metadata.

### Design Questions

1. **Should multi-field be one tensor or many?**
   - Option A: One tensor per position (current partial approach) - simpler, but loses dataset context
   - Option B: One tensor with position dimension - unified, but requires client-side navigation
   - Option C: Hierarchical metadata - tensor per position + dataset-level schema discovery

2. **How to expose position metadata?**
   - Add position index/name to `TensorDescriptor`?
   - New `get_positions()` API on adapters?
   - Schema discovery protocol (see TODO section)?

3. **Config representation**:
   - Explicit: `[[sources.path = "FOV_*/Default", type = "ome-tiff-multifile-multifield"]]`
   - Auto-discovery: Detect position patterns in directory structure

### High Priority

- [ ] **Fix Micro-Manager multi-file detection**
  - Update glob patterns: `img_channel*_position*_time*_z*.tif` (not just `.ome.tif`)
  - Check for `metadata.txt` (not just `_metadata.txt`)
  - Properly group files by position index

- [ ] **Research multi-field support in vendor formats**
  - Test ND2 with multiple positions
  - Test CZI/LIF with multiple scenes vs positions
  - Document how aicsimageio exposes position metadata

### Medium Priority

- [ ] **Design position metadata protocol**
  - Add position info to `TensorDescriptor` or new message
  - Consider relationship to OME-Zarr plate schema

- [ ] **Implement position-aware directory scanning**
  - Detect `FOV_*` / `Position_*` directory patterns
  - Group by acquisition session (same metadata format)

### Future

- [ ] **OME-Zarr HCS plate support**
  - Plate/well/position navigation
  - Specialized `OmeZarrPlateClient` (see Metadata Transmission Protocol TODO)

---

## Notes
