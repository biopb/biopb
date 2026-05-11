# IDR and EMBL Dataset Recommendations

This file lists recommended public microscopy datasets from IDR (Image Data Resource) and EMBL-EBI
that require manual download due to special endpoint requirements. These datasets are ideal for
benchmarking different file formats and sizes.

## Download Instructions

### IDR S3 Access (EMBL Endpoint)
IDR data is hosted on EMBL's S3-compatible storage with a custom endpoint:

```python
import s3fs

fs = s3fs.S3FileSystem(
    anon=True,
    endpoint_url='https://uk1s3.embassy.ebi.ac.uk'
)

# List available datasets
fs.ls('idr-public/ngff/')  # OME-Zarr datasets
fs.ls('idr-public/ome-tiff/')  # OME-TIFF datasets
```

### HTTP Download
For individual files, use HTTP URLs directly:

```bash
# Example: download IDR NGFF dataset
wget -r -np -nH --cut-dirs=3 https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/6001240.zarr/
```

---

## IDR OME-Zarr (NGFF) Datasets

### Small/Medium Datasets (< 500MB)

| IDR ID | Description | Biology | Estimated Size | URL |
|--------|-------------|---------|----------------|-----|
| `idr-6001240` | High-content screening plate | Cell screening assay | ~50MB | `s3://idr-public/ngff/6001240.zarr` |
| `idr-6001241` | High-content screening plate | Cell screening assay | ~40MB | `s3://idr-public/ngff/6001241.zarr` |
| `idr-0026` | Mitosis tracking time series | Cell division time-lapse | ~200MB | `s3://idr-public/ngff/0026.zarr` |
| `idr-0080` | USHI007A1 HCS plate | Human stem cell imaging | ~300MB | `s3://idr-public/ngff/0080.zarr` |

### Large Datasets (> 500MB)

| IDR ID | Description | Biology | Estimated Size | URL |
|--------|-------------|---------|----------------|-----|
| `idr-0062` | Planaria regeneration screen | Stem cell regeneration | ~2GB | `s3://idr-public/ngff/0062.zarr` |
| `idr-6001056` | Super-resolution nanoscopy | Sub-diffraction imaging | ~1GB | `s3://idr-public/ngff/6001056.zarr` |
| `idr-0101` | Developmental time series | Embryo imaging | ~3GB | `s3://idr-public/ngff/0101.zarr` |

---

## IDR OME-TIFF Datasets

| IDR ID | Description | Size | URL |
|--------|-------------|------|-----|
| `idr-6001240-tiff` | Same data as NGFF version in TIFF | ~50MB | `s3://idr-public/ome-tiff/6001240.tiff` |
| `idr-example-ome` | OME-TIFF example file | ~20MB | `s3://idr-public/ome-tiff/example.ome.tiff` |

---

## EMBL-EBI NGFF Datasets

| Dataset ID | Description | Size | URL |
|------------|-------------|------|-----|
| `embl-ngff-tiny` | NGFF test dataset | <10MB | `s3://embl-ngff-public/tiny.zarr` |
| `embl-ngff-demo` | NGFF demonstration | ~50MB | `s3://embl-ngff-public/demo.zarr` |

---

## Recommended Downloads by Use Case

### For Latency Benchmarks (small, fast downloads)
1. `idr-6001240.zarr` - ~50MB, simple 2D plate
2. `idr-6001241.zarr` - ~40MB, simple structure

### For Throughput Benchmarks (medium size)
1. `idr-0026.zarr` - ~200MB, time series data
2. `idr-0080.zarr` - ~300MB, HCS plate structure

### For Stress Tests (large files)
1. `idr-0062.zarr` - ~2GB, complex multi-level data
2. `idr-6001056.zarr` - ~1GB, super-resolution

### For Format Diversity
1. Download both NGFF and OME-TIFF versions of same dataset (e.g., `idr-6001240`)
2. Compare read performance across formats

---

## HTTP Download URLs (Direct Access)

If S3 access fails, use HTTP URLs:

```
https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/6001240.zarr/.zattrs
https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/6001241.zarr/
https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/0026.zarr/
https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/0062.zarr/
https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/0080.zarr/
https://uk1s3.embassy.ebi.ac.uk/idr-public/ngff/6001056.zarr/
```

---

## Additional Sources (Vendor-Specific Formats)

### CZI Files (Zeiss)
These require aicsimageio adapter. Download from:
- Zeiss sample data: https://www.zeiss.com/microscopy/en/products/software/zeiss-zen/blue-edition/sample-data.html
- AICSImageIO test files: https://downloads.aics.ai/AICSImageIO/ (may require SSL workaround)

### ND2 Files (Nikon)
- Nikon sample data: https://www.nikon.com/products/microscopes/sample-data/
- Often included in aicsimageio test suite

### LIF Files (Leica)
- Leica sample data: Contact Leica or use aicsimageio test files

---

## Notes

1. **IDR S3 Endpoint**: IDR uses EMBL's custom S3 endpoint, not standard AWS S3.
   Use `endpoint_url='https://uk1s3.embassy.ebi.ac.uk'` when accessing via s3fs.

2. **Dataset Availability**: Some IDR IDs may not have NGFF versions.
   Check `idr.openmicroscopy.org` for dataset availability.

3. **Network Requirements**: These tests are marked to skip automatically when
   network is unavailable (see conftest.py `is_s3_source()` checks).

4. **Size Estimates**: Sizes are approximate; actual size depends on pyramid
   levels and compression.