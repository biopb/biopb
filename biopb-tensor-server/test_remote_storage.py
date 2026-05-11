"""Test remote storage implementation with public S3 buckets."""

import tempfile
from pathlib import Path

# Test 1: Allen Cell public S3 (standard AWS, anonymous access)
print("=" * 60)
print("Test 1: Allen Cell OME-TIFF (tiny, ~0.38MB)")
print("=" * 60)

try:
    import s3fs

    url = "s3://allencell/aics/data_handoff_4dn/crop_seg/27a34691_segmentation.ome.tif"
    print(f"URL: {url}")

    # Direct S3 filesystem creation (anonymous)
    fs = s3fs.S3FileSystem(anon=True)
    fs_path = url[5:]  # Strip s3:// prefix

    print(f"S3FileSystem created")
    print(f"Path: {fs_path}")

    # Check if file exists
    exists = fs.exists(fs_path)
    print(f"  exists(): {exists}")

    if exists:
        info = fs.info(fs_path)
        print(f"  size: {info.get('size', 'N/A')} bytes ({info.get('size', 0) / 1024:.2f} KB)")

    print("[PASS] Allen Cell S3 anonymous access works\n")

except Exception as e:
    print(f"[FAIL] Allen Cell S3 test failed: {e}\n")


# Test 2: IDR/EMBL S3 with custom endpoint
print("=" * 60)
print("Test 2: IDR OME-Zarr with custom endpoint")
print("=" * 60)

try:
    import s3fs

    idr_url = "s3://idr-public/ngff/6001240.zarr"
    print(f"URL: {idr_url}")

    # IDR S3 with custom endpoint and anonymous access
    fs = s3fs.S3FileSystem(
        anon=True,
        endpoint_url="https://uk1s3.embassy.ebi.ac.uk"
    )
    fs_path = idr_url[5:]  # Strip s3:// prefix

    print(f"S3FileSystem created with custom endpoint")
    print(f"Path: {fs_path}")

    # Check if directory exists
    exists = fs.exists(fs_path)
    print(f"  exists(): {exists}")

    isdir = fs.isdir(fs_path)
    print(f"  isdir(): {isdir}")

    # Check for .zattrs
    zattrs_path = fs_path + '/.zattrs'
    has_zattrs = fs.exists(zattrs_path)
    print(f"  has .zattrs: {has_zattrs}")

    if has_zattrs:
        zattrs_content = fs.cat_file(zattrs_path).decode('utf-8')
        print(f"  .zattrs preview: {zattrs_content[:200]}...")

    print("[PASS] IDR/EMBL S3 with custom endpoint works\n")

except Exception as e:
    print(f"[FAIL] IDR S3 test failed: {e}\n")


# Test 3: SourceConfig and adapter creation
print("=" * 60)
print("Test 3: Create adapter and read metadata")
print("=" * 60)

try:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.adapters.tiff import OmeTiffAdapter
    import tifffile
    import s3fs

    # Test with Allen Cell OME-TIFF (tiny)
    allen_url = "s3://allencell/aics/data_handoff_4dn/crop_seg/27a34691_segmentation.ome.tif"
    source_config = SourceConfig(
        type="ome-tiff",
        url=allen_url,
        source_id="allen-test",
    )

    print(f"  SourceConfig.is_remote: {source_config.is_remote}")

    # Create filesystem and open file
    fs = s3fs.S3FileSystem(anon=True)
    fs_path = allen_url[5:]

    # Open TIFF file
    tiff = tifffile.TiffFile(fs.open(fs_path, mode='rb'))
    adapter = OmeTiffAdapter(tiff, source_config.source_id, None)

    # Get tensor descriptor
    desc = adapter.get_tensor_descriptor()
    print(f"  Tensor shape: {list(desc.shape)}")
    print(f"  Tensor dtype: {desc.dtype}")

    print("[PASS] OmeTiffAdapter creation works\n")

except Exception as e:
    print(f"[FAIL] OmeTiffAdapter test failed: {e}\n")


# Test 4: OME-Zarr adapter creation
print("=" * 60)
print("Test 4: OME-Zarr adapter creation")
print("=" * 60)

try:
    from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.remote import CredentialsConfig, CredentialProfile
    import zarr
    from zarr.storage import FSStore
    import s3fs
    import json

    idr_url = "s3://idr-public/ngff/6001240.zarr"
    source_config = SourceConfig(
        type="ome-zarr",
        url=idr_url,
        source_id="idr-test",
    )

    print(f"  SourceConfig.is_remote: {source_config.is_remote}")

    # Create credential profile for IDR
    idr_profile = CredentialProfile(
        name="idr-anon",
        storage_type="s3",
        endpoint_url="https://uk1s3.embassy.ebi.ac.uk",
    )
    idr_config = CredentialsConfig(default_profile="idr-anon", profiles=[idr_profile])

    # Create adapter
    adapter = OmeZarrAdapter.create_from_config_with_credentials(source_config, idr_config)

    # Get tensor descriptor
    desc = adapter.get_tensor_descriptor()
    print(f"  Tensor shape: {list(desc.shape)}")
    print(f"  Tensor dtype: {desc.dtype}")
    print(f"  Chunk shape: {list(desc.chunk_shape)}")

    # Get OME metadata
    metadata = adapter.get_metadata()
    if 'multiscales' in metadata:
        print(f"  Has multiscales: Yes ({len(metadata['multiscales'])} level(s))")

    print("[PASS] OmeZarrAdapter creation works\n")

except Exception as e:
    print(f"[FAIL] OME-Zarr adapter test failed: {e}\n")


# Test 5: Read actual data from OME-Zarr
print("=" * 60)
print("Test 5: Read actual data from IDR OME-Zarr")
print("=" * 60)

try:
    from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.remote import CredentialsConfig, CredentialProfile
    from biopb.tensor.ticket_pb2 import ChunkBounds

    idr_url = "s3://idr-public/ngff/6001240.zarr"
    source_config = SourceConfig(
        type="ome-zarr",
        url=idr_url,
        source_id="idr-read-test",
    )

    idr_profile = CredentialProfile(
        name="idr-anon",
        storage_type="s3",
        endpoint_url="https://uk1s3.embassy.ebi.ac.uk",
    )
    idr_config = CredentialsConfig(default_profile="idr-anon", profiles=[idr_profile])

    adapter = OmeZarrAdapter.create_from_config_with_credentials(source_config, idr_config)

    # Read a small slice
    desc = adapter.get_tensor_descriptor()
    shape = list(desc.shape)

    print(f"  Tensor shape: {shape}")

    # Read first 10x10 chunk (if 2D) or appropriate slice
    if len(shape) >= 2:
        slice_stop = [min(10, shape[0]), min(10, shape[1])]
        if len(shape) >= 3:
            slice_stop.append(min(10, shape[2]))
        bounds = ChunkBounds(start=[0] * len(shape), stop=slice_stop)
    else:
        bounds = ChunkBounds(start=[0], stop=[min(10, shape[0])])

    data = adapter.get_data(bounds)
    print(f"  Read data shape: {data.shape}")
    print(f"  Data dtype: {data.dtype}")
    print(f"  Data min/max: {float(data.min()):.2f}/{float(data.max()):.2f}")

    print("[PASS] Data read from remote OME-Zarr works\n")

except Exception as e:
    print(f"[FAIL] Data read test failed: {e}\n")


print("=" * 60)
print("Summary: Remote storage tests completed")
print("=" * 60)