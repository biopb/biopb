"""Test fixtures and fixture factory functions for biopb-tensor-server.

Provides synthetic test data fixtures that replace external data dependencies,
enabling comprehensive tests without requiring BIOPB_TEST_DATA_DIR.

This module is designed to be importable from both:
- biopb-tensor-server/tests
- src/test/python
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def create_multifile_micromanager_dataset(
    tmpdir: str,
    complete: bool = True,
    n_channels: int = 3,
    n_positions: int = 1,
    image_shape: Tuple[int, int] = (64, 64),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, List[str], Dict]:
    """Create Micro-Manager style multi-file dataset.

    Creates a directory with:
    - _metadata.txt (Micro-Manager JSON format with Coords/Metadata keys)
    - img_channel0.tif, img_channel1.tif, etc.

    Args:
        tmpdir: Temporary directory to create dataset in
        complete: If False, skip creating one expected file (incomplete dataset)
        n_channels: Number of channels
        n_positions: Number of positions (for multi-position datasets)
        image_shape: Shape of each image (height, width)
        dtype: Data type for images

    Returns:
        Tuple of (directory_path, file_list, metadata_dict)
    """
    import tifffile

    dir_path = Path(tmpdir)
    file_list = []
    metadata = {"Summary": {"Channels": n_channels, "Positions": n_positions}}

    # Generate expected files and create them
    skip_file_idx = n_channels - 1 if not complete else -1  # Skip last channel if incomplete

    for pos_idx in range(n_positions):
        for chan_idx in range(n_channels):
            # Micro-Manager naming convention
            if n_positions > 1:
                filename = f"img_pos{pos_idx}_channel{chan_idx}.tif"
            else:
                filename = f"img_channel{chan_idx}.tif"

            # Add metadata coords
            coord_key = f"Coords-Default/{filename}"
            metadata[coord_key] = {"ChannelIndex": chan_idx, "PositionIndex": pos_idx}
            meta_key = f"Metadata-Default/{filename}"
            metadata[meta_key] = {
                "UUID": f"uuid-{pos_idx}-{chan_idx}",
                "Width": image_shape[1],
                "Height": image_shape[0],
            }

            # Create the file unless it's the skipped one
            if chan_idx != skip_file_idx:
                filepath = dir_path / filename
                # Create unique data per channel/position for test verification
                data = np.full(image_shape, (pos_idx + 1) * 100 + chan_idx, dtype=dtype)
                tifffile.imwrite(str(filepath), data, photometric="minisblack")
                file_list.append(filename)

    # Write metadata file
    metadata_path = dir_path / "metadata.txt"
    metadata_path.write_text(json.dumps(metadata))

    return str(dir_path), file_list, metadata


def create_5d_6d_micromanager_dataset(
    tmpdir: str,
    n_positions: int = 2,
    n_times: int = 3,
    n_channels: int = 2,
    n_z: int = 4,
    image_shape: Tuple[int, int] = (32, 32),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, List[str], Dict]:
    """Create full 5D/6D MicroManager dataset with position, time, channel, and z dimensions.

    Creates a directory with:
    - metadata.txt (Micro-Manager JSON format with IntendedDimensions, AxisOrder, and Coords)
    - img_pos0_time0_channel0_slice0.tif, etc.

    Args:
        tmpdir: Temporary directory to create dataset in
        n_positions: Number of positions
        n_times: Number of time points
        n_channels: Number of channels
        n_z: Number of z slices
        image_shape: Shape of each image (height, width)
        dtype: Data type for images

    Returns:
        Tuple of (directory_path, file_list, metadata_dict)
    """
    import tifffile

    dir_path = Path(tmpdir)
    file_list = []

    # Build metadata with full 5D/6D structure
    metadata = {
        "Summary": {
            "AxisOrder": ["position", "time", "channel", "z"],
            "IntendedDimensions": {
                "position": n_positions,
                "time": n_times,
                "channel": n_channels,
                "z": n_z,
            },
            "Channels": n_channels,
            "Slices": n_z,
            "Frames": n_times,
            "Positions": n_positions,
        }
    }

    # Generate files for all combinations
    for pos_idx in range(n_positions):
        for time_idx in range(n_times):
            for chan_idx in range(n_channels):
                for z_idx in range(n_z):
                    # Micro-Manager naming convention for 5D/6D
                    filename = f"img_pos{pos_idx}_time{time_idx}_channel{chan_idx}_slice{z_idx}.tif"

                    # Add coords
                    coord_key = f"Coords-Default/{filename}"
                    metadata[coord_key] = {
                        "PositionIndex": pos_idx,
                        "TimeIndex": time_idx,
                        "ChannelIndex": chan_idx,
                        "SliceIndex": z_idx,
                    }
                    meta_key = f"Metadata-Default/{filename}"
                    metadata[meta_key] = {
                        "UUID": f"uuid-{pos_idx}-{time_idx}-{chan_idx}-{z_idx}",
                        "Width": image_shape[1],
                        "Height": image_shape[0],
                    }

                    # Create file with unique data for verification
                    filepath = dir_path / filename
                    # Encode position, time, channel, z in the data for verification
                    value = pos_idx * 1000 + time_idx * 100 + chan_idx * 10 + z_idx
                    data = np.full(image_shape, value, dtype=dtype)
                    tifffile.imwrite(str(filepath), data, photometric="minisblack")
                    file_list.append(filename)

    # Write metadata file
    metadata_path = dir_path / "metadata.txt"
    metadata_path.write_text(json.dumps(metadata))

    return str(dir_path), file_list, metadata


def create_multifile_ome_dataset(
    tmpdir: str,
    complete: bool = True,
    n_files: int = 3,
    image_shape: Tuple[int, int] = (64, 64),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, List[str], Dict]:
    """Create multi-file OME-TIFF dataset with companion _metadata.txt file.

    Creates a directory with:
    - _metadata.txt (OME-XML companion file referencing TIFF files)
    - img_001.tif, img_002.tif, etc. (plain TIFFs, no embedded OME-XML)

    Uses companion-file approach (separate OME-XML) rather than embedded
    OME-XML in each file. This tests the adapter's ability to use TiffSequence
    for multi-file aggregation while parsing companion file metadata.

    Args:
        tmpdir: Temporary directory to create dataset in
        complete: If False, skip creating one expected file (incomplete dataset)
        n_files: Number of TIFF files
        image_shape: Shape of each image (height, width)
        dtype: Data type for images

    Returns:
        Tuple of (directory_path, file_list, metadata_dict)
    """
    import tifffile

    dir_path = Path(tmpdir)
    file_list = []
    skip_file_idx = n_files - 1 if not complete else -1

    # Build OME-XML metadata (companion file format)
    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="test">
"""
    for i in range(n_files):
        filename = f"img_{i+1:03d}.tif"
        uuid = f"urn:uuid:test-{i}"
        ome_xml += f"""    <TiffData FirstC="{i}" FirstT="0" FirstZ="0">
      <UUID FileName="{filename}">{uuid}</UUID>
    </TiffData>
"""
        if i != skip_file_idx:
            filepath = dir_path / filename
            data = np.full(image_shape, i + 1, dtype=dtype)
            tifffile.imwrite(str(filepath), data)
            file_list.append(filename)

    ome_xml += """  </Image>
</OME>"""

    # Write companion metadata file
    metadata_path = dir_path / "_metadata.txt"
    metadata_path.write_text(ome_xml)

    # Parse OME-XML for metadata dict (simplified)
    metadata = {"_format": "OME-XML", "n_files": n_files, "files": file_list}

    return str(dir_path), file_list, metadata


def create_multiresolution_ome_zarr(
    tmpdir: str,
    n_levels: int = 4,
    base_shape: Tuple[int, int] = (256, 256),
    chunk_size: Tuple[int, int] = (64, 64),
    dtype: np.dtype = np.uint8,
    with_axes_names: bool = True,
) -> Tuple[str, List[str], Dict]:
    """Create OME-Zarr with multiple resolution levels.

    Creates a directory with:
    - .zattrs with multiscales metadata
    - Level arrays: 0/, 1/, 2/, ... with proper downsampling ratios

    Args:
        tmpdir: Temporary directory to create dataset in
        n_levels: Number of resolution levels
        base_shape: Shape of base (full resolution) array
        chunk_size: Chunk size for arrays
        dtype: Data type
        with_axes_names: Include axis names in .zattrs

    Returns:
        Tuple of (zarr_path, level_paths, zattrs_dict)
    """
    import zarr

    zarr_path = Path(tmpdir) / "test.ome.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    # Create a zarr group first
    root = zarr.open_group(str(zarr_path), mode="w")

    level_paths = []
    datasets = []

    for level in range(n_levels):
        # Compute level shape by dividing by scale factor
        scale_factor = 2 ** level
        level_shape = (
            base_shape[0] // scale_factor,
            base_shape[1] // scale_factor,
        )
        if level_shape[0] == 0 or level_shape[1] == 0:
            continue  # Skip levels with zero-size dimensions

        # Create level array within the group
        arr = root.create_dataset(
            str(level),
            shape=level_shape,
            chunks=chunk_size,
            dtype=dtype,
        )

        # Populate with distinguishable data (level index as value for easy testing)
        arr[:] = level

        level_paths.append(str(zarr_path / str(level)))

        # Add dataset metadata
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{"type": "scale", "scale": [scale_factor, scale_factor]}]
        })

    # Create .zattrs with multiscales
    if with_axes_names:
        axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    else:
        axes = [{"name": "y"}, {"name": "x"}]

    zattrs = {
        "multiscales": [{
            "name": "test",
            "axes": axes,
            "datasets": datasets,
            "version": "0.4",
        }]
    }

    zattrs_path = zarr_path / ".zattrs"
    zattrs_path.write_text(json.dumps(zattrs))

    return str(zarr_path), level_paths, zattrs


def create_tiled_ome_tiff(
    tmpdir: str,
    shape: Tuple[int, int, int] = (3, 128, 128),
    tile_size: Tuple[int, int] = (32, 32),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, Tuple[int, int, int], Dict]:
    """Create tiled OME-TIFF with embedded metadata.

    Creates a tiled OME-TIFF file with:
    - Multiple channels/planes
    - Tiled structure for chunk-based access
    - Embedded OME-XML metadata

    Args:
        tmpdir: Temporary directory to create file in
        shape: Shape of data (n_planes, height, width) or (height, width)
        tile_size: Tile size (tile_height, tile_width)
        dtype: Data type

    Returns:
        Tuple of (tiff_path, shape, tile_info)
    """
    import tifffile

    # Ensure 3D shape for multi-plane
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])

    tiff_path = Path(tmpdir) / "test.ome.tif"

    # Create data with distinguishable values per plane
    data = np.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        data[i] = i + 1  # Each plane has unique value

    tifffile.imwrite(
        str(tiff_path),
        data,
        photometric="minisblack",
        tile=tile_size,
        metadata={"axes": "CYX"},
    )

    tile_info = {
        "tile_width": tile_size[1],
        "tile_height": tile_size[0],
        "tiles_per_row": (shape[2] + tile_size[1] - 1) // tile_size[1],
        "tiles_per_col": (shape[1] + tile_size[0] - 1) // tile_size[0],
        "n_planes": shape[0],
    }

    return str(tiff_path), shape, tile_info


def create_multi_series_ome_tiff(
    tmpdir: str,
    n_series: int = 3,
    series_shape: Tuple[int, int, int] = (2, 64, 64),
    tile_size: Tuple[int, int] = (32, 32),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, List[str], Dict]:
    """Create OME-TIFF with multiple series (multi-field/multi-position).

    Creates a tiled OME-TIFF file with multiple series, each representing
    a different field/position. Each series has distinguishable data.

    Args:
        tmpdir: Temporary directory to create file in
        n_series: Number of series (fields)
        series_shape: Shape of each series (n_planes, height, width)
        tile_size: Tile size (tile_height, tile_width)
        dtype: Data type

    Returns:
        Tuple of (tiff_path, series_names, series_info)
    """
    import tifffile

    tiff_path = Path(tmpdir) / "multiseries.ome.tif"

    # Create data for each series with distinguishable values
    series_names = []
    series_info = {'n_series': n_series, 'shapes': []}

    # Write multi-series TIFF using tifffile's metadata support
    data_stack = []
    for series_idx in range(n_series):
        series_name = f"Field_{series_idx}"
        series_names.append(series_name)

        # Create data for this series - each series has unique values
        series_data = np.zeros(series_shape, dtype=dtype)
        for plane_idx in range(series_shape[0]):
            # Each series-plane has unique value: series_idx * 100 + plane_idx + 1
            series_data[plane_idx] = series_idx * 100 + plane_idx + 1

        data_stack.append(series_data)
        series_info['shapes'].append(series_shape)

    # Write as OME-TIFF with multiple images (series)
    # tifffile handles this via metadata
    with tifffile.TiffWriter(str(tiff_path), bigtiff=True) as tif:
        for series_idx, series_data in enumerate(data_stack):
            # Write each series as a separate image
            tif.write(
                series_data,
                photometric="minisblack",
                tile=tile_size,
                metadata={
                    'axes': 'CYX',
                    'Name': f"Field_{series_idx}",
                },
            )

    return str(tiff_path), series_names, series_info


def create_companion_ome_dataset(
    tmpdir: str,
    n_files: int = 3,
    image_shape: Tuple[int, int] = (64, 64),
    dtype: np.dtype = np.uint16,
) -> Tuple[str, List[str], Dict]:
    """Create companion OME dataset with .companion.ome file.

    Creates a directory with:
    - sample.companion.ome (OME-XML companion file referencing TIFF files)
    - data_001.tif, data_002.tif, etc. (plain TIFFs referenced by companion)

    Args:
        tmpdir: Temporary directory to create dataset in
        n_files: Number of TIFF files to create
        image_shape: Shape of each image (height, width)
        dtype: Data type

    Returns:
        Tuple of (companion_path, tiff_files, metadata_info)
    """
    import tifffile

    dir_path = Path(tmpdir)

    # Create TIFF files with unique values
    tiff_files = []
    for i in range(n_files):
        filename = f"data_{i+1:03d}.tif"
        filepath = dir_path / filename
        data = np.full(image_shape, i + 1, dtype=dtype)
        tifffile.imwrite(str(filepath), data)
        tiff_files.append(str(filepath))

    # Create companion.ome file with OME-XML referencing TIFFs
    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="companion_test">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" SizeX="{width}" SizeY="{height}" SizeZ="{n_files}" SizeC="1" SizeT="1" Type="uint16">
""".format(width=image_shape[1], height=image_shape[0], n_files=n_files)

    for i in range(n_files):
        filename = f"data_{i+1:03d}.tif"
        uuid = f"urn:uuid:data-{i+1}"
        ome_xml += f"""      <TiffData FirstZ="{i}" FirstC="0" FirstT="0" IFD="0">
        <UUID FileName="{filename}">{uuid}</UUID>
      </TiffData>
"""

    ome_xml += """    </Pixels>
  </Image>
</OME>"""

    # Write companion file
    companion_path = dir_path / "sample.companion.ome"
    companion_path.write_text(ome_xml)

    metadata_info = {
        'n_files': n_files,
        'image_shape': image_shape,
        'companion_path': str(companion_path),
    }

    return str(companion_path), tiff_files, metadata_info


def create_hdf5_dataset(
    tmpdir: str,
    shape: Tuple[int, ...] = (100, 100),
    chunks: Tuple[int, ...] = (50, 50),
    dtype: np.dtype = np.uint8,
    dataset_name: str = "data",
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...]]:
    """Create HDF5 dataset with chunked array.

    Args:
        tmpdir: Temporary directory to create file in
        shape: Shape of dataset
        chunks: Chunk size
        dtype: Data type
        dataset_name: Name of dataset inside HDF5 file

    Returns:
        Tuple of (h5_path, shape, chunks)
    """
    import h5py

    h5_path = Path(tmpdir) / "test.h5"

    with h5py.File(str(h5_path), "w") as f:
        # Create chunked dataset with distinguishable values
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        f.create_dataset(dataset_name, data=data, chunks=chunks)

    return str(h5_path), shape, chunks


def create_zarr_array(
    tmpdir: str,
    shape: Tuple[int, ...] = (128, 128),
    chunks: Tuple[int, ...] = (64, 64),
    dtype: np.dtype = np.uint8,
    dim_labels: Optional[List[str]] = None,
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...]]:
    """Create simple Zarr array for testing.

    Args:
        tmpdir: Temporary directory to create array in
        shape: Shape of array
        chunks: Chunk size
        dtype: Data type
        dim_labels: Optional dimension labels (not stored, just returned for reference)

    Returns:
        Tuple of (zarr_path, shape, chunks)
    """
    import zarr

    zarr_path = Path(tmpdir) / "test.zarr"
    arr = zarr.open_array(
        str(zarr_path),
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
    )

    # Populate with data where each chunk has unique value
    n_chunks_per_dim = [s // c for s, c in zip(shape, chunks)]
    for chunk_indices in np.ndindex(*n_chunks_per_dim):
        # Compute chunk value based on indices
        chunk_value = sum(idx * (max(n_chunks_per_dim) ** i) for i, idx in enumerate(chunk_indices))
        slices = tuple(
            slice(idx * c, min((idx + 1) * c, s))
            for idx, c, s in zip(chunk_indices, chunks, shape)
        )
        arr[slices] = chunk_value + 1  # Avoid zero for better test visibility

    return str(zarr_path), shape, chunks