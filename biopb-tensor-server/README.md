# Biopb-Tensor-Server

**TensorFlightServer** is a bio-imaging data server that provides efficient, uniform access to pixel data and metadata across many storage formats.

This sub-project predates the other biopb components and was folded into the biopb stack so the deterministic part of image processing — data access — need not be re-implemented by the agents for every problem. That said, it can also be deployed independently of biopb; see [Deployment](#deployment).

## Core concept

TensorFlightServer has a simple goal: to **efficiently** read very large imaging datasets, and to do it regardless of the storage format the data are in.

Internally, TensorFlightServer works by **_transcoding_** — converting the original data into **Apache Arrow** format — because many legacy formats are simply too difficult to read both lazily and efficiently. Notably, TensorFlightServer's transcode process is:

  - **on demand** - so it only transcodes the data the user cares about
  - at **data-chunk** (not whole-file) granularity — so it skips the parts of a dataset (e.g. a channel that isn't useful) that the user ignores
  - into **multi-resolution pyramid** - so whole-slide gigapixel images can still be visualized

The Arrow data are then served to the user via an Arrow Flight server, which brings a few additional benefits:

  - **Network transparency**: the data can sit anywhere on the network and still be accessible.
  - **Metadata database**: all metadata is centralized in one database and queryable using standard SQL.
  - **Language-agnostic**: the data can be read from any of the languages Arrow Flight supports.

## Supported formats

| Format | Extension(s) | Reader / notes |
|--------|--------------|----------------|
| OME-Zarr | `.zarr/` | Multiscale pyramid, incl. HCS plates; native |
| OME-TIFF | `.ome.tiff`, `.ome.tif` | Single- and multi-file; native (`tifffile`) |
| TIFF | `.tif`, `.tiff` | Standard TIFF and TIFF sequences (heuristic filename pattern) |
| Akoya | `.qptiff` | Native, Akoya PhenoImager multiplex whole-slide format |
| Micro-Manager | NDTiff (`NDTiff.index`), legacy (`metadata.txt`) | Multi-file MM acquisitions; native (`ndtiff`) |
| Zeiss | `.czi`, `.lsm` | Native (`.czi` via `pylibCZIrw`, `.lsm` via `tifffile`) |
| Leica | `.lif` | Native (`bioio-lif`) |
| Nikon | `.nd2` | Native (`bioio-nd2`) |
| DeltaVision | `.dv` | Native (`bioio-dv`) |
| DICOM | `.dcm` | Single files and multi-file series; native (`pydicom`) |
| NIfTI | `.nii`, `.nii.gz` | Native (`nibabel`) |
| MRC | `.mrc` | Native (`rosettasciio`) |
| EMD | `.emd` | Native (`rosettasciio`) both Berkeley and Velox flavors |
| HDF5 | `.h5`, `.hdf5` | Requires explicit dataset path in config |
| Olympus | `.oif`, `.oib` | Java Bio-Formats (`bioio-bioformats`) |
| Imaris | `.ims` | Java Bio-Formats (`bioio-bioformats`) |
| Zeiss (legacy) | `.zvi` | Java Bio-Formats (`bioio-bioformats`) |

## Client

  - Protocol-level support is via Arrow Flight (all major languages).
  - **Python** and **Java** support are first-class, with an SDK that maps tensors to native data types.
  - The Python SDK maps the image data to a Dask array (thread-safe and `dask.distributed`-compatible).
  - The Java SDK maps the image data to an ImgLib2 CellArray.
  - There is no direct Arrow Flight support for JavaScript, but TensorFlightServer ships a small HTTP sidecar for web-browser access.

### Python client SDK

```
pip install biopb[tensor]
```

```python
from biopb.tensor import TensorFlightClient

# Connect to a running server (token is optional in local mode)
client = TensorFlightClient("grpc://localhost:8815", token="your-token")

# List available sources
sources = client.list_sources()

# Get a lazy dask array for a specific tensor, by its globally-unique array_id:
# "source_id/field" for a multi-tensor source, or "source_id" for a single one.
arr = client.get_tensor("my-zarr/0", scale_hint=[1, 2, 2])

# Slice lazily (chunks are fetched on demand)
plane = arr[5, :, :]
print(plane.shape, plane.dtype)

# The lazy array is serializable (graph only, dask.distributed-compatible)
import cloudpickle
serialized = cloudpickle.dumps(arr)

# .compute() triggers the actual data load
mean = plane.mean().compute()
print(f"mean = {mean}")
```

### Java client SDK

```xml
<dependencies>
    <dependency>
        <groupId>io.github.jiyuuchc</groupId>
        <artifactId>biopb</artifactId>
        <version>CURRENT_VERSION</version>
    </dependency>
</dependencies>
```

## Deployment

### Standard

```sh
# install biopb
curl -fsSL https://biopb.org/install.sh | bash

# or on windows
# irm https://biopb.org/install.ps1 | iex

# start control plane
biopb control start
```

### Docker — remote data server

Docker is the standard way to run a **remote, headless data server**: the image
is a Flight-only data plane (one gRPC port, no HTTP sidecar).

```bash
docker run -d --restart unless-stopped \
    --name biopb-tensor \
    -p 8815:8815 \
    -v ${YOUR_DATA_LOCATION}:/data \
    -v biopb-state:/root/.local/state \
    -e BIOPB_TENSOR_TLS=1 \
    jiyuuchc/biopb-tensor-server:latest

docker logs biopb-tensor    # copy the access token, printed once
```

`BIOPB_TENSOR_TLS=1` enables encryption. The server writes the TLS cert to
`/root/.local/state` so it can be reused across restarts.

See [containerize.md](containerize.md) for a complete list of deployment options, including methods for HPC deployment with singularity.

## Configuration

You can create a custom config file to fine-tune server behavior, e.g. specifying multiple data sources. The config covers *what* to serve; *where and how* to expose it (`--host`/`--port`/`--tls`) is set on the command line, not here.

```json
{
  "server": { "log_level": "INFO" },
  "cache": {
    "backend": "file",
    "file_max_segment_mb": 256,
    "file_max_total_gb": 128
  },
  "sources": [
    { "url": "/data", "monitor": true },
    {
      "url": "/experiment.zarr",
      "alias": "my-zarr",
      "type": "zarr",
      "dim_labels": ["z", "y", "x"]
    }
  ]
}
```

## Development

### Requirements

- Python >= 3.10, < 3.13
- pyarrow >= 14.0.0

### Setup
```bash
# From repository root
pip install -e biopb-tensor-server/
```

### CLI Reference

```
biopb-tensor-server serve         Start the gRPC Flight server only
biopb-tensor-server launch        Start the Flight server + HTTP sidecar
biopb-tensor-server validate      Validate a config file (biopb.json)
biopb-tensor-server list-tensors  List all data sources and tensors in a config
biopb-tensor-server config-schema Print the config file's JSON Schema
biopb-tensor-server version       Show version information
biopb-tensor-server diagnose ...  Diagnostic commands for a running server
```

#### Launch

```bash
# Local mode (the default loopback bind — no token required)
biopb-tensor-server launch --config biopb.json

# Remote mode (a public bind — token required, auto-generated if omitted)
biopb-tensor-server launch --config biopb.json --host 0.0.0.0 --token mytoken...

# Over TLS (clients dial grpcs:// and pin the cert on first connect)
biopb-tensor-server launch --config biopb.json --tls

# gRPC only (no web sidecar)
biopb-tensor-server serve --config biopb.json
```

### Testing

```bash
# Server tests (from biopb-tensor-server/)
pip install -e ".[test]"
pytest
```

## License

MIT
