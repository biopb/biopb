# Biopb-Tensor-Server

A high-performance Arrow Flight + HTTP server for serving multi-dimensional microscopy image data with lazy, chunked, multi-resolution, zero-copy access. Supports OME-Zarr, OME-TIFF, HDF5, CZI, LIF, ND2, DICOM, NIfTI, and more.

## Core Concept

TensorFlightServer sit on your data server to lazily convert your microscopy data of various formats into arrow IPC, thus allowing distributed zero-copy access with very high throughput, all the while maintaining a database (DuckDB) for metadata query.

On client side, the data can be accessed in any langugae supported by arrow flight. We primarily support **python**, for which the data is mapped to dask arrays, and **Java**, for which the data is mapped to ImgLib2.CellArray.

## Features

| | aicsimageio | omero | biopb |
|----|----|----|----|
| unified data access | ✓ | ✗ | ✓ |
| centralized metadata | ✗ | ✓ | ✓ |
| lazy read | ✓ | ✗ | ✓ |
| multi-resolution | ✗ | ✓ | ✓ |
| thread-safe | ✗ | ✗ | ✓ |
| serializable access | ✗ | ✗ | ✓ |
| language agnostic | ✗ | - | ✓ |
| web app | - | ✓ | ✓ |
| desktop app | - | ✗ | ✓ |
| zero-copy | ✗ | ✗ | ✓ |

## Quick Start
```bash
curl -fsSL https://biopb.org/install.sh | bash
```

## Deploy to data server
```bash
docker run -d --rm --init \
    --name biopb-tensor \
    -p 8814:8814 -p 8815:8815 \
    -v ${YOUR_DATA_LOCATION}:/data \
    -e BIOPB_TENSOR_TOKEN=your_secure_token \
    jiyuuchc/biopb-tensor-server:latest
```

The container is a headless data plane: `biopb-tensor-server launch` runs the
Arrow Flight gRPC endpoint (`:8815`, for SDK clients) and the FastAPI HTTP
sidecar (`:8814`, the data-plane API). There is no bundled webapp. The browser
UI lives in the top-level `web/` workspace — see `web/README.md` for how to
build and serve it. `--init` gives the container a reaping init as PID 1 —
optional, since `launch` (PID 1) handles `docker stop` gracefully, but it
cleans up any stray orphans.

See [containerize.md](containerize.md) for a complete list of deployment options, including methods for HPC deployment with singularity.

## Security

- Token-based authentication on both gRPC and HTTP for browser access
- Transport is **unencrypted** by default! Only deploy on trusted intranet!
- If BIOPB_TENSOR_TOKEN is not given, the server generates a random token that can be viewed with `docker logs biopb-tensor`
- **Local mode** (loopback `server.host`, the default) enforces no token — the single-machine case. **Remote mode** (public `server.host`) requires a token, auto-generated if none is supplied.
- Bind to localhost-only `-p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815` if not on trusted net, and access via ssh tunnel

## Configuration

You can create custom config file to fine-tune server behavior, e.g.,specifying multiple data sources.

```json
{
  "server": { "host": "127.0.0.1", "port": 8815, "log_level": "INFO" },
  "cache": {
    "backend": "file",
    "file_max_segment_mb": 256,
    "file_max_total_gb": 4096
  },
  "sources": [
    { "url": "/data" },
    {
      "alias": "my-zarr",
      "type": "zarr",
      "url": "/experiment.zarr",
      "dim_labels": ["z", "y", "x"]
    }
  ]
}
```

Directories (the `/data` source above) are recursively scanned for data
discovery; a specific source like `my-zarr` lets you override its metadata.

To use your custom configuration:

```bash
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/biopb.json:/custom.json \
    -v ~/data:/data \
    -v ~/experiment.zarr:/experiment.zarr \
    -e CONFIG_FILE=/custom.json \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    jiyuuchc/biopb-tensor-server:latest
```

### Monitoring

Directory monitoring uses a claim-based discovery protocol with periodic rescans:

```json
{
  "sources": [
    { "url": "/data/acquisition/", "monitor": true }
  ]
}
```

## Development

### Requirements

- Python >= 3.10, < 3.13
- pyarrow >= 14.0.0

### Installation
```bash
# From repository root
pip install -e biopb-tensor-server/

# With optional web dependencies (FastAPI sidecar)
pip install -e "biopb-tensor-server/[web]"

# With all format support
pip install -e "biopb-tensor-server/[web,aics,ome-zarr,medical]"
```

### Launch

```bash
# Local mode (loopback server.host — no token required)
biopb-tensor-server launch --config biopb.json

# Remote mode (public server.host — token required, auto-generated if omitted)
biopb-tensor-server launch --config biopb.json --token mytoken...

# gRPC only (no web sidecar)
biopb-tensor-server serve --config biopb.json
```

### CLI Reference

```
biopb-tensor-server serve    Start the gRPC Flight server only
biopb-tensor-server launch   Start Flight server + HTTP sidecar for web
biopb-tensor-server validate Check a config file (biopb.json)
biopb-tensor-server list-tensors  List all data sources and tensors in a config
biopb-tensor-server version  Show version information
```

Both commands share the Flight-server options; `launch` adds the HTTP sidecar on
top.

### Key options for `serve`:

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | (required) | Path to config file (`biopb.json`) |
| `--host, -h` | (from config) | Flight server host (overrides config) |
| `--port, -p` | (from config) | Flight server port (overrides config) |
| `--writable` | false | Enable write mode for data upload |
| `--token` | (auto) | Flight access token; auto-generated on a public bind if omitted |
| `--log-level, -l` | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `--log-file` | (none) | Rotating log file path |

### Key options for `launch`:

All of `serve`'s options above, plus the HTTP sidecar:

| Flag | Default | Description |
|------|---------|-------------|
| `--web-host` | 127.0.0.1 | HTTP sidecar bind address |
| `--web-port` | 8816 | HTTP sidecar port |
| `--cors` | (loopback only) | CORS origin to allow (repeatable); needed for a browser app on another origin |

### Python Client

```python
from biopb_tensor_server import TensorFlightClient, TensorArray

# Connect to running server
client = TensorFlightClient("grpc://localhost:8815", token="your-token")

# List available sources
sources = client.list_sources()

# Get a lazy array for a specific tensor (by its globally-unique array_id:
# "source_id/field" for a multi-tensor source, or "source_id" for a single one)
arr = client.get_tensor("my-zarr/0", scale_hint=[1, 2, 2])

# Obtain a slice (lazy, chunks fetched on demand)
data = arr[5, :, :]
print(data.shape, data.dtype)

# Serializable (lazy, graph only)
import cloudpickle as pickle
serialized = pickle.dumps(arr)

# Compute triggers data load
import dask as da
data_mean = da.compute(data.mean())
print(f"mean = {data_mean}")
```

### Testing

```bash
# Server tests
pip install -e ".[test]"
pytest

# http/web tests
cd web/packages/tensor-flight-client
pnpm test
```

## License

MIT
