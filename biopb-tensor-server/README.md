# Biopb-Tensor-Server

A high-performance Arrow Flight + HTTP server for serving multi-dimensional microscopy image data with lazy, chunked, multi-resolution, zero-copy access. Supports OME-Zarr, OME-TIFF, HDF5, CZI, LIF, ND2, DICOM, NIfTI, and more — with a built-in web viewer.

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
    -p 8813:8813 -p 8815:8815 \
    -v ${YOUR_DATA_LOCATION}:/data \
    -e BIOPB_TENSOR_TOKEN=your_secure_token \
    jiyuuchc/biopb-tensor-server:latest
```

The container runs the **control plane** as its single web origin: open
`http://localhost:8813` for the dashboard, with the data viewer at
`http://localhost:8813/viewer`. The
Arrow Flight gRPC endpoint stays on `:8815` for SDK clients. The HTTP sidecar
(`:8814`) is now internal to the container and is not published. `--init` gives
the container a reaping init as PID 1 — optional, since the control plane (PID 1)
handles `docker stop` and reaps its own child, but it cleans up any stray orphans.

See [deploy.md](deploy.md) for a complete list of deployment options, including methods for HPC deployment with singularity.

## Security

- Token-based authentication on both gRPC and HTTP for browser access
- Transport is **unencrypted** by default! Only deploy on trusted intranet!
- If BIOPB_TENSOR_TOKEN is not given, the server generates a random token that can be viewed with `docker logs biopb-tensor`
- **Local mode** (loopback `server.host`, the default) enforces no token — the single-machine case. **Remote mode** (public `server.host`) requires a token, auto-generated if none is supplied.
- Bind to localhost-only `-p 127.0.0.1:8813:8813 -p 127.0.0.1:8815:8815` if not on trusted net, and access via ssh tunnel

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
docker run -d -p 8813:8813 -p 8815:8815 \
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

- Python >= 3.8
- pyarrow >= 14.0.0
- Node.js / pnpm (for web app and TS client)

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

### Web App

The browser UI is one Vite + React SPA in the top-level `web/` workspace (not the
FastAPI sidecar, which is API-only). Build it before running the server:

```bash
# Production build (output → web/packages/app/dist/)
pnpm -C web install && pnpm -C web build
```

The bundle is copied into the Docker image and served by the **control plane**,
the single web origin (`:8813`): the dashboard at `/` and the data viewer at
`/viewer`. On first load you will be prompted for the access token (printed in
the container logs, or set via `BIOPB_TENSOR_TOKEN`).

For development with hot reload, run the dev server separately:
```bash
pnpm -C web dev   # runs on :5173, proxies to a live control on :8813
```

### CLI Reference

```
biopb-tensor-server serve    Start the gRPC Flight server only
biopb-tensor-server launch   Start Flight server + HTTP sidecar for web
biopb-tensor-server validate Check a config file (JSON; legacy TOML)
biopb-tensor-server list     List all data sources and tensors in a config
biopb-tensor-server version  Show version information
```

### Key options for `launch`:

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | (required) | Path to config file (JSON; legacy TOML) |
| `--web-port` | 8814 | HTTP server port |
| `--web-host` | 127.0.0.1 | HTTP server bind address |
| `--token` | (auto) | Website access token (remote mode; auto-generated if omitted) |
| `--open` | false | Open browser after startup |
| `--web-url` | http://localhost:5173 | Base URL of web app (CORS + --open) |
| `--cors` | (derived from --web-url) | Extra CORS origins (repeatable) |
| `--log-level, -l` | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Key options for `serve`:

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | (required) | Path to config file (JSON; legacy TOML) |
| `--host, -h` | (from config) | gRPC server host |
| `--port, -p` | (from config) | gRPC server port |
| `--writable` | false | Enable write mode for data upload |

### Python Client

```python
from biopb_tensor_server import TensorFlightClient, TensorArray

# Connect to running server
client = TensorFlightClient("grpc://localhost:8815", token="your-token")

# List available sources
sources = client.list_sources()

# Get a lazy array for a specific tensor
arr = client.get_tensor("my-zarr", tensor_id="0", scale_hint=[1, 2, 2])

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
