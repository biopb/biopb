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
docker run -d --rm \
    --name biopb-tensor \
    -p 8814:8814 -p 8815:8815 \
    -v ${YOUR_DATA_LOCATION}:/data \
    -e BIOPB_TENSOR_TOKEN=your_secure_token \
    jiyuuchc/biopb-tensor-server:latest
```

See [deploy.md](deploy.md) for a complete list of deployment options, including methods for HPC deployment with singularity.

## Security

- Token-based authentication on both gRPC and HTTP for browser access
- Transport is **unencrypted** by default! Only deploy on trusted intranet!
- If BIOPB_TENSOR_TOKEN is not given, the server generates a random token that can be viewed with `docker logs biopb-tensor`
- Dev mode (`--dev` flag or `BIOPB_WEB_DEV_BYPASS=1`) disables token enforcement!
- Bind to localhost-only `-p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815` if not on trusted net, and access via ssh tunnel

## Configuration

You can create custom config file to fine-tune server behavior, e.g.,specifying multiple data sources.

```toml
[server]
host = "127.0.0.1"
port = 8817
log_level = "INFO"

[cache]
backend = "file"
file_max_segment_mb = 256
file_max_total_gb = 4096      # increase flight storage limit

# directories will be recursively scanned for data discovery
[[sources]]
url = "/data"

# specific data sources allow metadata override
[[sources]]
source_id  = "my-zarr"
type       = "zarr"
url        = "/experiment.zarr"
dim_labels = ["z", "y", "x"]
```
To use your custom configuration:

```bash
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/my-config.toml:/custom.toml \
    -v ~/data:/data \
    -v ~/experiment.zarr:/experiment.zarr \
    -e CONFIG_FILE=/custom.toml \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    jiyuuchc/biopb-tensor-server:latest
```

### Monitoring

Directory monitoring uses a claim-based discovery protocol with periodic rescans:

```toml
[[sources]]
url = "/data/acquisition/"
monitor = true   # enable live filesystem monitoring
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
# Interactive launch with auto-generated token
biopb-tensor-server launch --config biopb-tensor.toml

# Dev mode (localhost, no token required)
biopb-tensor-server launch --config biopb-tensor.toml --dev

# gRPC only (no web sidecar)
biopb-tensor-server serve --config biopb-tensor.toml
```

### Web App

The React web viewer runs alongside the server. From the repository root:

```bash
# Development (hot reload on :5173)
pnpm --filter @biopb/web dev

# Production build
pnpm --filter @biopb/web build   # output → packages/web/dist/
```

The web app connects to the FastAPI sidecar (default `http://localhost:8816`). On first load you will be prompted for the access token printed by the `launch` command.

### CLI Reference

```
biopb-tensor-server serve    Start the gRPC Flight server only
biopb-tensor-server launch   Start Flight server + HTTP sidecar for web
biopb-tensor-server validate Check a TOML config file
biopb-tensor-server list     List all data sources and tensors in a config
biopb-tensor-server version  Show version information
```

### Key options for `launch`:

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | (required) | Path to TOML config |
| `--web-port` | 8816 | HTTP sidecar port |
| `--web-host` | 127.0.0.1 | HTTP sidecar bind address |
| `--token` | (prompt/auto) | Website access token |
| `--dev` | false | Skip token check (localhost only) |
| `--open` | false | Open browser after startup |
| `--web-url` | http://localhost:5173 | Base URL of web app (CORS + --open) |
| `--cors` | (derived from --web-url) | Extra CORS origins (repeatable) |
| `--log-level, -l` | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Key options for `serve`:

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | (required) | Path to TOML config |
| `--host, -h` | (from config) | gRPC server host |
| `--port, -p` | (from config) | gRPC server port |
| `--compute-backend` | auto | auto, cpu, or gpu |
| `--writable` | false | Enable write mode for data upload |

### Python Client

```python
from biopb_tensor_server import TensorFlightClient, TensorArray

# Connect to running server
client = TensorFlightClient("grpc://localhost:8817", token="your-token")

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
cd packages/tensor-flight-client
pnpm test
```

## License

MIT
