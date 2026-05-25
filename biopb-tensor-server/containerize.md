# BioPB Tensor Server Docker Deployment

## Overview

This document describes how to deploy the BioPB Tensor Server as a Docker/Singularity container. The container includes:

- **FastAPI HTTP Server** (port 8814) - Serves React webapp, API endpoints, and health checks
- **TensorFlightServer** (gRPC on port 8815) - Arrow Flight server for tensor data
- **Webapp** - React-based image browser UI (served by FastAPI)

## Build Instructions

### Prerequisites

- Docker installed
- buf CLI installed (for local wheel build)
- Python 3.11+ with pip
- pnpm installed

### Dependency Version Notes

**numpy < 2.0 required**: The tensor server requires `numpy < 2.0` due to compatibility issues with `tifffile` and `aicsimageio`. numpy 2.0+ removed `ndarray.newbyteorder()` which older tifffile versions rely on. This constraint is enforced in `pyproject.toml` and `benchmarks/biopb-bench.def`.

If you encounter `AttributeError: 'numpy.ndarray' object has no attribute 'newbyteorder'` when reading TIFF files, ensure numpy is pinned to < 2.0:

```bash
pip install "numpy<2.0"
```

### Step 1: Build Wheel Locally

```bash
# From repository root
pip wheel . --no-deps -w wheels/
pip wheel ./biopb-tensor-server --no-deps -w wheels/
```

### Step 2: Build Webapp Locally

The webapp is served directly by FastAPI. Build with:

```bash
# From repository root
VITE_TENSOR_API="" pnpm --filter @biopb/web build
```

This creates `biopb-tensor-server/packages/web/dist/` which is copied into the Docker image.

### Step 3: Build Docker Image

```bash
docker build --memory=4g --memory-swap=8g -t biopb-tensor-server:latest -f biopb-tensor-server/Dockerfile .
```

**Note:** The `--memory` and `--memory-swap` flags are recommended because the build extracts a large Python site-packages layer (~1.5GB) during the COPY step from the builder stage. Without sufficient memory, the build may hang or fail with exit code 137 (OOM killed). If your system has limited memory, try building with `--no-cache` first.

**Image size:** ~1.24GB

## Docker Usage

### Basic Run

```bash
docker run -d \
    --name biopb-tensor \
    -p 8814:8814 \
    -p 8815:8815 \
    -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=your_secure_token \
    biopb-tensor-server:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (none) | Path to TOML config file (if set and exists, uses this file; otherwise generates from env vars) |
| `DATA_DIR` | `/data` | Directory containing microscopy files (used when generating config) |
| `MONITOR` | `true` | Enable live filesystem monitoring (NFS/Lustre: set to false) |
| `BIOPB_BASE_PORT` | `8810` | Base port - HTTP=BASE+4, gRPC=BASE+5 |
| `COMPUTE_BACKEND` | `auto` | Compute backend: auto, cpu, or gpu |
| `BIOPB_TENSOR_TOKEN` | (prompted) | Access token for webapp |
| `BIOPB_WEB_DEV_BYPASS` | (unset) | Set to `true` for dev mode (no token check) |
| `BIOPB_BIND_LOCALHOST` | (unset) | Set to `true` to bind HTTP to localhost (Singularity/HPC only; ignored in Docker) |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Base temp directory (avoids multi-user collisions on shared /tmp) |

### Port Derivation

All ports are derived from `BIOPB_BASE_PORT`:

| Service | Port | Formula |
|---------|------|---------|
| HTTP (FastAPI + webapp) | 8814 | BASE + 4 |
| gRPC (TensorFlightServer) | 8815 | BASE + 5 |

Ports are auto-discovered to avoid conflicts (especially for Singularity on HPC where host network is shared).

### Configuration Methods

The container uses a unified entrypoint that supports two configuration methods:

1. **Config file**: If `CONFIG_FILE` is set and the file exists, uses that TOML file
2. **Environment variables**: Otherwise, generates config from `DATA_DIR`, `MONITOR`, etc.

When `MONITOR=true`, initial dataset discovery follows the same stability checks as later rescans. On startup, the server may therefore come up healthy before newly written files under the monitored directory are exposed as sources. With the default settings, expect monitored datasets to appear only after they have remained unchanged for the configured stability window.

### Examples

```bash
# Basic run (auto port discovery, default base 8810)
docker run -d -p 8814:8814 -p 8815:8815 -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# Custom base port (HTTP=9004, gRPC=9005, etc.)
docker run -d -p 9004:9004 -p 9005:9005 -v ~/data:/data \
    -e BIOPB_BASE_PORT=9000 \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# With custom config file
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/my-config.toml:/custom.toml \
    -v ~/data:/data \
    -e CONFIG_FILE=/custom.toml \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# Dev mode (localhost only, no token required)
docker run -d -p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815 -v ~/data:/data \
    -e BIOPB_WEB_DEV_BYPASS=true \
    biopb-tensor-server:latest
```

### Health Checks

These checks report service readiness, not completion of monitored dataset discovery. A healthy container can still have zero visible monitored sources briefly after startup while stability gating defers initial registration.

```bash
# Liveness (HTTP)
curl http://localhost:8814/livez

# Readiness (HTTP)
curl http://localhost:8814/readyz

# Flight server health action (gRPC)
# Use Flight's do_action("health") for gRPC health status
```

### Access the Webapp

1. Open `http://localhost:8814/` in browser
2. Enter the token (shown once in container logs, or set via `BIOPB_TENSOR_TOKEN`)
3. Browse microscopy datasets

## Singularity Usage (HPC)

### Build from Docker Image

```bash
# From local Docker image
singularity build biopb-tensor-server.sif docker-daemon://biopb-tensor-server:latest

# Or from published Docker image
singularity build biopb-tensor-server.sif docker://ghcr.io/jiyuuchc/biopb-tensor-server:latest
```

### Basic Singularity Run

```bash
# Simple run - auto port discovery from default base (8810)
singularity run \
    --bind ~/data:/data \
    --env BIOPB_TENSOR_TOKEN=your_secure_token \
    biopb-tensor-server.sif
```

Ports are auto-discovered to avoid conflicts on shared HPC nodes. The container will print discovered ports on startup.

### Configuration Options

**Method 1: Default (auto port discovery)**
```bash
singularity run \
    --bind ~/data:/data \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif
```

**Method 2: Custom base port**
```bash
# Use BIOPB_BASE_PORT=9000 → HTTP=9004, gRPC=9005, Sidecar=9006, Flight=9007
singularity run \
    --bind ~/data:/data \
    --env BIOPB_BASE_PORT=9000 \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif
```

**Method 3: Custom config file**
```bash
singularity run \
    --bind ~/my-config.toml:/custom.toml \
    --bind ~/data:/data \
    --env CONFIG_FILE=/custom.toml \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif
```

### HPC Cluster Examples

```bash
# SLURM interactive session - auto port discovery
srun --pty singularity run \
    --bind /scratch/user/data:/data \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# Custom base port range (avoid conflicts with other users)
singularity run \
    --bind ~/data:/data \
    --env BIOPB_BASE_PORT=9000 \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# Dev mode for debugging (no token, localhost only on shared HPC node)
singularity run \
    --bind ~/data:/data \
    --env BIOPB_WEB_DEV_BYPASS=true \
    --env BIOPB_BIND_LOCALHOST=true \
    biopb-tensor-server.sif
```

## Architecture

```
Container (external ports 8814, 8815)
├── FastAPI HTTP Server (8814)
│   ├── /                      → React SPA (static files from /app/webapp)
│   ├── /api/*                 → API endpoints (sources, slice, render)
│   ├── /livez, /readyz        → health endpoints
│
└── TensorFlightServer (8815)  → Arrow Flight gRPC (direct access)
    ├── do_action("health")    → health check action
    ├── list_flights           → list available tensors
    └── do_get                 → fetch tensor data
```

FastAPI serves both the webapp and API endpoints on port 8814. TensorFlightServer exposes gRPC directly on port 8815 (no proxy needed).

### Network Binding Control

By default, FastAPI binds to all interfaces inside the container. Docker's port forwarding (`-p PORT:PORT`) then exposes the service to the host's network.

**For localhost-only access:**
- **Docker**: Use `-p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815` to restrict to host's localhost
- **Singularity/HPC**: Use `BIOPB_BIND_LOCALHOST=true` to bind HTTP to localhost (useful on shared nodes)

Note: `BIOPB_BIND_LOCALHOST=true` is **ignored in Docker** with a warning, since it would break external access (services bound to 127.0.0.1 inside a container cannot be reached from outside).

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| OME-TIFF | `.ome.tiff`, `.ome.tif` | Single file |
| OME-Zarr | `.zarr/` | Multiscale pyramid |
| Zarr | `.zarr/` | Generic zarr arrays |
| HDF5 | `.h5`, `.hdf5` | Requires explicit dataset path in config |
| CZI | `.czi` | Via aicsimageio + bioformats-jar |
| LIF | `.lif` | Via aicsimageio + bioformats-jar |
| ND2 | `.nd2` | Via aicsimageio[nd2] + bioformats-jar (included) |
| TIFF | `.tiff`, `.tif` | Standard TIFF |

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker logs biopb-tensor
```

Common causes:
- No data directory mounted: ensure `-v ~/data:/data`
- Token not provided: set `BIOPB_TENSOR_TOKEN` or check logs for generated token

### Webapp shows "unlock" page but token doesn't work

- Verify token matches `BIOPB_TENSOR_TOKEN`
- Check token is 16-128 characters, URL-safe (`[A-Za-z0-9_-]`)
- Dev mode bypasses token check: use `-p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815` with `--env BIOPB_WEB_DEV_BYPASS=true`

### Files not appearing in webapp

- Check mount path matches `DATA_DIR`: `-v ~/data:/data` with default `DATA_DIR=/data`
- On NFS/Lustre, set `MONITOR=false` and restart container to refresh catalog
- Check file format is supported

### ND2 files fail to load

The image now includes `bioformats-jar` and `aicsimageio[nd2]` for ND2 support. If issues persist:
- Verify file is not corrupted
- Check Docker logs for specific error
- Try loading with Python directly: `aicsimageio.AICSImage('/data/your_file.nd2')`

## Ports Summary

All ports derived from `BIOPB_BASE_PORT` (default: 8810):

| Port | Service | External Access | Formula |
|------|---------|-----------------|---------|
| BASE+4 | FastAPI HTTP | Yes (webapp + API + health) | `BIOPB_BASE_PORT + 4` |
| BASE+5 | TensorFlightServer gRPC | Yes (tensor data) | `BIOPB_BASE_PORT + 5` |

Ports are auto-discovered at startup to avoid conflicts. Expose HTTP (BASE+4) for webapp access and gRPC (BASE+5) for programmatic clients.