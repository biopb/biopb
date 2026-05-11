# BioPB Tensor Server Docker Deployment

## Overview

This document describes how to deploy the BioPB Tensor Server as a Docker/Singularity container. The container includes:

- **TensorFlightServer** (gRPC on port 8815) - Arrow Flight server for tensor data
- **HTTP Sidecar** (port 8816) - FastAPI proxy for browser access
- **nginx** (port 80) - Serves React webapp and proxies API requests
- **Webapp** - React-based image browser UI

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

### Step 1: Build biopb Wheel Locally

The biopb package must be built locally because buf's remote plugins have network issues inside Docker:

```bash
# From repository root
pip wheel . --no-deps -w wheels/
```

This creates `wheels/biopb-<version>-py3-none-any.whl`.

### Step 2: Build Webapp Locally

The webapp must be built with `VITE_TENSOR_API=""` so nginx proxy works:

```bash
# From repository root
VITE_TENSOR_API="" pnpm --filter @biopb/web build
```

This creates `biopb-tensor-server/packages/web/dist/`.

### Step 3: Build Docker Image

```bash
docker build -t biopb-tensor-server:latest -f biopb-tensor-server/Dockerfile .
```

**Image size:** ~1.24GB

## Docker Usage

### Basic Run

```bash
docker run -d \
    --name biopb-tensor \
    -p 80:80 \
    -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=your_secure_token \
    biopb-tensor-server:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | `/app/config/default-config.toml` | Path to TOML config file (if set and exists, uses this file; otherwise generates from env vars) |
| `DATA_DIR` | `/data` | Directory containing microscopy files (used when generating config) |
| `MONITOR` | `true` | Enable live filesystem monitoring (NFS/Lustre: set to false) |
| `HOST` | `0.0.0.0` | gRPC server host |
| `PORT` | `8815` | gRPC server port |
| `WEB_PORT` | `8816` | HTTP sidecar port (internal) |
| `NGINX_PORT` | `80` | nginx/webapp port (use higher port on HPC, e.g., 8080) |
| `COMPUTE_BACKEND` | `auto` | Compute backend: auto, cpu, or gpu |
| `BIOPB_TENSOR_TOKEN` | (prompted) | Access token for webapp |
| `BIOPB_WEB_DEV_BYPASS` | (unset) | Set to `true` for dev mode (no token check) |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Base temp directory (avoids multi-user collisions on shared /tmp) |

### Configuration Methods

The container uses a unified entrypoint that supports two configuration methods:

1. **Config file**: If `CONFIG_FILE` is set and the file exists, uses that TOML file
2. **Environment variables**: Otherwise, generates config from `DATA_DIR`, `HOST`, `PORT`, etc.

Docker sets `CONFIG_FILE=/app/config/default-config.toml` by default, so it uses the baked-in config. Singularity inherits this behavior; use `--cleanenv` to generate from env vars instead.

### Examples

```bash
# Basic run (uses baked-in default-config.toml)
docker run -d -p 80:80 -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# With custom config file
docker run -d -p 80:80 \
    -v ~/my-config.toml:/custom.toml \
    -v ~/data:/data \
    -e CONFIG_FILE=/custom.toml \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# Generate config from env vars (unset CONFIG_FILE)
docker run -d -p 80:80 -v ~/data:/data \
    -e CONFIG_FILE="" \
    -e DATA_DIR=/data \
    -e COMPUTE_BACKEND=cpu \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# Dev mode (localhost only, no token required)
docker run -d -p 80:80 -v ~/data:/data \
    -e BIOPB_WEB_DEV_BYPASS=true \
    biopb-tensor-server:latest

# gRPC only (no webapp)
docker run -d -p 8815:8815 -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest serve
```

### Health Checks

```bash
# Liveness
curl http://localhost/livez

# Readiness
curl http://localhost/readyz
```

### Access the Webapp

1. Open `http://localhost/` in browser
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
singularity run \
    --bind ~/data:/data \
    --env BIOPB_TENSOR_TOKEN=your_secure_token \
    biopb-tensor-server.sif
```

By default, Singularity inherits the Docker image's `CONFIG_FILE` environment variable, so it uses the baked-in config. To generate config from env vars instead:

```bash
# Use --cleanenv to not inherit Docker env vars, then set your own
singularity run --cleanenv \
    --bind ~/data:/data \
    --env DATA_DIR=/data \
    --env NGINX_PORT=8080 \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif
```

### Configuration Options

**Method 1: Use baked-in config (default)**
```bash
singularity run \
    --bind ~/data:/data \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif
```

**Method 2: Generate from environment variables**
```bash
singularity run --cleanenv \
    --bind ~/data:/data \
    --env DATA_DIR=/data \
    --env HOST=0.0.0.0 \
    --env PORT=8815 \
    --env NGINX_PORT=8080 \
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
# SLURM interactive session with custom webapp port (port 80 requires root)
srun --pty singularity run \
    --bind /scratch/user/data:/data \
    --env NGINX_PORT=8080 \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# With custom gRPC port
singularity run \
    --bind ~/data:/data \
    --env PORT=9000 \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# Dev mode for debugging (no token, custom port)
singularity run \
    --bind ~/data:/data \
    --env NGINX_PORT=8080 \
    --env BIOPB_WEB_DEV_BYPASS=true \
    biopb-tensor-server.sif
```

## Architecture

```
Container (port 80 externally)
├── nginx (80)                 → serves webapp, proxies /api/* to sidecar
│   ├── /                      → React SPA (static files)
│   └── /api/*                 → proxy to 127.0.0.1:8816
│
├── TensorFlightServer (8815)  → Arrow Flight gRPC (internal)
│
└── HTTP Sidecar (8816)        → FastAPI (localhost only, nginx proxies)
```

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
- Dev mode bypasses token check: `--env BIOPB_WEB_DEV_BYPASS=true`

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

| Port | Service | External Access | Configurable via |
|------|---------|-----------------|------------------|
| 80 (default) | nginx | Yes (webapp + API) | `NGINX_PORT` env var |
| 8815 | gRPC | Yes (if exposed) | `PORT` env var |
| 8816 | FastAPI sidecar | No (internal only) | `WEB_PORT` env var |

On HPC systems, use `NGINX_PORT=8080` or similar since port 80 typically requires root privileges.
Expose only the nginx port for webapp access. gRPC port 8815 is for programmatic access (Python clients).