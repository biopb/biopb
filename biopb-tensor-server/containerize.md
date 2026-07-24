# BioPB Tensor Server Docker/Singularity Deployment

## Overview

This document describes how to deploy the BioPB Tensor Server as a Docker/Singularity container. The container is a **headless data plane** — `biopb-tensor-server launch` runs directly as PID 1. It includes:

- **TensorFlightServer** (gRPC on port 8815) - Arrow Flight server for tensor data
- **HTTP sidecar** (port 8814) - FastAPI data-plane API (sources, slice, render, health); the browser-friendly HTTP surface

There is **no control plane and no bundled webapp** in this image. A browser UI is served elsewhere (a dev server, a static host, or a separate control-plane deployment) and points at the two published ports above.

## Build Instructions

_Pre-built docker image is uploaded to docker hub (docker://jiyuuchc/biopb-tensor-server). Skip this step if you only need the image for deployment._

### Prerequisites

- Docker installed
- buf CLI installed (for local wheel build)
- Python 3.11+ with pip

### Dependency Version Notes

The tensor server runs on **numpy 2.x**. Two version pins are load-bearing and
enforced in `pyproject.toml` (see `docs/aicsimageio-to-bioio-migration.md` for
the full rationale):

- **`zarr < 3`** — biopb's Zarr/OME-Zarr adapters target the Zarr 2.x API.
- **`tifffile >= 2024.8.10, < 2025.5.21`** — the lower bound is numpy-2 safe;
  the upper bound keeps tifffile's `aszarr` store on Zarr 2 (2025.5.21 dropped
  Zarr 2), which the OME-TIFF read path depends on.

Vendor microscopy formats (CZI, LIF, ND2, DV, …) are read via **bioio** (the
maintained successor to aicsimageio), installed through the `[aics]` extra.

### Step 1: Build Wheels Locally

The image ships two wheels: core `biopb` (the tensor server depends on it) and
`biopb-tensor-server` itself. No control-plane wheel and no webapp bundle are
built for this image.

```bash
# From repository root
pip wheel . --no-deps -w wheels/
pip wheel ./biopb-tensor-server --no-deps -w wheels/
```

### Step 2: Build Docker Image

```bash
docker build --memory=4g --memory-swap=8g -t biopb-tensor-server:latest -f biopb-tensor-server/Dockerfile .
```

**Note:** The `--memory` and `--memory-swap` flags are recommended because the build extracts a large Python site-packages layer (~1.5GB) during the COPY step from the builder stage. Without sufficient memory, the build may hang or fail with exit code 137 (OOM killed). If your system has limited memory, try building with `--no-cache` first.

**Image size:** ~1.24GB

## Docker Usage

```bash
docker run -d --init \
    --name biopb-tensor \
    -p 8814:8814 \
    -p 8815:8815 \
    -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=your_secure_token \
    biopb-tensor-server:latest
```

> `--init` runs a small init as PID 1 to reap zombies. The container's PID 1 is
> `biopb-tensor-server launch`, which handles `docker stop` (SIGTERM) gracefully;
> `--init` is optional belt-and-suspenders for any unrelated orphaned
> grandchildren. Publish `8814` (the HTTP sidecar / data-plane API) and `8815`
> (Flight gRPC).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (unset) | Path to a JSON config file. If set, all other variables below are ignored. Its `[server].port` **must equal gRPC=`BIOPB_BASE_PORT`+5** so the published gRPC port matches; `[server].host` is the Flight bind (`0.0.0.0` to publish) |
| `DATA_DIR` | `/data` | Container path of microscopy files; mount the host dir onto it with `-v /host/data:/data` (used when generating config) |
| `MONITOR` | `true` | Enable live filesystem monitoring (poll-based) |
| `BIOPB_BASE_PORT` | `8810` | Base port in container. Derived: **HTTP sidecar=BASE+4** (publish this — the data-plane API), gRPC Flight=BASE+5 (publish for SDK clients) |
| `BIOPB_TENSOR_TOKEN` | (auto-generated) | Access token for the HTTP sidecar and gRPC; printed once in the logs when auto-generated |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Where the generated `runtime-config.json` is written. **Not to be confused with**  `$TMPDIR` |
| `TMPDIR/TEMP/TMP` | `/tmp` | Cache parent dir. Unset → cache lands on the container's **ephemeral writable layer** at `/tmp/biopb-cache-0`. Set it (e.g. `-e TMPDIR=/cache` with `-v vol:/cache`) to move the cache onto a volume — see [Cache Storage](#cache-storage) |
| `CACHE_MAX_TOTAL_GB` | `16` | Max total size of the on-disk file cache, in GB |
| `CACHE_MAX_SEGMENT_MB` | `256` | Max size of each cache segment file, in MB |

### Cache Storage

The server keeps a **file-backed cache of decoded chunks** (Arrow IPC segments). You need to know **where it lives and how big it can get**.

**Location.** `<system temp dir>/biopb-cache-<uid>` (the system temp dir honors `$TMPDIR` / `$TEMP` / `$TMP`). In **Docker**, this defaults to **`/tmp/biopb-cache-0` on the container's ephemeral writable layer** (overlay2). It is **not** a mounted volume — it consumes the Docker graph storage under `/var/lib/docker` and is discarded on `docker rm`.

**Size.** When config is generated from env vars, the cap defaults to
**`CACHE_MAX_TOTAL_GB=16`** (16 GB). Under Docker that means up to **16 GB can
accumulate on the writable layer**.

**Safely increase cache size** by putting the cache on a mounted volume instead of the writable layer.

```bash
docker run \
    -v tensor-cache:/cache \
    -e CACHE_MAX_TOTAL_GB=128 \
    -e TMPDIR=/cache \
    biopb-tensor-server:latest
```

> Note: `CACHE_MAX_TOTAL_GB` / `CACHE_MAX_SEGMENT_MB` only apply when the
> entrypoint **generates** the config from env vars. If you supply your own
> `CONFIG_FILE`, set the limits in its `cache` object (`file_max_total_gb`,
> `file_max_segment_mb`, `file_cache_dir`) instead.

### Examples

```bash
# Custom base port (BIOPB_BASE_PORT=9000 -> HTTP sidecar=9004, gRPC=9005)
docker run -d -p 9004:9004 -p 9005:9005 -v ~/data:/data \
    -e BIOPB_BASE_PORT=9000 \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# With custom config file. Its [server].port must match gRPC=BASE+5 (8815 here);
# [server].host is the Flight bind (0.0.0.0 to publish).
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/my-config.json:/custom.json \
    -v ~/data:/data \
    -e CONFIG_FILE=/custom.json \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# Localhost-only access. A token is still required.
docker run -d \
    -p 127.0.0.1:8814:8814 \
    -p 127.0.0.1:8815:8815 \
    -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest
```

## Singularity Usage (HPC)

### Build from Docker Image

```bash
# From local Docker image
singularity build biopb-tensor-server.sif docker-daemon://biopb-tensor-server:latest

# Or from published Docker image
singularity build biopb-tensor-server.sif docker://ghcr.io/jiyuuchc/biopb-tensor-server:latest
```

### Basic Usage

```bash
# Simple run - point DATA_DIR straight at the host path (no --bind needed).
# Singularity auto-mounts $HOME, /tmp, $PWD (and usually /scratch, /project),
# so data under those is already visible inside the container at the same path.
singularity run \
    --env DATA_DIR=$HOME/data \
    --env BIOPB_TENSOR_TOKEN=your_secure_token \
    biopb-tensor-server.sif
```

> Note: Ports are auto-discovered to avoid conflicts on shared HPC nodes. The container will print discovered ports on startup.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (unset) | Path to a JSON config file (if set and exists, uses this file; otherwise config is generated from env vars) |
| `DATA_DIR` | `/data` | Path of microscopy files. Singularity auto-mounts `$HOME`, `/tmp`, `$PWD` (and usually `/scratch`, `/project`). Use `--bind /host:/container` for locations the site doesn't auto-mount. |
| `MONITOR` | `true` | Enable live filesystem monitoring (poll-based) |
| `BIOPB_BASE_PORT` | `8810` | Base port - HTTP sidecar=BASE+4, gRPC=BASE+5 |
| `BIOPB_TENSOR_TOKEN` | (auto-generated) | Access token for the HTTP sidecar and gRPC; printed once in the logs when auto-generated |
| `BIOPB_BIND_LOCALHOST` | (unset) | Set to `true` to bind both HTTP and gRPC to loopback → **local mode, no token** (useful on shared nodes reached via localhost). A public container bind still auto-generates a token. |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Where the generated `runtime-config.json` is written |
| `TMPDIR/TEMP/TMP` | `/tmp` | Cache parent dir. Singularity auto-binds host `/tmp`, so the cache lands at `/tmp/biopb-cache-<uid>` on host disk (persistent). Set it to relocate — see [Cache Storage](#cache-storage) |
| `CACHE_MAX_TOTAL_GB` | `16` | Max total size of the on-disk file cache, in GB (only applies when generating config from env vars; ignored if `CONFIG_FILE` is set) |
| `CACHE_MAX_SEGMENT_MB` | `256` | Max size of each cache segment file, in MB (same applicability as above) |

### Examples

```bash
# Custom base port → HTTP=9004, gRPC=9005
singularity run \
    --env DATA_DIR=$HOME/data \
    --env BIOPB_BASE_PORT=9000 \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# Custom config file (point CONFIG_FILE at the host path directly; no --bind)
singularity run \
    --env CONFIG_FILE=$HOME/my-config.json \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# SLURM interactive session
srun --pty singularity run \
    --env DATA_DIR=/scratch/$USER/data \
    --env BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server.sif

# Local mode (no token, loopback bind only on a shared HPC node)
singularity run \
    --env DATA_DIR=$HOME/data \
    --env BIOPB_BIND_LOCALHOST=true \
    biopb-tensor-server.sif
```

## Health Checks

These checks report service readiness, not completion of monitored dataset discovery. A healthy container can still have zero visible monitored sources briefly after startup while stability gating defers initial registration.

```bash
# HTTP sidecar liveness/readiness (the data-plane API)
curl http://localhost:8814/livez
curl http://localhost:8814/readyz

# Flight server health action (gRPC)
# Use Flight's do_action("health") for gRPC health status
```

## Browser UI

This image does not serve a webapp. Run the SPA separately (a dev server, a
static host, or a control-plane deployment) and point it at this container's
published HTTP sidecar (`:8814`) and Flight gRPC (`:8815`) ports. You will be
prompted for the token (shown once in container logs, or set via
`BIOPB_TENSOR_TOKEN`).

## Architecture

```
Container (external ports 8814, 8815)
├── FastAPI HTTP Server (8814)  → data-plane API (browser-friendly HTTP)
│   ├── /api/*                 → API endpoints (sources, slice, render)
│   └── /livez, /readyz        → health endpoints
│
└── TensorFlightServer (8815)  → Arrow Flight gRPC (direct access)
    ├── do_action("health")    → health check action
    ├── list_flights           → list available tensors
    └── do_get                 → fetch tensor data
```

`biopb-tensor-server launch` runs both servers in a single process (PID 1): the
FastAPI HTTP sidecar (port 8814, the data-plane API) and TensorFlightServer
(gRPC, port 8815). Both are token-authenticated on a public bind. There is no
control plane and no static bundle in this image.

### Network Binding Control

By default, **both** the FastAPI HTTP server and the gRPC Flight server bind to
all interfaces (`0.0.0.0`) inside the container. Docker's port forwarding
(`-p PORT:PORT`) then exposes each service to the host's network; both are
token-authenticated. The HTTP sidecar connects to the Flight server over loopback whenever it binds to a wildcard address — which is what this entrypoint always generates — so the bind change never affects the sidecar.

**For localhost-only access:**
- **Docker**: Use `-p 127.0.0.1:8814:8814 -p 127.0.0.1:8815:8815` to restrict to host's localhost
- **Singularity/HPC**: Use `BIOPB_BIND_LOCALHOST=true` to bind both HTTP and gRPC to localhost (useful on shared nodes)

Note: `BIOPB_BIND_LOCALHOST=true` is **ignored in Docker** with a warning, since it would break external access (services bound to 127.0.0.1 inside a container cannot be reached from outside).

## Supported File Formats

| Format | Extension(s) | Reader / notes |
|--------|--------------|----------------|
| OME-Zarr | `.zarr/` | Multiscale pyramid, incl. HCS plates; native |
| OME-TIFF | `.ome.tiff`, `.ome.tif` | Single- and multi-file; native (`tifffile`) |
| TIFF | `.tif`, `.tiff` | Standard TIFF and TIFF sequences; native |
| Micro-Manager | NDTiff (`NDTiff.index`), legacy (`metadata.txt`) | Multi-file MM acquisitions; native (`ndtiff`) |
| Zeiss | `.czi`, `.lsm` | Native (`bioio-czi`; `.lsm` via `tifffile`) |
| Leica | `.lif` | Native (`bioio-lif`) |
| Nikon | `.nd2` | Native (`bioio-nd2`) |
| DeltaVision | `.dv` | Native (`bioio-dv`) |
| Olympus | `.oif`, `.oib` | Java Bio-Formats (`bioio-bioformats`) |
| Imaris | `.ims` | Java Bio-Formats (`bioio-bioformats`) |
| HDF5 | `.h5`, `.hdf5` | Requires explicit dataset path in config |
| DICOM | `.dcm` | Single files and multi-file series; native (`pydicom`) |
| NIfTI | `.nii`, `.nii.gz` | Native (`nibabel`) |
| Zeiss (legacy) | `.zvi` | No native Python reader — Java Bio-Formats (`bioio-bioformats`) |

Vendor readers are provided by **bioio** (successor to aicsimageio), each as its
own `bioio-*` plugin. Other formats are also handled but omitted here for
brevity: additional Bio-Formats types (`.lei`, `.vsi`) and assorted
scientific/image formats via the generic bioio fallback.

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker logs biopb-tensor
```

Common causes:
- A mounted `CONFIG_FILE` that is malformed (JSON parse or value-validation error — a legacy TOML is rejected outright; convert it with `biopb server migrate-config`) — check `docker logs` for the traceback
- A `CONFIG_FILE` with no `sources` → "No data sources configured" (exit 1)
- A custom config whose sources are all invalid/unreachable (bad paths, or remote sources that all fail auth) → "No sources loaded successfully" (exit 1)
- A port already in use inside the container (e.g. `--network host` colliding with a host process on 8814/8815)

### Token rejected (401 from the sidecar / gRPC)

- Verify token matches `BIOPB_TENSOR_TOKEN`
- Check token is 16-128 characters, URL-safe (`[A-Za-z0-9_-]`)
- Note: a Docker container always binds `0.0.0.0` (remote mode), so token enforcement stays on. The tokenless local mode via `BIOPB_BIND_LOCALHOST=true` works only under Singularity.

### Files not appearing

- Check mount path matches `DATA_DIR`: `-v ~/data:/data` with default `DATA_DIR=/data`
- New files take a moment to register: they appear only after the stability window passes and the next periodic rescan runs (this is poll-based and works on NFS/Lustre — no need to disable `MONITOR`)
- Check file format is supported

### Files fail to load

- Verify file format is supported (above)
- Verify file is not corrupted
- Check Docker logs for specific error
