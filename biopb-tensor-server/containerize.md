# BioPB Tensor Server Docker/Singularity Deployment

## Overview

This document describes how to deploy the BioPB Tensor Server as a Docker/Singularity container. The container is a **headless data plane** — `biopb-tensor-server launch` runs directly as PID 1. It includes:

- **TensorFlightServer** (gRPC on port 8815) - Arrow Flight server for tensor data
- **HTTP sidecar** (port 8814) - FastAPI data-plane API (sources, slice, render, health); the browser-friendly HTTP surface

## Docker Usage

```bash
docker run -d --rm \
    --name biopb-tensor \
    -p 127.0.0.1:8814:8814 \
    -p 127.0.0.1:8815:8815 \
    -v ${YOUR_DATA_DIR}:/data \
    -e BIOPB_TENSOR_ALLOW_NO_TOKEN=1 \
    biopb-tensor-server:latest
```

> replace `${YOUR_DATA_DIR}` with a real path on your computer

> The container binds `0.0.0.0` internally, so by default it auto-generates an
> access token (printed in `docker logs biopb-tensor`). Because the ports above
> are published to host loopback only (`127.0.0.1`), `BIOPB_TENSOR_ALLOW_NO_TOKEN=1`
> opts out of that and serves the data API without a token — convenient for a
> single-machine setup. Drop it (and grab the logged token) if the ports are
> reachable from other hosts.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (unset) | Path to a JSON config file. If set, all other variables below are ignored. Its `[server].port` **must equal gRPC=`BIOPB_BASE_PORT`+5** so the published gRPC port matches; `[server].host` is the Flight bind (`0.0.0.0` to publish) |
| `DATA_DIR` | `/data` | Container path of microscopy files; mount the host dir onto it with `-v /host/data:/data` |
| `MONITOR` | `true` | Enable live filesystem monitoring (poll-based) |
| `BIOPB_BASE_PORT` | `8810` | Base port in container. Derived: **HTTP sidecar=BASE+4** (publish this — the data-plane API), gRPC Flight=BASE+5 (publish for SDK clients) |
| `BIOPB_TENSOR_TOKEN` | (auto-generated) | Access token for the HTTP sidecar and gRPC; printed once in the logs when auto-generated |
| `BIOPB_TENSOR_ALLOW_NO_TOKEN` | (unset) | Set to `1`/`true` to serve the data API **without a token** even on the public `0.0.0.0` bind (insecure — trusted networks only). Ignored when `BIOPB_TENSOR_TOKEN` is set |
| `BIOPB_CORS_ORIGINS` | (unset) | Space-separated CORS origins (→ repeated `--cors`). Set this to allow a browser SPA served from a different origin to call the sidecar (e.g. `BIOPB_CORS_ORIGINS="http://localhost:5173 http://my.host:8813"`) |
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

### More examples

```bash
# Public server (with an access token) with custom base port (BIOPB_BASE_PORT=9000 -> HTTP sidecar=9004, gRPC=9005)
docker run -d -p 9004:9004 -p 9005:9005 \
    -v ~/data:/data \
    -e BIOPB_BASE_PORT=9000 \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# With custom config file. Its [server].port must match gRPC=BASE+5 (8815 here)
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/my-config.json:/custom.json \
    -v ~/data:/data \
    -e CONFIG_FILE=/custom.json \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest
```

## Singularity Usage (HPC)

### Build from Docker Image

```bash
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

> Note: This creates a publicly accessible server (binds to 0.0.0.0) **without data encryption**. Make sure you only do this on a trusted network.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (unset) | Path to a JSON config file (if set and exists, uses this file; otherwise config is generated from env vars) |
| `BIOPB_BASE_PORT` | `8810` | Base port - HTTP sidecar=BASE+4, gRPC=BASE+5 |
| `BIOPB_TENSOR_TOKEN` | (auto-generated) | Access token for the HTTP sidecar and gRPC; printed once in the logs when auto-generated |
| `BIOPB_TENSOR_ALLOW_NO_TOKEN` | (unset) | Set to `1`/`true` to serve the data API **without a token** even on a public bind (insecure — trusted networks only). Ignored when `BIOPB_TENSOR_TOKEN` is set or `BIOPB_BIND_LOCALHOST=true` |
| `DATA_DIR` | `/data` | Path of microscopy files. Singularity auto-mounts `$HOME`, `/tmp`, `$PWD` (and usually `/scratch`, `/project`). Use `--bind /host:/container` for locations the site doesn't auto-mount. |
| `MONITOR` | `true` | Enable live filesystem monitoring (poll-based) |
| `BIOPB_CORS_ORIGINS` | (unset) | Space-separated CORS origins (→ repeated `--cors`). Set this to allow a browser SPA served from a different origin to call the sidecar |
| `BIOPB_BIND_LOCALHOST` | (unset) | Set to `true` to bind both HTTP and gRPC to loopback → **local mode, no token** (useful on shared nodes reached via localhost). A public container bind still auto-generates a token. |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Where the generated `runtime-config.json` is written |
| `TMPDIR/TEMP/TMP` | `/tmp` | Cache parent dir. Singularity auto-binds host `/tmp`, so the cache lands at `/tmp/biopb-cache-<uid>` on host disk (persistent). Set it to relocate — see [Cache Storage](#cache-storage) |
| `CACHE_MAX_TOTAL_GB` | `16` | Max total size of the on-disk file cache, in GB (only applies when generating config from env vars; ignored if `CONFIG_FILE` is set) |
| `CACHE_MAX_SEGMENT_MB` | `256` | Max size of each cache segment file, in MB (same applicability as above) |

### More examples

```bash
# Local mode (no token, loopback bind only on a shared HPC node)
singularity run \
    --env DATA_DIR=$HOME/data \
    --env BIOPB_BIND_LOCALHOST=true \
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
```

## Health Checks

These checks report service readiness, not completion of monitored dataset discovery. A healthy container can still have zero visible monitored sources briefly after startup while stability gating defers initial registration.

```bash
# HTTP sidecar liveness/readiness (the data-plane API)
curl http://localhost:8814/livez
```

## Browser access

**CORS.** The HTTP sidecar allows requests from its own origin and localhost
variants by default. If you have a webapp running at a *different* origin, the
browser's CORS preflight will reject `Authorization: Bearer <token>` calls.
Set `BIOPB_CORS_ORIGINS` to the web-app's origin(s) (space-separated) so the
sidecar allows them:

```bash
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/data:/data \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    -e BIOPB_CORS_ORIGINS="http://localhost:5173 http://my.host:8813" \
    biopb-tensor-server:latest
```

## Build from source

### Prerequisites

- Docker installed
- buf CLI installed (for local wheel build)
- Python 3.11+ with pip

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

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker logs biopb-tensor
```

Common causes:
- A mounted `CONFIG_FILE` that is malformed (JSON parse or value-validation error — a legacy TOML is rejected outright; convert it with `biopb server migrate-config`) — check `docker logs` for the traceback
- A port already in use inside the container (e.g. `--network host` colliding with a host process on 8814/8815)

> A config with **no sources** (or whose sources all fail to load) is **not** a
> startup error — the server boots and serves an **empty catalog** (health
> `SERVING`, zero sources), since sources can arrive later via a monitored
> directory, runtime add, or upload. See *Files not appearing* below.

### Token rejected (401 from the sidecar / gRPC)

- Verify token matches `BIOPB_TENSOR_TOKEN`
- Check token is 16-128 characters, URL-safe (`[A-Za-z0-9_-]`)
- Note: a Docker container always binds `0.0.0.0` (remote mode), so token enforcement stays on by default. To run tokenless, either set `BIOPB_TENSOR_ALLOW_NO_TOKEN=1` (serves the data API open — publish to `127.0.0.1` and stay on a trusted network) or, under Singularity, use `BIOPB_BIND_LOCALHOST=true`.

### Files not appearing

- Check mount path matches `DATA_DIR`: `-v ~/data:/data` with default `DATA_DIR=/data`
- New files take a moment to register: they appear only after the stability window passes and the next periodic rescan runs (this is poll-based and works on NFS/Lustre — no need to disable `MONITOR`)
- Check file format is supported

### Files fail to load

- Verify the file format is supported (see the supported-formats table in the [README](README.md))
- Verify file is not corrupted
- Check Docker logs for specific error
