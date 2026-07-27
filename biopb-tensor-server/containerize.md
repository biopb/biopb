# BioPB Tensor Server Docker/Singularity Deployment

## Overview

This document describes how to deploy the BioPB Tensor Server as a Docker/Singularity container. The container is a **headless, Flight-only data plane** — `biopb-tensor-server serve` runs directly as PID 1:

- **TensorFlightServer** (gRPC on port 8815) — Arrow Flight server for tensor data. **The only listener.**
- **HTTP sidecar** (port 8814) — the FastAPI data-plane API is **off by default**; set `BIOPB_ENABLE_HTTP_SIDECAR=1` to serve it (the container then runs `launch` instead).

**Why Flight-only.** The container is a pure gRPC data-plane endpoint: no HTTP
surface, no browser origin, no CORS, one port to publish. Browsing its data
happens *downstream* — a machine running the full biopb stack adds
`grpc://this-host:8815` (or `grpcs://`, below) as a remote source, so the browser
only ever talks to its own loopback origin and never needs to trust this
container's cert. See biopb/biopb#604.

## Docker Usage

```bash
docker run -d --rm \
    --name biopb-tensor \
    -p 127.0.0.1:8815:8815 \
    -v ${YOUR_DATA_DIR}:/data \
    -e BIOPB_TENSOR_ALLOW_NO_TOKEN=1 \
    biopb-tensor-server:latest
```

> replace `${YOUR_DATA_DIR}` with a real path on your computer

> The container binds `0.0.0.0` internally, so by default it auto-generates an
> access token (printed in `docker logs biopb-tensor`). Because the port above
> is published to host loopback only (`127.0.0.1`), `BIOPB_TENSOR_ALLOW_NO_TOKEN=1`
> opts out of that and serves the data API without a token — convenient for a
> single-machine setup. Drop it (and grab the logged token) if the port is
> reachable from other hosts.

### TLS (remote deployment)

For a container reachable beyond loopback, serve Flight over TLS. The server
mints a **self-signed** cert on first use and clients pin it on first connect
(TOFU) — there is no CA to distribute:

```bash
docker run -d --rm \
    --name biopb-tensor \
    -p 8815:8815 \
    -v ${YOUR_DATA_DIR}:/data \
    -v biopb-state:/root/.local/state \
    -e BIOPB_TENSOR_TLS=1 \
    biopb-tensor-server:latest
```

Clients dial `grpcs://<host>:8815` with the token from the logs.

> **Mount the state dir.** The generated cert lives at
> `/root/.local/state/biopb/tls/`. Without a volume there it is lost on `docker rm`,
> and the next container mints a *different* cert — which every client that pinned
> the old one will refuse (by design). The volume above (or `-e XDG_STATE_HOME=/state`
> with `-v biopb-state:/state`) keeps the identity stable.

To serve a certificate you already have, mount it and point `BIOPB_TLS_CERT` /
`BIOPB_TLS_KEY` at the in-container paths instead — no state volume needed.

TLS and `BIOPB_ENABLE_HTTP_SIDECAR` are mutually exclusive today: the sidecar's
internal client does not yet speak TLS to the Flight server, so the entrypoint
refuses that combination rather than silently serving plaintext.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (unset) | Path to a JSON config file. If set, all other variables below are ignored. Its `[server].port` **must equal gRPC=`BIOPB_BASE_PORT`+5** so the published gRPC port matches; `[server].host` is the Flight bind (`0.0.0.0` to publish) |
| `DATA_DIR` | `/data` | Container path of microscopy files; mount the host dir onto it with `-v /host/data:/data` |
| `MONITOR` | `true` | Enable live filesystem monitoring (poll-based) |
| `BIOPB_BASE_PORT` | `8810` | Base port in container. Derived: **gRPC Flight=BASE+5** (publish this — the data plane), HTTP sidecar=BASE+4 (only with the opt-in below) |
| `BIOPB_ENABLE_HTTP_SIDECAR` | (unset) | Set to `1`/`true` to also serve the FastAPI data-plane HTTP API on BASE+4 (the container runs `launch` instead of `serve`). Off by default — the container is Flight-only. Publish the port explicitly (`-p 8814:8814`); it is not `EXPOSE`d |
| `BIOPB_TENSOR_TLS` | (unset) | Set to `1`/`true` to serve Flight over TLS with a self-signed cert (auto-generated in the state dir; clients dial `grpcs://` and pin it on first connect). Mount a volume at `/root/.local/state` to keep the cert stable across `docker rm`. Incompatible with `BIOPB_ENABLE_HTTP_SIDECAR` |
| `BIOPB_TLS_CERT` / `BIOPB_TLS_KEY` | (unset) | In-container paths to a PEM cert + key to serve instead of the self-signed one (mount them in). Must be set together; takes precedence over `BIOPB_TENSOR_TLS` |
| `BIOPB_TENSOR_TOKEN` | (auto-generated) | Access token for gRPC (and the sidecar, if enabled); printed once in the logs when auto-generated |
| `BIOPB_TENSOR_ALLOW_NO_TOKEN` | (unset) | Set to `1`/`true` to serve the data API **without a token** even on the public `0.0.0.0` bind (insecure — trusted networks only). Ignored when `BIOPB_TENSOR_TOKEN` is set |
| `BIOPB_CORS_ORIGINS` | (unset) | Space-separated CORS origins (→ repeated `--cors`). Only meaningful with `BIOPB_ENABLE_HTTP_SIDECAR`: allows a browser SPA served from a different origin to call the sidecar (e.g. `BIOPB_CORS_ORIGINS="http://localhost:5173 http://my.host:8813"`) |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Where the generated `runtime-config.json` is written. **Not to be confused with**  `$TMPDIR` |
| `TMPDIR/TEMP/TMP` | `/tmp` | Cache parent dir. Unset → cache lands on the container's **ephemeral writable layer** at `/tmp/biopb-cache-0`. Set it (e.g. `-e TMPDIR=/cache` with `-v vol:/cache`) to move the cache onto a volume — see [Cache Storage](#cache-storage) |
| `CACHE_MAX_TOTAL_GB` | `16` | Max total size of the on-disk file cache, in GB |
| `CACHE_MAX_SEGMENT_MB` | (unset) | Max size of each cache segment file, in MB. Unset → server default (~64 MB) |

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
# Public server (with an access token) with custom base port (BIOPB_BASE_PORT=9000 -> gRPC=9005)
docker run -d -p 9005:9005 \
    -v ~/data:/data \
    -e BIOPB_BASE_PORT=9000 \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# With custom config file. Its [server].port must match gRPC=BASE+5 (8815 here)
docker run -d -p 8815:8815 \
    -v ~/my-config.json:/custom.json \
    -v ~/data:/data \
    -e CONFIG_FILE=/custom.json \
    -e BIOPB_TENSOR_TOKEN=mytoken \
    biopb-tensor-server:latest

# With the HTTP sidecar restored (publish its port explicitly)
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/data:/data \
    -e BIOPB_ENABLE_HTTP_SIDECAR=1 \
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

> Note: This creates a publicly accessible server (binds to 0.0.0.0) **without data encryption**. Either keep it on a trusted network, or add `--env BIOPB_TENSOR_TLS=1` and dial `grpcs://` (see [TLS](#tls-remote-deployment)).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_FILE` | (unset) | Path to a JSON config file (if set and exists, uses this file; otherwise config is generated from env vars) |
| `BIOPB_BASE_PORT` | `8810` | Base port - gRPC=BASE+5 (HTTP sidecar=BASE+4 only with the opt-in below) |
| `BIOPB_ENABLE_HTTP_SIDECAR` | (unset) | Set to `1`/`true` to also serve the FastAPI data-plane HTTP API on BASE+4. Off by default — the container is Flight-only |
| `BIOPB_TENSOR_TLS` | (unset) | Set to `1`/`true` to serve Flight over TLS with a self-signed cert (clients dial `grpcs://` and pin it on first connect). The cert lives under `$HOME/.local/state/biopb/tls/`, which Singularity auto-mounts from the host, so it is stable across runs |
| `BIOPB_TLS_CERT` / `BIOPB_TLS_KEY` | (unset) | Paths to a PEM cert + key to serve instead of the self-signed one. Must be set together |
| `BIOPB_TENSOR_TOKEN` | (auto-generated) | Access token for gRPC (and the sidecar, if enabled); printed once in the logs when auto-generated |
| `BIOPB_TENSOR_ALLOW_NO_TOKEN` | (unset) | Set to `1`/`true` to serve the data API **without a token** even on a public bind (insecure — trusted networks only). Ignored when `BIOPB_TENSOR_TOKEN` is set or `BIOPB_BIND_LOCALHOST=true` |
| `DATA_DIR` | `/data` | Path of microscopy files. Singularity auto-mounts `$HOME`, `/tmp`, `$PWD` (and usually `/scratch`, `/project`). Use `--bind /host:/container` for locations the site doesn't auto-mount. |
| `MONITOR` | `true` | Enable live filesystem monitoring (poll-based) |
| `BIOPB_CORS_ORIGINS` | (unset) | Space-separated CORS origins (→ repeated `--cors`). Only meaningful with `BIOPB_ENABLE_HTTP_SIDECAR` |
| `BIOPB_BIND_LOCALHOST` | (unset) | Set to `true` to bind to loopback → **local mode, no token** (useful on shared nodes reached via localhost). A public container bind still auto-generates a token. |
| `BIOPB_TMP` | `/tmp/biopb-${USER}` | Where the generated `runtime-config.json` is written |
| `TMPDIR/TEMP/TMP` | `/tmp` | Cache parent dir. Singularity auto-binds host `/tmp`, so the cache lands at `/tmp/biopb-cache-<uid>` on host disk (persistent). Set it to relocate — see [Cache Storage](#cache-storage) |
| `CACHE_MAX_TOTAL_GB` | `16` | Max total size of the on-disk file cache, in GB (only applies when generating config from env vars; ignored if `CONFIG_FILE` is set) |
| `CACHE_MAX_SEGMENT_MB` | (unset) | Max size of each cache segment file, in MB. Unset → server default (~64 MB) (same applicability as above) |

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

A Flight-only container has no HTTP endpoint to `curl`. Readiness is the Flight
`health` action, which is token-gated like every other verb — so run it inside
the container, where `BIOPB_TENSOR_TOKEN` is set:

```bash
docker exec biopb-tensor python -c \
  "from biopb.tensor import TensorFlightClient as C; \
   print(C('grpc://localhost:8815').health_check()['status'])"
```

> `docker exec` sees the environment the container was **started** with, so this
> works when you pass `-e BIOPB_TENSOR_TOKEN=...` (or run tokenless). With an
> auto-generated token the value only exists in the entrypoint's own process, so
> pass the token explicitly if you want a scripted health check — or fall back to
> a liveness-only TCP check on the published port.

With `BIOPB_ENABLE_HTTP_SIDECAR=1` the old HTTP probe is available again:

```bash
curl http://localhost:8814/livez
```

## Browser access

The container serves no browser origin. To view its data in a browser, run the
full biopb stack on your own machine (`biopb control start`) and **add this
container as a remote source** (`grpc://host:8815`, or `grpcs://` with TLS) — the
browser then talks only to its own loopback control, and the container needs no
HTTP surface, no CORS, and no browser-trusted certificate.

**CORS (sidecar only).** If you opted the HTTP sidecar back in and a webapp calls
it from a *different* origin, the browser's CORS preflight will reject
`Authorization: Bearer <token>` calls. Set `BIOPB_CORS_ORIGINS` to the web app's
origin(s) so the sidecar allows them:

```bash
docker run -d -p 8814:8814 -p 8815:8815 \
    -v ~/data:/data \
    -e BIOPB_ENABLE_HTTP_SIDECAR=1 \
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
- A port already in use inside the container (e.g. `--network host` colliding with a host process on 8815)
- `BIOPB_TENSOR_TLS=1` together with `BIOPB_ENABLE_HTTP_SIDECAR=1` — the entrypoint refuses that combination (exit 2); the sidecar cannot yet reach a TLS Flight server

> A config with **no sources** (or whose sources all fail to load) is **not** a
> startup error — the server boots and serves an **empty catalog** (health
> `SERVING`, zero sources), since sources can arrive later via a monitored
> directory, runtime add, or upload. See *Files not appearing* below.

### Token rejected (401 from gRPC / the sidecar)

- Verify token matches `BIOPB_TENSOR_TOKEN`
- Check token is 16-128 characters, URL-safe (`[A-Za-z0-9_-]`)
- Note: a Docker container always binds `0.0.0.0` (remote mode), so token enforcement stays on by default. To run tokenless, either set `BIOPB_TENSOR_ALLOW_NO_TOKEN=1` (serves the data API open — publish to `127.0.0.1` and stay on a trusted network) or, under Singularity, use `BIOPB_BIND_LOCALHOST=true`.

### Client refuses to reconnect after recreating a TLS container

The client pinned the container's self-signed cert on first connect (TOFU) and
the recreated container minted a new one. Mount a volume at
`/root/.local/state` (or supply a fixed `BIOPB_TLS_CERT`/`BIOPB_TLS_KEY`) so the
identity survives; to accept the new cert, drop the stale entry from the
client's `~/.local/state/biopb/tls-known-hosts.json`.

### Files not appearing

- Check mount path matches `DATA_DIR`: `-v ~/data:/data` with default `DATA_DIR=/data`
- New files take a moment to register: they appear only after the stability window passes and the next periodic rescan runs (this is poll-based and works on NFS/Lustre — no need to disable `MONITOR`)
- Check file format is supported

### Files fail to load

- Verify the file format is supported (see the supported-formats table in the [README](README.md))
- Verify file is not corrupted
- Check Docker logs for specific error
