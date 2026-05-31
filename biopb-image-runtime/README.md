# biopb-image-runtime

Base Docker image and utilities for gRPC services implementing the `biopb.image` protocol.

## Overview

This subproject provides:
- **Base Docker image** for ObjectDetection/ProcessImage gRPC services
- **Mock service** for pytest and explicit infrastructure testing without real ML models
- **Embedded tensor cache** for larger-than-memory data handling
- **Utilities** for image encoding/decoding, authentication, and lazy data handling
- **Test client** for verifying gRPC connectivity

## Architecture

### Single Process with Embedded Tensor Cache

The `run_server()` helper optionally starts an embedded TensorFlightServer for ephemeral cache. Large results are uploaded to the file-based cache and returned as `SerializedTensor` references.

```
┌─────────────────────────────────────────────┐
│  image-server (single process)               │
│                                              │
│  ┌─────────────┐      ┌─────────────────┐   │
│  │ gRPC server │      │ embedded Flight │   │
│  │ port 50051  │◀────▶│ cache server    │   │
│  │             │      │ port 8817       │   │
│  └─────────────┘      └─────────────────┘   │
│                              │               │
│                              ▼               │
│                       /data/cache/           │
│                       (file-based)           │
└─────────────────────────────────────────────┘
```

## Usage

### Build Docker Images

Run from repo root:

```bash
cd /path/to/biopb  # repo root
./biopb-image-runtime/scripts/build.sh

# Force rebuild without cache
./biopb-image-runtime/scripts/build.sh --no-cache
```

Or manually (requires pre-built wheel):

```bash
cd /path/to/biopb  # repo root

# Build wheel first
pip wheel . --no-deps -w wheels/

# Build Docker image
docker build -t biopb-image-base -f biopb-image-runtime/Dockerfile .
```

### Run a Derived Service

`biopb-image-base` is a base image. It does not define a default entrypoint and is not meant to be run directly without specifying a servicer.

It also does not define a default health check. Health checks belong in `docker-compose.yaml` for the mock workflow or in derived service images that declare their own runtime contract.

For real servicers, enable and expose the gRPC health service from your own service process. The base utilities already support this through `create_server()` and `run_server()` with `health_check=True` and optional readiness checks; derived images should add the container-level health probe that matches their runtime entrypoint.

Build a derived image that adds your servicer and sets an entrypoint:

```dockerfile
FROM biopb-image-base

COPY my_servicer.py /opt/biopb/my_servicer.py

ENTRYPOINT ["python", "/opt/biopb/my_servicer.py"]
CMD ["--cache-dir", "/data/cache"]
```

Then run it:

```bash
# For deployment: use externally reachable hostname/IP
docker run --rm \
  -p 50051:50051 \
  -p 8817:8817  \
  -v tensor-cache:/data/cache \
  my-biopb-servicer \
    --cache-dir /data/cache \
    --cache-size 32GB \
    --tensor-external-location \
    grpc://$(hostname):8817

# For local testing only (clients on same host):
docker run --rm \
  -p 50051:50051 \
  -p 8817:8817  \
  -v tensor-cache:/data/cache \
  my-biopb-servicer \
    --cache-dir /data/cache \
    --cache-size 32GB \
    --tensor-external-location \
    grpc://localhost:8817
```

**Note:** `--tensor-external-location` must be an address that clients can reach.
Using `localhost` only works when clients run on the same host machine.
For deployment, use the server's hostname or IP (e.g., `grpc://server-host:8817`).

### Run the Mock Service Explicitly

The mock servicer is available for pytest and explicit development workflows. Run it by providing the Python module explicitly:

**With embedded cache:**

```bash
docker run --rm -p 50051:50051 -p 50052:8817 -v tensor-cache:/data/cache \
    biopb-image-base \
    python -m biopb_image_base.mock_servicer \
    --cache-dir /data/cache --cache-size 32GB
```

**Standalone (eager data only, no lazy support):**

```bash
docker run --rm -p 50051:50051 \
    biopb-image-base \
    python -m biopb_image_base.mock_servicer --local
```

**With docker-compose:**

```bash
cd /path/to/biopb
docker compose -f biopb-image-runtime/docker-compose.yaml up
```

The compose file defines the mock service health check explicitly.

### Test Client

The test client (`client.py`) is included in the base image for testing real servicers from within Docker.

**From within Docker (testing a real servicer):**

```bash
# Run against a servicer in another container or host
python /opt/biopb/tests/client.py --port 50051 --ip <server_ip>

# With authentication token
python /opt/biopb/tests/client.py --port 50051 --ip <server_ip> --token <your-token>

# Streaming test
python /opt/biopb/tests/client.py --port 50051 streaming --iterations 4
```

**Local pytest tests (for mock servicer):**

```bash
cd biopb-image-runtime
pip install -e .[test]

# Run against local mock server
pytest tests/ -v
```

### Build Options

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BIOPB_LOG_LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

## Python API

### Base Utilities

```python
from biopb_image_base import (
    create_server,       # Create configured gRPC server
    run_server,          # Run server with optional embedded cache
    setup_logging,       # Configure logging
    decode_image_data,   # Decode ImageData to numpy/dask
    encode_image,        # Encode numpy to ImageData
    return_lazy_or_eager,  # Return large results as lazy data
    BiopbServicerBase,   # Base class for servicers
)

# Example servicer
class MyServicer(BiopbServicerBase):
    def RunDetection(self, request, context):
        with self._server_context(context):
            img = decode_image_data(request.image_data)
            # ... process ...
            detections = [...]  # your detections
            return DetectionResponse(detections=detections)

    def Run(self, request, context):
        with self._server_context(context):
            img = decode_image_data(request.image_data)
            result = ...  # your result (numpy or dask array)
            return ProcessResponse(
                image_data=return_lazy_or_eager(result, self._tensor_cache)
            )
```

### Running a Service

```python
from biopb_image_base import run_server

servicer = MyServicer()

# With embedded tensor cache (recommended)
run_server(
    servicer,
    port=50051,
    cache_dir="/data/cache",  # Enables lazy data handling
    cache_size="32GB",
    health_check=True,
)

# Or without cache (eager data only)
run_server(servicer, port=50051)
```

## Dockerfile for Derived Services

Provide your servicer entrypoint in the derived image:

```dockerfile
FROM biopb-image-base

# Install your ML dependencies
RUN pip install torch cellpose

# Copy your servicer
COPY my_servicer.py /opt/biopb/my_servicer.py

ENTRYPOINT ["python", "/opt/biopb/my_servicer.py"]
CMD ["--cache-dir", "/data/cache"]

# Add a service-specific health probe in the derived image
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1
```

Or run directly without custom Dockerfile:

```bash
docker run --rm -p 50051:50051 \
    -v /path/to/my_servicer.py:/opt/biopb/my_servicer.py \
    biopb-image-base \
    python /opt/biopb/my_servicer.py --cache-dir /data/cache
```

## biopb.image Protocol

| Service | Methods |
|---------|---------|
| ObjectDetection | RunDetection, RunDetectionStream, RunDetectionOnGrid, RunModelAdaptation, GetOpNames |
| ProcessImage | Run, RunStream, GetOpNames |

## Files

```
biopb-image-runtime/
├── Dockerfile              # Base image for derived services
├── docker-compose.yaml     # Development setup
├── pyproject.toml          # Python package
├── requirements.txt        # Dependencies
├── src/biopb_image_base/
│   ├── __init__.py
│   ├── logging_config.py   # Logging setup
│   ├── server.py           # gRPC server + embedded tensor cache
│   ├── common.py           # Image utilities, base servicer
│   ├── health.py           # gRPC health check
│   ├── debug.py            # Stats, GPU info (nvidia-smi)
│   └── mock_servicer.py    # Mock implementation
├── tests/
│   ├── client.py           # Test client CLI (included in image)
│   ├── test_service.py     # pytest integration tests
│   └── test_image.png      # Sample image
├── scripts/
│   └── build.sh            # Build script
└── README.md
```