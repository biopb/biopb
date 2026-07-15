# The algorithm plane

Base Docker image and utilities for implementing algorithm plugins using the `biopb.image` protocol.

## Build a custom algorithm plugin

### Overview
`biopb-image-runtime` creates a base image for adding algorithm plugins, so the agent has more tools to use.

The base image does not define a default entrypoint and is not meant to be run directly without specifying a servicer.

For real servicers derive from the `BiopbServicerBase` class with your custom algorithm. You can also enable and expose the gRPC health service through the convenient function `run_server()`.

### Example servicer
```python
# my_servicer.py
import biopb.image as proto

from biopb_image_base import (
    run_server,            # Run server (optionally with embedded cache)
    decode_image_data,     # Decode ImageData to numpy/dask
    return_lazy_or_eager,  # Return result inline (small) or as a lazy tensor ref (large)
    BiopbServicerBase,     # Base class for servicers
)

class MyServicer(BiopbServicerBase):

    def GetOpNames(self, request, context):
        """Description of your algorithm"""
        with self._server_context(context):
            return proto.OpNames(
                names=["my_algorithm"],
                op_schemas={
                    "my_algorithm": proto.OpSchema(
                        description="Best image processing algorithm",
                    ),
                }
            )

    def Run(self, request, context):
        img = decode_image_data(request.image_data)
        result = ...  # your result (numpy or dask array)
        return proto.ProcessResponse(
            image_data=return_lazy_or_eager(result, self._tensor_cache)
        )

# Run the service
run_server(MyServicer(), port=50051)
```

See [biopb-server](https://github.com/biopb/biopb-server) project for fully functional implementation examples.

### Run the new servicer
```bash
docker run --rm -p 50051:50051 \
    -v /path/to/my_servicer.py:/opt/biopb/my_servicer.py \
    jiyuuchc/biopb-image-base \
    python /opt/biopb/my_servicer.py
```

### Register the server with biopb-mcp

Add your server's URL to the `services.process_image_servers` list in the
biopb-mcp config (`~/.config/biopb/mcp-config.json`):

``` json
{
    "services": {
        "process_image_servers": ["grpc://your_ip_address:50051"]
    }
}
```

Each URL is queried via `GetOpNames` and exposed as callables in the kernel's
`ops` dict. You can also edit this without touching the file: the browser
dashboard's **MCP Settings** page (served by the control) edits the same
`mcp-config.json`.

## Architecture
This subproject provides:
- **Base Docker image** for ProcessImage/ObjectDetection gRPC services
- **Mock service** for pytest and explicit infrastructure testing without real ML models
- **Embedded tensor cache** for larger-than-memory data handling
- **Utilities** for image encoding/decoding, authentication, and lazy data handling
- **Test client** for verifying gRPC connectivity

### Single Process with Embedded Tensor Cache

The `run_server()` helper optionally starts an embedded TensorFlightServer for ephemeral cache. Large results are uploaded to the file-based cache and returned as `SerializedTensor` references.

```
┌─────────────────────────────────────────────┐
│  image-server (single process)              │
│                                             │
│  ┌─────────────┐      ┌─────────────────┐   │
│  │ gRPC server │      │ embedded Flight │   │
│  │ port 50051  │◀───▶│ cache server    │   │
│  │             │      │ port 8817       │   │
│  └─────────────┘      └─────────────────┘   │
│                              │              │
│                              ▼              │
│                       /data/cache/          │
│                       (file-based)          │
└─────────────────────────────────────────────┘

run_server(
    servicer,
    port=50051,               # main grpc port
    cache_dir="/data/cache",  # Enables lazy data handling
    cache_size="32GB",
    health_check=True,
)

```

## Development

### Build Base Docker Images

Run from repo root:

```bash
cd /path/to/biopb  # repo root
./biopb-image-runtime/scripts/build.sh

# Force rebuild without cache
./biopb-image-runtime/scripts/build.sh --no-cache
```
Or manually (build wheels yourself):

```bash
cd /path/to/biopb  # repo root

# Build wheels first (the Dockerfile installs both biopb and biopb_tensor_server)
pip wheel . --no-deps -w wheels/
pip wheel biopb-tensor-server/ --no-deps -w wheels/

# Build Docker image
docker build -t biopb-image-base -f biopb-image-runtime/Dockerfile .
```

### Run the Mock Service Explicitly

The mock servicer is available for pytest and explicit development workflows. Run it by providing the Python module explicitly:

**With embedded cache:**

```bash
docker run --rm -p 50051:50051 -p 50052:8817 -v tensor-cache:/data/cache \
    biopb-image-base \
    python -m biopb_image_base.mock_servicer \
    --cache-dir /data/cache --cache-size 32GB \
    --tensor-external-location grpc://localhost:50052
```

`--tensor-external-location` is **required** when the embedded cache binds
`0.0.0.0` — it is the address clients use to reach the cache (here the mapped host
port). Pass `--local` instead to bind loopback and skip it.

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

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BIOPB_LOG_LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Python API

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
```

### biopb.image Protocol

| Service | Methods |
|---------|---------|
| ObjectDetection | RunDetection, RunDetectionStream, RunDetectionOnGrid, RunModelAdaptation, GetOpNames |
| ProcessImage | Run, RunStream, GetOpNames |

See [buf.build](https://buf.build/jiyuuchc/biopb/docs/main%3Abiopb.image) for full definition.

### Build a docker image for your servicer

```dockerfile
# Dockerfile
FROM biopb-image-base

COPY my_servicer.py /opt/biopb/my_servicer.py

ENTRYPOINT ["python", "/opt/biopb/my_servicer.py"]
CMD ["--cache-dir", "/data/cache"]
```

Then run it:

```bash
docker run --rm \
  -p 50051:50051 \
  my-biopb-servicer

# With cache server and lazy I/O
docker run --rm \
  -p 50051:50051 \
  -p 8817:8817  \
  -v tensor-cache:/data/cache \
  my-biopb-servicer \
    --cache-dir /data/cache \
    --cache-size 32GB \
    --tensor-external-location \
    grpc://$(hostname):8817

# For local deployment/testing
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
