# The algorithm plane

Serve functions to the biopb agent over the `biopb.image` `Ops` protocol, from
one file or a Docker image.

## Serve functions over `Ops`

One file, no packaging. A parameter annotated `Tensor(axes)` is a tensor
argument; every other parameter is a kwarg, advertised with its default.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["biopb-image-base[lazy]", "scikit-image"]
# ///
from biopb_image_base import Tensor, op, serve

@op(description="Mean intensity and area per label", labels=["measurement"])
def label_stats(image: Tensor("YX"), labels: Tensor("YX")) -> dict:
    from skimage.measure import regionprops_table
    return regionprops_table(labels, image, properties=["label", "area", "mean_intensity"])

@op(description="Gaussian denoise", labels=["denoising"], input="blocks", overlap=16)
def gaussian(image: Tensor("YX"), sigma: float = 2.0):
    from skimage.filters import gaussian as g
    return g(image, sigma=sigma, preserve_range=True)

if __name__ == "__main__":
    serve()
```

`uv run server.py --port 50051` serves it; `--describe` prints the op list and
exits.

- `input="eager"` (default) hands the function numpy arrays, `"lazy"` dask
  arrays, and `"blocks"` maps a pixelwise function over blocks with
  `block_shape` and `overlap`, iterating the axes not in `axes`.
- A single return value is the output `result`, a tuple `0`, `1`, .... Arrays
  are tensors; anything else is JSON. A generator yields progress strings
  and per-item outputs, and what it returns is the final event's outputs.
- Large results go to the embedded tensor server under `--cache-dir`, else to
  the plane named by `BIOPB_TENSOR_URL`/`BIOPB_TENSOR_TOKEN`, else inline.
- The core install serves inline pixels only; `[lazy]` adds lazy input and the
  plane sink. The server checks `$BIOPB_ALGORITHM_TOKEN` when set; bound off
  loopback without one, it mints one and prints it.

## Register a server with biopb-mcp

A server file copied to `~/.config/biopb/algorithms/<name>.py` is run by the
control in an environment of its own. A server running elsewhere is named with
a `<name>.json` file there:

``` json
{"url": "grpc://your_ip_address:50051"}
```

The control probes it and the kernel binds its ops into `ops`. A server that
does not implement `Ops` -- one still speaking the retired `ProcessImage`
protocol, say -- is listed as an error.

## The Docker base image

`biopb-image-base` carries biopb, the tensor server and this package, for a
server whose model is easier to ship as an image. It defines no entrypoint: a
derived image adds its model's packages and a server file.

```dockerfile
# Dockerfile
FROM biopb-image-base

RUN pip install --no-cache-dir cellpose
COPY server.py /opt/biopb/server.py

ENTRYPOINT ["python", "/opt/biopb/server.py", "--host", "0.0.0.0"]
CMD ["--cache-dir", "/data/cache"]
```

```bash
docker run --rm -p 50051:50051 -p 8817:8817 -v tensor-cache:/data/cache \
  my-biopb-server \
    --cache-dir /data/cache --cache-size 32GB \
    --tensor-external-location grpc://$(hostname):8817
```

Bound off loopback, the server mints a token and prints it unless
`BIOPB_ALGORITHM_TOKEN` is set. `--cache-dir` returns large results through an
embedded tensor server (port 8817), and `--tensor-external-location` is the
address clients reach it at: `localhost` works only for clients on the same
host.

## Development

### Build the base image

Run from repo root:

```bash
./biopb-image-runtime/scripts/build.sh            # from the latest tags
./biopb-image-runtime/scripts/build.sh --no-cache
```

Or build the wheels yourself:

```bash
pip wheel . --no-deps -w wheels/
pip wheel biopb-tensor-server/ --no-deps -w wheels/
docker build -t biopb-image-base -f biopb-image-runtime/Dockerfile .
```

`docker compose -f biopb-image-runtime/docker-compose.yaml up` runs
`echo_server.py`, an `Ops` server that echoes its input, with the embedded
cache.

### Tests

```bash
pip install -e "biopb-image-runtime[test]"
pytest biopb-image-runtime/tests/
```

The server tests run an `Ops` server file as a subprocess; no Docker or GPU.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BIOPB_LOG_LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `BIOPB_ALGORITHM_TOKEN` | The token the server checks |
| `BIOPB_TENSOR_URL`, `BIOPB_TENSOR_TOKEN` | The data plane large results go to (without `--cache-dir`) |

## Files

```
biopb-image-runtime/
├── Dockerfile              # Base image for derived services
├── docker-compose.yaml     # Development setup
├── echo_server.py          # The compose file's Ops server
├── pyproject.toml          # Python package
├── requirements.txt        # The image's dependencies
├── src/biopb_image_base/
│   ├── __init__.py
│   ├── ops.py              # @op, serve(): the Ops server
│   ├── server.py           # Embedded tensor cache
│   ├── common.py           # Token interceptor, error translation
│   ├── health.py           # gRPC health check
│   ├── logging_config.py   # Logging setup
│   ├── stitch.py           # Stitching tiled segmentations
│   └── dynamics_local.py   # Flow dynamics for stitching
├── tests/
├── scripts/
│   └── build.sh            # Build script
└── README.md
```
