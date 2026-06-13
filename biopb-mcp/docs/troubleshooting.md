# Troubleshooting

## Connecting to a tensor server

The Tensor Browser connects to a [biopb tensor server](https://github.com/biopb/biopb)
over Arrow Flight. The server URL is resolved in this order:

1. `BIOPB_TENSOR_URL` (and `BIOPB_TENSOR_TOKEN`) environment variables
2. the saved `tensor_browser.server_url` in your config
3. the default `grpc://localhost:8815`

### Auto-starting a local tensor server

If the initial connection fails, the URL is local, and the `biopb`
command-line tool is on your `PATH`, the browser offers to start a local server
for you by running `biopb server start`. If `biopb` is **not** installed, this
offer is skipped silently — install the full
[biopb](https://github.com/biopb/biopb) system, or point `BIOPB_TENSOR_URL` at a
server that is already running.

### Startup failures

When auto-start fails, the browser shows the underlying cause inline; the full
server output is written to `~/.local/share/biopb/logs/`. Common causes:

- **Port already in use** — most likely on a shared machine or HPC node where
  another user already holds the default port (gRPC `8815`). Either start your
  server on a different base port with `BIOPB_BASE_PORT` (e.g.
  `BIOPB_BASE_PORT=9000`), or set `BIOPB_TENSOR_URL` to the existing server
  instead of starting a new one.
- **Server started but not reachable in time** — startup exceeded the timeout;
  check the server log for the real error and try connecting again once it is up.

Each user gets a private on-disk cache (e.g. `/tmp/biopb-cache-<uid>`) with its
own lock, so multiple users running their own server on the same node do not
collide.
