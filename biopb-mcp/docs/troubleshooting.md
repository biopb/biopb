# Troubleshooting

## Connecting to a tensor server

The Tensor Browser needs to reach the biopb-control's endpoints in order to
talk to a [tensor server](https://github.com/biopb/biopb/tree/main/biopb-tensor-server)
(the *data plane*). This is normally a non-issue: An agent harness launching
biopb-mcp over the stdio shim gets this for free — its shim starts the control
automatically. But a plain `napari` session (direct `napari` run) will show
**"No biopb control plane is running"**. So, start one:

```
biopb control start
```

then press **Connect** in the Tensor Browser.

### Pointing at a server the control does not own

Occasionally you want to use a tensor server without the control plane — a
dev server, or one on another machine — set `BIOPB_TENSOR_URL` (plus
`BIOPB_TENSOR_TOKEN` if it needs a token). That bypasses the control completely.

### Startup failures

When the data plane fails to come up, the browser shows the underlying cause
inline; the full server output is written to `~/.local/state/biopb/logs/`
(the MCP session's own log is under `~/.local/state/biopb/mcp/`). Common causes:

- **Port already in use** — most likely on a shared machine or HPC node where
  another user already holds the default port (gRPC `8815`). Either point at the
  existing server (`BIOPB_TENSOR_URL`), or give your own a free port: the
  containerized/HPC server honors `BIOPB_BASE_PORT` (e.g. `BIOPB_BASE_PORT=9000`;
  HTTP=`BASE+4`, gRPC=`BASE+5`), while a local `biopb control start` takes the
  same number as `--base-port`.
- **Server started but not reachable in time** — startup exceeded the timeout;
  check the server log for the real error and try connecting again once it is up.

Each user gets a private on-disk cache (e.g. `/tmp/biopb-cache-<uid>`) with its
own lock, so multiple users running their own server on the same node do not
collide.
