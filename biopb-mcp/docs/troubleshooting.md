# Troubleshooting

## Connecting to a tensor server

The Tensor Browser reads image data from a
[biopb tensor server](https://github.com/biopb/biopb) (the *data plane*) over
Arrow Flight. **The control plane decides where that server is**, so there is no
server URL to configure: at connect time the client asks the control, which brings
the plane up if it is down and reports the address it bound.

So if the browser says **"No biopb control plane is running"**, start one:

```
biopb control start
```

then press **Connect** in the Tensor Browser (or restart the kernel). An agent
launching biopb-mcp over stdio gets this for free — its shim starts the control
automatically — but a plain `napari` session or `biopb mcp view` expects you to
have started it, and `biopb mcp view` refuses to open at all without one.

### Pointing at a server the control does not own

To use a tensor server nothing supervises — a dev server, or one on another
machine — set `BIOPB_TENSOR_URL` (plus `BIOPB_TENSOR_TOKEN` if it needs a token).
That bypasses the control completely: it is not consulted, and no local data plane
is started as a side effect.

Note that the control's own credential — the token file it writes for the plane it
owns — is **never** sent to such a server. If one needs authentication, say so with
`BIOPB_TENSOR_TOKEN`; the error message will tell you the same thing.

### Startup failures

When the data plane fails to come up, the browser shows the underlying cause
inline; the full
server output is written to `~/.local/state/biopb/logs/` (the MCP session's own
log is under `~/.local/state/biopb/mcp/`). Common causes:

- **Port already in use** — most likely on a shared machine or HPC node where
  another user already holds the default port (gRPC `8815`). Either point at the
  existing server (`BIOPB_TENSOR_URL`), or give your own a free port: the
  containerized/HPC server honors `BIOPB_BASE_PORT` (e.g. `BIOPB_BASE_PORT=9000`;
  HTTP=`BASE+4`, gRPC=`BASE+5`), while a local `biopb server start` takes its
  gRPC `port` from the config file (`biopb.json`) and its HTTP port from `--web-port`.
- **Server started but not reachable in time** — startup exceeded the timeout;
  check the server log for the real error and try connecting again once it is up.

Each user gets a private on-disk cache (e.g. `/tmp/biopb-cache-<uid>`) with its
own lock, so multiple users running their own server on the same node do not
collide.
