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
  another user already holds the default port (gRPC `8815`). Either point at the
  existing server (`BIOPB_TENSOR_URL`), or give your own a free port: the
  containerized/HPC server honors `BIOPB_BASE_PORT` (e.g. `BIOPB_BASE_PORT=9000`;
  HTTP=`BASE+4`, gRPC=`BASE+5`), while a local `biopb server start` takes its
  gRPC `port` from the TOML config and its HTTP port from `--web-port`.
- **Server started but not reachable in time** — startup exceeded the timeout;
  check the server log for the real error and try connecting again once it is up.

Each user gets a private on-disk cache (e.g. `/tmp/biopb-cache-<uid>`) with its
own lock, so multiple users running their own server on the same node do not
collide.

## Scrambled or "ghosted" images in the viewer

Occasionally a layer renders incorrectly: pixels from a previously-viewed or
already-removed layer bleed through, the image looks scrambled or composited with
unrelated data, or a stale frame "sticks" and does not update when you change the
slice, zoom, or 2D/3D mode.

**Your data is not corrupted.** This is a display-layer glitch in napari's GPU
canvas (vispy/OpenGL), not a problem with the pixels — the bytes served by the
tensor server and held by the layer are unaffected (a byte-for-byte comparison of
the on-disk file, the served tensor, and the layer's displayed slice matches
exactly). It tends to appear after adding and removing many layers in one session
(their GPU visuals are not always torn down cleanly, so a leftover visual keeps
drawing its last texture under the current layer), and can be aggravated by older
or virtualized GPU drivers.

### Fixing it

1. **Restart the viewer — the reliable fix.** Close the napari window and reopen
   it (or restart napari). This rebuilds the canvas from scratch and clears any
   leftover visuals. In an agent/MCP session, restart the kernel instead
   (`restart_kernel`).
2. Lighter attempts that *sometimes* clear it without a restart: toggle 2D ⇄ 3D
   (the square/cube button, or press `2` / `3`), resize the viewer window, or
   remove and re-add the affected layer. These are not guaranteed — a leftover
   visual can survive a layer removal — so fall back to a restart if the scramble
   persists.

Confirm the recovery by re-adding the layer: if it now renders correctly, the
earlier frame was a stale-visual artifact, not your data.

### Reducing how often it happens

- Update your GPU drivers.
- Avoid bulk-adding then removing large numbers of layers in a single session;
  the leftover-visual leak is most likely to surface in that workflow.
