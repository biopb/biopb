# biopb-admin

The **biopb control plane**: a lean, always-on process that is the durable root
of a biopb deployment. It

- **supervises the durable planes** as subprocesses — starting the tensor
  (data) plane, monitoring its liveness, and restarting it on crash with
  backoff — without ever importing them, and
- (later) fronts a **single web origin** for the dataviewer, the admin UI, and
  each session's observe page.

It is deliberately *not* a compute host: no napari, Qt, dask, kernel, or import
of `biopb-tensor-server` / `biopb-mcp`. Everything heavy is a supervised
subprocess. See `biopb-mcp/docs/mcp-dedaemonization-migration.md` for the full
architecture (the admin is the root of the dependency tree; MCP sessions are
ephemeral, shim-owned clients that *use* the planes and never start them).

## Usage

Managed through the core `biopb` CLI (which owns the pidfile / detach / stop
plumbing):

```
biopb admin start      # detach a persistent supervisor; brings up the data plane
biopb admin status     # is the admin up? is the data plane serving?
biopb admin stop
biopb admin run        # run in the foreground (Ctrl-C to stop)
```

`biopb admin start` brings the data plane up by default; pass `--no-data-plane`
to run the admin as an adopt-only supervisor (it will only monitor / restart a
tensor server that is already running, not spawn one).

The supervisor exposes a small loopback control API (default
`127.0.0.1:8813`) that clients use to ask "is the data plane up, and bring it up
if not" — this is what replaced `biopb-mcp` shelling out `biopb server start`.
