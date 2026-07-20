# biopb-control

The **biopb control plane**: a lean, always-on process that is the durable root
of a biopb deployment. It

- **supervises the durable planes** as subprocesses — starting the tensor
  (data) plane, monitoring its liveness, and restarting it on crash with
  backoff — without ever importing them, and
- **is the single web origin** — it serves the built `web/` SPA (dashboard `/`,
  dataviewer `/viewer`, admin, and each session's observe page at
  `/session/<id>/observe`) and reverse-proxies the data plane (`/data_plane/*`)
  and each session's API (`/session/<id>/api/*`).

It is deliberately *not* a compute host: no napari, Qt, dask, kernel, or import
of `biopb-tensor-server` / `biopb-mcp`. Everything heavy is a supervised
subprocess. See `biopb-mcp/ARCHITECTURE.md` for the full
architecture (the control is the root of the dependency tree; MCP sessions are
ephemeral, shim-owned clients that *use* the planes and never start them).

## Usage

Managed through the core `biopb` CLI (which owns the pidfile / detach / stop
plumbing):

```
biopb control start      # detach a persistent supervisor; brings up the data plane
biopb control status     # is the control up? is the data plane serving?
biopb control stop
biopb control run        # run in the foreground (Ctrl-C to stop)
```

`biopb control start` brings the data plane up by default; pass `--no-data-plane`
to run the control as an adopt-only supervisor (it will only monitor / restart a
tensor server that is already running, not spawn one).

The supervisor exposes the control API and serves the browser UI (default
`127.0.0.1:8813` in local mode; public behind a required token under `--remote`).
Clients use the API to ask "is the data plane up, and bring it up if not" — this
is what replaced `biopb-mcp` shelling out `biopb server start`.
