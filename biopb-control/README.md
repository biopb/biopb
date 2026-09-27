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
subprocess. See [ARCHITECTURE.md](ARCHITECTURE.md) for the supervision model, the
invariants, the web-origin routing table, and the session registry; and
`biopb-mcp/ARCHITECTURE.md` for the sessions themselves (ephemeral, shim-owned
clients that *use* the planes and never start them).

## Usage

Managed through the core `biopb` CLI (which owns the pidfile / detach / stop
plumbing):

```
biopb control start      # detach a persistent supervisor; brings up the data plane
biopb control status     # is the control up? is the data plane serving?
biopb control stop
```

`biopb control start` brings the data plane up by default; pass `--no-data-plane`
to run the control as an adopt-only supervisor (it will only monitor / restart a
tensor server that is already running, not spawn one). It resolves sensible
defaults for everything, including `--config` and `--static-dir`.

Foreground use (a systemd/launchd unit, an Open OnDemand app, debugging
supervision) runs `biopb-control run` (or `python -m biopb_control run` if that
script isn't on PATH) directly — the same entry point `start` spawns as its own
child. Unlike `start`, it has **no defaults**: `--config` and `--static-dir`
must both be passed explicitly (and every other setting individually, rather
than derived from `--base-port`) or the process refuses to start / comes up
with no web UI. This is deliberately not advertised as a drop-in replacement
for casual use — reach for `biopb control start` unless you specifically need
a foreground process (`biopb/biopb#736`: `biopb control run`, the typer wrapper
that used to fill in those same defaults, bypassed the fail-closed guard this
entry point already enforces on a public `--control-host` — retired rather than
kept as a second copy of the same check).

**Ports** come from `--base-port` (default 8810): control = base+3, sidecar =
base+4, flight = base+5, the container's convention. A control that moved off
8813 publishes where it landed in `state/biopb/control.json`, so `status` /
`logs` and biopb-mcp follow it; `stop` and the bind path deliberately do not, so
a stale record can never dictate the next bind.

The supervisor exposes the control API and serves the browser UI on
`127.0.0.1:<base+3>` — **with either bind**. `--grpc-bind` publishes the *flight*
plane (TLS-capable, token-gated), never this listener: the control is plaintext
HTTP with no TLS support, so publishing it would put the data-plane token on the
wire in the clear (biopb/biopb#614). Reach the UI from another machine over a
tunnel, `ssh -L 8813:localhost:8813 <host>`, which `control start` prints
whenever the plane goes public. To publish it anyway (behind your own TLS proxy),
pass an explicit `--control-host 0.0.0.0` to `biopb-control run`; that bind is
fail-closed and refuses to come up without a token.

A public `--grpc-bind` also requires `--grpc-external-location` — the address a
*different* machine dials to reach the plane (e.g. an HPC scheduler's assigned
FQDN), since a wildcard bind is not itself dialable. Forwarded verbatim to the
data plane, which advertises it to clients and is where it is actually enforced.

Clients use the API to ask "is the data plane up, and bring it up if not" — this
is what replaced `biopb-mcp` shelling out `biopb server start`.
