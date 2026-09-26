# biopb-mcp Architecture

## Overview

`biopb-mcp` is an **MCP server** that connects [napari](https://napari.org) and AI
agents to biopb servers. It gives an agent a live Python kernel holding the data
plane (`client`) and the algorithm plane (`ops` and plugin modules), plus a napari
viewer where the session has one. Thesis: *"agent first; provide tools only if they
help."* Image results go to the viewer or the control's web viewer, other results to
the agent's chat.

The viewer is optional: it needs the `[napari]` extra, `viewer.enabled`, and a
display. The launcher decides once per session and hands the kernel its reason when
there is none; the kernel then skips Qt and napari entirely. A display-less host runs
without one unless `viewer.virtual_display` opts into a launcher-owned Xvfb, which is
for tests.

The viewer docks the **Tensor Browser**, and the agent's `add_tensor` builds its
layers through the same pipeline; both come from the separate napari plugin
[biopb-napari-widget](https://github.com/biopb/biopb-napari-widget), which
depends on nothing here. biopb-mcp imports only its three root exports:
`TensorBrowserWidget`, `add_tensor_layer` and `wrap_levels`. `add_tensor_layer`
builds napari layers, so it cannot live in the napari-free SDK, and it must not
be copied back here: sharing it is what keeps the browser and `add_tensor` in
agreement on what a layer is (label sets, axes, pyramids).

---

## Process structure

Everything the package runs is a chain of four processes, each spawned and reaped
by the one above it. The whole chain is **client-scoped**: it comes up when an MCP
client connects and is gone when that client disconnects.

```
              AI agent / MCP client
                        │  stdio JSON-RPC (fd 0 / fd 1)
                        ▼
   ┌──────────────────────────────────────────────────┐
   │ shim                     per client connection   │
   │   owns fd 1; imports only the mcp SDK            │
   └────────────────────────┬─────────────────────────┘
                            │  http → /mcp, dynamic port
                            ▼
   ┌──────────────────────────────────────────────────┐
   │ session child            ephemeral, shim-owned   │
   │   FastMCP / uvicorn  — tools + resources         │
   │   KernelHost         — owns the kernel           │
   │   observe UI         — job history + cancel      │
   └───────┬──────────────────────────────────────────┘
           │ spawns; jupyter ZMQ
           ▼
   ┌────────────────────────────┐   ┌───────────────────────┐
   │ Jupyter kernel             │┈┈►│ dask LocalCluster     │
   │   napari viewer window (Qt)│   │   scheduler + workers │
   │   agent namespace          │   └───────────────────────┘
   │   job runner, viewer proxy │    spawned by the kernel,
   └──────────────┬─────────────┘    only when asked for
                  │ Flight / gRPC
                  ▼
      data plane · algorithm servers · control plane
      (outside this package — see ../development.md)
```

One ownership fact deliberately does not follow the spawn chain: the **planes at
the bottom are never started here** — the session is a pure client of them, and
only *registers* itself with the control.

### Why this shape

The chain is split where it is because of **fd-1 corruption**. Under stdio MCP,
**fd 1 *is* the JSON-RPC channel**, so any stray stdout from a heavy process
(uvicorn/Qt/dask/kernel) corrupts it. Hence the **shim/heavy split**: a
featherweight shim owns fd 1 and imports only the mcp SDK, and all heavy work runs
in a separate child it bridges to over http — making fd-1 corruption structurally
impossible.

The biopb-mcp package is the only biopb component that requires a GUI environment
(to run napari). This is why it is **client-scoped** rather than a shared daemon:
to avoid a stale env, e.g. `DISPLAY` (#98).

The session's relationship to the durable planes — it uses them, never starts them,
and only registers itself with the control — is the control plane's side of the
contract; see [`../biopb-control/ARCHITECTURE.md`](../biopb-control/ARCHITECTURE.md).

---

## Components

### Shim-owned MCP sessions

Shim (`--transport stdio`) is the interface the mcp clients (claude code) see, which

1. Start-and-forget the control plane; writes a session registry at the state dir,
   so the control can see it.
2. **spawns its own ephemeral session child** (FastMCP/uvicorn + the kernel host)
   on a **dynamic OS-assigned port**,
3. **bridges** stdio JSON-RPC ↔ that child's `/mcp` until the client closes stdin,
   replaying the child's initialize result **verbatim** — including `instructions`.
4. **reaps** the child and its kernel grandchild as a tree (POSIX process group +
   parent-death pipe; Windows Job Object, #403) on the way out.

### The kernel

The session owns a **single child Jupyter kernel** hosting the tensor client, dask,
`ops`, the plugin modules and, where there is one, the napari viewer. Agent code runs *in that kernel*, not on the MCP
thread or napari's Qt loop — so a runaway execution can be interrupted or
hard-restarted without killing the MCP server. The host's round trips are quick
snippets that may overlap: one threaded client routes each reply and iopub
message to the call it answers, and the kernel runs requests in arrival order.

The kernel is **launched lazily, not at boot**, so a long-running server binds
cheaply and never pops a napari viewer until user requested it; kernel-dependent
tools return a structured not-ready status until then. The health probe waits for
the bootstrap's `_jobs`, not for `viewer`. **Closing the napari window tears the
kernel back down to idle**, and `start_kernel` rebuilds it.

An agent's code runs as a cell on the kernel's main thread, like a notebook user's,
so the Qt loop pauses while it runs; a long compute can run on a worker thread through
`run_async`, keeping the viewer live. The agent-facing `viewer` object is a
**main-thread marshaling proxy** over the real `napari.Viewer`, because an off-main
napari mutation can segfault the kernel (#100).

### dask cluster

The kernel leaves dask as dask configures itself: computes run on the in-process
scheduler, shared with the napari viewer, and a cell that wants parallel
multi-process computation builds a dask `Client` itself. Defaulting to in-process is a
reliability choice: a long-lived cluster nobody opted into is one nobody watches,
and a suspended host left every `.compute()` blocked on a scheduler whose workers
were gone (**#970**). Stop needs no dask machinery either: a blocking dask
`Client` call cancels its own futures when interrupted.

A cluster a cell spins belongs to the **kernel** — its workers are that process
group's children, so they go down with it. There is no cluster machinery in the
session child or the kernel's bootstrap, and no MCP tool: attaching is
`Client(...)` and detaching is `close()`. The one piece kept is the viewer's pin:
its slice reads run in-process whatever the default scheduler is
(biopb-napari-widget's `wrap_levels`, #8).

### Data connection

The kernel and the Tensor Browser share one `biopb.tensor.Connection` (the SDK's):
the plane's URL, the live Flight client, and why there is none. The kernel builds
it and hands it to the widget, which connects it, so a reconnect in the widget is
the agent's next `client`. It caches nothing; the catalog the tree is drawn from
is the widget's own (biopb-napari-widget's `SourceList`), re-listed when the server's
`source_count` moves.

Where the plane is comes from `biopb.control`, the SDK's client of the control
(#628): one `ensure_data_plane` call brings the plane up if it is down and returns
its endpoint and credential. `$BIOPB_TENSOR_URL` is the one escape hatch, for a
data server the control does not supervise; it bypasses the control completely,
and the control's credential is **never** sent to it — authenticate it with
`$BIOPB_TENSOR_TOKEN`. A local TLS data server is trusted from disk. The contract
is `docs/discovery-contract.md`.

A session with no control has no data plane to be told about. The stdio shim
starts the control for an agent harness; a plain `napari` session does not, and
the Tensor Browser then offers a URL and token field; `biopb mcp view` refuses to
open without one.

### Ports, logs, and per-user isolation

The data plane's default gRPC port is **8815**, which is the one piece of shared
state on a multi-user machine — an HPC node where another user already holds it is
the common startup failure. Two ways out: point at the existing server with
`$BIOPB_TENSOR_URL`, or move your own. The containerized/HPC server derives its
ports from `$BIOPB_BASE_PORT` (HTTP `BASE+4`, gRPC `BASE+5`); a local
`biopb control start` takes the same number as `--base-port`.

Everything else is already per-user: each user gets a private on-disk cache
(`/tmp/biopb-cache-<uid>`) with its own lock, so co-tenants running their own
servers on one node do not collide. When a plane fails to come up the browser
shows the cause inline; the full server output is in
`~/.local/state/biopb/logs/`, and the MCP session's own log in
`~/.local/state/biopb/mcp/`.

### Extending the kernel namespace

The agent's capability surface **is** the kernel namespace (`client`, `ops`,
`np`/`da`, and `viewer` where there is one), so a user adds capability by simply *putting objects in scope*.
Two paths feed it: `*.py` files in a user kernel dir, and `biopb_mcp.namespace` entry
points for published plugin packages. Either way a plugin is loaded as a **module and
bound under one name** — its file stem or entry-point name (#664) — so its helpers and
imports stay off the namespace, and the **reserved-name guard** is one check per
plugin. Both paths are **fail-open per unit** (one bad plugin is skipped without
aborting the bootstrap). Plugin modules are registered for by-value pickling, so their
functions still run on a dask worker that cannot import them. The agent discovers
plugins with `dir()`/`inspect_object`, not from a "generated enumeration", so code and
doc cannot drift.

---

## Security model

**The kernel is a real IPython kernel with imports allowed — `execute_code` is
arbitrary code execution by design.** Everything the session exposes is therefore an RCE
threat, and it is protected in two places:

- **Its own listener** binds loopback only and enforces an `Origin`/`Host`
  allowlist, so a malicious page in the user's browser is rejected before it
  reaches the kernel. There is **no loopback token** — the `Origin` check already
  blocks the browser-attacker threat.
- **The control's proxy**, for anything reached through it. The `/session/<id>/*`
  hop strips the guard above, so the control's web-origin check is the child's
  **only** authentication there — and `/mcp` itself is deliberately never proxied,
  reachable only by the shim that owns the child. Both are enforced control-side:
  [`../biopb-control/ARCHITECTURE.md`](../biopb-control/ARCHITECTURE.md).
