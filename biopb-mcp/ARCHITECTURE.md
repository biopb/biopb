# biopb-mcp Architecture

## Overview

`biopb-mcp` connects [napari](https://napari.org) and AI agents to biopb servers.
It has two faces over one shared data layer:

1. **napari plugin** — a `Tensor Browser` widget that browses/loads images from
   the tensor server, plus two **demo widgets** (`Object Detection`, `Image
   Processing`) that exercise the `biopb.image` gRPC protocol. The demo widgets
   exist to test algorithm servers; they are **not** the primary interface.
2. **MCP server** — exposes a live napari viewer to an AI agent. Thesis: *"agent
   first; provide tools only if they help."* The agent drives napari through a real
   Python kernel; image results go to the viewer, other results to the agent's chat.

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
   │   DaskClusterHost    — owns the cluster          │
   │   observe UI         — job history + cancel      │
   └───────┬──────────────────────────────┬───────────┘
           │ spawns; jupyter ZMQ          │ spawns
           ▼                              ▼
   ┌────────────────────────────┐   ┌───────────────────────┐
   │ Jupyter kernel             │   │ dask LocalCluster     │
   │   napari viewer window (Qt)│──►│   scheduler + workers │
   │   agent namespace          │   └───────────────────────┘
   │   job runner, viewer proxy │    attaches by injected
   └──────────────┬─────────────┘    scheduler address
                  │ Flight / gRPC
                  ▼
      data plane · algorithm servers · control plane
      (outside this package — see ../development.md)
```

Two ownership facts deliberately do not follow the spawn chain: the **dask cluster
hangs off the session child, not the kernel**, so it survives kernel restarts; and
the **planes at the bottom are never started here** — the session is a pure client
of them, and only *registers* itself with the control.

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

The session owns a **single child Jupyter kernel** hosting the napari viewer,
dask, and the tensor client. Agent code runs *in that kernel*, not on the MCP
thread or napari's Qt loop — so a runaway execution can be interrupted or
hard-restarted without killing the MCP server. A single `RLock` serializes access,
held only for *quick* snippets, never during long compute.

The kernel is **launched lazily, not at boot**, so a long-running server binds
cheaply and never pops a napari viewer until user requested it; kernel-dependent
tools return a structured not-ready status until then. **Closing the napari window
tears the kernel back down to idle**, and `start_kernel` rebuilds it.

Because napari viewer is launched within the jupyter kernel, agents' jobs are run
in a background thread, so the Qt loop stays free to respond to user and agent inputs
alike. The agent-facing `viewer` object is a **main-thread marshaling proxy** over
the real `napari.Viewer`, because an off-main napari mutation can segfault the kernel
(#100).

### dask cluster

The **mcp server** — not the kernel — owns the dask `LocalCluster`, so it
survives kernel restart/respawn/window-close with no cold worker re-spawn per
restart (the dominant restart cost on Windows). The kernel attaches via an injected
scheduler address; worker/memory changes therefore need a *session* restart, not
just a kernel restart. An idle reaper bounds the decoupling: once the cluster has
sat with **no kernel attached** past its TTL it is torn down, and the next kernel
launch re-spins it.

### Data connection

`TensorConnection` is a **GUI-independent** data-access service — it imports
neither Qt nor napari — so the `TensorBrowserWidget` and the headless MCP kernel
share one implementation. It owns the tensor Flight client and the source catalog.
It resolves no endpoint from config and persists nothing.

Its connect policy: the **control is the only source of a data plane** (#628). One
`ensure_data_plane` call both brings the plane up if it is down and returns the
*authoritative* gRPC endpoint. `$BIOPB_TENSOR_URL` is the one escape hatch, for
connecting to a data-server _not_ supervised by the control. A local TLS data server
is trusted from disk (its cert is already on this machine).

### Extending the kernel namespace

The agent's capability surface **is** the kernel namespace (e.g., `viewer`, `client`,
`ops`, `np`/`da`), so a user adds capability by simply *putting objects in scope*.
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
