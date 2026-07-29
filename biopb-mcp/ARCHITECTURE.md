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

The MCP server process **is http-only** (loopback streamable-http). `--transport
stdio` — still the default, so installer-seeded client configs keep working — no
longer serves MCP from the launcher; it runs the **shim**, which

1. **spawns its own ephemeral session child** (FastMCP/uvicorn + the kernel host)
   on a **dynamic OS-assigned port**,
2. **bridges** stdio JSON-RPC ↔ that child's `/mcp` until the client closes stdin,
   replaying the child's initialize result **verbatim** — including `instructions`,
   the field the generic `mcp-proxy` drops, which is why the bridge is vendored,
3. **reaps** the child and its kernel grandchild on the way out.

There is **no probe-and-reuse, no shared daemon, no fixed port**: each stdio client
spawns and owns its own session, so N clients get N independent sessions (N
viewers), by design. The child **inherits the shim's live environment** (so the
agent's viewer lands on the human's real display — the #98 fix), **registers with
the control** on startup, and is **reaped as a tree** (POSIX process group +
parent-death pipe; Windows Job Object, #403).

Native http skips the shim entirely and is preferred where the client supports it.

### The kernel

The session child owns a **single child Jupyter kernel** hosting the napari viewer,
dask, and the tensor client. Agent code runs *in that kernel*, not on the MCP
thread or napari's Qt loop — so a runaway execution can be interrupted or
hard-restarted without killing the MCP server. A single `RLock` serializes access,
held only for *quick* snippets, never during long compute.

The kernel is **launched lazily, not at boot**, so a long-running server binds
cheaply and never pops a viewer with nobody connected; kernel-dependent tools
return a structured not-ready status until then. **Closing the napari window tears
the kernel back down to idle**, and `start_kernel` rebuilds it.

Two consequences of running agent code off the main thread shape the kernel's
internals: jobs run in a background thread so the Qt loop stays free to service
screenshots and status mid-job, and the agent-facing `viewer` is a **main-thread
marshaling proxy** over the real `napari.Viewer`, because an off-main napari
mutation can segfault the kernel (#100).

### dask cluster

The **session child** — not the kernel — owns the dask `LocalCluster`, so it
survives kernel restart/respawn/window-close with no cold worker re-spawn per
restart (the dominant restart cost on Windows). The kernel attaches via an injected
scheduler address; worker/memory changes therefore need a *session* restart, not
just a kernel restart. An idle reaper bounds the decoupling: once the cluster has
sat with **no kernel attached** past its TTL it is torn down, and the next kernel
launch re-spins it.

### Data connection

`TensorConnection` is a **GUI-independent** data-access service — it imports
neither Qt nor napari — so the `TensorBrowserWidget` and the headless MCP kernel
share one implementation. It owns the tensor Flight client, the source catalog, and
URL/token resolution.

Its connect policy is **control-first**: it asks the control to ensure the data
plane (which brings the plane up if down and returns the *authoritative* gRPC
endpoint), and only when no control answers falls back to a direct connect on a
locally-resolved `(url, token)`. It is a **pure client** — it never starts a server
itself. Because `connect()` blocks on I/O it must be driven off the caller's main
thread.

Two policies worth knowing: a **local TLS plane is trusted from disk, never
pinned** (its cert is already on this machine; a remote plane still TOFU-pins), and
the catalog is **self-healing** — a catalog cached at connect can be partial,
because the server reports `SERVING` before it finishes enumerating scenes, so a
daemon thread re-lists on any source-count change.

### Extending the kernel namespace

The agent's capability surface **is** the kernel namespace (`viewer`, `client`,
`ops`, `np`/`da`), so a lab adds capability by *putting objects in scope*, not by
extending a protocol. Two paths feed it: `*.py` files in a user kernel dir, exec'd
with IPython `startup/` semantics, and `biopb_mcp.namespace` entry points for
published plugin packages. Both are **fail-open per unit** (one bad plugin is
skipped, never aborting the bootstrap) and pass a **reserved-name guard** so a
plugin cannot overwrite a load-bearing handle. There is deliberately **no generated
enumeration** — the agent discovers plugins with `dir()`/`inspect_object`, so code
and doc cannot drift.

---

## Security model

**The kernel is a real IPython kernel with imports allowed — `execute_code` is
arbitrary code execution by design.** Do not describe it as sandboxed. The system
assumes a **localhost / trusted-intranet** deployment; untrusted-network exposure
is expected to be fronted by a separately-documented reverse proxy.

Everything the session exposes is therefore an RCE surface, and it is protected in
two places:

- **Its own listener** binds loopback only and enforces an `Origin`/`Host`
  allowlist, so a malicious page in the user's browser is rejected before it
  reaches the kernel. There is **no loopback token** — the `Origin` check already
  blocks the browser-attacker threat.
- **The control's proxy**, for anything reached through it. The `/session/<id>/*`
  hop strips the guard above, so the control's web-origin check is the child's
  **only** authentication there — and `/mcp` itself is deliberately never proxied,
  reachable only by the shim that owns the child. Both are enforced control-side:
  [`../biopb-control/ARCHITECTURE.md`](../biopb-control/ARCHITECTURE.md).
