# The biopb Project — Architecture Overview

A reference for readers new to biopb. It describes the whole project across the
`biopb` monorepo and the `biopb-server` repo, the reasoning behind the major
design decisions, and a few implementation details that are unusual enough to
surprise a newcomer.

---

## Scope

**biopb is an open protocol and toolchain for bioimage analysis** — moving large
multidimensional microscopy images around and running analysis algorithms
(segmentation, detection, restoration, …) on them, over the network, in a way
that is language-agnostic and that scales to data far larger than a client's
RAM.

The project deliberately separates two concerns and gives each its own protocol:

- **A data plane** — how (large) image tensors are stored, catalogued, and
  streamed. Built on Apache Arrow Flight.
- **A compute plane** — how analysis algorithms are invoked. Built on gRPC.

On top of those, it provides a **human/agent-facing client**: a napari plugin
that browses the data plane, and an MCP server that lets an AI agent drive a
live napari session.

### The repositories

The `biopb` repo is a **monorepo**. `biopb-mcp` was a separate repository and has
been folded in as the `biopb-mcp/` subdirectory; only `biopb-server` remains
external.

| Repo / component | Role |
|------|------|
| **`biopb`** (monorepo) | The protocol itself (`proto/`), plus reference servers and the clients, each a top-level subdir: the **tensor server** (`biopb-tensor-server/`, the data plane), the **image runtime** (`biopb-image-runtime/`, the base for compute-plane servers), the **agent client** (`biopb-mcp/`, the napari plugin + MCP server), the **control plane** (`biopb-control/`, the single web origin that supervises the data plane and serves the browser UI), and the **browser front end** (`web/`, one Vite + React SPA — dataviewer, admin, dashboard, observe — served by the control). Polyglot: protobuf/Flight stubs are generated for Python, Java, and JS/TS. The Python packages form a **uv workspace** ([`tool.uv.workspace`] in the root `pyproject.toml`); `web/` is a separate **pnpm workspace**. Versioning is **three lines, three tags**: the **SDK** (`biopb` Python→PyPI + Java→Maven Central, and `biopb-image-base` Docker) is tagged `v*`; the **tensor-server Docker image** has its own line tagged `server-v*`; and the **product bundle** (tensor-server wheel, mcp, control, web bundle → the GitHub release) is tagged `release-v*`. Docker images ship on `v*`/`server-v*` (their own CI workflows), not from `release.yaml`. See `docs/release-model.md`. |
| **`biopb-server`** | Concrete algorithm servers implementing the compute plane: `cellpose`, `cellpose-sam`, `lacss`, `samcell`, `ucell`. Each is a thin model wrapper on the shared image-runtime base, shipped as a container. |

The intended deployment is **personal or small-lab** use, on localhost or a
trusted intranet. Hardening for untrusted networks (TLS, authn/z, k8s) is
expected to be handled by a separately-documented reverse proxy in front of the
services, so the services themselves stay simple.

**Two deployment modes** (the deliberately-small security surface). `biopb
control start` runs **local mode** by default: every listener (control 8813,
tensor HTTP sidecar 8814, flight gRPC 8815) binds loopback — the single-machine
90% case. Local mode is tokenless by default (no unlock step), but a token is
**optional**: pass `--token` / `BIOPB_TENSOR_TOKEN` and it is enforced across the
loopback listeners too (the browser then gates behind the unlock page, exactly as
in remote). A local token is a supported mode — the control hands its credential
to local clients on the filesystem (see the credential handoff below); one
residual UI gap is noted there. `biopb control start
--remote` runs **remote mode**: the control's browser UI *and* the flight server
bind publicly behind a **required** token (supplied via `--token` /
`BIOPB_TENSOR_TOKEN`, else generated and printed), the sidecar stays on loopback
(the control proxies it), and the browser UI gates itself behind an unlock page
(driven by the control's public `GET /health` → `auth_required`). Token
enforcement is thus **independent** of the network mode; what `--remote` fixes is
the *bind address*. The one invariant is **fail-closed** — a public listener is
never left unauthenticated: `--remote` refuses to run without a token, and local
mode refuses to start if the config binds the flight server publicly *and* no
token is supplied, so "public + unauthenticated" is unrepresentable. The one
policy lives in the stdlib-only `biopb._web_auth` predicates that the control and
the sidecar both bind to (so they cannot drift); there is no separate "dev-mode"
token bypass.

**The local credential handoff** (biopb/biopb#470). Credential distribution is off
the HTTP API and on the filesystem, the standard local-daemon pattern (Jupyter's
runtime JSON, Docker): the control writes the resolved data-plane token to an
owner-only file in the user's state dir (`state/biopb/tensor-server.token`) when it
starts serving, and removes it on a clean stop. `_control_client` and `_connection`
read it there (via `biopb._credentials`), with `BIOPB_TENSOR_TOKEN` kept as an
explicit override. The endpoint never returns the token, so a loopback-reachable,
unauthenticated control is not a token oracle. The boundary is filesystem
permissions: `0600` on POSIX, and on Windows a DACL granting only the current user
with inheritance disabled (the faithful `0600` analogue — `os.chmod` can't express
owner-only there). This defends against *other uids* on a shared host, not a
same-uid process (neither does an env var, which additionally leaks via
`/proc/<pid>/environ`, `ps e`, and every inherited child).

With the handoff in place, biopb-mcp authenticates to a token-gated local plane on
its own (no env var needed), and `POST /api/data_plane/ensure` is gated like every
other `/api/*` route (the client carries the token; the old unauthenticated-state-
change exemption, #424 item 2, is gone).

One residual, tracked in #470 and separately fixable: **the admin UI's server-side
file chooser disappears** on a token-gated local plane (`/api/admin/browse` 404s)
because the sidecar computes `local: token is None`, so a loopback plane behind a
token looks remote to it. It should derive `local` from the bind instead; it fails
closed in the meantime.

---

## Environment

Two package managers, because the repo is polyglot.

**Python — one `uv` workspace, one shared `.venv`.** The root `pyproject.toml`
declares a [`tool.uv.workspace`] with four members (`biopb-image-runtime`,
`biopb-tensor-server`, `biopb-mcp`, `biopb-control`); the root `biopb` package is
the fifth. All resolve **in-tree** (`[tool.uv.sources]` … `workspace = true`),
never from PyPI, so a checkout builds against sibling source, not a published
release. Set the whole thing up — and restore it after adding a dependency — with
a single **all-packages** sync from the repo root:

```sh
uv sync --all-packages --all-extras
```

Run everything (tests, CLIs, scripts) through that root `.venv`. **Do not**
`uv sync --package <one>` against the shared venv — it *prunes* the venv down to
that one package's deps; always sync `--all-packages`. Python is pinned to
`>=3.10,<3.13`.

**Browser front end — a separate `pnpm` workspace** under `web/` (`@biopb/web` +
`@biopb/tensor-flight-client`): `pnpm -C web install`, then `pnpm -C web build` /
`test` / `lint` / `dev`. It is *not* part of the uv workspace; the two toolchains
are independent.

**Protobuf / Flight stubs are generated, not committed.** `buf generate` (config
in `buf.gen.yaml`, protos under `proto/`) writes the Python stubs into
`src/main/python/` and the Java stubs under `target/`. Regenerating at build time
keeps them from drifting in the tree — but a source checkout needs `buf` on PATH
(end users installing from release wheels do not; the wheels ship the generated
stubs).

**Testing.** `pytest` per package — the data plane in `biopb-tensor-server/tests/`,
the client in `src/test/python/`, and likewise `biopb-mcp` / `biopb-image-runtime`
/ `biopb-control`. Java tests run under `mvn -B test`, web tests under
`pnpm -C web test` (vitest). `src/test/python/README.md` has the map.

**Lint & format run on commit, not by hand.** `pre-commit` drives `ruff check` +
`ruff format` (v0.15.15) plus the end-of-file / trailing-whitespace / YAML /
napari-plugin checks. The **root `[tool.ruff]` is the single source of truth** for
all four Python packages (the per-package ruff/black blocks were removed), so
don't run ruff as a separate step — committing formats and autofixes for you.

---

## Architecture rationales

### Why split the data plane from the compute plane

Bioimage analysis has two very different cost structures: **bulk data movement**
(gigapixel images, multi-channel Z/T stacks) and **compute** (a GPU model run).
Coupling them — e.g. shipping pixels inside every RPC request/response — forces
every algorithm server to also be a high-throughput data mover, and forces the
client to hold whole images in memory to relay them.

biopb separates the two:

- The **tensor (data) plane** owns "where the pixels live and how they move."
- The **image (compute) plane** owns "what algorithm runs."

This lets each scale independently, lets multiple algorithm servers share one
data store, and — crucially — lets an algorithm server **pull its input pixels
directly from the data plane** instead of receiving them through the client (see
the lazy-input framework below).

### The tensor plane: Arrow Flight, a catalog, and lazy dask

The data plane is an **Arrow Flight** server (`biopb-tensor-server`). Flight,
rather than plain gRPC, because it is purpose-built for high-throughput,
columnar, near-zero-copy bulk transfer — exactly the profile of chunked image
tensors.

Key design choices:

- **Format-agnostic ingestion at the server.** Pluggable *adapters*
  (`tiff`, `zarr`, `ome-zarr`, `bioio`) read whatever microscopy format a
  lab has; a *discovery* pass plus a directory *watcher* register files (local
  paths or remote URLs) as **sources** without anyone hand-importing them.
  Clients never deal with proprietary formats — they always see uniform tensors.
- **A queryable catalog.** Sources and their metadata are tracked in a metadata
  database; clients discover data with a server-side **DuckDB SQL** query
  (complete, not truncated) or a capped `list_sources()`.
- **Chunked, lazy access.** The client (`TensorFlightClient`) exposes each
  source as a **thread-safe, picklable `dask.array`**. Pixels are fetched chunk
  by chunk on demand, so a client can compute over an image far larger than its
  RAM and only materialize the final result. Writing results back
  (`upload_array`) computes and uploads chunk by chunk symmetrically.

The net effect: **the data plane is a complete I/O layer.** "Read any format,
cache it, and hand me a lazy array" is a solved problem the rest of the system
builds on — it is not something clients or tools should re-implement.

### The compute plane: stateless algorithm servers with an eager/lazy duality

The compute plane is a gRPC contract (`proto/biopb/image`) with two services:

- **`ProcessImage`** — `Run`, `RunStream`, `GetOpNames`. General image→image
  operations (segmentation, denoising, …), where a server may expose several
  named *ops*.
- **`ObjectDetection`** — `RunDetection`, `RunDetectionStream`,
  `RunDetectionOnGrid`, `RunModelAdaptation`, `GetOpNames`. Detection/instance
  outputs (ROIs, labels).

The pivotal design point is that every request/response can carry image data in
one of **two modes** (`return_lazy_or_eager` in the image runtime):

- **Eager** — pixels are embedded inline in the message. Simple; fine for small
  images.
- **Lazy** — the message carries a **tensor source reference** instead of
  pixels. The algorithm server pulls the input straight from the tensor server,
  and writes its result back as a *new* source, returning that source id.

**The lazy-input framework is what makes large-data analysis practical.** With
it, ops compose on big images without the client ever holding a full array:
`segment(raw_source_id) -> labels_source_id -> measure(labels_source_id) -> …`,
each step streaming through the data plane. The algorithm server gets an
embedded Arrow Flight cache (a "lazy data side channel") so it can serve its
results back into the plane.

Algorithm servers are otherwise **stateless and uniform**: `biopb-server`
backends subclass a shared `BiopbServicerBase` from the image runtime and only
provide the model-specific inference, so adding a new algorithm is "wrap a model
+ point it at the protocol," not "build a server."

### The client: "trust the agent," with tools only where they help

`biopb-mcp` is built on a specific bet: **bench scientists' analysis tasks are
open-ended, so an AI agent with a general-purpose compute environment beats a
fixed set of GUI buttons.**

Rather than encode every workflow as a widget, the MCP server gives an AI agent:

- a **live napari viewer** (the scientist watches results appear),
- a **Python kernel** with the data plane and algorithms pre-wired into its
  namespace (`viewer`, `np`, `da`, `client`, `ops`), and
- a **small set of tools** — screenshot, run code, inspect, and kernel control.

The division of labor: image results go to the viewer; quantitative results go
to the agent's chat; the user decides what to save (via napari). Purpose-built
tools are added **only where the agent cannot do the job in plain Python** —
the canonical example being trained-model segmentation, which LLMs cannot do
accurately unaided. Classical operations (filtering, regionprops, blob
detection, spatial filtering) are left to the agent, because wrapping them would only
constrain it. The `ProcessImage` segmentation/processing **widgets in the napari
plugin are demos** of how to stand up an algorithm server — not the primary
interface.

The intended extension story is that different labs add their own
domain-specific algorithm servers (compute plane) and tools (agent side) for
their own problems. On the agent side the low-friction path is a **user plugin**
(`biopb/biopb-mcp#92`): drop a `*.py` file in `~/.config/biopb/kernel/` (or ship a
`biopb_mcp.namespace` entry-point package) and its callables load straight into
the kernel namespace, visible to the agent — capability is added by *putting
objects in scope*, not by extending a protocol. biopb-mcp ships one such built-in
as the reference example (`biopb_mcp/plugins/rolling_ball.py`, a fast ImageJ port
of rolling-ball background subtraction); the installer seeds a copy into
`~/.config/biopb/kernel/` so it is visible and editable there. See
`biopb-mcp/ARCHITECTURE.md`.

### Polyglot by construction

There is one source of truth — the `.proto` files — and language stubs are
generated with **buf**. This is why a Java (`pom.xml`), a JS/TS (pnpm), and a
Python toolchain all live in the `biopb` repo: the protocol is meant to be
consumed from analysis ecosystems in any of those languages (napari/Python,
ImageJ/Java, web/TS).

### The agentic coupling (biopb-mcp's main user-facing design)

`biopb-mcp` is where a scientist actually meets the system, and its defining
idea is *how* it couples an external AI agent to a live, shared analysis session
— rather than exposing a narrow remote API. This is the component most worth
understanding in detail.

```
        AI agent (Claude)                              Scientist
              │  MCP tools + resources                      ▲
              │  (stdio shim → child's dynamic /mcp)        │ watches / prompts
              ▼                                             │
   ┌────────────────────────┐                               │
   │  MCP server process    │                               │
   │  (FastMCP, _server.py) │                               │
   └───────────┬────────────┘                               │
               │ dispatch (serialized by one RLock)         │
               ▼                                            │
   ┌────────────────────────────────────────────────┐       │
   │  child Jupyter kernel (separate process)       │       │
   │    opens ── napari viewer window ──────────────┼───────┘
   │    namespace:  viewer, np, da, client, ops     │
   │        │              │               │        │
   └────────┼──────────────┼───────────────┼────────┘
            ▼              ▼               ▼
         viewer        data plane     compute plane
                     client / dask   ops (ProcessImage)
                          │               │
                          ▼               ▼
                 biopb-tensor-server   biopb-server algorithm servers
```

Key properties of the coupling:

- **The coupling surface is a shared Python namespace, not a fixed API.** The
  agent is not handed a menu of "segment / threshold / measure" endpoints; it is
  handed the live objects — `viewer`, `client` (data plane), `ops` (compute
  plane), plus `skimage`/`da` — and writes arbitrary Python against them through the
  `execute_code` tool. That is what keeps the system open-ended: any analysis
  expressible in Python over those handles is reachable, and new capability is
  added by putting new objects into the namespace, not by extending a protocol.

- **It is a *shared* session, not a headless one.** The viewer the agent mutates
  is the same window the scientist is watching. Agent and human operate on one
  live session: the agent adds a result layer, the scientist sees it appear,
  tweaks it by hand, and the agent reads it back. The product is collaboration
  on a shared canvas, not a batch job that returns a file.

- **The agent is coupled to all three planes at once.** Through the one
  namespace it can pull lazy data (`client.get_tensor` → a dask array), run
  trained-model algorithms (`ops[...]`), and present/inspect results
  (`viewer`). The lazy-input framework carries through: the agent can chain
  `ops` on source ids so large-data work never round-trips pixels through the
  kernel.

- **A perceive → act → verify loop.** The agent acts by running code that
  mutates the viewer, then *perceives* the effect with `take_screenshot` (and
  examines state with `inspect_object` / `server_status`). Seeing the rendered
  image — not just array values — is how it confirms a segmentation or a display
  change actually looks right, and how it shows the user.

- **Resources orient the agent.** Four `napari://` resources (`guide`, `viewer`,
  `tensor`, `annotations`) are served as living documentation the agent reads on
  demand, so it learns the namespace, the data API, and idiomatic patterns
  without trial and error.

- **Division of labor with the human.** Image results land in the viewer; scalar
  and tabular results go to the agent's chat; the scientist decides what to keep
  and saves it through napari. The agent orchestrates; the human stays in
  control of the canvas and of what becomes a durable artifact.

- **The session is ephemeral and shim-owned; the planes are durable and
  control-supervised.** Each MCP client's stdio shim spawns and owns *its own*
  session child (kernel + viewer) on a dynamic port and reaps it when the client
  disconnects; the child inherits the shim's live environment, so the viewer
  always lands on the user's real display. The data/compute planes the session
  uses are *not* part of it — a lean **control plane** supervises them as durable
  subprocesses and is the single web origin. It **serves the whole browser UI
  itself** — one Vite SPA (`web/`) with the dashboard at `/`, the dataviewer at
  `/viewer`, and each session's observe page at `/session/<id>/observe` — while
  proxying the data plane (`/data_plane/*`) and each session's API
  (`/session/<id>/api/*`), so N dynamic-port sessions need no N bookmarks. The
  agentless `biopb mcp view` is the exception: a self-contained foreground viewer
  that stays off the control entirely. See
  `biopb-mcp/ARCHITECTURE.md`.

This coupling — *arbitrary agent code against a live, shared session* — is
exactly why the kernel-isolation choices in §3 exist: the session must survive
the agent doing something wrong, so the kernel is a separate, interruptible,
restartable process.

---

## Implementation notes (the notable choices)

These are decisions a newcomer would not guess and that are worth knowing before
editing the code.

- **The agent's code runs in a *separate* child Jupyter kernel**, not in the MCP
  server process and not on napari's Qt event loop. The MCP server process
  spawns the kernel (via `jupyter_client`), and the **napari viewer window lives
  inside that kernel process** (`%gui qt` integration). Rationale: agent code is
  arbitrary and may hang or crash; isolating it in a child kernel means a runaway
  execution can be interrupted (`SIGINT`) or hard-restarted (process-group
  `SIGKILL` + respawn) without killing the MCP server. A single `RLock`
  serializes access to the one kernel. **That MCP server process is itself an
  ephemeral, shim-owned child:** the client's stdio shim spawns it per connection
  on a dynamic port (inheriting the shim's live environment — the #98 display fix)
  and reaps it — plus the kernel grandchild — on disconnect, so the whole
  session tree is client-scoped, not a shared daemon (see the de-daemonization /
  control-plane migration doc).

- **The kernel is bootstrapped via `IPKernelApp.exec_lines`.** A startup line
  injected at launch runs `biopb_mcp.mcp._bootstrap`, which enables Qt, opens
  the viewer, wires up dask and the data connection, and populates the namespace
  — all *before* the kernel serves any tool call. Bootstrap failure is detected
  by a health probe checking `'viewer' in dir()`.

- **The data-access service is deliberately GUI-free.** `TensorConnection`
  (`biopb-mcp`) imports neither Qt nor napari, so the same object is shared by
  the napari widget and the headless kernel and is unit-testable without a
  display.

- **A cache-file mmap fast path** for localhost clients of the tensor server
  (`chunk_locate` Flight action, `biopb/biopb#9`). The server's file cache
  already holds every decoded chunk as an Arrow IPC message in a segment file,
  so instead of re-sending those bytes through the loopback `do_get` socket the
  client asks for the chunk's on-disk byte range (`locate_entry`), `mmap`s the
  segment, reads just that message, and hands out a **zero-copy view** onto the
  mapping (Option C, `biopb/biopb#571`): the client closes its own
  `MemoryMappedFile` handle at once, but Arrow refcounts the mapping so the
  returned array keeps it alive (`ndarray → pyarrow.Buffer → MemoryMappedFile`),
  and untouched chunk pages are never faulted in (a partial read is nearly free).
  This beats the socket because the bytes are already warm in the page cache
  (the server wrote them for caching anyway) and it skips the loopback gRPC
  overhead — and it now also skips the whole-chunk copy the socket cannot
  (`.copy()` there fell off glibc's 32 MiB `mmap`-threshold cliff at a 64 MB
  chunk; both paths now return a **read-only** view, the uniform mutability
  contract). The view's safety rests on the server never truncating a mapped
  segment inode: segment ids are strictly monotonic and eviction only `unlink`s,
  so the one truncating `"wb"` open always targets a fresh path — an NFS
  `cache_dir` would break that and wants an explicit gate. The view's *cost* is a
  disk-leak: while a client holds it the server can't reclaim that segment's
  blocks even after eviction `unlink`s it (the inode survives to last close), so
  a client pinning many segments keeps the server's `cache_dir` above budget.
  The client bounds that leak with **pinned-segment accounting** (still
  `biopb/biopb#571`): it tracks the on-disk size of the distinct segments it
  keeps mapped (refcounted by inode, released by a `weakref.finalize` on the
  backing Arrow buffer) and, once over `BIOPB_CACHEFILE_PIN_LIMIT_BYTES`
  (default 16 GiB), copies the chunk out and drops the mapping instead of pinning
  another segment — still off the warm mmap, no `do_get`. The hot path stays
  cheap: the gate is a lock-free int compare, the segment size reuses the
  `stat` the fast path already does, and only the view branch pays a lock + one
  finalizer. Gated to **POSIX localhost** (Windows file-mmap blocks the server's
  segment `unlink` — `biopb/biopb#5` — so Windows clients use `do_get`); the
  client honors `BIOPB_CACHEFILE_TRANSFER_DISABLED=1` to force the socket, and
  falls back to `do_get` whenever a chunk can't be located (memory backend, old
  server, evicted segment). Replaces the old `/dev/shm` `shm_transfer` path,
  which was *slower* than the socket because it allocated a fresh POSIX segment
  per chunk. Byte offsets are captured **at write time** from the sink cursor
  (`biopb/biopb#541`) — the first append in a segment also flushes the writer's
  buffered schema message, so that one entry's start is recovered by reading the
  schema message's length off the file. Deriving them lazily instead made every
  cache *miss* re-walk the whole active segment (O(entry count), ~5 ms at 145 MB
  with 0.87 MB chunks and worse for small ones), a cost paid only by the fast
  path it was meant to accelerate — and that walk was the only place the read
  path took the cache's write lock, so a stalled write blocked locates. Between
  the write path and the boot `.idx` sidecar restore, an index entry now carries
  its range from birth, so `locate_entry` derives nothing and the walk is gone
  rather than kept as a fallback — an entry that somehow lacks a range simply
  falls back to `do_get`, the designed floor of this path.

- **Localhost read amplification — chunk size is conflated with access
  granularity.** The server sizes chunks to a fixed transfer cap
  (`MAX_ARROW_BATCH_BYTES = 64 MB` in `chunk.py`, splitting non-spatial axes first
  and keeping the Y-X plane whole), and that same grid is the *access* unit the
  client reads. A consumer reading a small sub-region — the napari viewer
  scrubbing one Z plane (~2.75 MB) out of a ~63 MB chunk — transfers the whole
  chunk, a ~23× amplification. The capability to read arbitrary sub-bounds cheaply
  already exists (`adapter.get_data(bounds)` decodes only the requested planes);
  only the chunk grid / `chunk_id` forces whole-chunk transfers. Decoupling the
  read grid from the 64 MB transfer cap (client-selectable granularity) is the
  structural fix — `biopb/biopb#8`.

- **The localhost client chunk cache is off by default, and locality-sensitive
  under distributed dask.** On localhost the per-process client cache
  (`biopb.tensor.client`, gated by `_resolve_cache_bytes` / `BIOPB_CACHE_LOCAL`)
  is disabled: the server already caches, and under `biopb-mcp`'s default
  *multi-process distributed* dask a per-process cache is replicated per worker.
  `biopb-mcp` bounds it with a cluster-wide `dask_cache_budget` (split
  `budget // n_workers` by a worker-init plugin; localhost still resolves to 0
  unless `BIOPB_CACHE_LOCAL=1` is set). **Caveat measured:**
  even with the cache on, the viewer's *serial* plane reads scatter across workers
  — dask's locality scheduler keys on tracked task *dependencies*, not the opaque
  per-worker cache side-effect, so repeated reads of the same chunk round-robin
  onto different workers (≈25% hit, N× redundant copies). The clean viewer fix is
  to compute its slices on a single-process scheduler (one shared cache) —
  `biopb/biopb-mcp#8`; the per-worker cache helps mainly when paired with that or
  with deterministic chunk→home-worker sharding.

- **The standard Flight verbs are extended with custom `do_action` types** on
  the tensor server: `health`, `create_source`, `upload_status`, `chunk_locate`
  — alongside the normal `do_get`/`do_put`/`get_flight_info`/`list_flights`.
  Startup is **progressive** (`biopb/biopb#212`): the server reaches `SERVING`
  immediately and runs/streams its data-folder scan in the background, so
  `SERVING` no longer implies a complete catalog — `health` carries
  `full_scan_in_progress` / `last_full_scan_finished_at` as the freshness signal
  (see `biopb-tensor-server/ARCHITECTURE.md` and `biopb-tensor-server/docs/progressive-discovery.md`).

- **A tensor is identified by its `array_id` alone** — the policy every server,
  SDK (Python/Java/TS), and the CLI must follow. The authoritative spec lives in
  `proto/biopb/tensor/descriptor.proto` (top-of-file comment block); the short
  version: `array_id` is **globally unique** and is the primary key, constructed
  as `source_id` (single-tensor source) or `source_id/field` (multi-tensor). It
  is the same value as `TensorReadOption.tensor_id`. `source_id` is globally
  unique **and slash-free**, so it is a *derived projection* of `array_id` —
  recoverable as `array_id.split("/", 1)[0]` — carried on the wire only as a
  routing convenience; `array_id` is authoritative. The `field` part may itself
  contain `/` (e.g. HCS `well/field`); the source boundary is always the **first**
  `/`, which is why `source_id` must be slash-free. The **same** `array_id` is
  used identically in the wire descriptor, the request `tensor_id`, the
  `chunk_id` (a length-prefixed binary field, so an embedded `/` is safe), and
  every cache key — there is **no** separate bare-vs-qualified form. Collapsing
  that former dual-form to one identifier is what removes the translation seam
  behind the cross-source cache collisions (`biopb/biopb#45`) and the upload
  path's `array_id`/`source_id` conflation. The server resolves a request by
  stripping the `source_id` prefix; for back-compat it also accepts a bare field,
  the `source_id`, or an empty `tensor_id` (→ the default/first tensor,
  `biopb/biopb#44`). *Implemented across the stack: `source_id` slash-free
  validation at registration (`#50`); the server emits the qualified `array_id`
  from `list_flights`/`GetFlightInfo`/the chunk descriptor and reduces a request
  `tensor_id` to the within-source field at one chokepoint
  (`_field_within_source`), with `get_tensor_adapter` tolerant of either form
  (`#51`); the HTTP sidecar's dim-label lookup and the Java/Python
  `SerializedTensor` endpoint-fetch derive `source_id` as the slash-free prefix
  and tolerate both forms (`#52`). Single-tensor `array_id == source_id` is the
  default construction (no `"0"` sentinel), and the CLI's `source_id/tensor_id`
  parsing was already conformant.*

- **OME/microscopy metadata travels as JSON, not protobuf.** It rides on
  `metadata_json` on a tensor descriptor and `ome_metadata` dicts — there is no
  `biopb.ome` protobuf package (an early, comprehensive OME-in-protobuf blueprint
  under `proto/biopb/ome/` was never adopted and was removed; see git history).

- **Generated protobuf/Flight stubs are not committed** — buf regenerates them
  at build time, which keeps the polyglot stubs from drifting in the tree.

- **There is a no-AVX/SSE4.2 build path** for the image runtime: on older CPUs
  `pyarrow` (and therefore the lazy Arrow-Flight side channel) is unavailable,
  so those builds fall back to eager-only operation.

---

## Where to look first

- **Protocol:** `biopb/proto/biopb/{image,tensor}/`.
- **Data plane:** `biopb/biopb-tensor-server/biopb_tensor_server/` —
  `server.py`, `adapters/`, `discovery.py`, the metadata DB.
- **Compute base:** `biopb/biopb-image-runtime/src/biopb_image_base/` —
  `BiopbServicerBase`, `return_lazy_or_eager`, the embedded cache.
- **An example algorithm server:** `biopb-server/cellpose/cellpose_server.py`
  (the only remaining separate repo).
- **Client / agent:** `biopb/biopb-mcp/src/biopb_mcp/` — `_connection.py` (data
  service), `tensor_browser/`, and `mcp/` (`_kernel.py`, `_bootstrap.py`,
  `_server.py`). See `docs/release-model.md` for how the product is built and
  released.
- **Control plane / web origin:** `biopb/biopb-control/src/biopb_control/` —
  `_control.py` (the ASGI app: serves the `web/` SPA + proxies the data plane and
  sessions), `_supervisor.py` (data-plane subprocess lifecycle).
- **Browser front end:** `biopb/web/` — one Vite + React SPA served by the
  control. `packages/app/src/` (`main.tsx` routes; `pages/{HomePage,AdminPage,
  DashboardPage,ObservePage}.tsx`) and `packages/tensor-flight-client/` (the TS
  data-plane SDK). See `web/README.md` and `web/ARCHITECTURE.md`.
