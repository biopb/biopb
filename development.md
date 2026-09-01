# The biopb Project — Architecture Overview

### Monorepo layout

`biopb` is a **monorepo**, with a sister repo biopb-server for specific algorithm
plane implementations. Each top-level subdir is one component:

| Path | Component |
|---|---|
| `proto/` | **The protocol** — `biopb.image` (compute plane) and `biopb.tensor` (data plane) `.proto` files. The single source of truth; stubs for Python, Java, and JS/TS are generated from it. |
| `src/` | **The core `biopb` SDK** — the Python package (`src/main/python/biopb`: the `tensor` Flight client, the `biopb` CLI, and the stdlib-only cross-process seams) and the Java client (`src/main/java`), plus their tests under `src/test/`. |
| `biopb-tensor-server/` | **The data plane** — the Arrow Flight server, its format adapters, catalog, cache, and HTTP sidecar. |
| `biopb-image-runtime/` | **The compute-plane base** — `BiopbServicerBase` and the base Docker image that algorithm servers derive from. |
| `biopb-mcp/` | **The agent client** — the napari plugin (Tensor Browser + demo widgets) and the MCP server that drives a live napari session. |
| `biopb-control/` | **The control plane** — the single web origin; supervises the data plane and serves the browser UI. |
| `web/` | **The browser front end** — one Vite + React SPA (dataviewer, admin, dashboard, observe), served by the control. |
| `install/` | Installers (`install.sh` / `install.ps1`) and the GUI launcher. |
| `docs/`, `docs-api/` | Project-wide design docs, and the generated API reference site. |

**Versioning is two lines, two tags.** The **SDK** (`biopb` Python→PyPI +
Java→Maven Central, and `biopb-image-base` Docker) is tagged `v*`; the **product**
(the `biopb-tensor-server` wheel **and** its Docker image, plus mcp, control, and
the web bundle → the GitHub release) is tagged `release-v*`. Docker images are
built by their own CI workflows (image-base on `v*`, tensor-server on
`release-v*`), not from `release.yaml`. See
[`docs/release-model.md`](docs/release-model.md).

---

## Runtime shape

A biopb deployment is a **tree rooted at a durable control plane**:

```
   control plane   (durable ROOT — lean: supervise + route + serve the web UI)
        ├── supervises ─► data plane      (tensor Flight server + HTTP sidecar)
        ├── supervises ─► algorithm plane (algorithm servers)          [pending]
        └── observes   ◄─ MCP sessions    (ephemeral, SHIM-owned; self-register)
                            env inherited from the shim
                            USE the planes; never START them
```

| Component | Lifetime | Owned by | Role |
|---|---|---|---|
| Control plane | durable (root) | OS service / `biopb` launcher | supervise, route, single-origin web front, session registry, auth |
| Data plane | durable | control (subprocess) | pixels, cache, remote-data proxy |
| Algorithm plane | durable | control (subprocess) | compute ops *(pending)* |
| MCP session | ephemeral | the **shim** | kernel + dask + viewer; env-inherited; registers with control |
| Shim | per client connection | the MCP client | stdio↔http bridge; spawns & reaps its session child |

---

## Environment

### Python

Managed with `uv`. One shared `.venv` workspace for all packages.
(see root `pyproject.toml`) Set the whole thing up — and restore it after adding
a dependency — with a single **all-packages** sync from the repo root:

```sh
uv sync --all-packages --all-extras
```

**Do not** `uv sync --package <one>` against the shared venv — it *prunes* the
venv down to that one package's deps.

### Browser front end

A separate `pnpm` workspace under `web/` (`@biopb/web` +
`@biopb/tensor-flight-client`): `pnpm -C web install`, then `pnpm -C web build` /
`test` / `lint` / `dev`. It is *not* part of the uv workspace; the two toolchains
are independent.

### Protobuf / Flight stubs

Generated, not committed. `buf generate` (config in `buf.gen.yaml`, protos under
`proto/`) writes the Python stubs into `src/main/python/` and the Java stubs under
`target/`.

A source checkout needs `buf` on PATH (end users installing from release wheels do
not; the wheels ship the generated stubs).

### Testing

**Python:** `pytest` per package — the data plane in `biopb-tensor-server/tests/`,
the client in `src/test/python/`, and likewise `biopb-mcp` / `biopb-image-runtime`
/ `biopb-control`.

**Java** tests run under `mvn -B test`

**web** tests under `pnpm -C web test` (vitest).

`src/test/python/README.md` has the map.

### Lint

Lint & format run through `pre-commit`, which drives `ruff check` +
`ruff format` (v0.15.15). The **root `[tool.ruff]` is the single source of
truth** for all four Python packages (the per-package ruff/black blocks were
removed).

Java and TS have no lint policy currently.

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

### The tensor plane: a complete I/O layer

The data plane is an **Arrow Flight** server (`biopb-tensor-server`). Flight,
rather than plain gRPC, because it is purpose-built for high-throughput,
columnar, near-zero-copy bulk transfer — exactly the profile of chunked image
tensors. Three choices give it its shape:

- **Format-agnostic ingestion at the server.** Pluggable adapters read whatever
  microscopy format a lab has, and discovery + a directory watcher register files
  as sources without anyone hand-importing them. Clients never deal with
  proprietary formats — they always see uniform tensors.
- **A queryable catalog**, discovered with a server-side DuckDB SQL query.
- **Chunked, lazy access.** The client exposes each source as a **thread-safe,
  picklable `dask.array`**, fetched chunk by chunk on demand, so a client can
  compute over an image far larger than its RAM and materialize only the final
  result. Writing results back (`upload_array`) is symmetric.

The net effect: **"read any format, cache it, and hand me a lazy array" is a
solved problem the rest of the system builds on** — it is not something clients or
tools should re-implement. Adapters, cache, discovery, and the CLI launcher are in
[`biopb-tensor-server/ARCHITECTURE.md`](biopb-tensor-server/ARCHITECTURE.md); the
client-side localhost read path in
[`docs/localhost-fast-path.md`](docs/localhost-fast-path.md).

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

Algorithm servers are otherwise **stateless and uniform**: `biopb-server`
backends subclass a shared `BiopbServicerBase` from the image runtime and only
provide the model-specific inference, so adding a new algorithm is "wrap a model
+ point it at the protocol," not "build a server." See
[`biopb-image-runtime/README.md`](biopb-image-runtime/README.md).

### biopb-mcp

Agent owns a **python kernel** to run code with. The harness provides additional
tools only when they help

`biopb-mcp` is built on a specific hypothesis: scientific data analysis tasks are
open-ended, so an AI agent with a general-purpose compute environment beats a
fixed set of GUI buttons. Rather than encode every workflow as a widget, the MCP
server hands an agent a **live napari viewer**, a **Python kernel** with the data
plane and algorithms pre-wired into its namespace (`viewer`, `np`, `da`, `client`,
`ops`), and a **small set of tools** — screenshot, run code, inspect, kernel
control.

- **The coupling surface is a shared Python namespace, not a fixed API.** The
  agent is handed live objects and writes arbitrary Python against them. Any
  analysis expressible in Python over those handles is reachable, and new
  capability is added by *putting objects in scope* — not by extending a protocol.
  That is also the extension story: a lab drops a `*.py` file in
  `~/.config/biopb/kernel/` (or ships a `biopb_mcp.namespace` entry-point package)
  and it is loaded into the kernel namespace as a module named after the file, so
  the agent calls its functions through it (biopb/biopb-mcp#92, #664).
- **It is a *shared* session, not a headless one.** The viewer the agent mutates
  is the same window the scientist is watching: the agent adds a result layer, the
  scientist sees it appear, tweaks it by hand, and the agent reads it back. The
  agent acts by running code, then *perceives* the effect with `take_screenshot` —
  seeing the rendered image, not just array values, is how it confirms a
  segmentation actually looks right. The product is collaboration on a shared
  canvas, not a batch job that returns a file.
- **Division of labor.** Image results land in the viewer; scalar and tabular
  results go to the agent's chat; the scientist decides what becomes a durable
  artifact. Purpose-built tools are added **only where the agent cannot do the job
  in plain Python** — the canonical example being trained-model segmentation.
  Classical operations (filtering, regionprops, blob detection) are left to the
  agent, because wrapping them would only constrain it. The `ProcessImage`
  widgets in the napari plugin are **demos** of how to stand up an algorithm
  server, not the primary interface.

Because the agent runs arbitrary code against a live session, the session must
survive the agent doing something wrong — which is why the kernel is a separate,
interruptible, restartable child process, and why sessions are ephemeral and
shim-owned while the planes are durable and control-supervised. The session's own
process chain, its security model, and its component map are in
[`biopb-mcp/ARCHITECTURE.md`](biopb-mcp/ARCHITECTURE.md); the supervision of the
durable planes and the web origin in
[`biopb-control/ARCHITECTURE.md`](biopb-control/ARCHITECTURE.md).

---

## Where to look first

- **Protocol:** `proto/biopb/{image,tensor}/`.
- **Data plane:** `biopb-tensor-server/biopb_tensor_server/` — `serving/server.py`,
  `adapters/`, `core/discovery.py`, the metadata DB.
- **Compute base:** `biopb-image-runtime/src/biopb_image_base/` —
  `BiopbServicerBase`, `return_lazy_or_eager`, the embedded cache.
- **An example algorithm server:** `biopb-server/cellpose/cellpose_server.py`
  (the only remaining separate repo).
- **Client / agent:** `biopb-mcp/src/biopb_mcp/` — `_connection.py` (data
  service), `tensor_browser/`, and `mcp/` (`_kernel.py`, `_bootstrap.py`,
  `_server.py`).
- **Control plane / web origin:** `biopb-control/src/biopb_control/` —
  `_control.py` (the ASGI app: serves the `web/` SPA + proxies the data plane and
  sessions), `_supervisor.py` (data-plane subprocess lifecycle).
- **Browser front end:** `web/` — one Vite + React SPA served by the control.
  `packages/app/src/` (`main.tsx` routes; `pages/`) and
  `packages/tensor-flight-client/` (the TS data-plane SDK). See `web/README.md`
  and `web/ARCHITECTURE.md`.
- **Release / build:** `docs/release-model.md`.
- **The skills catalog:** `biopb-mcp/docs/skills.md` — what a skill is, how it
  ships, and how it is checked: structure, retrieval and contract tests in CI;
  simulated-user interaction runs against a real session locally, as a benchmark
  rather than a gate.
- **Workflow verification:** `docs/verification-scratch-kernel.md` — proposed:
  why `verify_workflow` should run in a scratch *process* rather than a scratch
  namespace, what a second kernel costs, and the one-slot admission rule that
  keeps two kernels from becoming two schedulers.
- **Agent benchmarks:** `biopb-mcp/docs/fixtures.md` — what a run is given and
  how it is scored. One runner over one case directory, whether the case is a
  claim about a skill or about a piece of work (`_tests/agentbench/` for the
  machinery, `_tests/bench/` for the cases and the run).
