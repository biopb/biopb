# The algorithm plane under the control

Status: **proposed** (design only; not implemented).

**Components:** the image protocol (`proto/biopb/image/`, a new `Ops` service),
`biopb-image-runtime` (a function-level API and a PyPI wheel), `biopb-control`
(the registry and the supervision), the core SDK (`biopb.control` reads the
registry), `biopb-mcp` and biopb-napari-widget (the kernel's `ops` and the
widget read the control).

## Goal

1. The list of algorithm servers leaves the biopb-mcp config
   (`services.process_image_servers`) for the control, so a napari widget, the
   CLI and the dashboard read one list without an MCP session in the picture.
   `development.md` marks the algorithm plane *supervised by the control,
   pending*; this is that.
2. An agent can add an op it cannot run in the kernel. Most algorithms cannot
   be imported there: they pin a torch that conflicts with the session, want a
   different Python, or are not Python at all. The base server ships only as a
   Docker image, and Docker is the wrong tool on a workstation: a daemon, a
   group membership, GPU passthrough, minutes per build, and none of it on a
   plain Windows or macOS install. The agent's path to a new op has to be
   **one file, no packaging, no container**, the way a kernel plugin already
   is.
3. One protocol for both deployments: a server the control runs next to the
   user's data plane, and a server someone runs remotely. `ProcessImage` was
   written for the second. It takes exactly one image, returns one image and a
   string, and has no shape for a long call, which is why servers work around
   it (`unifmir`'s `async_result` returns a handle and computes on a thread).

## The shape

```
~/.config/biopb/algorithms/          the registry: one entry per server
    cellpose.py                      a script entry: the control runs it under uv
    remote-lacss.json                a url entry: someone else runs it

biopb control
    installs, describes and supervises each script entry
    probes each url entry
    GET  /api/algorithms             every entry with state + cached ops
    POST /api/algorithms/{ensure,stop,restart}
    GET  /api/algorithms/logs

biopb.control.algorithms()           [{name, url, state, ops, error}]

kernel `ops`, the Image Processing widget, `biopb image servers`, the dashboard
    all read the control; none reads a config file
```

## The protocol

A new service replaces `ProcessImage` and `ObjectDetection`:

```proto
// An argument or a result: pixels, or anything else as JSON.
message Arg {
  oneof kind {
    Tensor eager = 1;                        // inline pixels
    biopb.tensor.SerializedTensor lazy = 2;  // by reference, with its own read token
    google.protobuf.Value json = 3;
  }
}

message Call {
  string op = 1;
  map<string, Arg> args = 2;
}

message Event {
  map<string, Arg> outputs = 1;
  string progress = 2;                       // free text
}

message TensorArg {
  string axes = 1;     // the axes the function sees, e.g. "YX"
  bool mapped = 2;     // the server iterates the other axes; else they must be singleton
}

message OpInfo {
  string name = 1;
  string description = 2;
  repeated string labels = 3;
  google.protobuf.Struct kwargs = 4;         // non-tensor args with their defaults
  map<string, TensorArg> tensors = 5;        // which args are tensors
}

message OpList {
  repeated OpInfo ops = 1;
  string fingerprint = 2;                    // changes when any op changes
}

service Ops {
  rpc Call(Call) returns (stream Event);
  rpc Describe(google.protobuf.Empty) returns (OpList);
}
```

- **Any argument can be a tensor.** Image plus labels (per-object
  measurement), moving plus fixed (registration), raw plus labels (tracking)
  are ordinary calls. An op with no tensor argument (a retrieval, a summary)
  is one too.
- **Pixels are the only structured type.** They are too big for anything
  else. ROIs, scores, prompts, pixel sizes and progress are JSON or text the
  agent reads; a typed message for each is a catch-up game. A large ROI set
  goes to the plane's ROI sets and its name comes back as JSON.
- **One RPC, always a stream.** A plain op sends one event; a streaming op
  sends one per item. There is no unary/stream pair to fall back between, no
  call deadline (the client times out on silence between events), and a stop
  is a cancelled RPC.
- **The schema says which arguments are tensors.** The client cannot tell an
  `array_id` from a string argument otherwise, and `TensorArg` keeps the
  pre-call fit check the kernel and the widget do today.
- **`fingerprint`** lets the control cache an op list and know when it is
  stale.
- **Inline pixels are raw.** Compression is gRPC's, set per response by the
  sender: `serve()` gzips integer (label) outputs when it is deployed remotely,
  where they shrink 25-50x, and nothing on loopback, where it costs more than
  it saves. Compressing transfers from the plane (Arrow IPC with zstd) is the
  tensor plane's concern.

Where a large result goes is the server's deployment, not the call's:

| deployment | large results go to | set by |
|---|---|---|
| a script entry | the control's data plane | `BIOPB_TENSOR_URL`/`BIOPB_TENSOR_TOKEN` in the child's environment |
| remote (Docker) | the embedded in-process plane, with a per-result read token and a TTL | `--cache-dir`, as today |
| neither | inline | default |

A client keeps a lazy result that is already on its own plane as its
`array_id`, and copies any other into its scratch source, as the kernel's op
wrapper does today. The local path costs no copy; the protocol does not know
the difference.

The old services go once their callers move: the kernel's op wrapper, the
widget's gRPC client (biopb-napari-widget),
`biopb image` CLI, `_algorithms`, the mock servicer, the examples, and the
`ImageData` helpers in both SDKs. The prebuilt biopb-server images are
experimental and move when they are rebuilt. `ImageData`, `ImageAnnotation`,
`OpSchema` and the detection messages are deleted with them.

## The server file

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["biopb-image-base[lazy]", "scikit-image"]
# ///
from biopb_image_base import Tensor, op, serve

@op(description="Mean intensity and area per label", labels=["measurement"])
def label_stats(image: Tensor["YX"], labels: Tensor["YX"]) -> dict:
    from skimage.measure import regionprops_table
    return regionprops_table(labels, image, properties=["label", "area", "mean_intensity"])

@op(description="Gaussian denoise", labels=["denoising"], input="blocks", overlap=16)
def gaussian(image: Tensor["YX"], sigma: float = 2.0):
    from skimage.filters import gaussian as g
    return g(image, sigma=sigma, preserve_range=True)

if __name__ == "__main__":
    serve()
```

`serve()` builds the `OpList` from the decorated functions, decodes the
arguments, validates and merges kwargs, calls the function, encodes the
outputs, and parses `--host`/`--port`/`--describe`. `--describe` prints the
op list as JSON and exits without binding a port. The pieces exist in
`biopb_image_base.common` (`parse_kwargs`, `validate_kwargs`, the decoding,
the servicer base with its error translation).

- **Arguments come from the signature.** A parameter annotated `Tensor[axes]`
  is a tensor argument; every other parameter is a kwarg, and its default goes
  into `OpInfo.kwargs`. Everything the shape does not say (three channels for
  an RGB model, a dtype range, isotropic Z) stays in the function: it raises
  `ValueError`, and the base translates that to `INVALID_ARGUMENT`.
- **`axes` is what the function sees.** It gets exactly those axes, in that
  order, and the wrapper puts the others back on the result so the output
  carries the input's `dim_labels`.
- **`input` says how pixels arrive.** Eager or lazy is a resource decision, so
  it is declared, not inferred:
  - `"eager"` (default): numpy arrays. The other axes must be singleton, and a
    lazy input above a size cap is refused with an error naming the cap and
    the other two modes, so a forgotten slice never pulls a whole timelapse
    into the server.
  - `"lazy"`: the dask arrays as decoded; the function returns one. This and
    `"blocks"` need the `[lazy]` extra.
  - `"blocks"`, with `block_shape` and `overlap`: the wrapper maps the function
    over all tensor arguments with `map_overlap`, iterating the axes not in
    `axes` (`TensorArg.mapped`). The tensor arguments must share a shape. This
    is for **pixelwise** ops (filters, denoising, semantic probability maps),
    whose output at a pixel depends only on a neighbourhood the overlap covers.
    Instance segmentation over a large image is not pixelwise: labels have to
    agree across blocks, which is per-model work (the Cellpose servers link
    flow destinations). A server that does it takes `"lazy"` and does its own
    tiling.
- **Results need no declaration.** An array is a tensor output: a numpy
  result goes inline or to the sink by size, a dask result is written to the
  sink chunk by chunk and never assembled. Anything else (a string, a dict,
  numbers) is JSON, with numpy values converted. A single return value is the
  output `result`; a tuple gives the outputs `0`, `1`, ..., each by the same
  rule, so an op can return a label image and a table together.
- **A function that yields is a streaming op.** Tracking and SMLM carry state
  from frame to frame, which `blocks` cannot, and a generator holds it
  between yields. The function takes its input `"lazy"` and yields one item
  per step. With a plane as the sink, the server adds one tensor up front,
  writes each yielded frame into it, and every event carries that one
  reference and the progress, so the result is readable while the stream
  runs. Without one, each event carries its item inline and the client
  concatenates. A cancelled call closes the generator and keeps what was
  written.

### The runtime package

- **A wheel on PyPI.** `uv run` resolves `biopb-image-base` from the header, so
  it is built and published on the SDK tag with `biopb`.
- **A slim core.** `biopb` plus `grpcio-health-checking`, enough for inline
  pixels. A `[lazy]` extra adds `biopb[tensor]` (pyarrow, dask) for lazy input
  and a plane sink. scipy, imageio and typer move to the extras of the modules
  that use them. The embedded plane needs `biopb-tensor-server`, which is not
  on PyPI, so it stays in the Docker image; that is the remote deployment.

## The registry

A directory, `~/.config/biopb/algorithms/`, beside the kernel-plugin directory
and with the same rule: the file's stem is the entry's name, and a stem
starting with `_` is skipped.

- **`<name>.py`**, a script entry: a server file as above. The control runs it
  with uv in the environment built for its header alone, so nothing in it can
  conflict with the kernel or another entry.
- **`<name>.json`**, `{"url": "grpc://host:port"}`: a server someone else runs,
  Docker included. The control probes it with `Describe` and lists it; it
  never starts or stops it. The migration writes today's config entries as
  these on first start and deletes the mcp key.

The registry is the whole configuration, and `biopb._algorithms.configured()`
reads this directory.

## Supervision

`DataPlaneSupervisor` is already spec-driven: argv, environment, a port, a
liveness probe, backoff, a log file, parent-death arming. The tensor-specific
part splits out of `DataPlaneSpec` into a `ServiceSpec` (argv, env, port,
probe), and each script entry gets a supervisor.

A script entry goes through these states:

| state | meaning |
|---|---|
| `new` | never installed; no cached ops |
| `installing` | `uv lock --script` then `uv sync --script`; minutes the first time (torch) |
| `stopped` | installed and described, ops cached, no process |
| `starting` | `uv run <file> --port <N>`, waiting for the port |
| `up` | serving |
| `failed` | install, describe or start failed; `error` holds the log tail |

- **Install and describe happen once per file version.** After install the
  control runs `uv run <file> --describe` and caches the op list, keyed by the
  file's hash. The lock (`<name>.py.lock`, beside the file) makes a later
  start resolve the same wheels; an edited header re-locks.
- **Nothing runs until an op is called.** The data plane starts at boot; a
  script entry starts on `ensure`, and the kernel ensures on the first call to
  one of its ops. A GPU model holding memory nobody asked for is worse than a
  few seconds' wait. It stays up until stopped or the control exits.
- **An edit takes effect.** `ensure` on an entry whose file hash differs from
  the running one restarts it through install and describe; `restart` does
  the same unconditionally.
- **A config error does not loop.** A failure before the port binds (a bad
  header, an import error, a syntax error) is `failed` and waits for the next
  `ensure`; backoff restarts apply only to a server that was `up` and crashed.
- **Ports and tokens.** The control picks a loopback port per entry and a
  token the server checks (`TokenValidationInterceptor`); `/api/algorithms`
  gives both to its authenticated callers. Remote mode does not expose
  algorithm servers; a remote client reaches the ops through a session's
  kernel.

All of this is the control's, for script entries only; the protocol carries
none of it. A failure before a server can answer (install, import, bind) has
only one witness, the process that ran it, so it surfaces through the control
as `failed` and the log tail. A failure inside a call is a gRPC status with the
exception's message on the `Call` stream, the same from a managed server and a
remote one. A `url` entry has two states, `up` and `unreachable`, and no
`logs` or `restart`: its logs and lifecycle belong to whoever runs it.
`--describe` is how the control learns a script's ops without keeping it
running; a remote server answers `Describe` instead.

## What the agent does

1. Writes `~/.config/biopb/algorithms/label_stats.py` from a cell. Writing a
   file the user's own control will run is the power `execute_code` already
   has.
2. Calls `ops.refresh()`. The control installs and describes new or edited
   entries; the ops bind from the cached op lists without starting anything.
   An entry still `installing` is reported and binds on a later refresh.
3. Calls `ops.label_stats(image="<array_id>", labels="<array_id>")`. The first
   call starts the server.
4. When something fails, `ops.status()` shows each entry's state and error,
   `ops.logs("label_stats")` the log tail, and `ops.restart("label_stats")`
   picks up a fix; the last two refuse a `url` entry. `server_status` lists
   the entries and their states.

The docs store gets one reference page: the header, `op`'s keywords, the three
input modes, and the CUDA-index recipe for a torch pin.

## Client surfaces

| today | becomes |
|---|---|
| kernel `build_ops_from_config(CONFIG)` over `ProcessImage` | `ops` over `Ops`, bound from `biopb.control.algorithms()`, with `refresh`/`status`/`logs`/`restart` |
| widget `server_url` text field | a choice from `biopb.control.algorithms()`, with the text field as the no-control fallback |
| `biopb image servers` reads the mcp config | reads the control |
| dashboard `/api/algorithms` probes a config list | the supervisors' state and cached ops; probing remains for `url` entries |
| `biopb._algorithms.configured()` reads `mcp-config.json` | reads the registry directory |

The op wrapper calls `Call` and reads events until the stream ends, with an
inactivity timeout between events in place of a call deadline, and cancels the
RPC in a `finally` so a Stop (a `KeyboardInterrupt` on the main thread) does
not leave the server computing for a client that left. One output returns as
its value, several as a tuple.

`biopb.control` stays stdlib-only: `algorithms()` and the verbs are
authenticated HTTP calls. The gRPC probe stays in `biopb._algorithms`, on the
control's side.

## Why not

- **A serving framework** (BentoML, Ray Serve, LitServe, Triton, MLflow) brings
  its own transport and packaging. biopb owns the protocol; what is left is
  environment and process management, and uv does that with a header comment.
- **Docker for the agent path.** It stays the deployment format for published
  servers, reached as a `url` entry. The same server file runs under both.
- **The control supervising other runtimes** (Docker, pixi, conda). Their
  lifecycles are the user's; the control verifies them and passes them along.
  A pixi script kind is a later addition if a dependency off PyPI earns one.
- **Running the op in-process after all** (a subinterpreter, a venv the kernel
  imports from). The conflicts are the reason for the plane; a process boundary
  is the only one that holds against a torch pin.
- **A second, local-only protocol.** The deployments differ in where results
  go, which is server configuration, and in lifecycle, which is the control's.
  Neither needs a different call.

## Costs

- **uv on the machine.** The installer installs uv, so it is on PATH for every
  installed biopb. The frozen build ships the uv binary or reports script
  entries as unavailable.
- **One process per running entry**, started on demand and stopped with one
  call.
- **A protocol break.** Every client and server moves to `Ops`; see the list
  above.
- **The mcp config loses a key**, and its schema, tests and settings page
  follow. The gRPC message size stays a kernel client setting;
  `timeout.get_op_names` becomes the `Describe` timeout and
  `timeout.process_image` the inactivity timeout.

## Shape of the work

Each step leaves the monorepo working.

1. **Protocol and runtime.** `Ops` in the proto (Python and Java stubs);
   `op`/`serve` in `biopb-image-runtime` with the three sinks, `--describe`,
   the dependency split, and the wheel published on the SDK tag.
2. **Control.** `ServiceSpec` split out of `DataPlaneSpec`; the registry
   directory and its migration; install, describe and the state table;
   `ensure`/`stop`/`restart`/`logs` on `/api/algorithms`;
   `biopb.control.algorithms()`.
3. **Clients.** The kernel's `ops` on `Ops`, reading the control; the widget's
   server choice and client; `biopb image`; the docs-store page;
   `server_status`.
4. **Retirement.** `ProcessImage`, `ObjectDetection` and their messages
   deleted; the examples and the Docker base image on `Ops`.
