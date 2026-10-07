# The algorithm plane

How biopb runs algorithms it cannot import into the agent's kernel: servers of
the `Ops` gRPC protocol, listed in a registry the control reads, run by the
control or by someone else.

| piece | where |
|---|---|
| the protocol | `proto/biopb/image/rpc_ops.proto` |
| `op` / `serve`, the server side | `biopb-image-runtime` (`biopb_image_base.ops`), a wheel on PyPI and a Docker base image |
| the registry and the probe | `biopb_control._registry` (biopb-control) |
| supervision and `/api/algorithms` | `biopb-control` (`_algorithm_plane.py`, `_control.py`) |
| clients | `biopb.image.connect()` (calls a server), `biopb.algorithms()` (asks the control), the kernel's `ops` (biopb-mcp `_process_ops.py`), `biopb algorithm` and `biopb image`, the dashboard's algorithm card |

Most algorithms cannot run in the kernel: they pin a torch that conflicts with
the session, want another Python, or are not Python at all. So each runs in its
own process, and the way to add one is a single file with no packaging and no
container.

## The protocol

```proto
message Arg {
  oneof kind {
    Tensor eager = 1;                        // inline pixels
    biopb.tensor.SerializedTensor lazy = 2;  // by reference, with its own read token
    google.protobuf.Value json = 3;          // anything else
  }
}
message Call { string op = 1; map<string, Arg> args = 2; }
message Event { map<string, Arg> outputs = 1; string progress = 2; }

message TensorArg { string axes = 1; bool mapped = 2; }
message OpInfo {
  string name = 1; string description = 2; repeated string labels = 3;
  string kwargs = 4;                         // "name=default, ..."; text, not typed
  map<string, TensorArg> tensors = 5;        // which args are tensors
  enum InputMode { EAGER = 0; LAZY = 1; BLOCKS = 2; }
  InputMode input = 6;
  bool streaming = 7;
}
message OpList { repeated OpInfo ops = 1; string fingerprint = 2; }

service Ops {
  rpc Call(Call) returns (stream Event);
  rpc Describe(google.protobuf.Empty) returns (OpList);
}
```

- **Any argument can be a tensor**, so image plus labels, or moving plus fixed,
  is an ordinary call, and so is an op with no tensor at all.
- **Pixels are the only structured type**; they are too big to be anything
  else. Tables, ROIs, scores and progress are JSON or text the agent reads.
- **Every call is a stream.** A plain op sends one event, a streaming op one
  per item. There is no call deadline: the client times out on silence between
  events, and stopping is cancelling the RPC.
- **`OpInfo.tensors`** tells a client which arguments are tensors, so it can
  tell an `array_id` from a string and check a fit before calling.
  `TensorArg.mapped` is set for a `"blocks"` op, whose server iterates the axes
  outside `axes`; otherwise those axes must be singleton.
- **`OpInfo.kwargs` is documentation, not a value.** It is the non-tensor
  arguments written as they'd appear in a call (`"name=default, ..."`, a bare
  name when required) -- nothing decodes it back, so a default with no JSON
  shape (`nan`) is exactly as fine here as any other repr.
- **`OpInfo.input`/`streaming`** say how tensor arguments arrive (`EAGER`
  reads a lazy input whole, capped; `LAZY` may stay out-of-core; `BLOCKS` maps
  over blocks) and whether the call yields more than one event -- known ahead
  of the call, not just by making one.
- **`fingerprint`** changes when any op changes; the control caches op lists by
  it.
- **Compression is gRPC's.** A server bound off loopback gzips integer (label)
  outputs, which shrink 25-50x; on loopback it costs more than it saves.

The image protocol keeps `ImageData`, `ROI`, `annotation` and `bindata` beside
`Ops`; `ProcessImage` and `ObjectDetection` are gone.

## A server file

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["biopb-image-base[lazy]", "scikit-image"]
# ///
from biopb_image_base import Tensor, op, serve

@op(description="Mean intensity and area per label", labels=["measurement"])
def label_stats(image: Tensor("YX"), labels: Tensor("YX")) -> dict:
    from skimage.measure import regionprops_table
    return regionprops_table(labels, image, properties=["label", "area", "mean_intensity"])

@op(description="Gaussian denoise", labels=["denoising"], input="blocks", overlap=16)
def gaussian(image: Tensor("YX"), sigma: float = 2.0):
    from skimage.filters import gaussian as g
    return g(image, sigma=sigma, preserve_range=True)

if __name__ == "__main__":
    serve()
```

- **Arguments come from the signature.** A parameter annotated `Tensor(axes)`
  (or `Annotated[np.ndarray, Tensor(axes)]` in a type-checked file) is a
  tensor; every other parameter is a kwarg, advertised with its default. What
  the signature cannot say (three channels, a dtype range) the function checks,
  raising `ValueError`, which reaches the client as `INVALID_ARGUMENT`.
- **`axes` is what the function sees**, in that order. The wrapper puts the
  input's other axes back on a result of the same rank, so the output carries
  the input's `dim_labels`.
- **`input` says how pixels arrive**, a resource decision the file declares:
  - `"eager"` (default): numpy. The other axes must be singleton. The
    kernel reads an `array_id` itself and sends it inline, so the server needs
    no route to the plane (nor `[lazy]`); one over `EAGER_INPUT_CAP` (2 GiB)
    is refused rather than pulled whole. A server still accepts a reference
    from another client and applies the same cap.
  - `"lazy"`: dask; an inline input arrives as a one-chunk dask array. The
    function may return dask.
  - `"blocks"`: the function is mapped over the tensor arguments with
    `map_overlap` (`block_shape`, `overlap`), iterating the axes not in `axes`.
    Only for pixelwise ops, whose output at a pixel depends on a neighbourhood
    the overlap covers. Instance segmentation is not: its labels must agree
    across blocks, so a server that tiles one takes `"lazy"` and stitches
    itself (`biopb_image_base.stitch`).
- **Results need no declaration.** An array is a tensor output, anything else
  JSON (numpy values and DataFrames converted). One return value is the output
  `result`; a tuple gives `0`, `1`, ...
- **A function that yields streams.** A yielded string is a progress event,
  anything else one item's outputs; a returned value is the final event.
  Cancelling the call closes the generator.

`serve()` parses `--host` (default `127.0.0.1`), `--port`, `--workers`,
`--cache-dir` and `--describe`, which prints the op list as JSON and exits
without binding. Calls to one server run one at a time. Bound off loopback the
server requires a token, `$BIOPB_ALGORITHM_TOKEN` or one it mints and prints.

### Where a large result goes

The server's deployment decides, not the call:

| deployment | large results go to |
|---|---|
| `--cache-dir` | the server's embedded tensor server, each result with its own read token and a TTL |
| `$BIOPB_TENSOR_URL` (+ `$BIOPB_TENSOR_TOKEN`) set, as the control does for a script entry | that data plane's scratch source |
| neither | inline, up to one message (128 MiB) |

On a plane, a numpy result up to 64 MiB still goes inline; a dask result is
written block by block and never assembled. A client keeps a result already on
its own plane as the `array_id` and copies any other into its scratch source.

### The package

`biopb-image-base` is versioned and published on the SDK's `v*` tag, beside
`biopb`, so a server file's header resolves it from PyPI. The core
(`biopb`, gRPC, numpy) serves inline pixels; `[lazy]` adds `biopb[tensor]` for
lazy input and a plane sink, `[stitch]` adds scipy for the stitching helpers.
The embedded tensor server needs `biopb-tensor-server`, which is not on PyPI,
so `--cache-dir` works in the Docker base image only.

## The registry

`~/.config/biopb/algorithms/` (`biopb._config.locations.algorithms_dir()`), one file
per server, named by its stem; a stem starting with `_` is skipped.

- **`<name>.py`, a script entry**: a server file. The control runs it with uv,
  in an environment built from its header alone, so nothing in it conflicts
  with the kernel or another entry.
- **`<name>.json`, a url entry**: `{"url": "grpcs://host:port"}`, a server
  someone else runs, Docker included. The control probes it with `Describe`
  and never starts or stops it.

A name used by both kinds, or a `.json` without a url, is listed with an
`error` and the state `invalid`. On first start, while the directory does not
exist, the control moves an older install's `mcp-config.json`
`process_image_servers` into url entries and drops the key.

With consent, a fresh install adds `cellpose.json`, the Cellpose server at
`grpcs://cellpose.biopb.org:443` (off-site; it logs client IPs).

## Supervision (script entries)

| state | meaning |
|---|---|
| `new` | never installed |
| `installing` | `uv lock --script`, then `uv sync --script`; minutes the first time (torch) |
| `stopped` | installed and described, ops cached, no process |
| `starting` | `uv run --script <file> --port N`, waiting for the port |
| `up` | serving |
| `failed` | install, describe or start failed; `error` holds the log tail |

- **Install and describe once per file version.** After install the control
  runs the file with `--describe` and caches the op list by the file's sha256,
  in `<state dir>/algorithms/<name>.describe.json` beside `<name>.log`.
  Install is bounded at an hour, describe at ten minutes.
- **Nothing runs until an op is called.** A script entry starts on `ensure`,
  and the kernel ensures on the first call to one of its ops; a GPU model
  should not hold memory nobody asked for. It stays up until stopped or the
  control exits.
- **An edit takes effect.** `ensure` on an entry whose file hash changed
  reinstalls it and restarts a stale server; `restart` does so
  unconditionally.
- **A config error does not loop.** A server that exits before it was ever up
  (a bad header, an import error) is `failed` until the next `ensure`; backoff
  restarts are for one that was `up` and crashed.
- **Ports and tokens.** Each start gets a free loopback port and a fresh
  token, passed as `$BIOPB_ALGORITHM_TOKEN`; the child also gets the data
  plane's `$BIOPB_TENSOR_URL`/`$BIOPB_TENSOR_TOKEN`. An entry's row carries its
  url and token to authenticated callers.
- **Without uv** a script entry is `failed` with an error saying so; uv is
  found through `$BIOPB_UV`, PATH, `~/.local/bin` or `~/.cargo/bin`.

A url entry is `up`, `unreachable` or `error` by its probe, and `stop`,
`restart` and `logs` refuse it: its lifecycle and logs belong to whoever runs
it. A failure inside a call is a gRPC status on the `Call` stream, the same
from either kind; a failure before a server can answer surfaces only through
the control, as `failed` and the log tail.

### The API

| route | does |
|---|---|
| `GET /api/algorithms` | every entry: kind, state, cached ops, url, token, error |
| `POST /api/algorithms/refresh` | install and describe new or edited entries, probe url entries |
| `POST /api/algorithms/{ensure,stop,restart}` | one entry by name |
| `GET /api/algorithms/logs?name=&lines=` | a script entry's log tail |

The routes sit behind the control's token like the rest of `/api`. A verb
waits at most `?client_timeout` less five seconds (at most the install bound,
60 s without the hint), so a slow install answers before the caller gives up.
`biopb` wraps them stdlib-only (backed by the private `biopb._control`):
`algorithms()`, `refresh_algorithms()`, `ensure_algorithm()`,
`stop_algorithm()`, `restart_algorithm()`, `algorithm_logs()`. The `biopb
algorithm` commands (`list`, `refresh`, `start`, `stop`, `restart`, `logs`) are
thin faces over them; none starts a control, so each says so when none answers.

## Clients

- **The kernel's `ops`** binds one function per op from the control's cached op
  lists, without starting anything. A call ensures the entry (up to 900 s for
  a first install), then reads events with `timeout.process_image` as the
  inactivity timeout and cancels the RPC in a `finally`, so a stopped cell
  does not leave the server computing. An op whose only output is `result`
  returns its value; other outputs return as a tuple. `ops.refresh()`,
  `ops.status()`, `ops.logs(name)` and `ops.restart(name)` manage the entries;
  `server_status` includes `ops.status()`.
- **`biopb.image.connect(target)`** is the SDK client of one server. `target`
  is a `grpc://` or `grpcs://` URL, or a registry name, which the control
  brings up first and whose token it takes. `OpsClient.describe()` lists the
  ops; `call(op, **values)` sends arrays as pixels and everything else as JSON
  and returns the decoded result (`encode_arg` / `decode_arg` are the codec);
  `events()` is the raw stream, with `inactivity_timeout` bounding the silence
  and a cancel when the caller leaves. A failed call raises `ValueError` (the op
  refused its arguments), `LookupError` (no such op) or `RuntimeError`.
- **`biopb image`**: `ops <server>` and `call <server> [op]`, where `<server>`
  is a URL or a registry name. `--token` (or `BIOPB_IMAGE_TOKEN`) sets the
  bearer token for a URL; a registry name brings its own.
- **The dashboard**'s algorithm card lists `/api/algorithms`.
- **The docs store** page `algorithm-servers.md` is the agent's reference for
  writing a server file.

To add an op, the agent writes `~/.config/biopb/algorithms/<name>.py` from a
cell, calls `ops.refresh()`, and calls the new op; the first call starts the
server. When something fails, `ops.status()` and `ops.logs(name)` show why and
`ops.restart(name)` picks up a fix.

## Remote servers

The same server file runs in the `biopb-image-base` Docker image, reached as a
url entry. biopb-server's cellpose image is built that way, and
cellpose.biopb.org serves it behind nginx, which routes `/biopb.image.Ops/` to
it on the same URL that still serves `ProcessImage` to older installs.

## Why this shape

- **uv, not a serving framework or Docker, for local entries.** biopb owns the
  protocol; what is left is environment and process management, which uv does
  from a header comment. Docker stays the format for published servers.
- **A process per entry.** The conflicts are the reason for the plane; only a
  process boundary holds against a torch pin.
- **One protocol for local and remote.** The deployments differ in where
  results go, which is server configuration, and in lifecycle, which is the
  control's; neither needs a different call.
- **The control supervises only what it runs.** Docker, pixi or conda servers
  are their owners'; the control probes them and passes them along.
