---
description: Add an op by writing one server file — the header, `@op` and its keywords, input modes — and running a server elsewhere (a url entry).
---

# Writing an algorithm server

A package the kernel cannot import — it pins another torch, wants another
Python, or conflicts with what the session runs — goes behind the algorithm
plane instead: one file the control runs in an environment of its own, whose
functions become ops in [[ops]]. No packaging, no container.

## The loop

1. Write `~/.config/biopb/algorithms/<name>.py` from a cell. The stem is the
   server's name; a stem starting with `_` is ignored.
2. `print(ops.refresh())`. The control installs the file's dependencies and
   asks it for its ops, once per version of the file. The first install can
   take minutes (a torch); an entry still installing binds on a later refresh. `repr(ops)` lists the
   servers not built yet, and a server the control built since this kernel
   started is picked up the first time you name one of its ops.
3. Call the op. The first call starts the server.
4. On failure, `ops.status()` and `ops.logs("<name>")`; after an edit,
   `ops.restart("<name>")` (a changed file also restarts on its next call).

## The file

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

The header is uv's inline script metadata: everything the file imports goes in
`dependencies`, with `biopb-image-base` — `[lazy]` when an op takes `"lazy"`
or `"blocks"` input or returns large results. Import heavy packages inside the
function or at the top; either way they load in the server, never the kernel.

* **Arguments come from the signature.** A parameter annotated
  `Tensor("YX")` is a tensor argument; the string is the axes the function
  sees, in that order. Every other parameter is a plain argument, advertised
  with its default (no default = required).
* **Results need no declaration.** An array is a tensor output; anything else
  (a dict, a table, numbers, a string) is plain data. One return value is one
  output; return a tuple for several. An array of the op's own rank gets the
  input's other axes back.
* **Refuse bad input with `ValueError`** and a message the agent can act on
  (three channels expected, a dtype range); it reaches the caller as the
  error.

## `@op` keywords

| keyword | meaning |
|---|---|
| `description` | one line for listings; default: the docstring's first paragraph |
| `labels` | grouping, e.g. `["segmentation"]` |
| `name` | the op's name; default: the function's |
| `input` | how pixels arrive: `"eager"`, `"lazy"` or `"blocks"` (below) |
| `block_shape`, `overlap` | `"blocks"` only: block size and halo, per axis or one int |
| `dtype` | `"blocks"` only: the output dtype; saves computing one block to learn it |

## Input modes

With `"eager"` and `"lazy"`, the input's axes outside the op's `Tensor(...)`
must be size 1 — slice before calling. Only `"blocks"` maps over them.

* `"eager"` (default): numpy arrays. An `array_id` is read in the kernel and
  sent inline; one over 2 GiB is refused rather than pulled whole (the table in
  [[ops]] says where `lazy` and `blocks` input goes).
* `"lazy"`: dask arrays; the function may return one, which is written to the
  plane chunk by chunk.
* `"blocks"`: the function is mapped over blocks, and over every axis not in
  its `Tensor(...)`. For **pixelwise** ops only — filters, denoising,
  probability maps — whose output at a pixel depends on a neighbourhood the
  `overlap` covers. Instance segmentation is not: labels must agree across
  blocks, so take `"lazy"` and tile it yourself.

A function that `yield`s is a streaming op, for work that carries state frame
to frame (tracking, localization): a yielded string is progress, anything else
one item's result, and what it `return`s is the final result.

## A server someone runs

The control runs only `.py` entries. A server running elsewhere — another machine,
a batch job, a container — is named by `~/.config/biopb/algorithms/<name>.json`:

```json
{"url": "grpc://host:50051"}
```

`grpcs://` for TLS. Then `ops.refresh()` binds its ops. The entry has no other
fields. The control probes it with `Describe` and never starts, stops or restarts
it: `ops.restart` and `ops.logs` refuse it, and its lifecycle belongs to whoever
runs it. A `.json` with no url, or one sharing a name with a `.py`, is `invalid`;
the states are in [[ops]].

**What the deployment has to provide**

- **A running server.** The same file serves on its own:
  `uv run --script server.py --host <host> --port <port>`. `serve()` takes `--host`
  (default `127.0.0.1`), `--port` (default 50051), `--workers`, `--cache-dir` and
  `--describe`, which prints the op list as JSON and exits without binding: run it
  first to check the file where it will serve.
- **An address the control's machine can dial.** The url is connected to from the
  machine running the control, so it must reach the server from there, however the
  deployment arranges that.
- **No token.** A server bound off loopback requires a bearer token
  (`$BIOPB_ALGORITHM_TOKEN`, or one it mints and prints), and a url entry carries
  none, so the control cannot probe it and the kernel cannot call it. The server
  must be bound to loopback, and the entry's url a loopback address that reaches it.
- **Somewhere for large results.** It is the server's deployment, not the call's:

| deployment | a large result goes to |
|---|---|
| `--cache-dir` | the server's own embedded tensor server |
| `$BIOPB_TENSOR_URL` (and `$BIOPB_TENSOR_TOKEN`) set | that data plane's scratch source |
| neither | back inline, up to 128 MiB |

**Data crosses the kernel.** A remote server cannot read this machine's plane, so
an `array_id` input is read here and sent inline (2 GiB cap) unless the plane
advertises an external address ([[ops]] has the table). Slice first, or run the
server next to the data.

A server that goes away leaves its entry `unreachable`; once it is back,
`ops.refresh()` rebinds. How to deploy it is the environment's question, so look at
what the user has (a scheduler, an SSH host, a container runtime) and ask before
starting anything on a machine that is not theirs.

## Appendix: a GPU torch

On Linux PyPI's torch is built for one CUDA version, and on Windows it is
CPU-only. To choose the build, point torch at PyTorch's index in the header:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["biopb-image-base[lazy]", "torch", "cellpose"]
#
# [tool.uv]
# sources = { torch = { index = "pytorch-cu124" } }
# index = [
#   { name = "pytorch-cu124", url = "https://download.pytorch.org/whl/cu124", explicit = true },
# ]
# ///
```

Match the index to the machine's driver (`nvidia-smi` shows the highest CUDA
it supports); `cpu` instead of `cu124` gives the CPU build. Check from inside
an op, not the kernel: `torch.cuda.is_available()` in the server is what
counts.
