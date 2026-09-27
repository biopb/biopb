---
description: Add an op by writing one server file — the header, `@op` and its keywords, input modes, and a GPU torch.
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
   take minutes (a torch); an entry still installing binds on a later refresh.
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

* `"eager"` (default): numpy arrays. A large reference is refused rather
  than pulled whole.
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

## A GPU torch

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
