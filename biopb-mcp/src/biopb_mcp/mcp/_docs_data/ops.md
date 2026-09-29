---
description: Server-side image-processing ops: what `ops` holds, calling one, and the servers behind it.
---

## Image Processing Ops (`ops`)

`ops` holds the ops of the algorithm servers this machine's control knows:
server files it runs for you (`~/.config/biopb/algorithms/*.py`) and servers
someone else runs. Each op is a callable, by name or as an attribute:

```python
list(ops)                       # op names
inspect_object("ops.gaussian")  # description, tensor arguments and their axes, defaults
print(ops.status())             # every server: state, ops, the first line of an error
```

An op whose name is one of `ops`' own methods (`refresh`, `status`, `logs`,
`restart`) is reached as `ops["refresh"]`.

### Calling one

Arguments go by name; an op with one tensor argument also takes it first.

* A tensor argument is an `np.ndarray` (sent inline; its axes are read from
  `ndim` — 2D=YX, 3D=YXC, 4D=ZYXC, 5D=TZYXC — unless you pass
  `dim_labels="ZYX"`, or `{name: axes}` for several) or a tensor-server
  **array_id** (the server reads the pixels itself; nothing passes through the
  kernel).
* Every other argument is plain data (numbers, strings, lists, dicts).
* A tensor result is an `np.ndarray`, or an **array_id** when any input was one
  or the server returned it by reference — so ops chain on large data. Other
  results are plain data. Several outputs come back as a tuple.

```python
labels = ops.segment(arr, diameter=30)             # ndarray -> ndarray
seg_id = ops.segment(image="<array_id>")           # id -> id, lazy
stats = ops.label_stats(image=img_id, labels=seg_id)  # two tensors -> a table
```

A server file's first call starts its server, which can take a while (a model
loading). A long op prints its progress; there is no overall deadline, only a
limit on silence (`timeout.process_image`), and `interrupt_kernel` stops it on
the server too. A streaming op that sends several results returns a list.

### When something fails

* An argument the op refuses raises `ValueError` with the server's message —
  read it before retrying. Any other failure inside a call is a
  `RuntimeError`.
* A server file that cannot start raises `... is failed`: `ops.status()` shows
  the state, `ops.logs("<name>")` the install and start-up output. Fix the file,
  then `ops.restart("<name>")`.
* A url server that is `unreachable` belongs to whoever runs it; there is
  nothing to restart from here.

To add an op the kernel cannot import — its own torch, another Python, a
conflicting package — write a server file: see [[algorithm-servers]].
