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

* A tensor argument is an `np.ndarray` (always sent inline; its axes are read from
  `ndim` — 2D=YX, 3D=YXC, 4D=ZYXC, 5D=TZYXC — unless you pass
  `dim_labels="ZYX"`, or `{name: axes}` for several) or a tensor-server
  **array_id**, which travels as in the table below.
* Every other argument is plain data (numbers, strings, lists, dicts).
* A tensor result is an `np.ndarray`, or an **array_id** when any input was one
  or the server returned it by reference — so ops chain on large data. Other
  results are plain data. Several outputs come back as a tuple.

```python
labels = ops.segment(arr, diameter=30)             # ndarray -> ndarray
seg_id = ops.segment(image="<array_id>")           # id -> id, lazy
stats = ops.label_stats(image=img_id, labels=seg_id)  # two tensors -> a table
```

**Where an `array_id`'s pixels go** depends on the op's input mode (`op.input_mode`;
[[algorithm-servers]] says how a file picks one) and on where its server runs
(`op.url`; a loopback address is local):

| op input | local server | remote server |
|---|---|---|
| `eager` (default) | read here, sent inline (over 2 GiB is refused) | same |
| `lazy`, `blocks` | a reference: the server reads the plane itself, nothing passes through the kernel | a reference only if the plane advertises an external address, else read here and sent inline (same cap). The plane's bearer token is never sent to a remote server |

A server file's first call starts its server, which can take a while (a model
loading). A long op prints its progress; there is no overall deadline, only a
limit on silence (`timeout.process_image`), and `interrupt_kernel` stops it on
the server too. A streaming op that sends several results returns a list.

### Server states

`ops.status()` shows each server's state:

| state | meaning |
|---|---|
| **script entry** (a file the control runs) | |
| `new` | file seen, not installed yet: `ops.refresh()` |
| `installing` | the control is installing its dependencies and asking it for its ops; bind on a later `ops.refresh()` |
| `stopped` | described, not running; the first call starts it |
| `starting` | coming up (a model loading), or between a crash and its restart |
| `up` | running |
| `stale` | running an older version of the file; the next call restarts it, or `ops.restart("<name>")` |
| `failed` | install or start failed: `ops.logs("<name>")`, fix the file, then `ops.restart("<name>")` |
| **url entry** (a server someone else runs) | |
| `up` | answered |
| `unreachable` | nothing answered in time (down, bad host, TLS mismatch); it belongs to whoever runs it |
| `error` | answered with an error, or the probe itself failed |
| `invalid` | the entry is malformed (not a `grpc://` or `grpcs://` URL, or both `x.py` and `x.json` exist) |

`biopb algorithm list --json` prints the control's rows: `name`, `kind`
(`script` or `url`), `url`, `target`, `scheme`, `state`, `ops`, `op_count`,
`fingerprint` and `error` (a server's token is left out). `stopped` is only ever a script entry; a url entry is
`up`, `unreachable`, `error` or `invalid` by its last probe. To run a server yourself
and name it by a url entry, see [[algorithm-servers]].

**Two servers with the same op name** are both renamed `<server>_<op>` and the
plain name goes away, so `ops.invert` stops working when a second server also offers
`invert`.

### When a call fails

An argument the op refuses raises `ValueError` with the server's message — read it
before retrying. Any other failure inside a call is a `RuntimeError`; a server
that cannot start raises `... is failed` (see the states above).

To add an op the kernel cannot import — its own torch, another Python, a
conflicting package — write a server file: see [[algorithm-servers]].
