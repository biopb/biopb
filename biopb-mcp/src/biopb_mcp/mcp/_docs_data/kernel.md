---
description: The kernel: namespace, plugins, long-running jobs, where computes run.
---

# biopb-mcp IPython Kernel Guide

**Operation Guardrails** are in session `instructions`. Apply on every turn — follow them throughout.

Kernel is the main execution context for the agent. It is a live Python interpreter with access to
the TensorFlightClient for browsing and retrieving image data, and — where the session has a
display — a napari viewer window. The kernel is **stateful**: variables, imports, and viewer state
persist across turns. Agent code runs as a cell on the kernel's main thread, like a notebook cell:
while it runs the viewer does not repaint. Run a long compute you want to watch with `run_async(fn)`.

**There are two ways to show the user an image**, and which ones this session has is a fact about
the session, not about the work: the napari window ([[napari-viewer]]) and the browser page the control
serves ([[web-viewer]]). `server_status` says which. Do not assume a window exists.

## Namespace

| Name | Type | Description |
|------|------|-------------|
| `client` | TensorFlightClient or None | Connection to the data server for browsing/retrieving image data. Marshaled and thread-safe. |
| `viewer` | napari.Viewer | The napari window, where the session has one. `viewer.add_tensor(array_id)` puts a tensor on it; `viewer.tensor(layer)` reads one back as a plain dask array. Check `## Viewer` in `server_status` before relying on anything being *seen* — the object is bound even when nothing is on screen |
| `np/da` | module | imported packages: numpy and dask.array |
| `ops` | dict[str, callable] | biopb.image ProcessImage operations from configured servers (may be empty) |
| `run_on_main` | callable | runs `fn` on the Qt main thread and returns its result. Use it to **batch** many viewer mutations into one main-thread hop, or to touch raw Qt (`viewer.window`). |

- The `viewer` is a napari window, made **thread-safe** by marshaling known
  mutations (`viewer.dims`, `viewer.camera`, layer properties, `viewer.layers.remove()`,
  the `add_*()` family, …) to the Qt main thread. One caveat: raw Qt (`viewer.window`)
  still requires the main thread — off-thread access raises a clear error, so wrap it in
  `run_on_main()`. See [[napari-viewer]] for the full set of viewer operations, including
  mouse events.
- The `client` represents a `TensorFlightClient` instance. Data from the client are
  lazy, thread-safe, picklable dask arrays. See [[tensor-server-client]] for the full set of client
  operations, including browsing sources and reading tensors ([[upload]] is the write
  side); and see
  [[napari-viewer]] for how a layer's pixels differ from the server's. **Read a layer's pixels with `viewer.tensor(layer)`** — `layer.data` is
  packaged for the renderer, and handing a multiscale one to numpy silently computes on
  the lowest pyramid level.
- `ops` maps op name -> an inspectable callable that runs dedicated image-processing logic.
  The callable is a thin wrapper around a `biopb.image.ProcessImage` gRPC service on a configured
  server. The callable can take either a numpy array (eager) or a tensor-server array_id string
  (lazy). See [[ops]] for details.
- **Only `np` and `da` are pre-imported.** Everything else needs an explicit
  import, and these are guaranteed installed: `pandas`, `skimage`, `scipy`,
  `sklearn`, `matplotlib`, `cv2` (opencv-headless), `ome_zarr`, `napari`. The
  kernel is stateful, so one import per session is enough.

## Kernel plugins

**User plugins may add more names** beyond the table above: each `*.py` file in
`~/.config/biopb/kernel/`, and each installed `biopb_mcp.namespace` package, is
loaded at kernel start and bound as **one module, named after the file** —
`rolling_ball.py` becomes `rolling_ball`, and its functions are called as
`rolling_ball.subtract_background(...)`. They're lab-specific helpers, not
built-ins, so an unfamiliar module is likely one of these.

- **Which plugins loaded** — `server_status`, section `## Kernel plugins`. It
  lists the files and the packages that actually loaded; the loader is fail-open
  per unit, so a `*.py` sitting in that directory but missing from the report
  failed on load and the session log says why. The section reads
  `(disabled — services.namespace_enabled)` where plugins are switched off. The
  name it reports is the name bound in the namespace.
- **What a plugin offers** — introspection. `inspect_object` on the module prints
  its docstring plus every public callable with its signature and summary line:

```python
[n for n in dir() if not n.startswith("_")]   # everything actually in scope
inspect_object("rolling_ball")                 # the module: docstring + callables
```

A plugin function works inside `da.map_blocks` / `client.submit` like any other —
plugin modules are registered for by-value pickling, so a dask worker does not
need the plugin dir.

## Long-running jobs
A slow `execute_code` call returns a `job-N` handle while the cell keeps running on the main
thread. Meanwhile the viewer does not repaint, and `take_screenshot` / `inspect_object` refuse
until it ends -- asking for one then is a mistake in the plan. For a compute you want to watch,
end the cell with `run_async(fn, *args)` instead: `fn` runs on a worker thread, the call returns
a `task-...` id at once, and the viewer and screenshots stay live. Notes on tasks:
* One task at a time. What `fn` prints and returns is the task's record (`poll_job(task_id)`).
* Everything the cell prints after `run_async` is filed with the task, since the two share the
  cell's request: make `run_async` the cell's last statement.
* A task may mutate `viewer` (its calls are marshaled to the main thread). A user's cell can run
  meanwhile, and the two can race over the namespace and the viewer.

Stop a cell or a task with `interrupt_kernel` (a KeyboardInterrupt at the next bytecode; it also
cancels in-flight dask tasks while a cluster is attached) or `restart_kernel` (guaranteed, kills
the kernel). Notes:
* **`poll_job` waits for you — do not spin on it.** It watches a running job for
  `wait` seconds (10 by default, 30 max) and answers the moment the job ends, so
  calling it back-to-back asks the same question sooner and costs you a round trip
  for nothing. Raise `wait` if you have nothing to do until the job finishes; pass
  `wait=0` only for a snapshot you are not about to ask for again.
* **A blocking `.compute()` is interruptible** — while a cluster is attached
  `interrupt_kernel` cancels the in-flight dask tasks, so the `.compute()` raises and
  the job ends. On the in-process default (see below) the stop is best-effort.
* **Your own long loops** (per-chunk / per-file) are stopped by `interrupt_kernel`, which
  raises `KeyboardInterrupt` into the loop at the next iteration — no cooperative check needed.
* **Progress on a big graph:** submit with the distributed client
  (`_dask_client`, bound only while a cluster is attached — see below) and consume
  results as they land — this gives a live processed count via `poll_job`:
  ```python
  from dask.distributed import as_completed
  futs = _dask_client.compute(list_of_dask_results)   # list of Futures, non-blocking
  done = []
  for fut in as_completed(futs):
      done.append(fut.result())
      print(f"{len(done)}/{len(futs)} done", flush=True)   # visible via poll_job
  ```

## Where your computes run

The kernel starts on the **in-process scheduler**, the same one the napari viewer
reads through. That is the right default for this workload: a chunk read over
loopback is IO-bound and comes back as an mmap view, so routing it through worker
processes adds hops rather than speed — and a cluster nobody asked for is one that
quietly dies with the laptop lid and hangs every later `.compute()` (#970).

When the work is CPU-heavy and parallel — a per-tile filter over a big stack, a
segmentation sweep — put it on a cluster from a cell:

```python
_dask_ctl.attach()                 # spin a local cluster (sized from config)
_dask_ctl.attach("tcp://host:8786")  # or an external scheduler
_dask_ctl.detach()                 # back to in-process
```

The cluster lives as long as this kernel does. There is no tool for this because
there is nothing a tool would add: `.attach()` is `Client(...)` and `.detach()` is
`.close()`, and a client you build yourself works the same way — `_dask_ctl` just
also sizes the cluster from config and splits the chunk-cache budget across its
workers. `server_status`'s `## Dask` section always says which mode is in effect.
`restart_kernel` starts in-process again.

One real difference: **a `.compute()` is fully cancellable only while attached**
— `interrupt_kernel` cancels the in-flight futures. In-process the stop is
best-effort: a `KeyboardInterrupt` at the next bytecode, so a fetch inside a C
call ends only when it returns. Your own Python loops are stoppable either way.

## You are not the only writer of this namespace
The user can run their own code in this kernel, from a Jupyter notebook or console
attached to it. It is the same namespace and the same viewer, so their cells can rebind a variable you set, add
or remove a layer, or import something you did not.

* **You will be told, after the fact.** When user cells have run since your last call,
  a note listing them (`job-N (status)`) is appended to your `execute_code` / `poll_job`
  / `server_status` result. Read them with `poll_job`, and re-check what you rely on
  (`dir()`, `viewer.layers`, `inspect_object`) instead of trusting what you last saw.
* **One cell at a time.** While the user's cell runs, your new cells are refused and
  your other calls wait for it; while your cell runs, theirs waits for it. A user's cell
  can run while your `run_async` task does. Do not try to clear either.
* **Their cell is not yours to stop.** `interrupt_kernel` refuses a user job (it stops
  only your own). Do not reach for `restart_kernel` to get around that: it would
  destroy the user's variables and layers along with yours.
* **But no second agent.** The user is the only other writer you will meet: whichever
  agent runs code first holds the kernel until it restarts, and another client is
  refused by everything that changes kernel state — `execute_code`, `interrupt_kernel`,
  `restart_kernel` — keeping only the read-only tools. So a namespace change you did
  not make came from the person, and the note above is the whole story. If you are the
  one refused, say so and let the user decide; the restart is theirs.

Reading pixels off the server is [[tensor-server-client]], off a layer is
[[napari-viewer]], and the round trip for data too large to hold is [[upload]].
