# Plan: full main-thread marshaling of viewer access (the "viewer proxy")

Status: **implemented** (Phases 1–3 below: proxy in `mcp/_viewer_proxy.py`, wired
in `_bootstrap`, tests in `_tests/test_viewer_proxy.py`; Phases 4–5 — `batched()`
and the docs/guide pass — deferred). Fixes the class of kernel segfaults where
agent `execute_code` mutates the napari viewer from the background job thread
(e.g. `biopb/biopb#100`). Supersedes the partial `add_*`-only wrap.

## 1. The problem

`execute_code` runs agent code on a **background daemon thread** (`_jobs._run`)
so the kernel's Qt main thread stays free to pump the event loop, serve
`take_screenshot` / `server_status`, and stay interruptible. But napari/Qt
objects are **main-thread-only**. Any viewer mutation from the job thread that
synchronously emits a napari event into a Qt slot touches a Qt widget off-thread
and **segfaults the whole kernel** (window included).

Captured proof (`biopb/biopb#100`): `viewer.layers.clear()` on the job thread →
layer-removal cascade → `QtDims._resize_slice_labels` (`qt_dims.py:186`,
`findChildren`/`setFixedWidth`) → SIGSEGV. No async/GL frames involved — the
async-slicing theory was a red herring.

### Why the current mitigation is insufficient

`_jobs.wrap_viewer_for_threads` marshals only `_VIEWER_GUI_METHODS`
(`add_image`/`add_labels`/… + `add_tensor`) to the main thread via
`run_on_main`. Its own docstring concedes the gap: "Open-ended mutations
(`layer.data = ...`, contrast limits, camera, dims) are not wrapped." And
`viewer.layers.clear()/remove()/pop()`, `del viewer.layers[i]`,
`viewer.dims.set_current_step(...)` aren't either.

This is **thread-affinity leakage**: any method that *returns a live sub-object*
(`viewer.layers`, `viewer.dims`, `viewer.layers[0]`) hands the worker an
unguarded handle, and the next mutation on it crashes. An allowlist of wrapped
methods is structurally leaky, not merely incomplete.

## 2. Guarantee we are buying

**No code against the documented namespace (`viewer` and anything reachable from
it) can segfault the session from a background thread.** Either the op is
auto-marshaled onto the main thread (and succeeds), or it raises a catchable
`ViewerThreadError`. Never a process death.

Out of scope (the residual in-process limit): an agent that deliberately
`import napari`/`PyQt6` and pokes raw Qt from a worker. Only a separate-process
viewer would close that; we accept it (documented).

## 3. Mechanism: a transparent marshaling proxy

Replace the monkeypatch with a proxy placed in the kernel namespace as `viewer`.
The **real** `napari.Viewer` is untouched (napari/Qt internals keep their direct
references); the agent only ever holds the proxy.

Proxy behavior, keyed on the calling thread (no-op fast path when already on the
main thread — `run_on_main` already short-circuits there):

- `__setattr__(name, value)` → marshal `setattr(real, name, value)` to the main
  thread. This single hook covers **all** evented-model field mutations: every
  layer property (`data`, `contrast_limits`, `colormap`, `opacity`, …) and every
  `ViewerModel`/`Dims`/`Camera`/… field, because napari mutates via pydantic
  `validate_assignment` on `__setattr__`.
- `__getattr__(name)`:
  - bound method → return a callable that **marshals the call** to the main
    thread (covers mutating methods like `Labels.paint`, `Points.add`, and the
    `LayerList` ops, without enumerating which mutate);
  - returns a value that is a **known handle type** → return a wrapped proxy
    (re-wrap, so handles never leak unguarded);
  - returns an **inert** value (numpy/dask array, scalar, str, dict, None) →
    return as-is (cheap field reads are not marshaled);
  - returns an **unrecognized Qt-bearing object** (e.g. `viewer.window`,
    `_qt_viewer`) → return a **guard** that raises `ViewerThreadError` on any
    off-main call/attr (fail-loud, never a raw Qt handle).
- `LayerList` dunders: `__getitem__`/`__iter__` re-wrap yielded `Layer`s;
  `__setitem__`/`__delitem__`/`__iadd__` marshal; `__len__`/`__contains__`
  pass through.
- Transparency: spoof `__class__` (property → real type) so
  `isinstance(layer, napari.layers.Image)` still works; delegate
  `__repr__`/`__eq__`/`__hash__`; memoize `wrap(real)` in a
  `WeakValueDictionary` keyed by `id(real)` so `viewer.layers[0] is
  viewer.layers[0]` and identity checks hold.

Read vs mutate policy: **mutations and method calls are marshaled; plain field
reads pass through** (they are dict lookups on the pydantic model — low risk, and
marshaling every `.ndim` would be prohibitively slow). Returned handles are
always re-wrapped regardless. The fail-loud guard is the backstop for anything
unclassified.

Implementation: hand-rolled (~200 LOC, no new dependency) for full control of
the marshal/wrap policy. `wrapt.ObjectProxy` was considered but forwards calls
synchronously and we must override the call/attr/setattr paths anyway.

## 4. The handle registry (the "object graph" — measured, closed, shallow)

Measured against napari 0.7.0 (depth ≤ 2 from `viewer`):

- **6 evented models**: `ViewerModel`, `Camera`, `Cursor`, `Dims`,
  `GridCanvas`, `Tooltip`
- **1 container**: `LayerList` (+ its `Selection`)
- **8 layer classes**: `Layer`, `Image`, `Labels`, `Points`, `Shapes`,
  `Surface`, `Tracks`, `Vectors`

The proxy wraps these via **two generic proxy classes** (an `EventedModel`
proxy that also serves all 8 layer types, and a `LayerList` proxy) plus the
`wrap()` dispatcher. The ~150 settable properties + ~13 list ops collapse to the
`__setattr__`/`__call__` hooks — **no per-API enumeration**.

## 5. Completeness as an enforced invariant (the tripwire), not an assumption

A **graph-walk test** is the safety mechanism that makes "we know napari's
graph" true over time instead of assumed:

1. Build a real offscreen viewer; add one of each layer type.
2. BFS from `viewer` over attributes, `layers`, and sub-models; collect every
   reachable object whose type is an `EventedModel`, a napari `Layer`, or lives
   in a `PyQt6`/`vispy` module.
3. Assert each such type is **either** in the proxy registry **or** on an
   explicit `INERT`/`GUARDED` allowlist. Fail with the offending path + type.

A future napari that adds a model type or `LayerList` method then **breaks CI**,
not production. Because the release pins `napari[all]==0.7.0` (single source of
truth, see `release-model`/`versions.json`) and dev/CI sync to the same lock,
this test certifies exactly the graph that ships.

## 6. Integration points

- `_jobs.py`: add the proxy module (`_viewer_proxy.py` or in `_jobs`), keep
  `run_on_main` (used by the proxy and still exposed for power users). Delete
  `_VIEWER_GUI_METHODS` / replace `wrap_viewer_for_threads` with
  `make_viewer_proxy(real_viewer)`.
- `_bootstrap.py`: keep wiring the **real** `viewer` into internal subsystems
  (`patch_viewer_add_tensor`, `resync_view_for_capture`, `viewer_window_alive`,
  the Tensor Browser widget). Expose `wrap(real_viewer)` as the namespace
  `viewer`. Only the agent-facing handle is the proxy; internal code and tools
  that already run on the main thread use the real object directly.
- Headless: proxy is not installed (no Qt main loop); `_HeadlessViewer` stays as
  is, matching the existing `if not headless:` guard.
- `add_tensor` (monkeypatched on the real viewer) is reached through the proxy's
  `__getattr__` and marshaled like any other method — no special case.

## 7. Performance & deadlock (the real cost, not completeness)

Each marshaled op is a `QMetaObject.invokeMethod` QueuedConnection +
`future.result(timeout=_RUN_ON_MAIN_TIMEOUT)` — a main-thread round-trip, and it
serializes with rendering. Fine for normal use (a handful of viewer ops per
turn); the cost shows in **hot loops** (the `#100` STORM case: ~4000 per-frame
`set_current_step` + layer updates).

Mitigations:
- A **batching context** — `with viewer.batched(): ...` — collects mutations and
  applies them in **one** main-thread hop.
- Document `run_on_main(lambda: <bulk>)` for tight loops (one hop for the whole
  block).
- `future.result` keeps its timeout: a blocked main thread raises
  `ViewerThreadError` instead of hanging the job forever (a partial mutation is
  acceptable vs. a deadlock or segfault).

## 8. Phasing

1. **Proxy + registry + `wrap()` + fail-loud guard**; swap into bootstrap.
2. **Graph-walk tripwire test** (gates the registry; runs in the synced .venv).
3. **Unit tests**: from a job thread, assert no segfault and correct main-thread
   effect for `layers.clear()`, `del layers[i]`, `layers.remove`,
   `dims.set_current_step`, `layer.data = …`, `layer.contrast_limits = …`,
   `points.add(...)`; assert reads return wrapped handles, `isinstance` works,
   identity memoization holds, `viewer.window` access raises off-thread.
4. **Performance**: `batched()` context + a 1000-iter scrub benchmark.
5. **Docs**: `guide://kernel` and `development.md` — viewer is auto-marshaled;
   `run_on_main` is now rarely needed; state the raw-Qt residual limit.

## 9. Risks / open questions

- **Deadlock surface widens** vs. the add_*-only wrap; mitigated by the existing
  `run_on_main` timeout. Audit for agent code holding a lock the main thread
  needs.
- **Transparency gaps** (an unforwarded dunder) — covered by the tripwire +
  unit tests.
- **`napari.current_viewer()` / raw `import`** can still hand back an unwrapped
  viewer — documented residual; only a separate-process viewer (option C) closes
  it, which we are explicitly not doing.
- **Decisions** (settled):
  1. **Mutations-only** — marshal mutations + method calls; pass plain field
     reads through (handles still re-wrapped). The tripwire still guarantees no
     leaked handle.
  2. **Defer `batched()`** — document `run_on_main(lambda: <bulk>)` for hot loops
     for now; add the context only if the per-op hop actually bites.
  3. **Hand-rolled proxy** — no `wrapt` dependency.
