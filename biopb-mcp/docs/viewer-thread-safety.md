# Viewer thread-safety — the main-thread marshaling proxy

The kernel namespace's `viewer` is a transparent marshaling **proxy**
(`mcp/_viewer_proxy.py`, wired in `_bootstrap`; tripwire + unit tests in
`_tests/test_viewer_proxy.py`). Guarantee: **no code against `viewer` (or
anything reachable from it) can segfault the session from a background thread** —
each op is either marshaled onto the Qt main thread or raises a catchable
`ViewerThreadError`, never a process death.

## Why

`execute_code` runs agent code on a **background daemon thread** (`_jobs._run`)
to keep the Qt main thread free and interruptible. napari/Qt objects are
**main-thread-only**, so a viewer mutation off that thread that emits a napari
event into a Qt slot **segfaults the whole kernel**. Proof (`biopb/biopb#100`):
`viewer.layers.clear()` on the job thread → `QtDims._resize_slice_labels` →
SIGSEGV (no async/GL involved — async-slicing was a red herring).

The old `add_*`-only wrap (`wrap_viewer_for_threads`) is **structurally leaky**:
any call returning a live sub-object (`viewer.layers`, `viewer.layers[0]`) hands
the worker an unguarded handle whose next mutation crashes. The fix must wrap the
whole reachable graph, not a method list.

## Mechanism

The real `napari.Viewer` is untouched; only the agent holds the proxy. Keyed on
the calling thread (no-op fast path on the main thread):

- **`__setattr__`** marshals `setattr(real, …)` — one hook covers *every*
  evented-model field mutation, because napari mutates via pydantic
  `validate_assignment`.
- **`__getattr__`** dispatches on the yielded value: bound method → marshal the
  call; known handle type → re-wrapped proxy (handles never leak); inert value
  (array/scalar/str/None) → as-is; unrecognized Qt-bearing object
  (`viewer.window`, `_qt_viewer`) → a **guard** that raises `ViewerThreadError`
  off-main (fail-loud, never a raw Qt handle).
- **`LayerList` dunders**: read/iter re-wrap; set/del/iadd marshal; len/contains
  pass through.

Policy: **mutations + method calls marshaled, plain field reads pass through**
(dict lookups on the pydantic model — marshaling every `.ndim` is too slow);
returned handles always re-wrapped. Transparency: `__class__` spoofed to the real
type (so `isinstance` works), `__repr__`/`__eq__`/`__hash__` delegate, `wrap()`
memoized in a `WeakValueDictionary` by `id(real)` (so identity holds).

**Registry (napari 0.7.0, depth ≤ 2):** 6 evented models + `LayerList` + 8 layer
classes, all served by **two generic proxy classes** (an `EventedModel` proxy
covering the layers, a `LayerList` proxy) + the `wrap()` dispatcher — no per-API
enumeration.

**Tripwire:** a test BFSes a real offscreen viewer and asserts every reachable
`EventedModel` / napari `Layer` / `PyQt6`/`vispy` type is in the registry or an
explicit `INERT`/`GUARDED` allowlist. A future napari that adds a model or list
method **breaks CI, not production**; the pinned `napari[all]==0.7.0`
(`versions.json`) means the test certifies exactly the graph that ships.

## Gotchas

- **Hand-rolled (~200 LOC), no `wrapt`** — `wrapt.ObjectProxy` forwards
  synchronously, and we override call/attr/setattr anyway.
- **Cost:** each marshaled op is a `QMetaObject.invokeMethod` round-trip that
  serializes with rendering — negligible normally, but it bites in hot loops (the
  `#100` STORM case: ~4000 `set_current_step`/frame). Escape hatch:
  `run_on_main(lambda: <bulk>)` (one hop for the whole block). A
  `viewer.batched()` context is a possible future add, not built.
- **Timeout, not deadlock:** `future.result(timeout=_RUN_ON_MAIN_TIMEOUT)` raises
  `ViewerThreadError` on a blocked main thread rather than hanging. The widened
  marshal surface widens the deadlock surface vs. the old wrap — watch for agent
  code holding a lock the main thread needs.
- **Only the agent handle is proxied.** `_bootstrap` wires the *real* viewer into
  internal subsystems (they already run on the main thread); `add_tensor`
  (monkeypatched on real) is reached through the proxy like any method. Headless
  installs no proxy (`if not headless:` guard, `_HeadlessViewer`).
- **Residual (accepted):** a worker that `import napari` / `current_viewer()` /
  pokes raw `PyQt6` gets an unwrapped handle and can still crash — only a
  separate-process viewer would close it.
