# Viewer thread-safety — the main-thread marshaling proxy

The kernel namespace's `viewer` is a transparent marshaling **proxy**
(`mcp/_viewer_proxy.py`, wired in `_bootstrap`; tripwire + unit tests in
`_tests/test_viewer_proxy.py`). Guarantee: **no code against `viewer` (or
anything reachable from it) can segfault the session from a background
thread** — each op is either marshaled onto the Qt main thread or raises a
catchable `ViewerThreadError`, never a process death.

## Why

An `execute_code` cell runs on the Qt main thread, where the proxy calls
straight through. Agent code off that thread is where it matters: a `run_async`
task, which runs on a **worker thread** (`_jobs._run`) to leave the main thread
free, or any thread agent code starts itself. napari/Qt objects are
**main-thread-only**, so a viewer mutation off that thread that emits a napari
event into a Qt slot **segfaults the whole kernel** — confirmed by
`viewer.layers.clear()` off the main thread crashing through
`QtDims._resize_slice_labels` with no async or GL involved.

The old `add_*`-only wrap (`wrap_viewer_for_threads`) was **structurally
leaky**: any call returning a live sub-object (`viewer.layers`,
`viewer.layers[0]`) handed that thread an unguarded handle whose next
mutation would crash. The fix wraps the whole reachable graph, not a method
list.

## Mechanism

The real `napari.Viewer` is untouched; only the agent holds the proxy. Keyed
on the calling thread (no-op fast path on the main thread):

- **`__setattr__`** marshals `setattr(real, …)` — one hook covers *every*
  evented-model field mutation, because napari mutates via pydantic
  `validate_assignment`.
- **`__getattr__`** dispatches on the yielded value: bound method → marshal
  the call; known handle type → re-wrapped proxy (handles never leak);
  inert value (array/scalar/str/None) → as-is; unrecognized Qt-bearing
  object (`viewer.window`, `_qt_viewer`) → a **guard** that raises
  `ViewerThreadError` off-main (fail-loud, never a raw Qt handle).
- **`LayerList` dunders**: read/iter re-wrap; set/del/iadd marshal;
  len/contains pass through.

Policy: **mutations and method calls marshaled, plain field reads pass
through** (dict lookups on the pydantic model — marshaling every `.ndim` is
too slow); returned handles always re-wrapped. Transparency: `__class__` is
spoofed to the real type (so `isinstance` works), `__repr__`/`__eq__`/
`__hash__` delegate, `wrap()` is memoized in a `WeakValueDictionary` by
`id(real)` so identity holds.

**Registry (napari 0.7.0, depth ≤ 2):** 6 evented models + `LayerList` + 8
layer classes + the overlays, all served by **two generic proxy classes**
(an `EventedModel` proxy covering the layers, a `LayerList` proxy) plus the
`wrap()` dispatcher — no per-API enumeration.

The overlays are named separately in both the dispatcher and the tripwire,
for two independent reasons, and missing either one hides them completely:
they subclass **psygnal's** `EventedModel`, not napari's, so
`isinstance(obj, napari.utils.events.EventedModel)` is `False` for every
overlay; and napari publishes them as **properties** over a private
container (`viewer.text_overlay`, `layer.bounding_box`), so a walk over
pydantic fields never reaches them.

**Tripwire:** a test walks a headless viewer *through the proxy* and
asserts every reachable handle is wrapped, following **public attribute
access** — fields plus properties — because that is what agent code has,
and a field-only walk misses the overlays for the reason above. A future
napari that adds a model or list method **breaks CI, not production**; the
pinned `napari[all]==0.7.0` (`versions.json`) means the test certifies
exactly the graph that ships.

## Gotchas

- **Hand-rolled (~200 LOC), no `wrapt`** — `wrapt.ObjectProxy` forwards
  synchronously, and this needs to override call/attr/setattr anyway.
- **Cost:** each marshaled op is a `QMetaObject.invokeMethod` round-trip
  that serializes with rendering — negligible normally, but it bites in hot
  loops (thousands of `set_current_step` calls across a movie's frames).
  Only a `run_async` task pays it: a cell runs on the main thread, where the
  proxy calls straight through, so bulk viewer work belongs in a cell.
- **Timeout, not deadlock:** `future.result(timeout=_RUN_ON_MAIN_TIMEOUT)`
  raises `ViewerThreadError` on a blocked main thread rather than hanging.
  The widened marshal surface widens the deadlock surface vs. the old wrap
  — watch for agent code holding a lock the main thread needs.
- **Only the agent handle is proxied.** `_bootstrap` wires the *real*
  viewer into internal subsystems (they already run on the main thread);
  `add_tensor` (monkeypatched on real) is reached through the proxy like
  any method.
- **Residual (accepted):** code off the main thread that `import napari` /
  `current_viewer()` / pokes raw `PyQt6` gets an unwrapped handle and can still crash — only a
  separate-process viewer would close it.
