# Design note: viewer prefetch / cache warming

**Status:** **Deferred — not implemented.** This captures a design we worked
through and consciously shelved. The blocker is *not* technical impossibility;
it is ROI: doing it *correctly* needs a server-side change (a prefetch reader
pool in biopb-tensor-server), and the payoff is a human-scrubbing nicety in an
**agent-first** tool — the agent gains nothing from prefetch. Revisit when
either (a) biopb-mcp is deployed for interactive *human* use where scrub
latency matters, or (b) the tensor-server adapter is being touched for another
reason and a concurrent prefetch reader falls out cheaply.

## Why we'd want it

Loading an nd2 (no native pyramid) and scrubbing channels is decode-bound, with
a sharp page-cache order effect. Measured on `ND010.nd2` (18 positions, each
`[T1,C4,Z1,14234,14234]` uint16, **1.62 GB/position uncompressed**):

- **First channel of a position: ~5–6 s** = ~3 s disk I/O of the *whole 1.62 GB
  position* (channels are interleaved, and `reduction_method="area"` needs every
  full-res pixel, so the coarse overview secretly reads the full-res plane) +
  ~2.2 s deinterleave/downsample of one channel.
- **Each subsequent first-visit channel: ~2.2 s** — decode only; the raw bytes
  are already OS-page-cache-warm from the first read.
- **Revisiting a channel: ~18 ms** — the server caches the downsampled chunk per
  `chunk_id`.

So once a position is loaded, the user pays a flat ~2.2 s per *new* channel.
Prefetch would convert those into background work so scrubbing feels instant.
Full cost model: see the `nd2-cold-read-io-model` agent memory and biopb #72.

## Two modes (they are different optimizations)

- **Mode A — fill the client-local cache (transfers bytes).** Goal: chunk lives
  in the `TensorFlightClient` cache so a scrub needs zero round-trip. Bounded by
  the client cache (~1 GB) and requires transfer. On **localhost** the transfer
  is ~free (~0.13 s/50 MB over loopback), so Mode A's only marginal win over
  Mode B is skipping an ~18 ms round-trip — negligible. **Mode A earns its keep
  only against a remote server**, where it avoids repeated WAN round-trips.

- **Mode B — exercise the server backend (no transfer).** Goal: make the server
  decode + cache the downsampled chunk and warm the OS page cache. ~99 % of the
  latency is server-side, so this is where the win is — and you can be far more
  aggressive (server chunk cache is 128 GB vs the client's 1 GB) because you
  neither transfer nor pressure the client cache. **Mode B is exactly what the
  server `PrecacheWorker` already does internally** (`resolve_chunk_data(chunk_id,
  cache_manager)` — compute-and-cache, no client). So Mode B should not be a
  client feature; it should be a smarter, *demand-driven* version of the
  precache worker (the one currently disabled via `[precache] enabled=false`,
  because it is blanket + index-time and confounds profiling).

## The core risk: the `_io_lock` is non-preemptive

biopb-tensor-server's aicsimageio adapter serializes all reads to one file
behind a single `_io_lock` (`with self._io_lock:` in `get_data`). It exists
because the reader is **stateful and shared** — the adapter calls
`aics_image.set_scene(...)`, mutating one `AICSImage`, so concurrent reads would
corrupt each other. Consequences:

- You **cannot interrupt an in-flight chunk decode.** A speculative prefetch
  that holds the lock makes an interactive read wait for it to finish.
- The lock is held **per chunk** (released between chunks), so with
  cancel-*between*-chunks discipline the worst an interactive scrub suffers is
  **one in-flight prefetch chunk** (~1.1 s for a Y-band, ~2.2 s for a full
  channel). Bounded, not unbounded.
- Crucially, today's baseline is *already* ~2.2 s on every first-visit channel,
  so prefetch + cancel-between-chunks **never makes the average worse** — it
  trades guaranteed-2.2 s-always for usually-instant with occasional ~1 s
  collision stalls. The objection is *consistency*, not regression.

**The correct fix is not preemption (impossible) but de-contention:** give
prefetch its **own reader handle** (separate `ND2File`/`AICSImage`, or a small
prefetch reader pool) so prefetch I/O runs concurrently with interactive reads —
no shared lock to preempt. Most prefetch work is page-cache-warm decode, so
disk contention is minimal. Cost: 2× file handles + a little memory. This is a
**server-side** change and is the gate for doing prefetch *right*.

## Phased plan

- **Phase 0 — client-side v1 (no server change, localhost only).** On
  `add_image` in `add_tensor_layer`, spawn a daemon thread that walks the *other*
  channels of the *coarsest displayed level* and `.compute()`s each (warms server
  chunk cache + client cache; Mode B as a side effect of Mode A). Must:
  idle-gate (only run when no interactive activity), **cancel between chunks** on
  any interactive request / dims change, run **off the single agent job slot**
  (own executor, not the `_jobs` lane), and bound by the client cache budget.
  Removes the 2.2 s scrub stalls for flip-through-channels; accepts bounded
  collision stalls. Scope strictly to the viewer (the agent path does not
  benefit).
- **Phase 1 — server prefetch reader (de-contention).** Add a separate prefetch
  reader handle / pool in the adapter so prefetch does not share `_io_lock` with
  interactive reads. This is what removes the collision-stall risk and is the
  real prerequisite for being aggressive.
- **Phase 2 — demand-driven server warming + warm RPC (scalable / remote).**
  Expose a client→server "warm these `chunk_id`s" action (a Flight `DoAction`, or
  exposing `PrecacheWorker.enqueue`) serviced via `resolve_chunk_data` with **no
  transfer**. Re-purpose the precache worker from blanket-index-time to
  **view-triggered**: warm *this* position's siblings/neighbors, not the whole
  catalog. Reuse its existing politeness knobs (`idle_debounce_seconds`,
  `backlog_high_water`, idle-gating) so speculative warming never starves live
  reads.
- **Phase 3 — client prefetch policy.** View-event triggers: layer added → warm
  sibling channels; dims change → predict next slice; zoom intent → warm the
  current viewport's *next-finer-level* tiles (never eagerly — that is the
  full-res path). Cancel-on-move; cache-budget bounds. Reserve true Mode A
  (client-local cache filling) for remote deployments.

## Decision log

- **Don't build a client prefetch for Mode B** — it would reinvent the server
  `PrecacheWorker` over the wire and pay a transfer it's trying to avoid.
- **Don't rely on preemption** — the `_io_lock` can't be interrupted mid-chunk;
  the answer is a separate prefetch reader, not a smarter lock.
- **Reason for deferral is ROI, not feasibility** — the right version needs a
  server reader pool, and the audience (a human watching an agent-driven viewer)
  is narrow. Record it this way so a future revisit starts from "add the
  prefetch reader," not "is this possible."
