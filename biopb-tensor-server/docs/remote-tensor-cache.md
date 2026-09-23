# Remote tensor server as a source type

Scope: `biopb-tensor-server` (adapter + config/reconcile).

A config entry whose `url` is `grpc://host:port` is a source like any other, of
type `tensor-server`: the local server mirrors that upstream's catalog and
re-serves its data from its own local segment cache, fetching upstream only on
a miss. Any number of upstreams may sit in one config alongside ordinary
local/cloud sources, all behind the one cache.

This buys two things. A read that already landed in the local file cache is a
local `chunk_locate` mmap hit shared by every reader on the box, instead of a
private in-RAM copy per worker; and the upstream only ever sees a request for a
chunk this server doesn't already have.

## Config surface

```json
{
  "sources": [
    { "url": "grpc://lab-store.internal:8815", "alias": "lab", "credentials_profile": "lab-store" },
    { "url": "grpc://archive.internal:8815",   "alias": "arc", "credentials_profile": "archive"   },
    { "url": "/data/scratch/" }
  ],
  "credentials": { "profiles": [
    { "name": "lab-store", "storage_type": "biopb-tensor", "token": "…bearer…" },
    { "name": "archive",   "storage_type": "biopb-tensor", "token": "…bearer…" }
  ]}
}
```

`type` is `"tensor-server"`, auto-detected from the `grpc://`/`grpc+tls://`/
`grpcs://` scheme, so it is usually left out.

- **`url = grpc://host:port`** mirrors every source on the upstream (the
  network analogue of `url = "/data/"` directory discovery).
- **`url = grpc://host:port/<upstream_source_id>`** mirrors a single upstream
  source. The path is the upstream `source_id`, slash-free by the `array_id`
  spec, so the first `/` after the authority cleanly splits endpoint from
  source; every expanded concrete source carries this shape.
- **`alias`** (optional, slash-free) namespaces this upstream's mirrored
  `source_id`s. Optional for a lone upstream, required once a collision is
  possible.

Upstream auth rides `credentials_profile`, a `storage_type="biopb-tensor"`
profile:

| Key | Meaning |
|---|---|
| `token` | Bearer token for the upstream. Beats the `BIOPB_UPSTREAM_TENSOR_TOKEN` env fallback (single-upstream convenience only). |
| `tls_fingerprint` | Expected SHA-256 of the upstream's cert, as `cert init` prints it. Verified on every connect. |
| `tls_ca_file` | Path to a PEM to trust — a private CA, or the upstream's own leaf. |

TLS trust is optional; unset means TOFU pinning, which already works with zero
config. Configuring an anchor buys the one thing TOFU cannot: rejecting an
impostor at *first* contact. If both keys are set, the CA wins and a warning is
logged, so an operator is never left believing an unenforced fingerprint is
protecting them. An unreadable `tls_ca_file` raises rather than silently
degrading to TOFU — see *Misconfiguration is not unreachability* below.

`resolve_upstream_credentials()` (`adapters/remote_tensor.py`) produces one
frozen, hashable `UpstreamCredentials` from the source + profile, and every
dial site — the adapter's pooled client, the reconciler's bulk catalog fetch,
and the bare-host expansion — uses it, so a `grpcs://` upstream's configured CA
is honored everywhere it is dialed, not just on the adapter's own connection.

## The adapter — a passthrough that understands nothing

`RemoteTensorAdapter` (`adapters/remote_tensor.py`) fronts one source on one
upstream, bound to `(upstream_location, upstream_source_id, local_source_id)`.
It is format- and chunking-agnostic: it decodes no pixels, derives no chunk
grid of its own, and treats the upstream's `chunk_id` as opaque. The only thing
it does beyond passthrough is rewrite `array_id`s between its local
(namespaced) space and the upstream's.

Dispatch needs no changes to support this: the server already routes `do_get`
by the local `source_id` prefix on the chunk_id to the registered adapter, so
each mirrored source's own `RemoteTensorAdapter` is picked automatically and
rewrites to its own upstream. Multiple upstreams and local sources coexist in
one flat `source_id`-keyed registry with no multiplexing layer.

The local server's chunk-read path (`TensorAdapter.resolve_chunk_data` wrapping
`get_data` in the segment cache, keyed by `chunk_id`) is shared by every
adapter, so the proxy inherits the persistent file cache, eviction, crash
recovery and the `chunk_locate` mmap fast path unchanged — it adds no caching
code of its own.

- **Catalog surface** (`list_tensor_descriptors`, `get_metadata`,
  `get_tensor_descriptor`) mirrors the upstream with `array_id` rewritten
  local-ward, and degrades to an empty placeholder rather than raising when the
  upstream is unreachable — see *Unreachable upstream* below.
- **Read planning** (`plan_flight_info`) forwards the whole `GetFlightInfo` to
  the upstream and localizes the response: only the upstream knows the grid,
  the pyramid, and the physical scale for a given (possibly scaled) read, so
  the proxy re-derives none of it and instead relays the caller's field mask
  and hints upstream verbatim. On an upstream failure it falls back to the
  inherited local planner — never worse than treating the mirror as an
  ordinary, ungridded source.
- **Chunk reads** (`resolve_chunk_data`) peel a **proxy envelope** off the
  served chunk_id, forward the inner — the upstream's own chunk_id, carried
  byte-for-byte, never decoded — to the upstream's `do_get`, and cache the
  result under the envelope's own key. Forwarding the *scaled* inner means the
  upstream does any downsampling, so only the small result crosses the
  network.
- **Writes are not forwarded.** The proxy is read-only: `add_tensor` and other
  write verbs are refused on a mirrored source, exactly as on the wire.

## Identifier policy

Local ids are namespaced so multiple upstreams and local sources can share one
flat, `source_id`-keyed catalog:

```
local source_id = <alias>__<upstream_source_id>          (slash-free)
local array_id  = <alias>__<upstream_source_id>[/<field>]
```

`__` is a cosmetic separator — nothing parses it back apart; each adapter
already stores its `(alias, upstream_source_id)` explicitly. The `array_id`
spec still holds: the prefix is slash-free, so `source_id =
array_id.split("/", 1)[0]` recovers it whole. Namespacing costs exactly one
`array_id` rewrite; everything after it in a `chunk_id` is untouched, so the
cache and the mmap fast path are unaffected. The namespace is flat rather than
nested per-upstream deliberately: it lets the rest of the stack (registry
dispatch, the metadata DB, `list_flights`, precache, `do_get` routing) work
unchanged. A lone upstream with no `alias` keeps its ids verbatim; a second
upstream, or a colliding local id, requires one.

### The endpoint is deliberately not in the id

A local id is built from `(alias, upstream_source_id)` alone — no host, port
or scheme. Moving an upstream (new port, new host, `grpc://` → `grpcs://`)
therefore changes only that source's `url`: its `source_id`, its `array_id`s,
and the route inside every `chunk_id` are untouched, so the persistent segment
cache stays warm and ROI annotations stay attached. Contrast a *local* source,
whose id hashes its path — there, moving the file re-keys everything.

The `alias` is what makes this possible: two upstreams offering the same
`upstream_source_id` must be told apart somehow, and the obvious discriminator,
`host:port`, would fold the volatile half of the address into the identity and
re-key the whole mirror on a move. The alias is a stable, human-chosen
stand-in for the endpoint instead.

**An alias is part of the data's identity, not a display label.** Renaming one
re-keys every source mirrored from that upstream: cached chunks orphan (their
route changed) and ROI annotations detach from their `source_id`, going
invisible to a `roi` read and ageing toward `prune_unseen_days`. Two
corollaries follow: a lone upstream with no alias keeps verbatim ids, so
*adding* an alias later is itself a rename — set one from the start if a
second upstream is ever likely — and a retired alias should never be reused
for a different upstream, since a coinciding `upstream_source_id` would
re-attach old rows to new data.

## Catalog mirroring, expansion & refresh

A `tensor-server` source expands like a directory. The single-source form
registers under its namespaced local id; the bare-host form connects,
enumerates the upstream's source ids, and yields one concrete single-source
entry per upstream source — each then registers a `RemoteTensorAdapter` under
its namespaced `source_id` and is treated like any other source from there.
The upstream's own **scratch** source is never mirrored: it is a temp store
whose tensors have a deadline set by that server's policy, not a catalog worth
carrying.

**Enumeration and seeding are one bulk query.** `fetch_upstream_catalog` reads
every upstream source's id, tensors, metadata, `is_resolved` and `indexed_at`
in a single server-side `query_sources`, which is not truncated (unlike
`list_sources()`), so mirroring costs one upstream RPC regardless of catalog
size and a re-list can safely remove sources that disappeared. An upstream
with no SQL catalog falls back to id-only enumeration, and removals are then
skipped — a truncated or degraded list must never be treated as a complete
one, or a re-list would drop sources it simply failed to see.

**Cache staleness is versioned, not open.** The upstream's `indexed_at`
becomes this mirror's `content_version`, folded into every chunk_id's proxy
envelope (`b"iat:<ts>"`). A chunk_id minted against a since-superseded
`content_version` is rejected before any I/O (`check_chunk_version`) rather
than served stale, so an upstream that re-registers a source invalidates the
proxy's cached chunks for it instead of leaking through them.

**Refresh via `monitor=true`.** For a bare-host upstream, `monitor=true`
generalizes the filesystem rescan into a periodic re-list: each upstream has
its own adaptive cadence, re-listing every rescan tick (default 30s) while
changing or failing, with the period doubling per unchanged re-list up to
about an hour. Any change or connectivity failure resets it back to
every-tick, so a new or recovered upstream is mirrored within about one tick.

**Misconfiguration is not unreachability.** A bad `credentials_profile`
(unreadable or empty `tls_ca_file`) fails identically on every re-list, so it
backs the upstream straight off to the slow cadence instead of burning the
fast one retrying a typo, and is reported once as a config error naming the
back-off. Fixing it is reported too, and resets the upstream to the fast
cadence — an operator who edits the config needs to see that the edit took,
not just silence.

**Unreachable upstream.** A proxy "resolve" is a cheap reconnect, not a cloud
download, so recovery is transparent — there is no unresolved-source consent
step. The catalog surface degrades to a placeholder (`list_tensor_descriptors`
/ `get_metadata` return empty, so registration's metadata-DB sync succeeds
with a row of no tensors) while the serve surface stays live and raises a
retryable error on a miss, dropping the dead upstream connection so the next
call reconnects. Already-cached chunks keep serving through an outage.

Connections to each upstream are pooled process-wide by `(endpoint,
credentials)`, so N sources mirrored from one upstream share one connection
rather than opening N.

## Local-tensor-server config editing

The server reads its config once at startup, so reconfiguring `sources`
(including proxied remotes) is: edit `~/.config/biopb/biopb.json` — through
`GET`/`PUT /api/config` on the tensor sidecar, or by hand — then restart the
data plane and reconnect. `GET /api/admin/status` reports scan progress
(`full_scan_in_progress`, `last_full_scan_finished_at`, climbing
`source_count`) so a client can wait out the post-restart discovery scan
instead of guessing when it's done.

## Known limits

- **Chained proxies** (a proxy mirroring another proxy) stack aliases
  (`a__b__source`) and work, but a misconfigured cycle has no depth cap yet.
- **A bare-host upstream that is down at boot** can't be expanded — no
  per-source ids are knowable until it answers — so its sources appear only
  once `monitor=true` re-lists it successfully; the server still boots rather
  than failing outright when the only configured source is an unreachable
  monitored upstream.
