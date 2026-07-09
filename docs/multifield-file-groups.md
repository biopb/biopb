# Loose-File Groups as Multi-Field Sources (+ a Field-Group Aggregate RPC)

**Status:** Proposed (design). Nothing implemented yet. Two candidate mechanisms
for the aggregate view are documented: a new [Field-Group Aggregate RPC](#the-field-group-aggregate-rpc-core-protocol-addition)
and a lighter [virtual aggregate tensor](#alternative-implementation-the-aggregate-as-a-virtual-tensor-no-new-rpc)
that reuses the existing `GetFlightInfo` path (recommended for the homogeneous
loose-file case).
**Component:** `biopb-tensor-server` (adapters, metadata DB, Flight protocol),
`biopb-mcp` (tensor browser UX), `proto/biopb/tensor` (new request/response
message).
**Related:** `TiffSequenceAdapter` / `DicomSeriesAdapter` (`adapters/tiff.py`,
`adapters/dicom.py`), the multi-field/HCS machinery, the `array_id`-is-authoritative
policy (`proto/biopb/tensor/descriptor.proto`), `docs/cloud-storage-support.md`
(multi-file content-membership formats already degrade to N single-file sources
under cloud), the localhost `chunk_locate` mmap fast path (`#9`).

---

## Problem

The tensor server ships two adapters whose job is to stack a *directory of loose
image files* into one multidimensional tensor:

- **`TiffSequenceAdapter`** (`adapters/tiff.py:245`, registered `tiff-sequence`) —
  claims a directory of plain TIFFs via a **filename-coherence heuristic**
  (`_looks_like_tiff_sequence`, `tiff.py:126`: shared digit-run masks or a common
  stem), natural-sorts them (`_natural_key`, `tiff.py:84`), buckets by page count,
  and stacks the dominant bucket along an **opaque `i` axis**. It deliberately does
  **not** infer whether `i` is Z/T/C/position — it exposes the filename list via
  `get_metadata()` and leaves semantics to the caller.
- **`DicomSeriesAdapter`** (`adapters/dicom.py:407`, registered `dicom-series`) —
  groups files by `SeriesInstanceUID` (`dicom.py:470-482`) and sorts slices by
  `InstanceNumber` → `SliceLocation` → `ImagePositionPatient[z]`
  (`dicom.py:592-601`).

Two problems motivate a redesign:

1. **Heuristic/semantic mismatch.** The TIFF claim gate is a guess about *which*
   loose files form a set, and the stacking asserts an axis whose meaning the
   server cannot know. Coding that deterministically on the server is a mismatch:
   the entity best placed to decide "these files belong together and axis `i` is
   really the time dimension" is the **agent** (biopb-mcp), which has the analysis
   context. The server already concedes half of this (the `i` axis is opaque); the
   redesign finishes the split.

2. **We still want the directory to be one catalog entry.** The naive fix —
   "delete the adapters, let every file be its own source" — is verified to work
   mechanically (a lone plain `.tif` is claimed by `AicsImageIoAdapter`, since
   `.tif` is in `MICROSCOPY_EXTENSIONS` (`adapters/aicsimageio.py:382`) and
   `OmeTiffAdapter` returns `None` for non-OME tiffs (`aicsimageio.py:1229`)) but
   **explodes the catalog**: a 1,000-file acquisition becomes 1,000 sources,
   1,000 DuckDB rows, 1,000 `ListFlights` entries, and a per-file rescan storm.
   The tensor browser would be swamped — that directory becomes effectively the
   only thing it shows.

The design goal: **remove the heuristic without exploding the catalog, and keep
the data plane a complete I/O layer that serves every client (napari, web, Java),
not just the Python agent.**

## The model: a directory is one multi-field source

Model a directory of loose files as **one source that exposes N member tensors
("fields")**, reusing the existing multi-field/HCS machinery rather than
inventing anything:

- One source (`source_id` = the directory), N fields, `array_id = source_id/<field>`.
- The browser already renders this as a single collapsible node
  `"<name>  [N tensors]"` with expandable per-field children
  (`tensor_browser/_widget.py:1045-1068`).
- The catalog stays **one row per directory**, so the row-count-keyed protections
  that already exist (the `ListFlights` source cap at `server.py:948`, the
  browser's `>1000`-source server-query threshold) are not defeated.
- Because the directory remains the source, rescan change-detection keys on the
  **directory signature** (mtime), not per-file opens — so there is no
  steady-state rescan storm. This is the decisive advantage over "N independent
  sources," which would `stat`/open every file every rescan.

The two adapters are therefore **demoted, not deleted**:

| Kept (mechanical, deterministic)                     | Removed (heuristic / semantic)                          |
|------------------------------------------------------|---------------------------------------------------------|
| File enumeration; `_natural_key` / DICOM slice sort  | Single-tensor stacking with an asserted `i` axis        |
| Per-file lazy `get_data`, tile reads, per-file locks | `_looks_like_tiff_sequence` filename-coherence gate     |
| Per-file metadata (`InstanceNumber`, filename, …)    | Page-count bucketing to pick a "dominant" stack         |

The claim gate simplifies to something dumb and predictable, e.g. *"a directory
with ≥ N image files of a single extension is a group source."* Each file becomes
one field emitted as its own `TensorDescriptor`.

### What moves to the client

Stacking N fields into one viewable/analysable array becomes an explicit,
context-aware action above the data plane:

- **Agent:** enumerates fields with `UNNEST(tensors)` in a `query_sources` SQL,
  reads the ones it wants, and combines them with `da.stack` — assigning axis
  semantics itself.
- **Human, in napari:** the layer-list context action `napari.layer.merge_stack`
  ("Merge to stack") already exists and is lazy-preserving
  (`napari.layers.utils.stack_utils.images_to_stack`, verified dask-in→dask-out in
  napari 0.7).
- **Tensor browser:** a new **"View all stacked"** menu action beside the existing
  **"View all"** (`_view_all_tensors`, `_widget.py:1512`; menu `:1242`) — see
  [Client / UX](#client--ux).

## Ordering & identity policy

Ordering is **established once, at adapter `__init__`, and is canonical
everywhere.** The `tensors[]` list in the descriptor (and in the DuckDB
`tensors STRUCT[]`) is stored in that order and every consumer respects it — the
browser tree, "View all [stacked]", the aggregate RPC, and the agent. There is
**no per-request re-ordering**: a per-request `order_by` would let the stacked
axis disagree with the order the browser shows, confusing the user about which
field is at which index.

This keeps the DICOM ordering exactly where it already lives
(`DicomSeriesAdapter.__init__` sorts by `InstanceNumber`/`SliceLocation`) and
round-trips for free — the DuckDB list is ordered and `list_source_descriptors`
rebuilds `tensor_descs` by iterating it in stored order (`metadata_db.py:531`).

Two invariants make "respected everywhere" actually hold:

1. **The order must be total and deterministic across restarts.** Natural sort
   needs a stable tiebreak; DICOM needs a fallback when `InstanceNumber` collides
   or is absent (e.g. `SOPInstanceUID`). Otherwise two discovery passes could
   produce different "canonical" orders, and a persisted DB order could disagree
   with a freshly rebuilt adapter after a restart — a silent correctness bug.

2. **Field identity is decoupled from sort position.** A field's `array_id` is
   derived from a **stable key (the filename)** — `source_id/img_0042.tif`, never
   `source_id/field_7`. Inserting a file in the middle reorders the `tensors[]`
   list but **renames nothing**, so caches, the mmap fast path, and any
   agent-held `array_id` survive the common "someone dropped another TIFF in the
   folder" case. Order is a property of the list; identity is a property of the
   member; the two move independently.

## Scaling analysis: pushing multi-field ~100× past its design point

The multi-field machinery assumes a source has a handful-to-dozens of tensors
(the HCS case). Loose-file sequences routinely span **thousands** of files, which
stresses three places — none guarded by the existing caps, because every current
protection counts **sources**, and this design deliberately keeps source count
low. The cost moves *inside* a source.

Verified against the code:

1. **O(N) sequential `GetFlightInfo` at view construction — the one that
   matters.** Every `client.get_tensor` issues one blocking `GetFlightInfo` at
   dask-graph-*build* time, sequential and **not cached** (endpoints depend on the
   slice/scale request, so the client's descriptor/shape caches don't hold them;
   `client.py:1811`). `build_pyramid_levels` adds a per-level multiplier
   (`_tensor_utils.py:105-272`), so a naive "load all fields" pays **O(N·(L+1))**
   sequential round-trips *before any pixels move*. A 10,000-file directory is
   ~10,000+ RPCs: seconds-to-tens-of-seconds on localhost, minutes against a
   remote server. Today's single stacked tensor pays **one** `GetFlightInfo`.

   **Scope:** this bites **only** the interactive "view-all-at-once" path where a
   human waits and O(1) was the prior baseline. Viewing a single field is one
   RPC. The agent's compute path is unaffected — the N metadata calls are noise
   against O(N × chunk-bytes) of actual pixel reads.

2. **Fat `ListFlights` descriptor + catalog row.** `list_source_descriptors`
   embeds the **full** N-tensor list in the source descriptor
   (`metadata_db.py:531-545`); ListFlights returns it on every browser refresh /
   catalog sync (~100 bytes × N ≈ 1 MB per 10k-field dir), and the DB stores a
   10k-element `STRUCT[]` per row. Not fatal, but it makes a "list" call fat and
   the source cap does not touch it.

3. **Browser eager child creation.** `_populate_tree` builds a `QTreeWidgetItem`
   for *every* field up front, even while the node is collapsed
   (`_widget.py:1064`).

**What does not regress** (for calibration): server-side **file opens are O(1)** —
a single-file source is opened once at registration and per-field scene adapters
are lazily built and cached, with no re-open per `GetFlightInfo`
(`aicsimageio.py:604`, `:1081`); **rescan** stays cheap (directory signature);
**discovery** already reads all N files once today (page-count bucketing);
**display** after construction is unchanged (scrubbing pulls one field/chunk at a
time).

**Calibration before committing:** the same machinery already backs HCS plates.
Load the *largest* HCS plate available (hundreds–thousands of fields) through the
current browser and "View all" and measure — that is the empirical ceiling this
design inherits.

## The Field-Group Aggregate RPC (core protocol addition)

The clean fix for roadblock (1) is a **new Flight request that returns all fields
of a source in one round-trip, with endpoints** — simultaneously an optimization
*and* an intention channel. Signalling "I want the whole set at once" is what lets
the server do something it structurally cannot do per-field: **normalize
dtype/shape across the set**, because only now does it see the whole set and a
well-defined common grid.

This supersedes an earlier idea (persisting a synthesized "stack" tensor as an
extra catalog entry). A **request-time** aggregate is cleaner: the catalog stays a
pure inventory of N as-is fields; the stacked/normalized view is a transient
projection computed only when asked; nothing extra is stored.

> **A lighter alternative needs no new message at all** — model the aggregate as a
> *virtual tensor* fetched through the existing `GetFlightInfo`, because the
> default response shape (a) below is exactly what that verb already returns. It
> covers the stacked-cube case (facet 2) but not lazy field listing (facet 1); see
> [Alternative implementation](#alternative-implementation-the-aggregate-as-a-virtual-tensor-no-new-rpc).

### Request / response

Request (a new CMD proto carried on `get_flight_info`, or a `do_action` type —
biopb already extends `do_action` with custom types):

```
FieldGroupInfoRequest {
  string   source_id      // the directory group source
  repeated string fields  // optional subset; empty = all, in canonical order
  SliceHint  slice_hint   // applied uniformly to every field
  ScaleVector scale_hint  // applied uniformly to every field
  NormalizePolicy normalize  // NONE | DTYPE | DTYPE_AND_PAD
}
```

Response — two supported shapes, default **(a)**:

- **(a) One synthesized aggregate tensor** — a single `(i, Y, X)` (or
  `(i, Z, Y, X)`) `TensorDescriptor` **plus its chunk endpoints**, where index `i`
  walks the fields in canonical order. The client builds **one** lazy dask array;
  ideal for the "one stacked layer" UX. The `i` axis is opaque — semantics are the
  caller's to assign.
- **(b) N conformable per-field descriptors + endpoints** in one message — the
  client `da.stack`s (or subsets/reorders) them. More flexible for agents.

**Critical:** the response must carry **chunk endpoints**, not just descriptors —
otherwise the O(N) merely moves from `GetFlightInfo` to the do_get-planning stage.
Endpoints depend on slice/scale, so the request carries those hints and the server
applies them uniformly.

**No `order_by`** — fields come back in the canonical descriptor order
([policy above](#ordering--identity-policy)).

### Normalization policy (explicit, opt-in, conservative)

Only meaningful on an aggregate request. The server *may*:

- `DTYPE`: promote all fields to the widest common dtype (lossless).
- `DTYPE_AND_PAD`: additionally zero-pad each field's spatial extent to the group
  max (the old `#198` behavior, resurrected — but now explicit and scoped, not a
  silent always-on adapter side effect).
- If ranks or dim-labels differ incompatibly, **decline** aggregation and return
  ragged/per-field rather than guessing.

### Chunk identity & caching

Define the aggregate `chunk_id` so that in the **homogeneous / `normalize=NONE`**
case it **aliases each field's own chunk** (byte-identical), reusing the existing
per-field cache and the POSIX-localhost mmap fast path. When normalization *is*
applied, the padded/promoted bytes genuinely differ and get distinct derived cache
entries. Getting this right prevents aggregate reads from cold-missing a cache the
per-field path already warmed.

### This RPC also retires roadblock (2)

The same "give me source X's fields in one RPC" primitive is exactly the
mechanism for **lazy field listing**: `ListFlights` can then summarize a fat
source (`source X: N fields`, omitting the full list above a threshold) and the
client fetches the member list via this call on expand — keeping `ListFlights`
O(sources), not O(total fields). That leaves only roadblock (3) (browser Qt
items) as a pure client-side fix.

### Back-compatibility

Purely additive: a client or server without the new RPC falls back to N×
`GetFlightInfo`, so nothing breaks during rollout. New proto message → buf regen
across Python/Java/TS.

## Alternative implementation: the aggregate as a virtual tensor (no new RPC)

The Field-Group Aggregate RPC above is the maximal design. A lighter alternative
delivers its **default response shape (a)** — one synthesized `(i, Y, X)`
`TensorDescriptor` **plus chunk endpoints**, with slice/scale hints applied
uniformly — **without any new proto message**, because that response is *exactly*
what `GetFlightInfo` already returns and the hints already live on
`TensorReadOption`. Model the aggregate as a **virtual tensor** fetched through the
existing read path.

### A reserved field token, not an overloaded `source_id`

Address the aggregate as its own `array_id = source_id/<reserved>` (e.g.
`source_id/@all`), **not** by overloading `tensor_id == source_id`. The overload is
already taken: `_field_within_source` (`server.py:389`) maps `tensor_id ==
source_id` (and empty) to the source's **default/first tensor** (#44). Redefining
it to "the stacked aggregate" both breaks that back-compat contract and is actively
hazardous for the *heterogeneous* multi-field sources — an HCS plate or a CZI
scene-set would try to fuse thousands of incompatible fields, precisely what this
design says must **not** be stacked.

The reserved token sidesteps all of it:

- `tensor_id == source_id` keeps its #44 first-field meaning — no back-compat break.
- The aggregate is **virtual**: it is *not* an entry in `tensors[]` / the DuckDB
  `STRUCT[]` (so `ListFlights` and the catalog stay a pure inventory of N as-is
  fields — the "nothing extra stored" property is preserved), but
  `get_tensor_adapter("@all")` synthesizes an aggregate adapter on demand.
- Every client reads it with code it already has — `client.get_tensor("dir/@all")`
  — so there is **zero** new client method in Python/Java/TS. (A new RPC needs one
  per language.)
- The token must use a byte no field key can produce; filename-derived fields never
  yield it (`_natural_key` output), and it is documented in the `array_id` spec
  (below).

### Chunk aliasing is free in the `NONE` tier

Define the aggregate `chunk_id` as **field k's own `chunk_id`**. Then
`GetFlightInfo` for `dir/@all` merely concatenates each field's endpoints, and
**`DoGet` needs no aggregate code path at all**: the existing
`_get_adapter_for_chunk(chunk_id)` routes each chunk straight to its per-field
adapter, reusing the warm per-field cache and the POSIX-localhost mmap fast path
(#9). The synthetic aggregate adapter exists only for the duration of endpoint
enumeration in `GetFlightInfo`. This is the doc's "aggregate `chunk_id` aliases each
field's chunk" property, obtained for free rather than engineered.

The **normalized** tiers (`DTYPE` / `DTYPE_AND_PAD`) genuinely produce different
bytes (promoted / padded), so those *do* need a real synthetic read path at `DoGet`
with their own derived chunk ids. Ship the `NONE` endpoint-aliasing tier first;
treat normalization as a phase-2 follow-on.

### Scope: this covers facet 2, not facet 1

The RPC secretly bundles two needs; the virtual-tensor form serves only one:

| | What it is | One `GetFlightInfo`? |
|---|---|---|
| **Facet 2 — aggregate/stack** | one synthesized `(i, Y, X)` tensor + endpoints (homogeneous cube) | **Yes** — it *is* one tensor, one descriptor, one schema |
| **Facet 1 — batch / lazy listing** | N *heterogeneous* per-field descriptors in one message; retire the fat `ListFlights` (roadblocks 2/3) | **No** — a `FlightInfo` carries one schema; N differently-shaped descriptors don't fit |

So the virtual tensor **does not retire roadblock (2)** — that is the price of
avoiding the new message. It is an acceptable cut: the loose-file-group problem is
*homogeneous*, so facet 2 covers it completely, and lazy field-listing for fat
*heterogeneous* sources (HCS / scenes, whose O(N) pain already ships) becomes
separate, later work — and if that ever needs a wire change, that is the place to
spend one.

### Capability signalling: advertise, don't probe the error channel

"Optimistically request `dir/@all`, catch the error, fall back to View-all" is
**unsafe today** — the illegal-field path is inconsistent, and its most common
branch is a silent false-positive:

| Source kind | `get_tensor_adapter("@all")` | Client sees |
|---|---|---|
| **Single-tensor** (`ZarrAdapter` / base) | ignores the arg, `return self` | **No error** — returns the base tensor; `descriptor.array_id == "dir"`, not `dir/@all` |
| **HCS OME-Zarr** | `raise ValueError("Unknown well: @all")` | generic internal error, message-only |
| **OME-TIFF / bioio scenes** | `_scene_index_for_field` → `raise ValueError("Unknown scene")` | generic internal error, message-only |
| Missing **source** | `_get_adapter_for_tensor` → `None` | clean `FlightServerError("Tensor not found")` (`server.py:1106`) |

The only reliably catchable error (`FlightServerError("Tensor not found")`) is the
path a *field* miss almost never takes; the single-tensor case doesn't error at
all. So capability must be a **positive signal**, two ways:

- **Option A — harden the not-found path, then ride it.** Add an aggregate
  chokepoint in `get_flight_info` *before* `_field_within_source`: `field ==
  RESERVED_TOKEN` → `source_adapter.get_aggregate_adapter()`, returning `None` when
  unsupported → the existing `FlightServerError("Tensor not found")`. No proto
  change, but it still distinguishes "no aggregate" from other failures by
  message-matching, and it forces fixing the single-tensor silent-self and the
  HCS/scene raw-`ValueError` leaks so a stray `@all` can't slip through — delicate
  against the #44 rules (which intentionally let source_id / bare / empty resolve to
  the default tensor; only a genuinely unknown *nonempty* field should 404, and
  nothing enforces that today).
- **Option B (recommended) — capability bit.** Add `supports_aggregate` (bool) to
  `DataSourceDescriptor`, set by a group adapter that can stack, carried on the
  `ListFlights` / `GetFlightInfo` the client already fetches. The browser enables
  "View all stacked" only when set; the agent reads it via `query_sources`. The
  client never sends a doomed `@all`, so the fragile error semantics never enter the
  picture. Cost: one additive proto *field* (minor buf-regen) — far less than a new
  message, and a clean positive signal instead of inferring capability from whether
  an error came back.

  Two richer variants were considered and rejected. **Listing the aggregate as an
  `@all` entry in `tensors[]`** is self-describing (carries the `(N, Y, X)` shape)
  but pollutes the member list: `tensors[]` means "the real, independently-
  addressable members" everywhere it is consumed (DuckDB `STRUCT[]`, the agent's
  `UNNEST(tensors)` / `len(tensors)`, the browser's per-field children, "add all as
  separate layers"), so a folded-over projection sitting among them forces every
  reader to know and *filter* the reserved token — a silent-if-forgotten
  over-count, exactly the failure class of #378. **A separate self-describing
  `aggregate` `TensorDescriptor` field** avoids that pollution but must compute the
  aggregate descriptor at *list* time and hands readers another descriptor-shaped
  thing to reason about. The **bool** carries the one bit that discovery needs
  ("this source can stack"), pollutes nothing, and is ignorable by any consumer
  that does not care; the aggregate's shape is one `GetFlightInfo` on `<src>/@all`
  away — on the exact path the client was about to take anyway.

### Identity-spec note

The reserved token is a new clause in the `array_id` policy
(`proto/biopb/tensor/descriptor.proto`, top-of-file block) — **not** a proto
*message* change, but a policy addition: "`source_id/<reserved>` names a
**transient aggregate projection**, not a stored field." It must be written there
for the same reason the first-`/` source boundary rule is.

### How the two approaches compare

| | Field-Group Aggregate RPC | Virtual aggregate tensor |
|---|---|---|
| New proto **message** | yes (`FieldGroupInfoRequest` + response) | no |
| New per-language client method | yes (Python / Java / TS) | no — reuses `get_tensor` |
| Proto change | new message | one additive descriptor field (Option B) or none (Option A) |
| Facet 2 (stacked cube) | yes | yes |
| Facet 1 (lazy listing of N heterogeneous fields) | yes | **no** — stays future work |
| Subset selection (`fields=[…]`) | yes | no — agent does `N reads + da.stack` |
| Explicit `NormalizePolicy` on the wire | yes | encoded in token / adapter-decided |
| Chunk aliasing (`NONE`) | must be engineered | falls out of per-field `chunk_id` |
| Capability signalling | in the request/response | needs the `supports_aggregate` bit |

**Recommendation:** if the immediate goal is the loose-file-group cube, prefer the
**virtual aggregate tensor with the `supports_aggregate` bit** (facet 2, `NONE`
tier first) — it is the smaller, lower-coupling change and needs no new message or
client code. Adopt the full RPC only when facet 1 (lazy listing of fat
*heterogeneous* sources) becomes the driving requirement.

## Cross-adapter impact: two orthogonal concerns

This redesign carries two separable concerns, and they touch different adapters.

**Concern A — heuristic grouping (should the adapter be demoted?).** Only
`TiffSequenceAdapter` guesses membership from loose filenames. Every other
directory/multi-file adapter groups from an **authoritative manifest** and is
*not* a demotion candidate:

- MicroManager — `metadata.txt` Coords map (`tiff.py:945-964`); the filename glob
  only confirms the dataset exists.
- NDTiff — the `NDTiff.index` binary index (`ndtiff.py:154`).
- OME-Zarr (regular and HCS) — multiscales / plate+well `.zattrs`.
- aicsimageio and multi-file OME-TIFF — embedded OME-XML / reader scene list
  (`aicsimageio.py:281`, `:1261`).

(DICOM-series groups by the standard-defined `SeriesInstanceUID`, so it is
borderline rather than heuristic.)

**Concern B — multi-field scaling (many fields → O(N) costs).** This is **not
introduced by this redesign — it already exists today** in any adapter that emits
many tensors. The field-vs-axis modeling decides who is affected:

| Adapter | Model | Typical field count | Hit by scaling concern? |
|---|---|---|---|
| MicroManager, NDTiff | **one tensor, position = axis** (`_single_tensor_source`) | 1 | No — immune by construction |
| OME-Zarr regular image | one tensor | 1 | No |
| **OME-Zarr HCS plate** (`ome_zarr.py:432`) | many tensors (well × FOV) | hundreds–**thousands** | **Yes, today** |
| **aicsimageio scenes** (CZI/LIF/ND2) (`aicsimageio.py:520`) | many tensors (scene) | tens–**thousands** | **Yes, today** |
| tiff-sequence (after this redesign) | many tensors (per file) | thousands | Yes (new member of the class) |

So the Field-Group Aggregate RPC, lazy field listing, and lazy browser population
are **general multi-field infrastructure**, not sequence-specific. **HCS OME-Zarr
is the pre-existing worst case** — hundreds-to-thousands of fields *and* it opens
each field's level-0 zarr array to describe it (`ome_zarr.py:597`), so its
discovery is already O(N) per-field opens on top of O(N) view-all round-trips. The
HCS calibration this doc recommends is measuring pain that already ships.

**The field-vs-axis lesson.** When grouping is authoritative *and* members are
homogeneous on one coordinate grid, collapse them into an **axis of one tensor**
(MicroManager/NDTiff's `p` position axis) — O(1) descriptors, "view all" is free.
Use **fields** when members are heterogeneous / independently addressable (HCS
wells, scenes). `TiffSequenceAdapter` today asserts an `i` axis *without* the
authority to know the files are homogeneous — that is the mismatch. Moving it to
fields is correct for heuristic loose files and is what puts it in the N-fields
scaling class; the aggregate RPC (`DTYPE_AND_PAD`) then reconstructs the
single-tensor "cube" view on demand.

**Consequence for the RPC's normalize policy** — the `NormalizePolicy` switch is
what makes one RPC serve both classes:

- **`NONE`** — pure batch endpoint/descriptor fetch + lazy listing. What **HCS and
  vendor-scene** sources want: heterogeneous fields that should *not* be stacked,
  but still need fast browse/expand and fast "add all as separate layers."
- **`DTYPE` / `DTYPE_AND_PAD`** — normalize + synthesized stacked tensor. What
  **homogeneous sequences** want (the cube). Heterogeneous sources request `NONE`,
  or the server declines normalization.

Facet 1 (batch fetch) is universal multi-field infra; facet 2 (normalize/stack)
is opt-in for the homogeneous case.

## Client / UX

- **"View all stacked" menu action** in the tensor browser, beside "View all"
  (`tensor_browser/_widget.py:1242`). It should build the stack **lazily and
  directly** — one Field-Group Aggregate RPC → one lazy dask array →
  `viewer.add_image(..., multiscale=True)` — rather than adding N layers and
  merging (which instantiates N napari layers). When the server declines
  aggregation (heterogeneous), fall back to "View all" (N separate layers). The
  new leading axis is unlabeled by design; the human/agent assigns its meaning.
- **Lazy browser child population** (roadblock 3): populate a fat source's field
  children on expand, not eagerly at tree build.
- **napari `images_to_stack`** remains the right tool for the *human* who already
  has separate layers open and hits "Merge to stack"; the menu action is the
  build-in-one-shot path for large N.

## What changes, by component

The demotion of the loose-file adapters is common to both plans; the two differ
only in **how the aggregate view is served** (RPC vs virtual tensor).

**Common to both plans:**

- **`biopb-tensor-server`** — `adapters/tiff.py`, `adapters/dicom.py`: demote to
  multi-field group adapters — enumerate + order + assign stable filename-derived
  `array_id`s + emit N field `TensorDescriptor`s; drop single-tensor stacking and
  the `_looks_like_tiff_sequence` gate; simplify the claim to a count/extension
  test.
- **`biopb-mcp`** — "View all stacked" action; lazy browser child population;
  guide/resource note documenting the group-source idiom for agents.

**Plan 1 — Field-Group Aggregate RPC:**

- `server.py`: handle the aggregate request (endpoints for all fields / a
  synthesized aggregate tensor); optional server-side normalization.
- `metadata_db.py`: optionally summarize fat sources in `list_source_descriptors`
  (lazy field listing, facet 1).
- **`proto/biopb/tensor`**: `FieldGroupInfoRequest` / response message.
- **`biopb-mcp`**: client method wrapping the aggregate RPC.

**Plan 2 — virtual aggregate tensor (recommended for the homogeneous case):**

- `server.py`: a reserved-token dispatch in `get_flight_info` *before*
  `_field_within_source` → `source_adapter.get_aggregate_adapter()` synthesizing the
  aggregate descriptor + endpoints (aliasing each field's `chunk_id` in the `NONE`
  tier, so `DoGet` is untouched); `None` when unsupported.
- **`proto/biopb/tensor`**: one additive `DataSourceDescriptor.supports_aggregate`
  bool (Option B) — no new message; plus the reserved-token clause in the `array_id`
  spec comment.
- **`biopb-mcp`**: no new client method — "View all stacked" reads
  `supports_aggregate` and calls the existing `client.get_tensor("<source>/@all")`.
  Facet 1 (lazy listing) is **not** addressed by this plan.

## Open questions / non-goals

- **Cloud/unresolved sources.** Enumerating members is a content read, so an
  unresolved (dehydrated) directory cannot populate its field list until resolved
  — the same limitation the cloud path already documents
  (`docs/cloud-storage-support.md`: multi-file content-membership formats degrade
  to N single-file sources under cloud). No worse than today; not solved here.
- **A file that is itself multi-field** (a multi-scene loose `.tif`) collapses to
  one field showing its first series. Judged rare and acceptable; the escape hatch
  already exists in the `array_id` spec (the field part may contain `/`, so such a
  file could expand to `source_id/file.tif/scene0`) but is not built until needed.
- **Normalization semantics** for `DTYPE_AND_PAD` beyond spatial pad (crop vs pad,
  fill value, differing rank) need a precise spec.
- **Non-goal:** inferring axis semantics on the server. The `i` axis is opaque by
  design; meaning is assigned above the data plane.
