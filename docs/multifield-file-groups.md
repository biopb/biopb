# Loose-File Groups as Multi-Field Sources (+ a Field-Group Aggregate RPC)

**Status:** Proposed (design). Nothing implemented yet.
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

- **`biopb-tensor-server`**
  - `adapters/tiff.py`, `adapters/dicom.py`: demote to multi-field group
    adapters — enumerate + order + assign stable filename-derived `array_id`s +
    emit N field `TensorDescriptor`s; drop single-tensor stacking and the
    `_looks_like_tiff_sequence` gate; simplify the claim to a count/extension test.
  - `server.py`: handle the Field-Group Aggregate request (endpoints for all
    fields / a synthesized aggregate tensor); optional server-side normalization.
  - `metadata_db.py`: optionally summarize fat sources in `list_source_descriptors`
    (lazy field listing).
- **`proto/biopb/tensor`**: `FieldGroupInfoRequest` / response message.
- **`biopb-mcp`**: "View all stacked" action; lazy browser child population;
  client method wrapping the aggregate RPC; guide/resource note documenting the
  group-source idiom for agents.

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
