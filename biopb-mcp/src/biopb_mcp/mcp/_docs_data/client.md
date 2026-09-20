---
kind: reference
description: The `client` handle: browsing the catalog, loading tensors, uploading results.
---

# The `client` Handle: Catalog and Tensor Data

Arrays from here are lazy dask arrays in the canonical `[..., Z, Y, X]` axis
order the server guarantees, and a tensor loaded into the viewer stops being
lazy — see [[data]] before moving pixels between the server, a layer,
and your own variables.

## Check Connection
```python
if client is None:
    print("Not connected. Open Tensor Browser widget and connect first.")
else:
    print(client.health_check())
```

## Browse Sources
```python
# Preferred: server-side DuckDB query (complete, not truncated).
# The sources table columns: source_id, source_url, source_type, dtype,
# indexed_at, metadata_json, shape_summary, is_resolved, and `tensors`
# (a LIST of STRUCT(array_id, dim_labels, shape, dtype) -- one per tensor;
# `dtype`/`shape_summary` are just the first-tensor projection).
# There is no residency column, and no catalog-wide way to ask: whether a
# source's bytes are local is a live filesystem check, so it is answered per
# source, for one you are about to read (see below).
# The catalog is structural: the transfer grid a tensor is delivered on is not
# stored here -- `client.get_descriptor(array_id).chunk_shape` answers it.
df = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'", format="pandas")
print(df)

# Per-tensor queries (multi-field / HCS sources): use the nested `tensors`
# column with UNNEST or list_filter -- the scalar dtype/shape_summary only
# describe tensors[0].
client.query_sources(  # every tensor, one row each
    "SELECT source_id, t.array_id, t.shape, t.dtype "
    "FROM sources, UNNEST(tensors) AS u(t)", format="pandas")
client.query_sources(  # sources having ANY uint16 tensor
    "SELECT source_id FROM sources "
    "WHERE len(list_filter(tensors, t -> t.dtype = 'uint16')) > 0", format="pandas")

# One row per source, with its tensors (query_sources is the only browse)
for row in client.query_sources(
    "SELECT source_id, source_url, source_type, tensors FROM sources",
    format="records"):
    tensors = [(t["array_id"], t["shape"], t["dtype"]) for t in row["tensors"]]
    print(f"{row['source_id']}: {row['source_url']} ({row['source_type']}) {tensors}")

# Detailed metadata (OME_JSON) for one source
meta = client.get_source_metadata("source_id")
print(meta)
```

## Cloud / unresolved sources (experimental)
Cloud / remote source support is **experimental** and may change.
Some sources (cloud / synced-folder, e.g. OneDrive "Files-On-Demand") are
catalogued by URL only: their shape/dtype/fields are *unknown* until first read.
They list with `is_resolved == False` and an empty `tensors`, and reading one
(`get_tensor`/`add_tensor`) raises until you resolve it. Resolving asks the
server to **download the whole file** (slow, uses disk, fails offline), so it is
explicit -- never triggered by browsing.
```python
row, = client.query_sources(
    "SELECT is_resolved FROM sources WHERE source_id = 'source_id'",
    format="records")
if not row["is_resolved"]:                   # never resolved
    src = client.resolve("source_id")        # downloads + resolves (may take minutes)
    tensors = [(t["array_id"], t["shape"]) for t in src["tensors"]]  # now populated
```
`is_resolved` is the column to ask; residency is not a column at all. The two
are different questions: `is_resolved` says the server has read this source's
structure and is monotonic (false to true once, never back), which is what lets
it live in a row. Whether the bytes are local is true only right now -- a synced
folder re-dehydrates under storage pressure -- so it is checked live, per
source, on the descriptor:
```python
client.get_descriptor(array_id, with_pyramid=False, with_residency=True).is_resident
```
Ask it of a source you are about to read, never in a loop over a listing: the
answer is a stat walk of the source, so asking it per row made a browse scale
with the catalog rather than with what you were looking at.

Don't cache what it returns, and don't read residency as resolution: an empty
`tensors` answers neither, since a source can resolve cleanly and hold nothing
readable.

Hydrate-ahead (optional): `resolve()` fetches a multi-file source's *metadata*
only -- the bulk data files (e.g. zarr/ome-zarr chunks) still recall one-by-one,
slowly, the first time a read touches them, which makes the first pass over a big
image stall repeatedly. If you're about to work through the whole source, warm it
up front so the server pulls every member file resident in one go (server-side;
no pixels cross to the kernel). It's idempotent and reports progress:
```python
done = client.warm("source_id",
                   on_progress=lambda p: print(f"{p.files_done}/{p.files_total}"))
# Long-running; interrupt_kernel cancels it (the stream closes, the server stops).
# Single-file sources are a no-op (resolve already recalled the one file).
```
Filter footgun: an unresolved source has NULL `dtype`/`shape_summary` in the
`sources` table, so `query_sources("... WHERE dtype='uint8'")` silently *drops*
it -- it's hidden for being unresolved, not for not matching. Filter on it on
purpose instead:
```python
# what hasn't been resolved (downloaded) yet?
client.query_sources("SELECT source_id, source_url FROM sources WHERE NOT is_resolved",
                     format="pandas")
```

## Load into Viewer
Arrays are referenced by their `array_id`: `"source_id/t1"` for a tensor within a
multi-tensor source, a bare `"source_id"` for a single-tensor one.
```python
# As a layer -- auto-handles the multiscale pyramid for large images.
layer_name = viewer.add_tensor("source_id")
layer_name = viewer.add_tensor("source_id/t1")

# Or as a lazy dask array, without adding a layer:
arr = client.get_tensor("source_id/t1")
```

## Upload to Server
Declare the tensor, then fill it. Use `"cache:my_result"` as destination for
ephemeral results that don't need to be persisted long-term.
```python
desc = client.create_tensor("cache:my_result", arr)   # shape, dtype, chunks from arr
client.upload_array(desc, arr)
array_id = desc.array_id
```
A name is taken while its source exists: creating under a name that already
exists is refused. Re-running a cell needs a new name, or `"cache:"` for a
server-minted one.
