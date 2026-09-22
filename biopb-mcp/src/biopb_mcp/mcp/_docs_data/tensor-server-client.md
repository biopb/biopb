---
description: The tensor server through `client` — browsing the catalog, loading a tensor, and what lazy costs.
---

# The tensor server, through `client`

Arrays from here are lazy dask arrays in the canonical `[..., Z, Y, X]` axis
order the server guarantees. A tensor loaded onto the napari window is repackaged
for the renderer and is read back differently — [[napari-viewer]] has that half.

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
# The sources table columns: source_id, source_url, source_type,
# indexed_at, metadata_json, is_resolved, and `tensors`
# (a LIST of STRUCT(array_id, dim_labels, shape, dtype) -- one per tensor).
# Shape and dtype live in `tensors` only: a source can hold many, so there is
# no source-wide answer to ask for.
# The catalog is structural: the transfer grid a tensor is delivered on is not
# stored here -- `client.get_descriptor(array_id).chunk_shape` answers it.
df = client.query_sources("SELECT source_id FROM sources WHERE source_type='ome-zarr'", format="pandas")
print(df)

# Per-tensor queries (multi-field / HCS sources): use the nested `tensors`
# column with UNNEST or list_filter. `tensors[1]` is the source's first tensor,
# which for an image source is its picture (DuckDB lists are 1-indexed).
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

**Experimental — this surface is unstable and may change.** Keep to what is
below rather than building on the details.

Some sources are catalogued by URL only (cloud, or a synced folder like OneDrive
"Files-On-Demand"): their shape, dtype and field list are unknown until first
read. They list with `is_resolved == False` and an empty `tensors`, and
`get_tensor` on one raises.

```python
client.resolve("source_id")     # downloads the whole thing; minutes, disk, needs network
```

Resolving is deliberately explicit — browsing never triggers it — because it is
a full download. It returns the source's `sources` row, now populated. For a
multi-file source it fetches metadata only; `client.warm(source_id)` pulls the
member files resident up front, server-side, if you are about to read all of it.

**Resolved is not the same as local.** `is_resolved` says the server has read
the source's structure; it says nothing about where the bytes are, and a synced
folder re-dehydrates under storage pressure. Assume any cloud or synced-folder
source may need to fetch on first read — slow, and impossible offline — and plan
for it: warn the user before a long read rather than after it, and crop or warm
rather than reaching for the whole thing.

**Filter footgun:** an unresolved source has an empty `tensors`, so
`query_sources("... WHERE tensors[1].dtype = 'uint8'")` silently drops it —
hidden for being unresolved, not for failing to match. `WHERE NOT is_resolved`
is how you ask for them on purpose.

## Loading a tensor

Arrays are referenced by their `array_id`: `"source_id/t1"` for a tensor within a
multi-tensor source, a bare `"source_id"` for a single-tensor one.

```python
arr = client.get_tensor("source_id/t1")   # lazy dask array, nothing read yet

# Or straight onto the napari window, where the session has one -- it handles
# the pyramid, and returns the layer name ([[napari-viewer]]).
layer_name = viewer.add_tensor("source_id")
```

What you get is a **lazy dask array at full resolution, in canonical
`[..., Z, Y, X]` order** (`S` last for interleaved colour). The pyramid is the
server's; `get_tensor` gives you level 0 and `add_tensor` gives the viewer all
of it.

**Lazy means the bill arrives at the end.** `.shape` and `.dtype` are free while
the pixels are not there yet; a scikit-image call, `np.asarray`, or a `for` loop
over the array materializes all of it at once — unchunked, without progress, and
that is how a session allocates a volume it cannot hold. Crop first, keep the
chain lazy, `.compute()` once. Past `promote_after` that compute is a job you can
watch and cancel ([[kernel]]).

`client.get_descriptor(array_id)` is the only call that answers the transfer
`chunk_shape`; `client.get_physical_scale(array_id)` answers pixel size.

## Upload to Server

The write side is [[upload]] — tensors, label sets and ROI annotations, and what
each refuses. The short form:

```python
desc = client.add_tensor("zarr://scratch/@fields/my_result", arr)
client.upload_array(desc, arr)
```

An upload adds a tensor to a source that already exists and creates none, so a
result of your own goes on `scratch` — the temp store every writable server
serves at that fixed id. What lands there has a deadline (`desc.ttl_seconds` is
the one you got). A field is taken while its tensor is served, so re-running a
cell needs a new field name.
