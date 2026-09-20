---
kind: reference
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

**Residency is not a catalog column, and is not resolution.** `is_resolved` is
monotonic and lives in a row; whether the bytes are local is true only right now
— a synced folder re-dehydrates — so it is a live check on the descriptor, asked
of one source you are about to read and never in a loop over a listing:

```python
client.get_descriptor(array_id, with_pyramid=False, with_residency=True).is_resident
```

**Filter footgun:** an unresolved source has NULL `dtype`/`shape_summary`, so
`query_sources("... WHERE dtype='uint8'")` silently drops it — hidden for being
unresolved, not for failing to match. `WHERE NOT is_resolved` is how you ask for
them on purpose.

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
desc = client.create_tensor("cache:my_result", arr)
client.upload_array(desc, arr)
```

A name is taken while its source exists, so re-running a cell needs a new one or
a bare `"cache:"` for a server-minted name.
