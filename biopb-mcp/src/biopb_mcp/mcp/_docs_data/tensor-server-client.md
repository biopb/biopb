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

`client.location` is the address this client dials; `client.advertised_location` is
the one the server says it can be reached at from elsewhere, or `None` if it
published none.

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
df = client.query("SELECT source_id FROM sources WHERE source_type='ome-zarr'", format="pandas")
print(df)

# Per-tensor queries (multi-field / HCS sources): use the nested `tensors`
# column with UNNEST or list_filter. `tensors[1]` is the source's first tensor,
# which for an image source is its picture (DuckDB lists are 1-indexed).
client.query(  # every tensor, one row each
    "SELECT source_id, t.array_id, t.shape, t.dtype "
    "FROM sources, UNNEST(tensors) AS u(t)", format="pandas")
client.query(  # sources having ANY uint16 tensor
    "SELECT source_id FROM sources "
    "WHERE len(list_filter(tensors, t -> t.dtype = 'uint16')) > 0", format="pandas")

# One row per source, with its tensors (query is the only browse)
for row in client.query(
    "SELECT source_id, source_url, source_type, tensors FROM sources",
    format="records"):
    tensors = [(t["array_id"], t["shape"], t["dtype"]) for t in row["tensors"]]
    print(f"{row['source_id']}: {row['source_url']} ({row['source_type']}) {tensors}")

# Detailed metadata (OME_JSON) for one source
meta = client.get_source_metadata("source_id")
print(meta)
```

## Unresolved sources

A row with `is_resolved == False` has an empty `tensors`: its structure is not
known yet. `unresolved_reason` says why:

| `unresolved_reason` | what it is | what to do |
|---|---|---|
| *(resolved)* | the server has read its structure | read it |
| `pending` | found at startup, not read yet; nothing is downloaded | wait (`health_check()["registration_pending"]` counts what is left; `0` is a whole catalog), or read it: `get_tensor`, `get_descriptor` or `resolve_source` registers it at once |
| `needs_recall` | cloud / synced-folder placeholder (OneDrive Files-On-Demand): shape and dtype unknown until read | `client.resolve_source(id)` downloads the whole file, so ask first; it returns the populated row |
| `failed` | could not be read (a corrupt file, say) | `resolve_source` raises with the reason |

- **A structural filter hides them.** `WHERE tensors[1].dtype = 'uint8'` drops
  every unresolved source silently. While `registration_pending > 0`, an empty
  result means "not known yet", not "none". `WHERE NOT is_resolved` asks for them
  on purpose.
- **Resolved is not local.** It says the server has read the structure, not where
  the bytes are, and a synced folder re-dehydrates under storage pressure. Assume a
  cloud source may fetch on first read, slow and impossible offline: warn before a
  long read and crop instead of reaching for the whole thing.
- Resolving is explicit and browsing never triggers it; for a multi-file source it
  fetches metadata only, and the member files recall as they are read. Cloud
  support is experimental; keep to what is here.

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

## Writing back

[[upload]] covers tensors, label sets and ROI annotations, and what each refuses.
