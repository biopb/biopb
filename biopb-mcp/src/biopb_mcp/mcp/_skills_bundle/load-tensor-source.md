---
id: load-tensor-source
title: Load a tensor source into napari
description: Browse the tensor catalog, pick a source with the user, and load it as a napari image layer.
tags: [tensor, io, visualization]
version: 1.0.0
requires: [viewer, tensor]
---

# Load a tensor source into napari

**When to use.** The user wants to open image data that lives on a biopb tensor
server (the `client` catalog), rather than a file on disk. This is the usual first
step before any analysis.

## Steps

1. **Browse the catalog, don't guess.** Query server-side with DuckDB — it is
   complete, unlike `client.list_sources()`:

   ```python
   df = client.query_sources(
       "SELECT source_id, source_url, dtype, shape_summary, data_resident "
       "FROM sources ORDER BY indexed_at DESC LIMIT 50",
       format="pandas",
   )
   ```

   Unresolved cloud / synced-folder sources have NULL `dtype`/`shape_summary`; if
   the user is looking for something not yet resolved, filter on
   `WHERE NOT data_resident` rather than a dtype predicate that would drop them.

2. **Confirm the choice with the user.** Show the candidate rows (id, shape, dtype)
   and let them pick — do not assume which source they mean.

3. **Load lazily and add to the viewer** so the user can validate it:

   ```python
   arr = client.get_tensor(source_id)          # dask-backed, lazy
   layer = viewer.add_image(arr, name=source_id)
   ```

   For large tensors, prefer the multiscale path (see `guide://tensor`) so napari
   pages in only what the current view needs.

4. **Report** the layer name, shape, dtype, and axis order back to the user.

## Guardrails

- Use data from `client` / `viewer`; avoid the filesystem unless the user asks.
- Prefer lazy dask operations; only `.compute()` a final result.
- See `guide://tensor` for uploads and multiscale pyramids, `guide://viewer` for
  layer/dims/camera control.
