# napari `convert_to_uint8` and non-native byte order — fixed upstream

Status: **fixed upstream, workaround still in place.** Filed as
[napari#9144](https://github.com/napari/napari/issues/9144), fixed by
[napari#9345](https://github.com/napari/napari/pull/9345) (merged 2026-08-06),
released in **napari 0.9.0**. `biopb-mcp` pins `napari[all]==0.7.0`, so the
consumer-side workaround in `_tensor_utils.py` is still required here. Tracks
biopb/biopb#296.

## The bug

`napari.layers.utils.layer_utils.convert_to_uint8` raised `TypeError` on any
**non-native-endian** integer or float array (e.g. a big-endian `>i2` / `>f4`
FITS image). Every layer add goes through it via the thumbnail path
(`Image._update_thumbnail`), so a big-endian source could not be displayed at
all.

Root cause: two sites passed a **byte-order-qualified** dtype to a ufunc's
`dtype=` argument, which NumPy only allows to select the *general* DType:

```python
image_out = np.multiply(data, out_max, dtype=data.dtype)   # '>f4' -> TypeError
np.maximum(data, 0, out=data, dtype=data.dtype)            # '>i2' -> TypeError
```

## What landed

Upstream took the alternative this draft had listed as also-correct: **drop
`dtype=` entirely** rather than pass `data.dtype.type`. The `out=` array already
pins the dtype for `np.maximum`, and `np.multiply`'s result feeds
`np.rint(..., out=image_out)`, so the compute width is unchanged.

```python
image_out = np.multiply(data, out_max)
np.maximum(data, 0, out=data)
```

## Removing the workaround

Until the pin moves, `biopb-mcp` normalizes each pyramid level to native byte
order before `add_image` — `_to_native_byteorder` (`_tensor_utils.py:419`,
called at `:487`), a lazy `astype`; the wire/source bytes stay faithful per the
#293 binary schema.

When `napari[all]==0.7.0` in `biopb-mcp/pyproject.toml` is raised to 0.9.0 or
later, delete in one change:

- `_to_native_byteorder` and its call site in `_tensor_utils.py`;
- `TestToNativeByteorder` in `_tests/test_tensor_utils.py` — note that
  `test_unblocks_napari_convert_to_uint8` calls the real napari function, so it
  passes for the wrong reason once napari is fixed, and should go rather than be
  kept as regression cover.

Then napari consumes the faithful big-endian array directly, with no divergent
native-order copy.
