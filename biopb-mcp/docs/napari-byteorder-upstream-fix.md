# DRAFT — upstream napari fix: `convert_to_uint8` rejects non-native byte order

> Local draft only. **Not yet filed upstream.** Basis for a future napari
> issue/PR. Tracks biopb/biopb#296; lets us eventually drop the consumer-side
> workaround in `biopb_mcp/_tensor_utils.py::_to_native_byteorder`.

## Summary

`napari.layers.utils.layer_utils.convert_to_uint8` raises `TypeError` on any
**non-native-endian** integer or float array (e.g. a big-endian `>i2` / `>f4`
FITS image). This is hit on every layer add via the thumbnail path
(`Image._update_thumbnail` → `convert_to_uint8`), so a big-endian source cannot
be displayed at all.

Reproduced on napari `0.7.0` and still present on `main` (as of 2026-07).

## Repro (standalone)

```python
import numpy as np
from napari.layers.utils.layer_utils import convert_to_uint8

convert_to_uint8(np.arange(6, dtype=">i2").reshape(2, 3))   # big-endian int16
# TypeError: The `dtype` and `signature` arguments to ufuncs only select the
# general DType and not details such as the byte order or time unit ...
```

Same failure for a big-endian float (`np.arange(6, dtype=">f4")`), via the float
branch below.

## Root cause

`convert_to_uint8` passes a **byte-order-qualified** dtype to a ufunc's `dtype=`
argument, which NumPy only allows to select the *general* DType (not byte order
or time unit). Two sites in `src/napari/layers/utils/layer_utils.py`:

```python
# float branch
image_out = np.multiply(data, out_max, dtype=data.dtype)   # data.dtype == '>f4' -> TypeError
...
# signed-integer branch
np.maximum(data, 0, out=data, dtype=data.dtype)            # data.dtype == '>i2' -> TypeError
```

`data.dtype` here can be e.g. `dtype('>i2')`; NumPy rejects the byte order in
`dtype=`.

## Fix

Pass the dtype's **general type** (`data.dtype.type`, i.e. `np.int16` / `np.float32`),
which selects the same DType width without the byte-order qualifier. Minimal,
preserves the original intent of pinning the compute width:

```diff
--- a/src/napari/layers/utils/layer_utils.py
+++ b/src/napari/layers/utils/layer_utils.py
@@ convert_to_uint8
     if in_kind == 'f':
-        image_out = np.multiply(data, out_max, dtype=data.dtype)
+        image_out = np.multiply(data, out_max, dtype=data.dtype.type)
         np.rint(image_out, out=image_out)
         np.clip(image_out, 0, out_max, out=image_out)
         image_out = np.nan_to_num(image_out, copy=False)
         return image_out.astype(out_dtype)
@@
-        np.maximum(data, 0, out=data, dtype=data.dtype)
+        np.maximum(data, 0, out=data, dtype=data.dtype.type)
```

Alternative (also correct): drop `dtype=` entirely — the `out=` array already
pins the output dtype for `np.maximum`, and `np.multiply`'s result feeds
`np.rint(..., out=image_out)`. Preferring `.type` keeps the change surgical and
the intent explicit.

### Verified (NumPy 1.26.4)

Both sites, big-endian input, before vs after:

| call | `dtype=data.dtype` | `dtype=data.dtype.type` |
|------|--------------------|--------------------------|
| `np.maximum('>i2', 0, out=…)` | `TypeError` | OK → `[0,0,0,1,2,3]` (correct) |
| `np.multiply('>f4', 255)` | `TypeError` | OK → `[0.,51.,102.]` (correct) |

Values are unchanged from the native-endian case (byte order does not affect the
logical result).

## Suggested upstream test

```python
@pytest.mark.parametrize("dt", [">i2", ">u2", ">f4", ">i4"])
def test_convert_to_uint8_handles_non_native_byteorder(dt):
    data = (np.arange(6, dtype=dt).reshape(2, 3) * 5000).astype(dt)
    out = convert_to_uint8(data)
    assert out.dtype == np.uint8
    # matches the native-endian result
    np.testing.assert_array_equal(out, convert_to_uint8(data.astype(dt.replace(">", "<"))))
```

## Relationship to the biopb workaround

Until this lands upstream, `biopb-mcp` normalizes each pyramid level to native
byte order before `add_image` (`_to_native_byteorder` in `_tensor_utils.py`,
lazy `astype` — the wire/source bytes stay faithful per the #293 binary schema).
Once a fixed napari is the floor version, delete that helper and its call so
napari consumes the faithful big-endian array directly — no divergent
native-order copy.
