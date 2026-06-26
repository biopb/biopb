import warnings
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
from biopb.image import BinData, ImageData, Pixels
from biopb.image.utils import (
    _canonicalize_dtype,
    _np_from_pb,
    _pb_from_np,
    _serialize_from_numpy,
    deserialize_image_data,
    deserialize_to_numpy,
    get_image_data_dim_labels,
    get_image_data_shape,
    normalize_array_dims,
    serialize_from_numpy_to_image_data,
)


def test_import():
    import biopb.image as proto

    assert proto.__version__


def test_canonicalize_dtype():
    """Test that dtype canonicalization strips byteorder prefixes."""
    # Test various dtype strings with different prefixes
    assert _canonicalize_dtype("|u1") == "u1"
    assert _canonicalize_dtype("<u1") == "u1"
    assert _canonicalize_dtype(">u1") == "u1"
    assert _canonicalize_dtype("=u1") == "u1"
    assert _canonicalize_dtype("u1") == "u1"

    assert _canonicalize_dtype("<f4") == "f4"
    assert _canonicalize_dtype(">f4") == "f4"
    assert _canonicalize_dtype("=f4") == "f4"
    assert _canonicalize_dtype("f4") == "f4"

    assert _canonicalize_dtype("<i2") == "i2"
    assert _canonicalize_dtype(">i2") == "i2"
    assert _canonicalize_dtype("i2") == "i2"

    # Larger types
    assert _canonicalize_dtype("<u4") == "u4"
    assert _canonicalize_dtype("<f8") == "f8"
    assert _canonicalize_dtype("<i4") == "i4"


def test_serialize_produces_canonical_dtype():
    """Test that serialize_from_numpy produces canonical dtype without prefix."""
    # uint8 on little-endian systems normally produces '|u1' dtype.str
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pixels = _serialize_from_numpy(img)
    assert pixels.dtype == "u1", f"Expected 'u1' but got '{pixels.dtype}'"

    # uint16 produces '<u2' on little-endian
    img = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    pixels = _serialize_from_numpy(img)
    assert pixels.dtype == "u2", f"Expected 'u2' but got '{pixels.dtype}'"

    # float32
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pixels = _serialize_from_numpy(img)
    assert pixels.dtype == "f4", f"Expected 'f4' but got '{pixels.dtype}'"

    # float64
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pixels = _serialize_from_numpy(img)
    assert pixels.dtype == "f8", f"Expected 'f8' but got '{pixels.dtype}'"

    # int16
    img = np.array([[1, 2], [3, 4]], dtype=np.int16)
    pixels = _serialize_from_numpy(img)
    assert pixels.dtype == "i2", f"Expected 'i2' but got '{pixels.dtype}'"


def test_roundtrip_uint8():
    """Test round-trip serialization for uint8."""
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.uint8
    np.testing.assert_array_equal(img_new.squeeze(), img)


def test_roundtrip_uint16():
    """Test round-trip serialization for uint16."""
    img = np.random.randint(0, 65536, size=(64, 64, 3), dtype=np.uint16)
    pixels = _serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.uint16
    np.testing.assert_array_equal(img_new.squeeze(), img)


def test_roundtrip_int16():
    """Test round-trip serialization for int16."""
    img = np.random.randint(-1000, 1000, size=(64, 64, 3), dtype=np.int16)
    pixels = _serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.int16
    np.testing.assert_array_equal(img_new.squeeze(), img)


def test_roundtrip_float32():
    """Test round-trip serialization for float32."""
    img = np.random.random(size=(64, 64, 3)).astype(np.float32)
    pixels = _serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.float32
    np.testing.assert_array_almost_equal(img_new.squeeze(), img)


def test_roundtrip_float64():
    """Test round-trip serialization for float64."""
    img = np.random.random(size=(64, 64, 3)).astype(np.float64)
    pixels = _serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.float64
    np.testing.assert_array_almost_equal(img_new.squeeze(), img)


def test_endianness_conflict_warning():
    """Test that a warning is issued when dtype prefix conflicts with BinData.endianness."""
    # Create a Pixels message with conflicting endianness
    # dtype has '<' (little-endian) but BinData says BIG
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pixels = _serialize_from_numpy(img)

    # Manually override to create conflict
    # We'll create a Pixels with dtype '<u1' but endianness BIG
    conflicting_pixels = Pixels(
        bindata=BinData(data=img.tobytes(), endianness=BinData.Endianness.BIG),
        dtype="<u1",  # Little-endian prefix but BinData says BIG
        size_x=2,
        size_y=2,
        size_z=1,
        size_c=1,
        dimension_order="CXYZT",
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deserialize_to_numpy(conflicting_pixels)
        assert len(w) == 1
        assert "Endianness conflict" in str(w[0].message)


def test_utils():
    import numpy as np
    from biopb.image.utils import _serialize_from_numpy, deserialize_to_numpy

    img = np.random.random(size=[64, 64, 3])
    img = (img * 65536).astype("<u2")

    pixels = _serialize_from_numpy(img)

    img_new = deserialize_to_numpy(pixels)

    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype.str == "<u2"


def test_np_index_order_default():
    """Test that default np_index_order produces ZYXC order."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    # Default is ZYXC, so shape should be (Z, Y, X, C) = (1, 32, 32, 3)
    assert img_new.shape == (1, 32, 32, 3)


def test_np_index_order_custom():
    """Test custom np_index_order values."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)

    # Test CZYX order (C, Z, Y, X)
    img_czyx = deserialize_to_numpy(pixels, np_index_order="CZYX")
    assert img_czyx.shape == (3, 1, 32, 32)

    # Test XYZC order (X, Y, Z, C)
    img_xyzc = deserialize_to_numpy(pixels, np_index_order="XYZC")
    assert img_xyzc.shape == (32, 32, 1, 3)

    # Test YXCZ order
    img_yxcz = deserialize_to_numpy(pixels, np_index_order="YXCZ")
    assert img_yxcz.shape == (32, 32, 3, 1)


def test_np_index_order_invalid():
    """Test that invalid np_index_order raises ValueError."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)

    with pytest.raises(ValueError, match="np_index_order"):
        deserialize_to_numpy(pixels, np_index_order="ABCD")

    with pytest.raises(ValueError, match="np_index_order"):
        deserialize_to_numpy(pixels, np_index_order="ZYX")  # missing C

    with pytest.raises(ValueError, match="np_index_order"):
        deserialize_to_numpy(pixels, np_index_order="ZYXCC")  # duplicate C


def test_serialize_np_index_order():
    """Test serialization with custom np_index_order."""
    # 5D input with T dimension (TZYXC order in numpy C-order)
    img = np.random.randint(0, 256, size=(2, 64, 64, 3, 5), dtype=np.uint8)
    pixels = _serialize_from_numpy(img, np_index_order="TZYXC")

    assert pixels.size_t == 2
    assert pixels.size_z == 64
    assert pixels.size_y == 64
    assert pixels.size_x == 3
    assert pixels.size_c == 5

    # 2D input with custom order (XY instead of YX)
    img2d = np.random.randint(0, 256, size=(32, 64), dtype=np.uint8)
    pixels2d = _serialize_from_numpy(img2d, np_index_order="XY")
    assert pixels2d.size_x == 32  # first dimension
    assert pixels2d.size_y == 64  # second dimension


def test_deserialize_5d_with_t():
    """Test deserialization with non-singleton T dimension."""
    # Create 5D data with default TZYXC order
    img = np.random.randint(0, 256, size=(2, 32, 32, 3, 4), dtype=np.uint8)
    pixels = _serialize_from_numpy(img, np_index_order="TZYXC")

    # Deserialize with 5D output
    img_5d = deserialize_to_numpy(pixels, np_index_order="TZYXC")
    assert img_5d.shape == (2, 32, 32, 3, 4)

    # Deserialize with 4D output should fail (T is not singleton)
    with pytest.raises(ValueError, match="Dimension T has size"):
        deserialize_to_numpy(pixels, np_index_order="ZYXC")


def test_deserialize_2d_output():
    """Test deserialization to 2D output (squeeze Z, C, T)."""
    # Create 2D data (Y, X)
    img = np.random.randint(0, 256, size=(32, 64), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)  # Z=1, C=1, T=1

    # Deserialize to 2D
    img_2d = deserialize_to_numpy(pixels, np_index_order="YX")
    assert img_2d.shape == (32, 64)


def test_serialize_transpose_dimension_order():
    """Test that serialize correctly transposes when np_index_order differs from dimension_order.

    This tests the case where the input array's dimension order doesn't match the
    output protobuf's dimension_order, requiring a transpose operation.
    """
    # Create a 3D array where each element encodes its position: arr[z,y,x] = z*10000 + y*100 + x
    # Using small sizes for clarity: 2 z-slices, 3 y-rows, 4 x-cols
    z_size, y_size, x_size = 2, 3, 4
    img = np.zeros((z_size, y_size, x_size), dtype=np.uint16)
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                img[z, y, x] = z * 10000 + y * 100 + x

    # Serialize with np_index_order="ZYX" (C-order: axis 0=Z, axis 1=Y, axis 2=X)
    # and dimension_order="XYZCT" (F-order: X varies fastest)
    pixels = _serialize_from_numpy(
        img,
        dimension_order="XYZCT",  # Output F-order: X first varies fastest
        np_index_order="ZYX",  # Input C-order: axis 0=Z (slowest), axis -1=X (fastest)
    )

    # Verify sizes
    assert pixels.size_z == z_size
    assert pixels.size_y == y_size
    assert pixels.size_x == x_size
    assert pixels.size_c == 1
    assert pixels.size_t == 1

    # Deserialize back with matching np_index_order
    img_back = deserialize_to_numpy(pixels, np_index_order="ZYX")

    # Verify the data round-trips correctly
    assert img_back.shape == img.shape
    np.testing.assert_array_equal(img_back, img)

    # Also test with different np_index_order to verify transpose worked
    img_back_xyz = deserialize_to_numpy(pixels, np_index_order="XYZ")
    assert img_back_xyz.shape == (x_size, y_size, z_size)
    # Verify specific values: at position (x,y,z), we should get z*10000 + y*100 + x
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                expected = z * 10000 + y * 100 + x
                assert img_back_xyz[x, y, z] == expected, (
                    f"Mismatch at ({x},{y},{z}): expected {expected}, got {img_back_xyz[x, y, z]}"
                )


def test_serialize_transpose_with_channel():
    """Test transpose with channel dimension included."""
    # Create array with shape (Z, Y, X, C) = (2, 3, 4, 3)
    # Values encode position: z*10000 + y*100 + x*10 + c
    z_size, y_size, x_size, c_size = 2, 3, 4, 3
    img = np.zeros((z_size, y_size, x_size, c_size), dtype=np.uint16)
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                for c in range(c_size):
                    img[z, y, x, c] = z * 10000 + y * 100 + x * 10 + c

    # Test different dimension_order combinations
    for out_order in ["XYZCT", "CXYZT", "ZYXCT"]:
        pixels = _serialize_from_numpy(
            img, dimension_order=out_order, np_index_order="ZYXC"
        )

        assert pixels.size_z == z_size
        assert pixels.size_y == y_size
        assert pixels.size_x == x_size
        assert pixels.size_c == c_size

        # Round-trip
        img_back = deserialize_to_numpy(pixels, np_index_order="ZYXC")
        np.testing.assert_array_equal(
            img_back, img, err_msg=f"Round-trip failed for dimension_order={out_order}"
        )


def test_singleton_t_deprecation_warning():
    """Test that singleton_t=False raises deprecation warning."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deserialize_to_numpy(pixels, singleton_t=False, np_index_order="ZYXCT")
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)


def test_serialize_non_contiguous_array():
    """Test that serialization works with non-C-contiguous arrays.

    This tests that np.ascontiguousarray() correctly handles arrays
    that are F-order or have other non-contiguous memory layouts.
    """
    # Create a C-order array and make it F-order via transpose
    img_c = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    img_f = img_c.T  # Now F-order, shape (4, 3, 2)

    assert not img_f.flags["C_CONTIGUOUS"], "img_f should not be C-contiguous"
    assert img_f.flags["F_CONTIGUOUS"], "img_f should be F-contiguous"

    # Serialize with np_index_order matching the transposed shape
    # img_f has shape (4, 3, 2) which is (X, Y, Z) in our notation
    pixels = _serialize_from_numpy(img_f, np_index_order="XYZ")

    # Verify sizes
    assert pixels.size_x == 4
    assert pixels.size_y == 3
    assert pixels.size_z == 2

    # Round-trip should work correctly
    img_back = deserialize_to_numpy(pixels, np_index_order="XYZ")
    np.testing.assert_array_equal(img_back, img_f)


def test_serialize_f_order_input():
    """Test serialization with explicitly F-order array."""
    # Create F-order array directly (4D to match np_index_order)
    img = np.asfortranarray(np.arange(24, dtype=np.uint8).reshape(2, 3, 4, 1))
    assert img.flags["F_CONTIGUOUS"]

    # Serialize with np_index_order matching shape (Z=2, Y=3, X=4, C=1)
    pixels = _serialize_from_numpy(img, np_index_order="ZYXC")

    # Verify sizes match input shape
    assert pixels.size_z == 2
    assert pixels.size_y == 3
    assert pixels.size_x == 4
    assert pixels.size_c == 1

    # Round-trip
    img_back = deserialize_to_numpy(pixels, np_index_order="ZYXC")
    np.testing.assert_array_equal(img_back, img)


# ============================================================================
# deserialize_image_data Tests
# ============================================================================


def test_deserialize_image_data_eager_data():
    """Test deserialize_image_data with eager_data (Tensor)."""

    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    # Use the new Tensor-based serialization
    image_data = serialize_from_numpy_to_image_data(img, dim_labels=["Y", "X", "C"])
    result = deserialize_image_data(image_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32, 3)
    np.testing.assert_array_equal(result, img)


def test_deserialize_image_data_lazy_data():
    """Test deserialize_image_data with lazy_data (SerializedTensor)."""
    from biopb.image import ImageData
    from biopb.tensor.descriptor_pb2 import TensorDescriptor
    from biopb.tensor.serialized_pb2 import SerializedEndpoint, SerializedTensor
    from biopb.tensor.ticket_pb2 import ChunkBounds, TensorTicket

    # Create a mock SerializedTensor
    descriptor = TensorDescriptor(
        array_id="test-tensor",
        shape=[64, 64],
        dtype="uint8",
        chunk_shape=[32, 32],
    )

    serialized_tensor = SerializedTensor(
        tensor_descriptor=descriptor,
        location="grpc://localhost:8815",
        auth_token="",
        endpoints=[
            SerializedEndpoint(
                ticket=TensorTicket(chunk_id=b"chunk-0"),
                chunk_bounds=ChunkBounds(start=[0, 0], stop=[32, 32]),
            ),
        ],
    )

    image_data = ImageData(lazy_data=serialized_tensor)

    # Mock TensorFlightClient.tensor_from_pb to avoid needing live server
    mock_dask_arr = MagicMock()
    mock_dask_arr.shape = (64, 64)
    mock_dask_arr.dtype = np.uint8

    with patch(
        "biopb.tensor.client.TensorFlightClient.tensor_from_pb",
        return_value=mock_dask_arr,
    ):
        result = deserialize_image_data(image_data)

        # Should return the mocked dask array
        assert result.shape == (64, 64)


def test_deserialize_image_data_legacy_pixels():
    """Test deserialize_image_data with legacy pixels field."""
    from biopb.image import ImageData

    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pixels = _serialize_from_numpy(img)

    # Use deprecated pixels field directly
    image_data = ImageData(pixels=pixels)
    result = deserialize_image_data(image_data)

    assert isinstance(result, np.ndarray)
    # Legacy pixels uses dimension_order reversed for np_index_order
    # Default dimension_order="CXYZT" -> np_index_order="TXYZC"
    # The resulting shape preserves all dimensions in the order specified
    assert result.shape == (1, 1, 32, 32, 3)
    # The data round-trips correctly when squeezed
    np.testing.assert_array_equal(result.squeeze(), img)


def test_deserialize_image_data_no_data_raises():
    """Test that deserialize_image_data raises when no data field is set."""
    from biopb.image import ImageData

    # Create ImageData with no data fields set
    image_data = ImageData()

    with pytest.raises(ValueError, match="ImageData has no data field set"):
        deserialize_image_data(image_data)


def test_serialize_from_numpy_to_image_data():
    """Test serialize_from_numpy_to_image_data."""
    from biopb.image.utils import serialize_from_numpy_to_image_data

    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    image_data = serialize_from_numpy_to_image_data(img)

    assert isinstance(image_data, ImageData)
    assert image_data.HasField("eager_data")

    # Verify dimensions - Tensor uses dims list
    assert list(image_data.eager_data.dims) == [32, 32, 3]
    assert image_data.eager_data.dtype == "u1"


def test_serialize_from_numpy_to_image_data_with_dim_labels():
    """Test serialize_from_numpy_to_image_data with dim_labels."""
    from biopb.image.utils import serialize_from_numpy_to_image_data

    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    image_data = serialize_from_numpy_to_image_data(img, dim_labels=["Y", "X", "C"])

    assert list(image_data.eager_data.dim_labels) == ["Y", "X", "C"]


def test_get_image_data_dim_labels():
    """Test get_image_data_dim_labels helper."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    # With dim_labels
    image_data = serialize_from_numpy_to_image_data(img, dim_labels=["Y", "X", "C"])
    labels = get_image_data_dim_labels(image_data)
    assert labels == ["Y", "X", "C"]

    # Without dim_labels - returns None when not set
    image_data2 = serialize_from_numpy_to_image_data(img)
    labels2 = get_image_data_dim_labels(image_data2)
    assert labels2 is None


def test_get_image_data_dim_labels_implicit():
    """Test get_image_data_dim_labels with implicit=True."""
    # 2D -> ['y', 'x']
    img_2d = np.random.randint(0, 256, size=(32, 64), dtype=np.uint8)
    image_data_2d = serialize_from_numpy_to_image_data(img_2d)
    labels_2d = get_image_data_dim_labels(image_data_2d, implicit=True)
    assert labels_2d == ["y", "x"]

    # 3D with dim[0] <= 4 -> ['c', 'y', 'x'] (channel-first)
    img_3d_c_first = np.random.randint(0, 256, size=(3, 32, 64), dtype=np.uint8)
    image_data_3d_c_first = serialize_from_numpy_to_image_data(img_3d_c_first)
    labels_3d_c_first = get_image_data_dim_labels(image_data_3d_c_first, implicit=True)
    assert labels_3d_c_first == ["c", "y", "x"]

    # 3D with dim[-1] <= 4 -> ['y', 'x', 'c'] (channel-last)
    img_3d_c_last = np.random.randint(0, 256, size=(32, 64, 3), dtype=np.uint8)
    image_data_3d_c_last = serialize_from_numpy_to_image_data(img_3d_c_last)
    labels_3d_c_last = get_image_data_dim_labels(image_data_3d_c_last, implicit=True)
    assert labels_3d_c_last == ["y", "x", "c"]

    # 3D with both dim[0] <= 4 and dim[-1] <= 4 -> prefers dim[0] (channel-first)
    img_3d_both = np.random.randint(0, 256, size=(3, 32, 3), dtype=np.uint8)
    image_data_3d_both = serialize_from_numpy_to_image_data(img_3d_both)
    labels_3d_both = get_image_data_dim_labels(image_data_3d_both, implicit=True)
    assert labels_3d_both == ["c", "y", "x"]

    # 3D with neither dim[0] nor dim[-1] <= 4 -> None
    img_3d_none = np.random.randint(0, 256, size=(32, 64, 32), dtype=np.uint8)
    image_data_3d_none = serialize_from_numpy_to_image_data(img_3d_none)
    labels_3d_none = get_image_data_dim_labels(image_data_3d_none, implicit=True)
    assert labels_3d_none is None

    # 4D -> None (no heuristic)
    img_4d = np.random.randint(0, 256, size=(2, 32, 64, 3), dtype=np.uint8)
    image_data_4d = serialize_from_numpy_to_image_data(img_4d)
    labels_4d = get_image_data_dim_labels(image_data_4d, implicit=True)
    assert labels_4d is None

    # With explicit dim_labels, implicit=True should return explicit labels
    img_explicit = np.random.randint(0, 256, size=(32, 64, 3), dtype=np.uint8)
    image_data_explicit = serialize_from_numpy_to_image_data(
        img_explicit, dim_labels=["Z", "Y", "X"]
    )
    labels_explicit = get_image_data_dim_labels(image_data_explicit, implicit=True)
    assert labels_explicit == ["Z", "Y", "X"]  # explicit labels take precedence


def test_get_image_data_shape():
    """Test get_image_data_shape helper."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    image_data = serialize_from_numpy_to_image_data(img)
    shape = get_image_data_shape(image_data)
    assert shape == (32, 32, 3)


def test_pb_from_np_and_np_from_pb():
    """Test _pb_from_np and _np_from_pb helpers."""
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    tensor = _pb_from_np(img, dim_labels=["Y", "X", "C"])
    assert list(tensor.dims) == [32, 32, 3]
    assert tensor.dtype == "u1"
    assert list(tensor.dim_labels) == ["Y", "X", "C"]

    img_back = _np_from_pb(tensor)
    assert img_back.shape == img.shape
    np.testing.assert_array_equal(img_back, img)


def test_deserialize_image_data_cache_bytes_parameter():
    """Test deserialize_image_data with cache_bytes parameter for lazy_data."""
    from biopb.image import ImageData
    from biopb.tensor.descriptor_pb2 import TensorDescriptor
    from biopb.tensor.serialized_pb2 import SerializedTensor

    descriptor = TensorDescriptor(
        array_id="test-tensor",
        shape=[64, 64],
        dtype="uint8",
    )

    serialized_tensor = SerializedTensor(
        tensor_descriptor=descriptor,
        location="grpc://localhost:8815",
    )

    image_data = ImageData(lazy_data=serialized_tensor)

    mock_dask_arr = MagicMock()

    with patch(
        "biopb.tensor.client.TensorFlightClient.tensor_from_pb"
    ) as mock_tensor_from_pb:
        mock_tensor_from_pb.return_value = mock_dask_arr

        # Call with custom cache_bytes
        deserialize_image_data(image_data, cache_bytes=500_000_000)

        # Verify cache_bytes was passed
        mock_tensor_from_pb.assert_called_once()
        call_args = mock_tensor_from_pb.call_args
        assert call_args[1]["cache_bytes"] == 500_000_000


# ============================================================================
# normalize_array_dims Tests
# ============================================================================


def test_normalize_array_dims_transpose():
    """Test that normalize_array_dims correctly transposes dimensions."""
    # Create array with shape (Z, Y, X, C) = (2, 3, 4, 3)
    arr = np.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3)

    # Transpose to (C, Z, Y, X)
    result = normalize_array_dims(arr, ["Z", "Y", "X", "C"], ["C", "Z", "Y", "X"])

    assert result.shape == (3, 2, 3, 4)
    # Verify data is correctly transposed
    np.testing.assert_array_equal(result, arr.transpose(3, 0, 1, 2))


def test_normalize_array_dims_squeeze():
    """Test that normalize_array_dims correctly squeezes singleton dimensions."""
    # Create array with singleton Z dimension
    arr = np.arange(96, dtype=np.uint8).reshape(1, 3, 4, 8)  # (Z=1, Y=3, X=4, C=8)

    # Remove Z dimension
    result = normalize_array_dims(arr, ["Z", "Y", "X", "C"], ["Y", "X", "C"])

    assert result.shape == (3, 4, 8)
    np.testing.assert_array_equal(result, arr.squeeze(0))


def test_normalize_array_dims_expand():
    """Test that normalize_array_dims correctly expands dimensions."""
    # Create 3D array
    arr = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)  # (Y=3, X=4, C=2)

    # Add Z and T dimensions
    result = normalize_array_dims(arr, ["Y", "X", "C"], ["T", "Z", "Y", "X", "C"])

    assert result.shape == (1, 1, 3, 4, 2)
    np.testing.assert_array_equal(result.squeeze(), arr)


def test_normalize_array_dims_combined_operations():
    """Test squeeze, expand, and transpose combined."""
    # Array: (Z=1, Y=3, X=4, C=2) -> target: (C, Y, X, T=1)
    arr = np.arange(24, dtype=np.uint8).reshape(1, 3, 4, 2)

    result = normalize_array_dims(arr, ["Z", "Y", "X", "C"], ["C", "Y", "X", "T"])

    assert result.shape == (2, 3, 4, 1)
    # Verify by computing expected shape manually
    # Z is squeezed, C moves to front, T is added at end
    assert result.squeeze(-1).shape == arr.squeeze(0).transpose(2, 0, 1).shape


def test_normalize_array_dims_case_insensitive():
    """Test that dimension label comparison is case-insensitive."""
    arr = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)

    # Mix case in labels
    result = normalize_array_dims(arr, ["y", "X", "c"], ["C", "Y", "X"])

    assert result.shape == (2, 3, 4)
    np.testing.assert_array_equal(result, arr.transpose(2, 0, 1))


def test_normalize_array_dims_preserves_case():
    """Test that original label case is preserved in error messages."""
    arr = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)  # (Y=3, X=4, C=1)

    # Try to squeeze non-singleton dimension - original case preserved
    with pytest.raises(ValueError, match="'y'"):
        normalize_array_dims(arr, ["y", "x", "c"], ["x", "c"])


def test_normalize_array_dims_dask_array():
    """Test that normalize_array_dims works with dask arrays (lazy)."""
    # Create dask array
    arr = da.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3)

    result = normalize_array_dims(arr, ["Z", "Y", "X", "C"], ["C", "Z", "Y", "X"])

    assert isinstance(result, da.Array)
    assert result.shape == (3, 2, 3, 4)
    # Compute and verify
    np.testing.assert_array_equal(result.compute(), arr.compute().transpose(3, 0, 1, 2))


def test_normalize_array_dims_dask_squeeze_expand():
    """Test squeeze and expand with dask arrays."""
    arr = da.arange(24, dtype=np.uint8).reshape(1, 3, 4, 2)

    result = normalize_array_dims(arr, ["Z", "Y", "X", "C"], ["C", "Y", "X", "T"])

    assert isinstance(result, da.Array)
    assert result.shape == (2, 3, 4, 1)


def test_normalize_array_dims_none_raises():
    """Test that None dim_labels raises ValueError."""
    arr = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)

    with pytest.raises(ValueError, match="no dim_labels"):
        normalize_array_dims(arr, None, ["Y", "X", "C"])


def test_normalize_array_dims_length_mismatch():
    """Test that mismatched dim_labels length raises ValueError."""
    arr = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)

    with pytest.raises(ValueError, match="dim_labels length"):
        normalize_array_dims(arr, ["Y", "X"], ["Y", "X", "C"])


def test_normalize_array_dims_duplicate_labels():
    """Test that duplicate labels raise ValueError."""
    arr = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)

    with pytest.raises(ValueError, match="dim_labels has duplicate"):
        normalize_array_dims(arr, ["Y", "Y", "C"], ["Y", "X", "C"])

    with pytest.raises(ValueError, match="target_dim_labels has duplicate"):
        normalize_array_dims(arr, ["Y", "X", "C"], ["Y", "Y", "C"])


def test_normalize_array_dims_cannot_squeeze_non_singleton():
    """Test that squeezing non-singleton dimension raises ValueError."""
    arr = np.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3)  # Z=2 (not singleton)

    with pytest.raises(ValueError, match="Cannot squeeze non-singleton"):
        normalize_array_dims(arr, ["Z", "Y", "X", "C"], ["Y", "X", "C"])


def test_normalize_array_dims_no_change_needed():
    """Test that identical source and target returns same array."""
    arr = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)

    result = normalize_array_dims(arr, ["Y", "X", "C"], ["Y", "X", "C"])

    assert result.shape == arr.shape
    np.testing.assert_array_equal(result, arr)


def test_normalize_array_dims_complex_reorder():
    """Test complex dimension reorder with all 5 dims."""
    # Create 5D array
    arr = np.arange(480, dtype=np.uint8).reshape(
        2, 3, 4, 5, 4
    )  # (T=2, Z=3, Y=4, X=5, C=4)

    # Reorder from TZYXC to ZYXCT
    result = normalize_array_dims(
        arr, ["T", "Z", "Y", "X", "C"], ["Z", "Y", "X", "C", "T"]
    )

    assert result.shape == (3, 4, 5, 4, 2)
    np.testing.assert_array_equal(result, arr.transpose(1, 2, 3, 4, 0))
