import pytest
import numpy as np
import warnings

from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy, _canonicalize_dtype
from biopb.image import BinData


def test_import():
    import biopb.image as proto
    assert proto.__version__


def test_canonicalize_dtype():
    """Test that dtype canonicalization strips byteorder prefixes."""
    # Test various dtype strings with different prefixes
    assert _canonicalize_dtype('|u1') == 'u1'
    assert _canonicalize_dtype('<u1') == 'u1'
    assert _canonicalize_dtype('>u1') == 'u1'
    assert _canonicalize_dtype('=u1') == 'u1'
    assert _canonicalize_dtype('u1') == 'u1'

    assert _canonicalize_dtype('<f4') == 'f4'
    assert _canonicalize_dtype('>f4') == 'f4'
    assert _canonicalize_dtype('=f4') == 'f4'
    assert _canonicalize_dtype('f4') == 'f4'

    assert _canonicalize_dtype('<i2') == 'i2'
    assert _canonicalize_dtype('>i2') == 'i2'
    assert _canonicalize_dtype('i2') == 'i2'

    # Larger types
    assert _canonicalize_dtype('<u4') == 'u4'
    assert _canonicalize_dtype('<f8') == 'f8'
    assert _canonicalize_dtype('<i4') == 'i4'


def test_serialize_produces_canonical_dtype():
    """Test that serialize_from_numpy produces canonical dtype without prefix."""
    # uint8 on little-endian systems normally produces '|u1' dtype.str
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pixels = serialize_from_numpy(img)
    assert pixels.dtype == 'u1', f"Expected 'u1' but got '{pixels.dtype}'"

    # uint16 produces '<u2' on little-endian
    img = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    pixels = serialize_from_numpy(img)
    assert pixels.dtype == 'u2', f"Expected 'u2' but got '{pixels.dtype}'"

    # float32
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pixels = serialize_from_numpy(img)
    assert pixels.dtype == 'f4', f"Expected 'f4' but got '{pixels.dtype}'"

    # float64
    img = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pixels = serialize_from_numpy(img)
    assert pixels.dtype == 'f8', f"Expected 'f8' but got '{pixels.dtype}'"

    # int16
    img = np.array([[1, 2], [3, 4]], dtype=np.int16)
    pixels = serialize_from_numpy(img)
    assert pixels.dtype == 'i2', f"Expected 'i2' but got '{pixels.dtype}'"


def test_roundtrip_uint8():
    """Test round-trip serialization for uint8."""
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    pixels = serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.uint8
    np.testing.assert_array_equal(img_new.squeeze(), img)


def test_roundtrip_uint16():
    """Test round-trip serialization for uint16."""
    img = np.random.randint(0, 65536, size=(64, 64, 3), dtype=np.uint16)
    pixels = serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.uint16
    np.testing.assert_array_equal(img_new.squeeze(), img)


def test_roundtrip_int16():
    """Test round-trip serialization for int16."""
    img = np.random.randint(-1000, 1000, size=(64, 64, 3), dtype=np.int16)
    pixels = serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.int16
    np.testing.assert_array_equal(img_new.squeeze(), img)


def test_roundtrip_float32():
    """Test round-trip serialization for float32."""
    img = np.random.random(size=(64, 64, 3)).astype(np.float32)
    pixels = serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.float32
    np.testing.assert_array_almost_equal(img_new.squeeze(), img)


def test_roundtrip_float64():
    """Test round-trip serialization for float64."""
    img = np.random.random(size=(64, 64, 3)).astype(np.float64)
    pixels = serialize_from_numpy(img)
    img_new = deserialize_to_numpy(pixels)
    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype == np.float64
    np.testing.assert_array_almost_equal(img_new.squeeze(), img)


def test_endianness_conflict_warning():
    """Test that a warning is issued when dtype prefix conflicts with BinData.endianness."""
    # Create a Pixels message with conflicting endianness
    # dtype has '<' (little-endian) but BinData says BIG
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    pixels = serialize_from_numpy(img)

    # Manually override to create conflict
    # We'll create a Pixels with dtype '<u1' but endianness BIG
    from biopb.image import Pixels
    conflicting_pixels = Pixels(
        bindata=BinData(data=img.tobytes(), endianness=BinData.Endianness.BIG),
        dtype='<u1',  # Little-endian prefix but BinData says BIG
        size_x=2,
        size_y=2,
        size_z=1,
        size_c=1,
        dimension_order='CXYZT',
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deserialize_to_numpy(conflicting_pixels)
        assert len(w) == 1
        assert "Endianness conflict" in str(w[0].message)


def test_utils():
    import numpy as np
    from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy

    img = np.random.random(size=[64,64,3])
    img = (img * 65536).astype("<u2")

    pixels = serialize_from_numpy(img)

    img_new = deserialize_to_numpy(pixels)

    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype.str == "<u2"

