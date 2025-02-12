import pytest

def test_import():
    import biopb.image as proto
    assert proto.__version__

def test_utils():
    import numpy as np
    import biopb.image as proto
    from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy

    img = np.random.random(size=[64,64,3])
    img = (img * 65536).astype("<u2")

    pixels = serialize_from_numpy(img)

    img_new = deserialize_to_numpy(pixels)

    assert img_new.shape == (1, 64, 64, 3)
    assert img_new.dtype.str == "<u2"

