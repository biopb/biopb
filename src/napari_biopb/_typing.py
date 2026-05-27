from __future__ import annotations

from typing import Union

import dask.array
import numpy

napari_data = Union[dask.array.Array, numpy.ndarray]
