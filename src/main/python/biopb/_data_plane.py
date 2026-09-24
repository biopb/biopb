"""Alias of :mod:`biopb.control._data_plane`, kept while the monorepo migrates to it."""

import sys

from .control import _data_plane

sys.modules[__name__] = _data_plane
