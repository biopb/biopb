"""Alias of :mod:`biopb.control._endpoints`, kept while the monorepo migrates to it."""

import sys

from .control import _endpoints

sys.modules[__name__] = _endpoints
