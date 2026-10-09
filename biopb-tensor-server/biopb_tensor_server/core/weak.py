"""Weak references that tolerate an object that cannot have one."""

from __future__ import annotations

import weakref
from typing import Any, Callable, Optional


def weak_or_none(obj: Any) -> Optional[Callable[[], Any]]:
    """A call that returns *obj* while it lives, or None when *obj* cannot be
    weakly referenced (a ``__slots__`` class, a test double)."""
    try:
        return weakref.ref(obj)
    except TypeError:
        return None
