"""One module per task, each exporting a module-level ``CASE``.

There is no registration step: a module is discovered by being here. Adding a
task is writing one file, and this suite's tests start checking it by its
arriving.

A module whose name starts with ``_`` is skipped, which is how a task can be
written and banked before the data it needs exists on any machine.
"""

from __future__ import annotations

import importlib
import pkgutil

from .._runner import TaskCase


def _discover() -> tuple[TaskCase, ...]:
    found = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        case = getattr(module, "CASE", None)
        if case is None:
            raise ImportError(
                f"{module.__name__} is in cases/ but exports no CASE. A module "
                "here is a task by being here; there is nowhere else to say so."
            )
        found.append(case)
    return tuple(found)


CASES: tuple[TaskCase, ...] = _discover()
