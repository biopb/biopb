"""Nobody reaches agentbench's modules by a path that no longer exists.

This guards a failure that git cannot see and a reviewer has no reason to look
for. A new case module under `bench/cases/` is a **new file**, so
it merges cleanly against a branch that moved `_fixture` and `_respondent` out
from under it — and then `cases/__init__.py` imports every case, one stale
`from .._fixture import ...` raises, and the whole interaction package stops
collecting. Four test modules go down together and nothing ran.

That is exactly what happened across #711/#712 and would have happened again to
four open branches. The cost of finding it late is a rebase that looks clean,
a CI run that dies in collection, and an ImportError that names a module rather
than the mistake.

So this reads the source rather than importing it, and fails with the rewrite
to make. It is deliberately a *text* check on the tree: importing everything
would report the same breakage as an opaque cascade, which is the thing being
replaced.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TESTS_ROOT = HERE.parent

#: The modules this package owns. Reaching any of them from outside means
#: naming `agentbench`, whatever the depth.
OWNED = {
    "_agent",
    "_bridge",
    "_conversation",
    "_fixture",
    "_models",
    "_plane",
    "_respondent",
    "_session",
}


def _modules_outside_agentbench() -> list[Path]:
    return sorted(
        path
        for path in TESTS_ROOT.rglob("*.py")
        if HERE not in path.parents and path != HERE
    )


def _stale_imports(source: str) -> list[tuple[int, str]]:
    """``(line, what it said)`` for every relative import of an owned module
    that does not go through `agentbench`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        module = node.module or ""
        if "agentbench" in module.split("."):
            continue
        dots = "." * node.level
        # `from .._fixture import X`
        if module.split(".")[-1] in OWNED:
            found.append((node.lineno, f"from {dots}{module} import ..."))
            continue
        # `from .. import _fixture`
        for alias in node.names:
            if alias.name in OWNED:
                found.append((node.lineno, f"from {dots}{module} import {alias.name}"))
    return found


@pytest.mark.parametrize(
    "path", _modules_outside_agentbench(), ids=lambda p: str(p.relative_to(TESTS_ROOT))
)
def test_agentbench_is_reached_by_name_not_by_depth(path: Path):
    stale = _stale_imports(path.read_text(encoding="utf-8"))
    if not stale:
        return
    rel = path.relative_to(TESTS_ROOT)
    # How many dots this file needs to get up to `_tests/` and back down.
    depth = len(rel.parts)
    lines = "\n".join(f"    line {n}: {text}" for n, text in stale)
    pytest.fail(
        f"{rel} imports agentbench modules as if they were siblings:\n{lines}\n\n"
        f"They live in `_tests/agentbench/` and are shared with every suite. "
        f"From here that is `from {'.' * depth}agentbench.<module> import ...`.\n\n"
        "This is the import that merges cleanly and then stops the whole "
        "package collecting, so it is checked rather than remembered."
    )


def test_the_guard_would_catch_the_real_regression():
    """The guard's own test, because a scanner that finds nothing looks exactly
    like a clean tree. This is the literal text that broke four branches."""
    stale = _stale_imports(
        "from .._benchmark import Case, Layer\n"
        "from .._fixture import Attempt, Fixture\n"
        "from .._respondent import Persona\n"
        "from .. import _plane\n"
    )
    assert [text for _, text in stale] == [
        "from .._fixture import ...",
        "from .._respondent import ...",
        "from .. import _plane",
    ], "the guard no longer recognises the import it exists to catch"


def test_the_guard_accepts_the_canonical_form():
    assert not _stale_imports(
        "from ....agentbench._fixture import Attempt\n"
        "from ...agentbench import _plane\n"
        "from .._benchmark import Case\n"
        "from .cases import CASES\n"
    )
