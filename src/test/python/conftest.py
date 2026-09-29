"""Pytest configuration for the SDK tests.

Neutralises terminal colour for the whole session. Several tests assert on the
*text* a CLI prints -- "Deleted 2", a version string, an endpoint URL -- and
Rich emits ANSI escapes into captured output whenever it believes colour is
wanted, which turns `assert "1.2.3" == captured` into a comparison against
`"\x1b[1;36m1.\x1b[0m..."`.

`FORCE_COLOR` is the one that bites: Rich checks it *before* `NO_COLOR`, so
setting `NO_COLOR` alone does not help, and the variable is set by several
terminal tools and agent harnesses. CI has neither set, which is why these
tests pass there and fail on a developer's machine -- the worst place for a
test to disagree with CI.

Deliberately *not* at the repo root, though that would cover this directory
too: several `install/test` modules do `from conftest import ...`, importing
their own conftest as a top-level module, and a root `conftest.py` makes ruff's
isort resolve that name as first-party -- reclassifying the import block in
four unrelated files. `biopb-tensor-server/tests/conftest.py` carries its own
copy for the same reason its rootdir differs.
"""

import os

os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ.setdefault("TERM", "dumb")
