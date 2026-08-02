"""One module per skill, each holding a single `Case`. Data, not code.

Adding a skill to the §6 benchmark is:

1. register an ``interaction``-tier fixture in ``outcomes/`` — the one that
   strips a fact from the data so it can only be obtained by asking, proved
   there without a model (§6b);
2. write ``cases/<skill>.py`` with a `Case`: the task prompt, the persona
   holding that fact, which layer the fixture loads into, and the names its
   verifier wants back;
3. add it to :data:`CASES`.

No test code. `test_benchmark.py` parametrizes over this tuple, so a new case
brings its own arms, report and transcripts with it.
"""

from __future__ import annotations

from .._benchmark import Case
from . import drift_correction

#: Every skill benchmarked by §6.
CASES: tuple[Case, ...] = (drift_correction.CASE,)

__all__ = ["CASES"]
