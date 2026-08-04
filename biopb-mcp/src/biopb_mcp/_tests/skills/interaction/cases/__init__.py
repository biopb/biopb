"""One module per skill, each holding a single `CASE`. Data, not code.

Adding a skill to the benchmark is:

1. write ``cases/<skill>.py`` exporting a module-level `CASE`: an id for the
   case, the task prompt, the persona holding the fact the fixture withholds,
   the one `FixtureSpec` that case is written against, and a verifier for what
   the run leaves in the kernel;
2. there is no step 2. The module is discovered by being here.

One module is one `CASE`, so covering a skill two ways — a procedural fixture
and a real acquisition, which are two experiments and not two settings of one —
is two modules with two `case_id`s.

No test code, no registration line, no engine change. `test_benchmark.py`
parametrizes over :data:`CASES`, so a new case brings its own arms, report and
transcripts with it, and `test_cases.py` starts checking its persona and its
fixture by its arriving.

A module whose name starts with `_` belongs to a **deferred** skill — one the
runtime does not serve — and lands in :data:`DEFERRED_CASES` instead. It is not
benchmarked, because there is no catalog entry to withhold and so no delta to
measure; it is checked exactly as hard.

**Every shipped skill has to appear somewhere.** A skill that cannot be
benchmarked is a decision, not an oversight, so it goes in
:data:`NOT_BENCHMARKED` with the reason. `test_cases.py` asserts the catalogue
is covered by one or the other — the same shape as the contract layer's
"a declared package this layer says nothing about fails the suite", and what
keeps a 30-skill catalogue from quietly having 3 benchmarked skills.
"""

from __future__ import annotations

import importlib
import pkgutil

from .._benchmark import Case

#: Skills deliberately outside this layer, and why. A reason here should be
#: about the skill's *output*, not about the effort: "no number to score" is a
#: fact, "nobody has written it yet" is a TODO and belongs in an issue.
NOT_BENCHMARKED: dict[str, str] = {
    "write-a-skill": (
        "it emits a markdown file. There is no number with a knowable right "
        "answer, so there is nothing here for a programmatic verifier to score."
    ),
}


def _discover(deferred: bool) -> tuple[Case, ...]:
    """Every `CASE` in this package, in module-name order.

    Import-time discovery rather than a hand-maintained tuple: with a catalogue
    heading for 30 skills, a list someone has to remember to extend is a list
    that silently stops matching the directory.
    """
    found = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_") is not deferred:
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        case = getattr(module, "CASE", None)
        if not isinstance(case, Case):
            raise TypeError(
                f"{__name__}.{info.name} is in cases/ but exports no `CASE` "
                "(a module here is one skill's benchmark data, nothing else)"
            )
        found.append(case)
    return tuple(found)


#: Every skill benchmarked by this layer.
CASES: tuple[Case, ...] = _discover(deferred=False)

#: Cases for skills the runtime does not serve — the `_` prefix, `write-a-skill`'s
#: "private" marker, on the case module as well as on the skill file.
#:
#: **Kept apart from :data:`CASES`, and checked exactly as hard.** They are apart
#: because a deferred skill is absent from the catalog, so every arm of a 2x2 over
#: it would be the same arm: there is nothing to ablate and nothing to pay for.
#: They are checked because "deferred" is a statement about what the *runtime*
#: serves, not a licence for the data to rot — a case nobody verifies is a case
#: that is correct until the first time anyone looks, and the whole point of
#: banking one is that promoting the skill later does not begin by rebuilding it.
#:
#: `test_cases.py` runs every hermetic check over both tuples. The checks needing
#: the fixture itself skip when its data is not on this machine, which is the
#: ordinary state for a curated case and says so.
DEFERRED_CASES: tuple[Case, ...] = _discover(deferred=True)

__all__ = ["CASES", "DEFERRED_CASES", "NOT_BENCHMARKED"]
