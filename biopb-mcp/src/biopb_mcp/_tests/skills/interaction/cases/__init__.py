"""One module per skill, each holding a single `CASE`. Data, not code.

Adding a skill to the benchmark is:

1. write ``cases/<skill>.py`` exporting a module-level `CASE`: the task prompt,
   the persona holding the fact the fixture withholds, a builder for that
   fixture, and a verifier for what the run leaves in the kernel;
2. there is no step 2. The module is discovered by being here.

No test code, no registration line, no engine change. `test_benchmark.py`
parametrizes over :data:`CASES`, so a new case brings its own arms, report and
transcripts with it, and `test_cases.py` starts checking its persona and its
fixture by its arriving.

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


def _discover() -> tuple[Case, ...]:
    """Every `CASE` in this package, in module-name order.

    Import-time discovery rather than a hand-maintained tuple: with a catalogue
    heading for 30 skills, a list someone has to remember to extend is a list
    that silently stops matching the directory.
    """
    found = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
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
CASES: tuple[Case, ...] = _discover()

__all__ = ["CASES", "NOT_BENCHMARKED"]
