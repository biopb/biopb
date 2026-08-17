"""One module per case, each holding a single `CASE`. Data, not code.

**This is the one place a case is defined.** There were two — one per suite,
under two engines that had drifted — and "where is the benchmark data" had two
answers a reader had to know to ask about separately. There is one kind of case
now, and one engine behind it.

Adding one is:

1. write ``cases/<name>.py`` exporting a module-level `CASE`: an id, the task
   prompt, the persona, the one `FixtureSpec` the case is written against, and
   a verifier for what the run leaves in the kernel;
2. there is no step 2. The module is discovered by being here.

One module is one `CASE`, so covering a subject two ways — a procedural fixture
and a real acquisition, which are two experiments and not two settings of one —
is two modules with two `case_id`s.

No test code, no registration line, no engine change. `test_bench.py`
parametrizes over :data:`CASES`, so a new case brings its own samples, report
and transcripts with it, and `test_cases.py` starts checking its persona and its
fixture by its arriving.

**Every module here is a case, and every case runs.** There used to be a second
tuple, `DEFERRED_CASES`, holding the cases of skills the runtime does not serve,
checked hermetically and never run. That distinction is gone along with the
field that expressed it.

**This package knows nothing about the skills catalog.** A case used to carry
``skill=``, and three things read it: a `--bench-cases=skills|tasks` filter, a
coverage ledger asserting every shipped skill was benchmarked or exempted, and a
rule about which agent could score it. All three are gone. Nothing here imports
`mcp/_skills_layout.py`, globs `_skills_data`, or can tell a served skill from a
banked one — so promoting or banking a skill is a change to the catalog and to
no file in this tree, and a case's `namespace` is a subject on disk that stays
put either way.

What is lost with the ledger is worth naming: nothing now notices a shipped
skill that no case covers. That was the check keeping a 30-skill catalogue from
quietly having 3 benchmarked skills, and the honest answer to "what does the
benchmark cover" is again "whatever anyone got round to". It was removed
deliberately — it was the coupling — and if it is wanted back it belongs on the
skills side, asserting from the catalog outwards, not here.

A case exists by someone having written it, and that is the whole registration.
"""

from __future__ import annotations

import importlib
import pkgutil

from .._case import Case


def _discover() -> tuple[Case, ...]:
    """Every `CASE` in this package, in module-name order.

    Import-time discovery rather than a hand-maintained tuple: with a catalogue
    heading for 30 skills, a list someone has to remember to extend is a list
    that silently stops matching the directory.

    A `_`-prefixed module is skipped as shared helper code, and that is the only
    thing it may be. It is deliberately *not* a way to bank a case out of the
    run: a case nothing runs is correct until the first time anyone looks.
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
                "(a module here is one case's data, and nothing else — name it "
                "with a leading underscore if it is shared helper code)"
            )
        found.append(case)
    return tuple(found)


#: Every case this layer runs. There is no second tuple.
CASES: tuple[Case, ...] = _discover()

__all__ = ["CASES"]
