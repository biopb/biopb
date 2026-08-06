"""One module per case, each holding a single `CASE`. Data, not code.

**This is the one place a case is defined.** There were two — one per suite,
under two engines that had drifted — and "where is the benchmark data" had two
answers a reader had to know to ask about separately. A case that names a skill
and a case that names none differ by one field now, so they live together and
the run options decide which of them a given invocation pays for.

Adding one is:

1. write ``cases/<name>.py`` exporting a module-level `CASE`: an id, the task
   prompt, the persona, the one `FixtureSpec` the case is written against, and
   a verifier for what the run leaves in the kernel — plus ``skill=`` if it is
   a claim about a skill rather than about the work;
2. there is no step 2. The module is discovered by being here.

One module is one `CASE`, so covering a subject two ways — a procedural fixture
and a real acquisition, which are two experiments and not two settings of one —
is two modules with two `case_id`s.

No test code, no registration line, no engine change. `test_bench.py`
parametrizes over :data:`CASES`, so a new case brings its own samples, report
and transcripts with it, and `test_cases.py` starts checking its persona and its
fixture by its arriving.

**Every module here is a case, and every case runs.** There used to be a second
tuple, `DEFERRED_CASES`, holding the cases of skills the runtime does not serve:
banked behind the `_` marker, checked hermetically, and never run, because a 2x2
over an absent catalog entry is four copies of one arm. Decoupling the case from
the skill dissolved that — such a case names no `skill`, keeps the skill's name
as its :attr:`~.._case.Case.namespace`, and runs the shipped corner like any
other case with no ablation. The work is real whether or not a skill for it is
served, which is the whole reason it was worth banking.

**Every shipped skill has to appear somewhere.** A skill that cannot be
benchmarked is a decision, not an oversight, so it goes in
:data:`NOT_BENCHMARKED` with the reason. `test_cases.py` asserts the catalogue
is covered by one or the other — the same shape as the contract layer's
"a declared package this layer says nothing about fails the suite", and what
keeps a 30-skill catalogue from quietly having 3 benchmarked skills.

That gate is also what catches a **promotion**: drop the `_` from a skill file
and the skill is suddenly shipped and uncovered, and the fix is to add `skill=`
to the case already sitting here under its name. Demote one and
`test_nothing_claims_to_cover_a_skill_that_does_not_ship` fires from the other
side. Both directions used to be pinned by two filename prefixes agreeing; they
are pinned by the shipped catalogue now, which is the thing that actually
changed.

Nothing equivalent constrains a case with no skill: there is no catalogue of
work to be complete against, and a case exists by someone having written it.
"""

from __future__ import annotations

import importlib
import pkgutil

from .._case import Case

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

__all__ = ["CASES", "NOT_BENCHMARKED"]
