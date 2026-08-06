"""Which files in a skills directory are skills. The one copy of that rule.

Five readers have to agree about this, and each used to carry its own spelling:
the runtime loader (:mod:`._skills`), the authoring gate
(``_tests/skills/_validate``) and its fixtures, and
``.github/scripts/skill_contracts.py``, which proves in CI that the packages a
skill declares actually install. They drifted -- the last one skipped only the
prose docs, so it spent the deferral mechanism's whole first month proving the
packages of skills no agent can retrieve, and nothing failed until one of them
declared a dependency that would not resolve on 3.10.

The CI script runs *before any env exists*, so it cannot import this package; it
loads this file by path instead. Two constraints follow, and breaking either one
fails CI before there is an env to report it in:

* import nothing but the stdlib, and nothing relative;
* take a *name*, not a path. The packaged directory is an ``importlib.resources``
  Traversable, whose entries have ``.name`` but no ``.stem``.
"""

from __future__ import annotations

# Prose docs that may live alongside the skill files. Skipped by exact name
# rather than by "not kebab-case", which would silently swallow a misnamed real
# skill.
NOT_SKILLS = {"README", "ROADMAP"}


def is_skill_file(name: str) -> bool:
    """True if *name* is a skill the catalog serves.

    A leading ``_`` is the private marker (as in the kernel-plugin loader), used
    here for a skill written and banked but not served -- one whose value has
    not been shown for the model tier that consumes the catalog. Nothing
    downstream may read a deferred file: it is absent from the catalog, so a
    retrieval pin naming it or a package gate proving its dependencies is
    asserting something about a file no agent can reach.
    """
    return (
        name.endswith(".md")
        and not name.startswith("_")
        and name[: -len(".md")] not in NOT_SKILLS
    )
