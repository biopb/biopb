"""Which files in a skills directory are skills. The one copy of that rule.

Five readers have to agree about this, and each used to carry its own spelling:
the runtime loader (:mod:`._skills`), the authoring gate
(``_tests/skills/_validate``) and its fixtures, and
``.github/scripts/skill_contracts.py``, which proves in CI that the packages a
skill declares actually install. They drifted -- the last one skipped only the
prose docs, so it spent the deferral mechanism's whole first month proving the
packages of skills no agent can retrieve, and nothing failed until one of them
declared a dependency that would not resolve on 3.10.

Two rules, not one. :func:`is_skill_file` holds for any skills directory;
:func:`is_catalog_file` adds the prose-doc names, which are a property of the
repo directory those docs sit in and not of the user's own (#725).

The CI script runs *before any env exists*, so it cannot import this package; it
loads this file by path instead. Two constraints follow, and breaking either one
fails CI before there is an env to report it in:

* import nothing but the stdlib, and nothing relative;
* take a *name*, not a path. The packaged directory is an ``importlib.resources``
  Traversable, whose entries have ``.name`` but no ``.stem``.
"""

from __future__ import annotations

# Prose docs that may live alongside the skill files *in the repo*. Skipped by
# exact name rather than by "not kebab-case", which would silently swallow a
# misnamed real skill. Repo-only, hence :func:`is_catalog_file` rather than
# :func:`is_skill_file`: the user's local dir holds nothing but their own
# skills, so a ``readme.md`` there is one of them (#725). Compared case-folded.
NOT_SKILLS = {"readme", "roadmap"}

_SUFFIX = ".md"


def is_markdown_name(name: str) -> bool:
    """True if *name* is a markdown file, whatever the case of its extension.

    Case-folded because the local skills dir is the one uncontrolled input, and
    a user who writes ``Recipe.MD`` there means a skill. Windows makes that
    likelier -- the OS treats filename case as irrelevant everywhere else, so
    this would be the single place it is not, with nothing to say so.
    """
    return name.lower().endswith(_SUFFIX)


def is_skill_file(name: str) -> bool:
    """True if *name* is a skill file: markdown, and not deferred.

    A leading ``_`` is the private marker (as in the kernel-plugin loader), used
    here for a skill written and banked but not served -- one whose value has
    not been shown for the model tier that consumes the catalog. Nothing
    downstream may read a deferred file: it is absent from the catalog, so a
    retrieval pin naming it or a package gate proving its dependencies is
    asserting something about a file no agent can reach.

    That is a rule about the *file*, not about the subject. The benchmark runs a
    case for a banked skill like any other (`_tests/bench/cases/`) -- it names no
    skill, reads nothing here, and measures whether the work gets done at all.
    """
    return is_markdown_name(name) and not name.startswith("_")


def is_catalog_file(name: str) -> bool:
    """True if *name* is a skill of the **shipped** catalog.

    :func:`is_skill_file` plus the prose docs that ship beside it in
    ``_skills_data/``. Readers of the user's local dir want the narrower rule:
    excluding a name there drops a skill that works.
    """
    return is_skill_file(name) and name[: -len(_SUFFIX)].lower() not in NOT_SKILLS
