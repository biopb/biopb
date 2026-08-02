"""Rules the shipped skill files must satisfy, beyond well-formedness.

:mod:`._validate` answers "is this parseable and complete". These answer "is what
it claims still coherent" -- the `requires:` grammar the agent resolves at
runtime, cross-skill links that must land somewhere, and the authoring
guardrails from `write-a-skill` that are mechanically checkable.

Failures here are authoring bugs, and they surface in the author's PR rather
than in a stranger's session. Since the skills moved into this repo, that PR is
also the one that can change the runtime they describe.
"""

from __future__ import annotations

import re

import pytest

from ._validate import Report, split_frontmatter, validate
from .conftest import SKILLS_DIR, read_skill

# The live `requires:` vocabulary. The agent resolves these against
# `server_status`, so a token outside the grammar is not a lint nit -- it is a
# requirement that silently never resolves.
BARE_TOKENS = {"viewer", "tensor", "dask"}
NAMESPACED = re.compile(r"^(ops|plugin|pkg):(.+)$")
# `>=` is a floor; `~=` is PEP 440's compatible release, i.e. a floor plus an
# upper bound at the next minor. Deliberately no comma-separated pair: the
# runtime reader (mcp/_skills.py) splits a `[a, b]` frontmatter list on every
# comma before it strips quotes, so `"pkg:x>=1,<2"` reaches the agent as two
# broken tokens while the strict parser here reads it correctly -- a mis-parse
# that would pass review and only appear in the field.
PKG_SPEC = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*((>=|~=)\d+(\.\d+)*)?$")
PLUGIN_STEM = re.compile(r"^[a-z_][a-z0-9_]*$")  # a module name, not kebab-case

# `[[link]]`s, ignoring any inside an inline code span -- `write-a-skill` quotes
# the syntax itself as `[[skill-id]]`, which is documentation, not a link.
CODE_SPAN = re.compile(r"`[^`\n]*`")
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")

# The catalog is reused across labs; a body naming one acquisition cannot be.
DATASET_SPECIFIC = re.compile(
    r"(/home/|/Users/|[A-Za-z]:\\\\|"
    r"\b(source_id|array_id)\s*=\s*[\"'][^\"']+[\"'])"
)


@pytest.fixture(scope="session")
def shipped():
    """(entries, report) for the shipped skills tree."""
    return validate(SKILLS_DIR)


@pytest.fixture(scope="session")
def bodies(shipped_skill_files) -> dict:
    out = {}
    for path in shipped_skill_files:
        _, body = split_frontmatter(
            path.read_text(encoding="utf-8"), path.name, Report()
        )
        out[path.stem] = body
    return out


def test_the_shipped_catalog_validates(shipped):
    entries, rep = shipped
    assert rep.errors == [], "\n".join(rep.errors)
    assert entries, "no entries emitted"


def test_no_warnings_either(shipped):
    """Warnings are non-fatal by design, but the curated set should carry none --
    an inferred title in here means a field was forgotten."""
    _, rep = shipped
    assert rep.warnings == [], "\n".join(rep.warnings)


def test_what_validates_is_what_the_runtime_loads(shipped):
    """The two readers must agree on which files are skills and what their ids
    are. The strict one gates authoring; the tolerant one in mcp/_skills.py is
    what the agent actually sees, and a file only the gate can read would pass
    review and then be invisible.
    """
    from biopb_mcp.mcp import _skills

    entries, _ = shipped
    loaded = {e["id"] for e in _skills._scan_shipped()}
    assert loaded == {e.id for e in entries}


# --- requires: grammar ----------------------------------------------------


@pytest.mark.parametrize("key", ["requires", "suggests"])
def test_every_token_is_in_the_live_vocabulary(shipped, key):
    """One grammar, two keys. `suggests:` is the same vocabulary resolved the
    same way -- what differs is the verdict when it is unmet, not the syntax."""
    entries, _ = shipped
    bad = []
    for e in entries:
        for token in getattr(e, key):
            if token in BARE_TOKENS:
                continue
            m = NAMESPACED.match(token)
            if not m:
                bad.append(f"{e.id}: {token!r}")
                continue
            kind, value = m.groups()
            if kind == "pkg" and not PKG_SPEC.match(value):
                bad.append(f"{e.id}: {token!r} (malformed package spec)")
            elif kind == "plugin" and not PLUGIN_STEM.match(value):
                bad.append(
                    f"{e.id}: {token!r} (plugin is a module stem, "
                    f"e.g. plugin:rolling_ball)"
                )
            elif kind == "ops" and not value:
                bad.append(f"{e.id}: {token!r} (ops needs a kind)")
    assert not bad, f"{key}: tokens outside the vocabulary:\n" + "\n".join(bad)


def test_neither_list_has_duplicates(shipped):
    entries, _ = shipped
    for e in entries:
        assert len(e.requires) == len(set(e.requires)), f"{e.id}: {e.requires}"
        assert len(e.suggests) == len(set(e.suggests)), f"{e.id}: {e.suggests}"


def test_the_workspace_floor_is_never_optional(shipped):
    """`pkg:biopb-mcp>=X` says which release the body is written against, so a
    session below it cannot run the skill at all -- there is no degraded path
    from an interface that does not exist yet. Suggesting it would also put the
    workspace into the availability grid, which resolves against PyPI and would
    report the checkout being "downgraded" to the last release."""
    entries, _ = shipped
    for e in entries:
        bad = [t for t in e.suggests if t.startswith("pkg:biopb-mcp")]
        assert not bad, f"{e.id} suggests the workspace floor: {bad}"


def test_a_skill_that_drives_the_kernel_pins_a_biopb_mcp_floor(shipped):
    """`write-a-skill` sets this: a body that runs code in the session declares
    the first release exposing the interface it is written against. The
    metaskill is the exception -- it authors a file and touches nothing."""
    entries, _ = shipped
    for e in entries:
        drives_kernel = any(t in BARE_TOKENS for t in e.requires)
        pinned = any(t.startswith("pkg:biopb-mcp>=") for t in e.requires)
        if drives_kernel:
            assert pinned, f"{e.id} requires the session but no pkg:biopb-mcp>=X floor"
        else:
            assert not pinned, f"{e.id} pins biopb-mcp but declares nothing it drives"


def test_a_declared_plugin_is_called_through_its_module_name(shipped, bodies):
    """`plugin:<stem>` is also the name it is bound under in the kernel, so the
    body must call through it. A body that calls a bare function name was
    written against a namespace it does not get."""
    entries, _ = shipped
    for e in entries:
        for token in e.requires:
            if token.startswith("plugin:"):
                stem = token.split(":", 1)[1]
                assert f"{stem}." in bodies[e.id], (
                    f"{e.id} declares {token} but never calls {stem}.<something>"
                )


# --- cross-skill links ----------------------------------------------------


def test_every_wikilink_resolves_to_a_shipped_skill(shipped, bodies):
    entries, _ = shipped
    known = {e.id for e in entries}
    dangling = []
    for skill_id, body in bodies.items():
        for target in WIKILINK.findall(CODE_SPAN.sub("", body)):
            if target not in known:
                dangling.append(f"{skill_id} -> [[{target}]]")
    assert not dangling, "links to skills that do not exist:\n" + "\n".join(dangling)


def test_no_skill_links_to_itself(bodies):
    for skill_id, body in bodies.items():
        targets = WIKILINK.findall(CODE_SPAN.sub("", body))
        assert skill_id not in targets, f"{skill_id} links to itself"


# --- authoring guardrails -------------------------------------------------


def test_no_body_names_a_specific_dataset(bodies):
    """`write-a-skill` guardrail: a hard-coded source_id, array_id, or path
    makes the recipe unusable by anyone else."""
    offenders = []
    for skill_id, body in bodies.items():
        for m in DATASET_SPECIFIC.finditer(body):
            offenders.append(f"{skill_id}: {m.group(0)!r}")
    assert not offenders, "dataset-specific references:\n" + "\n".join(offenders)


def test_descriptions_are_one_sentence_and_read_as_a_request(shipped):
    """`description` is most of what `find_skills` matches on. Kept short and
    phrased as what the user wants, not as an implementation summary."""
    entries, _ = shipped
    for e in entries:
        assert e.description.endswith("."), f"{e.id}: description lacks a period"
        assert len(e.description) <= 200, (
            f"{e.id}: description is {len(e.description)} chars"
        )
        # One sentence: no interior period followed by a capital.
        assert not re.search(r"\.\s+[A-Z]", e.description), (
            f"{e.id}: description is more than one sentence"
        )


def test_titles_are_a_single_imperative_line(shipped):
    entries, _ = shipped
    for e in entries:
        assert "\n" not in e.title
        assert len(e.title) <= 80, f"{e.id}: title is {len(e.title)} chars"
        assert not e.title.endswith("."), f"{e.id}: title should not end in a period"


def test_every_skill_carries_at_least_one_tag(shipped):
    entries, _ = shipped
    for e in entries:
        assert e.tags, f"{e.id} has no tags"


def test_bodies_stay_under_the_length_proxy(shipped_skill_files):
    """~200 lines is `write-a-skill`'s proxy for "an algorithm is living in the
    skill". Enforced with headroom, since it is a proxy and not the rule."""
    counts = {p.stem: len(read_skill(p).splitlines()) for p in shipped_skill_files}
    oversized = [f"{stem}: {n} lines" for stem, n in counts.items() if n > 250]
    assert not oversized, "\n".join(oversized)


def _first_step(body: str, skill_id: str) -> str:
    steps = body.split("## Steps", 1)
    assert len(steps) == 2, f"{skill_id}: no '## Steps' section"
    return steps[1].lstrip().split("\n2.", 1)[0]


def test_the_first_step_is_the_requirement_check(bodies, shipped):
    """Whatever the tier, step 1 is resolving `requires:` -- there is no point
    asking the user which layer is truth if the scorer was never going to be
    there. A skill requiring nothing has nothing to check."""
    entries, _ = shipped
    for e in entries:
        if not (e.requires or e.suggests):
            continue
        first_step = _first_step(bodies[e.id], e.id)
        assert "requires:" in first_step and "server_status" in first_step, (
            f"{e.id}: step 1 does not resolve requires: against server_status"
        )


def test_a_suggested_package_has_its_absence_answered_in_step_1(bodies, shipped):
    """What makes `suggests:` sound rather than a way to duck the gate.

    An optional package is allowed to be unavailable on some platforms
    (skill-testing.md §4b), and `find_skills` still retrieves the skill there --
    so an agent on an unlucky platform reads this body and has to find the
    fallback in it. If step 1 does not say what to do without the package, that
    agent improvises from a half-followed recipe, which is the failure a curated
    catalog exists to prevent.

    Named in step 1 specifically, because that is where the gap is discovered.
    """
    entries, _ = shipped
    for e in entries:
        for token in e.suggests:
            if not token.startswith("pkg:"):
                continue
            name = re.split(r"[><~=]", token.split(":", 1)[1])[0]
            first_step = _first_step(bodies[e.id], e.id).lower()
            assert name.lower() in first_step, (
                f"{e.id}: step 1 never mentions the suggested {name}"
            )
            assert any(w in first_step for w in ("without", "degraded", "fallback")), (
                f"{e.id}: step 1 mentions {name} but never says what happens "
                "without it -- an optional package needs a named degraded path"
            )
