"""Rules the shipped skill files must satisfy, beyond well-formedness.

:mod:`._validate` answers "is this parseable and complete". These answer "is what
it claims still coherent" -- the `checklist:` grammar the agent resolves at
runtime, cross-skill links that must land somewhere, and the authoring
guardrails from `write-a-skill` that are mechanically checkable.

Failures here are authoring bugs, and they surface in the author's PR rather
than in a stranger's session. Since the skills moved into this repo, that PR is
also the one that can change the runtime they describe.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from ._validate import Report, split_frontmatter, validate
from .conftest import SKILLS_DIR, read_skill

# The live `checklist:` vocabulary. The agent resolves these against
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


def _checkout_root() -> Path:
    """The repository this module runs from, searched for by marker.

    Counted depths (`parents[5]`) are right until the file moves and then wrong
    without failing -- the agentbench fixtures were bitten by exactly that and
    switched to a search; see `_tests/agentbench/_fixture.checkout_root`. A
    missing marker is not an error here: the caller skips instead.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path("/nonexistent")


def test_the_package_gate_reads_the_same_skill_files(shipped_skill_files):
    """The third reader is `.github/scripts/skill_contracts.py`, which decides
    whose `pkg:` tokens CI installs and proves.

    It cannot import the package -- it runs before any env exists -- so it loads
    the layout rule from the checkout by path. That path is a string, and this
    suite is the only thing positioned to check it still resolves: the contracts
    workflow triggers on `_skills_data/**` and `_tests/skills/**`, so a rename
    under `mcp/` would not even run the gate that would have noticed.

    Asserting on the *files* rather than on the resulting package set is the
    point. A deferred skill that declares only workspace packages drops out of
    that set anyway, which is how the gate spent its first month walking the
    catalog wrongly with nothing to show for it.
    """
    script_path = _checkout_root() / ".github" / "scripts" / "skill_contracts.py"
    if not script_path.exists():
        pytest.skip(f"no checkout around this package ({script_path} is absent)")

    spec = importlib.util.spec_from_file_location("skill_contracts", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    assert {p.name for p in script.skill_files()} == {
        p.name for p in shipped_skill_files
    }


# --- checklist: grammar -------------------------------------------------------


def test_every_check_token_is_in_the_live_vocabulary(shipped):
    entries, _ = shipped
    bad = []
    for e in entries:
        for token in e.checklist:
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
    assert not bad, "checklist: tokens outside the vocabulary:\n" + "\n".join(bad)


def test_check_has_no_duplicates(shipped):
    entries, _ = shipped
    for e in entries:
        assert len(e.checklist) == len(set(e.checklist)), f"{e.id}: {e.checklist}"


def test_a_skill_that_drives_the_kernel_pins_a_biopb_mcp_floor(shipped):
    """`write-a-skill` sets this: a body that runs code in the session declares
    the first release exposing the interface it is written against. The
    metaskill is the exception -- it authors a file and touches nothing."""
    entries, _ = shipped
    for e in entries:
        drives_kernel = any(t in BARE_TOKENS for t in e.checklist)
        pinned = any(t.startswith("pkg:biopb-mcp>=") for t in e.checklist)
        if drives_kernel:
            assert pinned, f"{e.id} drives the session but no pkg:biopb-mcp>=X floor"
        else:
            assert not pinned, f"{e.id} pins biopb-mcp but declares nothing it drives"


def test_a_declared_plugin_is_called_through_its_module_name(shipped, bodies):
    """`plugin:<stem>` is also the name it is bound under in the kernel, so the
    body must call through it. A body that calls a bare function name was
    written against a namespace it does not get."""
    entries, _ = shipped
    for e in entries:
        for token in e.checklist:
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
    """Whatever the tier, step 1 is resolving `checklist:` -- there is no point
    asking the user which layer is truth if the scorer was never going to be
    there. A skill requiring nothing has nothing to check."""
    entries, _ = shipped
    for e in entries:
        if not e.checklist:
            continue
        first_step = _first_step(bodies[e.id], e.id)
        assert "checklist:" in first_step and "server_status" in first_step, (
            f"{e.id}: step 1 does not resolve checklist: against server_status"
        )
