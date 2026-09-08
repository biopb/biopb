"""The admin settings nav must cover every config section (biopb/biopb#948).

``ADMIN_NAV`` in ``web/packages/app/src/components/admin/adminSections.ts`` is a
hand-curated presentation list -- section order, prose, and the common/advanced
field split, none of which the schema carries. It also decides *visibility*: a
section with no entry is reachable only through the Raw JSON panel, and, because
``navIdForErrorPath`` maps a validation error onto a nav item by its section, an
error under such a section marks nothing while the page disables Save and points
at "the marked sections in the sidebar".

Nothing announced that, because omission is indistinguishable from a deliberate
hide -- ``annotations`` sat unreachable from #932 until #948. So a deliberate
omission has to say so in ``ADMIN_HIDDEN_SECTIONS``, and this test asserts the
schema's sections are exactly the nav's plus that set. The check lives here, on
the Python side, because that is where a new section is *added*: the failure
lands in the run of whoever adds it.
"""

import re
from pathlib import Path

import pytest
from biopb_tensor_server.core.config_schema import build_config_schema

ADMIN_SECTIONS_TS = "web/packages/app/src/components/admin/adminSections.ts"


def _repo_file(relative: str) -> Path:
    """The repo's copy of *relative*, or skip (installed package, no checkout)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    pytest.skip(f"{relative} not found -- not running from a repo checkout")


def _string_set(source: str, pattern: str, what: str) -> set:
    """Every double-quoted literal inside the region *pattern* captures."""
    match = re.search(pattern, source, re.S)
    assert match, f"could not find {what} -- this parser needs updating"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


@pytest.fixture(scope="module")
def admin_ts() -> str:
    return _repo_file(ADMIN_SECTIONS_TS).read_text()


@pytest.fixture(scope="module")
def nav_sections(admin_ts) -> set:
    """The config sections ADMIN_NAV renders.

    A nav item addresses its section by `section:`, or by `id:` alone for the two
    bespoke panels (sources / credentials), so both keys count. Ids that name no
    section (`raw`) are harmless: nothing in the schema answers to them.
    """
    body = re.search(r"export const ADMIN_NAV[^=]*=\s*\[(.*?)\n\];", admin_ts, re.S)
    assert body, "could not find the ADMIN_NAV array -- this parser needs updating"
    return set(re.findall(r'\b(?:id|section):\s*"([^"]+)"', body.group(1)))


@pytest.fixture(scope="module")
def hidden_sections(admin_ts) -> set:
    return _string_set(
        admin_ts,
        r"ADMIN_HIDDEN_SECTIONS[^=]*=\s*new Set(?:<[^>]*>)?\((.*?)\);",
        "ADMIN_HIDDEN_SECTIONS",
    )


def test_the_parser_still_reads_the_file(nav_sections):
    # Guards the guard: a refactor that defeats these regexes must fail loudly
    # rather than pass by finding nothing.
    assert {"sources", "credentials", "cache", "raw"} <= nav_sections


def test_every_config_section_is_navigable_or_hidden_on_purpose(
    nav_sections, hidden_sections
):
    sections = set(build_config_schema()["properties"])
    missing = sections - nav_sections - hidden_sections
    assert not missing, (
        f"config sections with no admin nav entry: {sorted(missing)}. Add an "
        f"ADMIN_NAV item in {ADMIN_SECTIONS_TS} (a validation error under a "
        "section the nav doesn't list marks nothing in the sidebar), or name it "
        "in ADMIN_HIDDEN_SECTIONS to say the omission is deliberate."
    )


def test_the_nav_does_not_point_at_a_section_that_is_gone(admin_ts):
    sections = set(build_config_schema()["properties"])
    declared = set(re.findall(r'\bsection:\s*"([^"]+)"', admin_ts))
    stale = declared - sections
    assert not stale, (
        f"ADMIN_NAV renders sections the config no longer has: {sorted(stale)} "
        "(the panel would be empty)."
    )


def test_hidden_is_not_a_way_to_also_list_a_section(nav_sections, hidden_sections):
    overlap = nav_sections & hidden_sections
    assert not overlap, f"listed in ADMIN_NAV and hidden at once: {sorted(overlap)}"
