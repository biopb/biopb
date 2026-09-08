"""The MCP settings nav must cover every config section (biopb/biopb#854).

``MCP_NAV`` in ``web/packages/app/src/components/admin/mcpSections.ts`` is a
hand-written presentation list, and a section appears on the settings page only
if it is named there. ``chat`` arrived in #845 and was never added, so
``chat.model`` -- which deliberately has no default, since guessing one bills you
for a model you did not pick -- was reachable only through the Raw JSON panel
until #853.

Omission is the default behaviour and three omissions are deliberate, so
"forgotten" and "intentionally hidden" are indistinguishable from the outside.
``MCP_HIDDEN_SECTIONS`` makes the deliberate ones explicit and this test asserts
the schema's sections are exactly the nav's plus that set. It lives on the Python
side because ``_SECTION_CLASSES`` is where a section is added; the schema only
reaches the browser at runtime via ``GET /api/mcp_config``.
"""

import re
from pathlib import Path

import pytest

from biopb_mcp._config_schema import build_mcp_config_schema

MCP_SECTIONS_TS = "web/packages/app/src/components/admin/mcpSections.ts"


def _repo_file(relative: str) -> Path:
    """The repo's copy of *relative*, or skip (installed package, no checkout)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    pytest.skip(f"{relative} not found -- not running from a repo checkout")


@pytest.fixture(scope="module")
def mcp_ts() -> str:
    return _repo_file(MCP_SECTIONS_TS).read_text()


@pytest.fixture(scope="module")
def nav_sections(mcp_ts) -> set:
    body = re.search(r"export const MCP_NAV[^=]*=\s*\[(.*?)\n\];", mcp_ts, re.S)
    assert body, "could not find the MCP_NAV array -- this parser needs updating"
    return set(re.findall(r'\b(?:id|section):\s*"([^"]+)"', body.group(1)))


@pytest.fixture(scope="module")
def hidden_sections(mcp_ts) -> set:
    block = re.search(
        r"MCP_HIDDEN_SECTIONS[^=]*=\s*new Set(?:<[^>]*>)?\((.*?)\);", mcp_ts, re.S
    )
    assert block, "could not find MCP_HIDDEN_SECTIONS -- this parser needs updating"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def test_the_parser_still_reads_the_file(nav_sections, hidden_sections):
    # Guards the guard: a refactor that defeats these regexes must fail loudly
    # rather than pass by finding nothing.
    assert {"pyramid", "chat", "raw"} <= nav_sections
    assert "widget" in hidden_sections


def test_every_config_section_is_navigable_or_hidden_on_purpose(
    nav_sections, hidden_sections
):
    sections = set(build_mcp_config_schema()["properties"])
    missing = sections - nav_sections - hidden_sections
    assert not missing, (
        f"config sections with no MCP settings nav entry: {sorted(missing)}. Add "
        f"an MCP_NAV item in {MCP_SECTIONS_TS}, or name it in MCP_HIDDEN_SECTIONS "
        "to say the omission is deliberate rather than forgotten."
    )


def test_the_nav_does_not_point_at_a_section_that_is_gone(mcp_ts):
    sections = set(build_mcp_config_schema()["properties"])
    declared = set(re.findall(r'\bsection:\s*"([^"]+)"', mcp_ts))
    stale = declared - sections
    assert not stale, (
        f"MCP_NAV renders sections the config no longer has: {sorted(stale)} "
        "(the panel would be empty)."
    )


def test_hidden_is_not_a_way_to_also_list_a_section(nav_sections, hidden_sections):
    overlap = nav_sections & hidden_sections
    assert not overlap, f"listed in MCP_NAV and hidden at once: {sorted(overlap)}"
