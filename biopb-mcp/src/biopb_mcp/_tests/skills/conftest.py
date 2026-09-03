"""Shared fixtures for the skills authoring gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from biopb_mcp.mcp._skills_layout import is_catalog_file

from ._schema import REQUIRED_SECTIONS

# The skills this package ships. A real path, not a Traversable: these tests run
# from the checkout, and the strict validator wants glob/read_text.
SKILLS_DIR = Path(__file__).resolve().parents[2] / "mcp" / "_skills_data"


def read_skill(path: Path) -> str:
    """Read a skill file the way the runtime does -- as UTF-8, always.

    Never `Path.read_text()` bare here. Skill bodies carry µm, superscripts and
    em dashes, and a bare read uses the *locale* codec: on a Windows runner
    (cp1252) `10⁹` raises UnicodeDecodeError, and a body whose non-ASCII happens
    to be in cp1252's range decodes to mojibake with no error at all. The runtime
    reader (`mcp/_skills.py`) passes utf-8 at all three of its read sites; this
    keeps the gate reading the same bytes it does.
    """
    return path.read_text(encoding="utf-8")


def make_body(
    *,
    h1: str = "Do the thing",
    sections: tuple[str, ...] | None = None,
    extra: str = "",
) -> str:
    """A body carrying every required H2, so a test can vary one thing at a time."""
    names = REQUIRED_SECTIONS if sections is None else sections
    out = [f"# {h1}", ""]
    for name in names:
        out += [f"## {name.title()}", "", f"Prose for {name}.", ""]
    if extra:
        out += [extra, ""]
    return "\n".join(out)


def write_skill(
    directory: Path,
    stem: str,
    *,
    frontmatter: str | None = None,
    body: str | None = None,
    raw: str | None = None,
) -> Path:
    """Write `<stem>.md` into `directory`. `raw` bypasses assembly entirely, for
    the malformed-file cases."""
    path = directory / f"{stem}.md"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    fm = (
        frontmatter
        if frontmatter is not None
        else (
            f"id: {stem}\n"
            f"title: A Title\n"
            f"description: One sentence describing what the user gets.\n"
            f"tags: [testing]\n"
            f"version: 1.0.0\n"
            f"checklist: []\n"
        )
    )
    path.write_text(
        f"---\n{fm}---\n\n{body if body is not None else make_body()}",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


@pytest.fixture
def skill_factory(skills_dir: Path):
    """`skill_factory("my-skill", version="9.9.9")` -> Path, in a temp tree."""

    def _make(stem: str = "sample-skill", **kw) -> Path:
        return write_skill(skills_dir, stem, **kw)

    return _make


@pytest.fixture(scope="session")
def shipped_skill_files() -> list[Path]:
    """Every shipped skill file, excluding any prose docs beside them."""
    files = [
        p
        for p in sorted(SKILLS_DIR.glob("*"))
        if p.is_file() and is_catalog_file(p.name)
    ]
    assert files, f"no skill files found under {SKILLS_DIR}"
    return files
