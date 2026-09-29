"""Shared fixtures: the shipped store, as files on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from biopb_mcp.mcp import _docs

# A real path, not a Traversable: these tests read the checkout, and they want
# glob/read_text.
DOCS_DIR = Path(__file__).resolve().parents[2] / "mcp" / "_docs_data"


def read_doc_file(path: Path) -> str:
    """Read a doc the way the runtime does -- as UTF-8, always.

    Never `Path.read_text()` bare here. Bodies carry µm, superscripts and em
    dashes, and a bare read uses the *locale* codec: on a Windows runner
    (cp1252) `10⁹` raises UnicodeDecodeError, and a body whose non-ASCII happens
    to be in cp1252's range decodes to mojibake with no error at all.
    """
    return path.read_text(encoding="utf-8")


def shipped_files(directory: Path = DOCS_DIR) -> list[Path]:
    """Every doc in *directory*, index included. All of them ship and read back."""
    return sorted(directory.rglob("*.md"))


def offered_files(directory: Path = DOCS_DIR) -> list[Path]:
    """The docs the release lists: `_`-prefixed banked ones and the index out.

    A banked doc ships and reads back by id, but no session is told it exists,
    so nothing downstream may assert about it -- a package gate proving its
    dependencies would be certifying a doc the release deliberately does not
    offer, and a doc is usually banked because its evidence is thin.
    """
    return [
        p
        for p in shipped_files(directory)
        if doc_id_of(p, directory) != _docs.INDEX_ID
        and not any(part.startswith("_") for part in p.relative_to(directory).parts)
    ]


def doc_id_of(path: Path, directory: Path = DOCS_DIR) -> str:
    return path.relative_to(directory).as_posix()[: -len(".md")]


def write_doc_file(directory: Path, doc_id: str, frontmatter: str = "") -> Path:
    """Write `<doc_id>.md` with *frontmatter* and a one-line body."""
    path = directory / f"{doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    front = f"---\n{frontmatter}---\n\n" if frontmatter else ""
    path.write_text(f"{front}# {doc_id}\n\nProse.\n", encoding="utf-8")
    return path


def declared_packages(path: Path) -> list[str]:
    """The `packages:` frontmatter of one doc, as requirement strings."""
    front = _docs.parse_frontmatter(read_doc_file(path))
    value = front.get("packages")
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value] if isinstance(value, list) else []


@pytest.fixture(scope="session")
def shipped_docs() -> list[Path]:
    """Every shipped doc file, index excluded. Banked ones included."""
    files = [p for p in shipped_files() if doc_id_of(p) != _docs.INDEX_ID]
    assert files, f"no docs found under {DOCS_DIR}"
    return files


@pytest.fixture(scope="session")
def offered_docs() -> list[Path]:
    """The shipped docs the release offers: banked ones excluded."""
    files = offered_files()
    assert files, f"the release offers nothing under {DOCS_DIR}"
    return files


@pytest.fixture(scope="session")
def seed_index() -> str:
    return read_doc_file(DOCS_DIR / "index.md")


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    return d
