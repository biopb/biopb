"""mkdocs-gen-files script: pull the project's development/design docs into
the site under Development/.

Sources: the root ``docs/`` (the whole product's own design docs) and each
sibling package's own ``docs/`` (nested under that package's name). A file
whose name starts with ``_`` is a draft/internal note, not meant for the
public site, and is skipped. Nothing this script writes is checked in — pages
live only in the in-memory build (see mkdocs-gen-files), same as
gen_ref_pages.py.

Relative markdown links between two included docs (e.g. the tensor server's
roi-annotations.md <-> the webapp's roi-annotations-ui.md) are rewritten to
match the flattened output layout; a link to anything not in this doc set
(ARCHITECTURE.md, a source README, an excluded ``_``-prefixed doc) is left
as-is -- it was never going to resolve on the built site either way.
"""

import re
from pathlib import Path
from posixpath import relpath as posix_relpath

import mkdocs_gen_files

REPO_ROOT = Path(__file__).resolve().parent.parent

# (source docs/ dir, nav group under Development -- None means flat, for the
# root docs/, which is about the product as a whole rather than one package).
SOURCES = [
    (REPO_ROOT / "docs", None),
    (REPO_ROOT / "biopb-tensor-server" / "docs", "biopb-tensor-server"),
    (REPO_ROOT / "biopb-mcp" / "docs", "biopb-mcp"),
    (REPO_ROOT / "biopb-control" / "docs", "biopb-control"),
    (REPO_ROOT / "biopb-image-runtime" / "docs", "biopb-image-runtime"),
    (REPO_ROOT / "web" / "docs", "web"),
]

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_LINK = re.compile(r"(\]\()([^)]+)(\))")


def _title(path: Path, text: str) -> str:
    match = _H1.search(text)
    return match.group(1) if match else path.stem


# path.resolve() (the doc's real location) -> its doc_path in the output tree
# (relative to development/, e.g. "biopb-tensor-server/roi-annotations.md").
sources = []  # (path, group, doc_path)
targets = {}
for src_dir, group in SOURCES:
    if not src_dir.is_dir():
        continue
    for path in sorted(src_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        doc_path = Path(group, path.name) if group else Path(path.name)
        sources.append((path, group, doc_path))
        targets[path.resolve()] = doc_path


def _rewrite_links(path: Path, doc_path: Path, text: str) -> str:
    def replace(match: re.Match) -> str:
        target = match.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        new_target = targets.get(resolved)
        if new_target is None:
            return match.group(0)
        new_relative = posix_relpath(new_target.as_posix(), doc_path.parent.as_posix())
        return f"{match.group(1)}{new_relative}{match.group(3)}"

    return _LINK.sub(replace, text)


nav = mkdocs_gen_files.Nav()

for path, group, doc_path in sources:
    text = _rewrite_links(path, doc_path, path.read_text(encoding="utf-8"))
    title = _title(path, text)

    full_doc_path = Path("development", doc_path)
    parts = (group, title) if group else (title,)
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(text)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(REPO_ROOT))

with mkdocs_gen_files.open("development/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
