"""Copy the repo's own architecture docs into the shipped knowledge store.

    python biopb-mcp/tools/sync_architecture_docs.py          # rewrite
    python biopb-mcp/tools/sync_architecture_docs.py --check  # exit 1 if stale

The sources stay as they are, readable on GitHub. The store gets copies under
``_docs_data/architecture/`` whose relative ``.md`` links are rewritten to the
store's ``[[id]]`` form; a link to a file that is not copied becomes plain text.
A ``_``-prefixed source is a draft and is not copied.

Ids: ``docs/<n>.md`` -> ``architecture/<n>``; a package's ``ARCHITECTURE.md`` ->
``architecture/<pkg>``; its ``docs/<n>.md`` -> ``architecture/<pkg>/<n>``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "biopb-mcp/src/biopb_mcp/mcp/_docs_data/architecture"
PACKAGES = {
    "biopb-control": "control",
    "biopb-mcp": "mcp",
    "biopb-tensor-server": "tensor-server",
    "web": "web",
}

_LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\s#]+\.md)(#[^)\s]*)?\)")
_FENCE = re.compile(r"^\s*(```|~~~)")


def sources() -> dict[Path, str]:
    """Source path -> store id (without the ``architecture/`` prefix)."""
    out: dict[Path, str] = {}
    for path in sorted((REPO / "docs").glob("*.md")):
        out[path] = path.stem
    out[REPO / "development.md"] = "development"
    for directory, short in PACKAGES.items():
        arch = REPO / directory / "ARCHITECTURE.md"
        if arch.exists():
            out[arch] = short
        for path in sorted((REPO / directory / "docs").glob("*.md")):
            out[path] = f"{short}/{path.stem}"
    return {p: i for p, i in out.items() if not p.name.startswith("_")}


def rewrite(text: str, source: Path, ids: dict[Path, str]) -> str:
    def link(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if "://" in target:
            return m.group(0)
        dest = (source.parent / target).resolve()
        doc_id = ids.get(dest)
        if doc_id is None:
            return label
        ref = f"[[architecture/{doc_id}]]"
        if "/" in label or label.strip("`").endswith(".md"):
            return ref  # the label was only the path
        return f"{label} ({ref})"

    out, fenced = [], False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        out.append(line if fenced else _LINK.sub(link, line))
    return "".join(out)


def build() -> dict[Path, str]:
    ids = {p.resolve(): i for p, i in sources().items()}
    return {
        OUT / f"{doc_id}.md": rewrite(src.read_text(encoding="utf-8"), src, ids)
        for src, doc_id in sources().items()
    }


def main(argv: list[str]) -> int:
    want = build()
    have = set(OUT.rglob("*.md")) if OUT.exists() else set()
    stale = sorted(
        str(p.relative_to(REPO))
        for p in want.keys() | have
        if want.get(p) != (p.read_text(encoding="utf-8") if p.exists() else None)
    )
    if "--check" in argv:
        if stale:
            print("stale (run biopb-mcp/tools/sync_architecture_docs.py):")
            print("\n".join(f"  {s}" for s in stale))
        return 1 if stale else 0
    for path in have - want.keys():
        path.unlink()
    for path, text in want.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"{len(want)} docs under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
