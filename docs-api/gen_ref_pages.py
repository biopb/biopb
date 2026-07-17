"""mkdocs-gen-files script: auto-generate the biopb Python SDK API reference.

Walks ONLY the ``biopb.image`` and ``biopb.tensor`` packages and emits one
mkdocstrings page per public module. Skipped:

- generated protobuf stubs (``*_pb2`` / ``*_pb2_grpc``),
- private modules (leading underscore, e.g. ``_version``, ``__main__``).

Private *members* within each module are filtered by the mkdocstrings
``filters: ["!^_"]`` option in mkdocs.yml. Nothing this script writes is checked
in — pages live only in the in-memory build (see mkdocs-gen-files).
"""

from pathlib import Path

import mkdocs_gen_files

# docs-api/ -> repo root -> the src layout used by [tool.setuptools.packages.find].
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "main" / "python"

# Only these subpackages are part of the documented SDK surface.
PACKAGES = ("image", "tensor")

nav = mkdocs_gen_files.Nav()


def _is_proto(parts: tuple[str, ...]) -> bool:
    return any(p.endswith(("_pb2", "_pb2_grpc")) for p in parts)


for pkg in PACKAGES:
    for path in sorted((SRC / "biopb" / pkg).rglob("*.py")):
        parts = tuple(path.relative_to(SRC).with_suffix("").parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = Path(*parts, "index.md")
        elif parts[-1].startswith("_"):
            # private module, __main__, _version, etc.
            continue
        elif _is_proto(parts):
            continue
        else:
            doc_path = Path(*parts).with_suffix(".md")

        if not parts or _is_proto(parts):
            continue

        ident = ".".join(parts)
        full_doc_path = Path("reference", doc_path)
        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(f"# `{ident}`\n\n::: {ident}\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(REPO_ROOT))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
