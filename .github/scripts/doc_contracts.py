#!/usr/bin/env python3
"""Run the doc contract module once per declared package, each in its own env.

Driver for `.github/workflows/doc-contracts.yaml`. See that file for why this is
not a step in mcp-ci; the short version is that one shared resolution would
force every doc's package to co-exist with every other's, and the first pair
that cannot would break the suite rather than the doc.

For each third-party `packages:` entry a shipped doc declares:

  1. build a throwaway venv,
  2. install the workspace (`biopb[tensor]` + `biopb-mcp`, both from this
     checkout) plus that one package,
  3. run `_tests/docs/test_contracts.py` in it.

The workspace goes in because the module imports the store's frontmatter reader,
and because `biopb_mcp/__init__.py` reads its version from installed metadata --
`_version.py` is gitignored, so an uninstalled import fails in a fresh checkout.
Co-resolving one declared package with the workspace is exactly what the
satisfiability gate certifies is safe; co-resolving them with *each other* is
what this script exists to avoid.

Deliberately stdlib + packaging only: it runs before any env exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
MCP_PKG = ROOT / "biopb-mcp" / "src" / "biopb_mcp" / "mcp"
DOCS_DIR = MCP_PKG / "_docs_data"
# What runs in each per-package env: the signature contracts, and nothing else.
# These assertions are *derived from the shipped store* -- the packages below
# come out of the docs' own frontmatter, and each assertion pins an API a body
# quotes. Delete a doc and the work here changes.
#
# The interaction benchmark is deliberately NOT here, though it also needs a
# doc's package: it drives a real napari session against a real model, so it
# needs a GL display and API keys, and it reports rather than gates. It runs on
# a workstation -- see `biopb-mcp/src/biopb_mcp/_tests/bench/README.md`.
CONTRACTS = Path("biopb-mcp/src/biopb_mcp/_tests/docs/test_contracts.py")

# The workspace's own distributions: `packages:` is for third-party APIs, so a
# workspace name there has no external surface to assert.
WORKSPACE = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Only `packages:` is read, by line, rather than the block by a YAML parser. The
# store's own reader is tolerant by design -- a `description:` carrying a colon
# is valid there and is not valid YAML -- so parsing the whole block here would
# fail the gate on a doc the runtime serves happily.
PACKAGES = re.compile(r"^packages\s*:\s*(.*)$", re.MULTILINE)


def doc_files() -> list[Path]:
    """The shipped docs this gate may prove packages for: the ones the release
    offers.

    A `_`-prefixed doc is banked. It ships and reads back by id, but no session
    is told it exists -- and it is usually banked because its evidence is thin,
    which is exactly when a green gate would read as an endorsement. The rule is
    a name test rather than an import because this runs before any env exists.
    """
    return sorted(
        p
        for p in DOCS_DIR.rglob("*.md")
        if p.stem != "index"
        and not any(part.startswith("_") for part in p.relative_to(DOCS_DIR).parts)
    )


def declared_packages() -> list[str]:
    """Every third-party `packages:` spec in the shipped store, deduplicated."""
    specs: set[str] = set()
    for path in doc_files():
        match = FRONTMATTER.match(path.read_text(encoding="utf-8"))
        if not match:
            continue
        for line in PACKAGES.findall(match.group(1)):
            for spec in line.strip().strip("[]").split(","):
                spec = spec.strip().strip("\"'")
                if spec and Requirement(spec).name.lower() not in WORKSPACE:
                    specs.add(spec)
    return sorted(specs)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def check_one(spec: str, python: str) -> bool:
    """True if the contracts pass with *spec* installed in a fresh env."""
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "env"
        if run(["uv", "venv", str(venv), "--python", python]) != 0:
            print(f"::error::could not create an env for {spec}")
            return False
        install = run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python(venv)),
                f"{ROOT}[tensor]",
                str(ROOT / "biopb-mcp"),
                "pytest",
                "pyyaml",
                "packaging",
                spec,
            ]
        )
        if install != 0:
            # Not the same failure as a broken assertion: the doc declares a
            # package that will not install on this interpreter/platform at all.
            print(f"::error::{spec} does not install on {sys.platform} py{python}")
            return False
        return (
            run(
                [
                    str(venv_python(venv)),
                    "-m",
                    "pytest",
                    "-v",
                    "--color=yes",
                    "-p",
                    "no:cacheprovider",
                    str(CONTRACTS),
                ]
            )
            == 0
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", default=f"{sys.version_info[0]}.{sys.version_info[1]}"
    )
    args = parser.parse_args()

    specs = declared_packages()
    if not specs:
        # A real answer, not a failure: no shipped doc declares a third party.
        # Say so loudly, because a silently empty loop reads as "all passed".
        print(
            "::notice::no shipped doc declares a third-party package; nothing to check"
        )
        return 0

    print(f"declared packages: {', '.join(specs)}", flush=True)
    failed = [spec for spec in specs if not check_one(spec, args.python)]
    for spec in failed:
        print(f"::error::doc contracts failed for {spec}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
