#!/usr/bin/env python3
"""Run the skills contract module once per declared package, each in its own env.

Driver for `.github/workflows/skill-contracts.yaml`. See that file for why this
is not a step in mcp-ci; the short version is that one shared resolution would
force every skill's package to co-exist with every other's, and the first pair
that cannot would break the suite rather than the skill.

For each third-party `pkg:` token a shipped skill declares:

  1. build a throwaway venv,
  2. install the workspace (`biopb[tensor]` + `biopb-mcp`, both from this
     checkout) plus that one package,
  3. run `_tests/skills/test_contracts.py` in it.

The workspace goes in because the module imports the skills validator, and
because `biopb_mcp/__init__.py` reads its version from installed metadata --
`_version.py` is gitignored, so an uninstalled import fails in a fresh checkout.
Co-resolving one skill package with the workspace is exactly what the
satisfiability gate certifies is safe; co-resolving skill packages with *each
other* is what this script exists to avoid.

Deliberately stdlib + pyyaml + packaging only: it runs before any env exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / "biopb-mcp" / "src" / "biopb_mcp" / "mcp" / "_skills_data"
CONTRACTS = Path("biopb-mcp/src/biopb_mcp/_tests/skills/test_contracts.py")

# Prose docs that live beside the skills. Keep in step with `_validate.NOT_SKILLS`.
NOT_SKILLS = {"README", "ROADMAP"}

# The workspace's own distributions: a floor on one is a statement about this
# repo's release history, not a third party's API.
WORKSPACE = {"biopb", "biopb-mcp", "biopb-tensor-server", "biopb-control"}

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def declared_packages() -> list[str]:
    """Every third-party `pkg:` spec in the shipped catalog, deduplicated."""
    specs: set[str] = set()
    for path in sorted(SKILLS_DIR.glob("*.md")):
        if path.stem in NOT_SKILLS:
            continue
        match = FRONTMATTER.match(path.read_text(encoding="utf-8"))
        if not match:
            continue
        frontmatter = yaml.safe_load(match.group(1)) or {}
        for token in frontmatter.get("requires") or []:
            token = str(token)
            if not token.startswith("pkg:"):
                continue
            spec = token.split(":", 1)[1]
            if Requirement(spec).name.lower() in WORKSPACE:
                continue
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
            # Not the same failure as a broken assertion: the skill declares a
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
        # A real answer, not a failure: no shipped skill declares a third party.
        # Say so loudly, because a silently empty loop reads as "all passed".
        print(
            "::notice::no shipped skill declares a third-party package; nothing to check"
        )
        return 0

    print(f"declared packages: {', '.join(specs)}", flush=True)
    failed = [spec for spec in specs if not check_one(spec, args.python)]
    for spec in failed:
        print(f"::error::skill contracts failed for {spec}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
