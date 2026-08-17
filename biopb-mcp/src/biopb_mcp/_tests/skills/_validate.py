"""Strict reader for skill files: the authoring gate.

The single choke point for skill-file format variation
(``biopb-mcp/docs/skills.md`` §4): tolerant read -> strict result.
Warnings are advisory; ERRORS mean the file does not ship.

Pure with respect to the filesystem — it writes nothing. It used to end by
emitting a ``catalog.json``; skills now ship as package data and the frontmatter
*is* the metadata, so validation is all that is left, and pytest is the gate
that used to be a CLI.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from biopb_mcp.mcp._skills_layout import is_skill_file

from ._schema import (
    CURRENT_SPEC_VERSION,
    KEBAB,
    REQUIRED_SECTIONS,
    SEMVER,
    SkillEntry,
    coerce_list,
)

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class Report:
    """Per-run diagnostics. Deliberately not module state: the validator is
    called more than once per process by the test suite, and a global
    accumulator would carry one run's errors into the next."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, fname: str, msg: str) -> None:
        self.errors.append(f"ERROR {fname}: {msg}")

    def warn(self, fname: str, msg: str) -> None:
        self.warnings.append(f"warn  {fname}: {msg}")


def split_frontmatter(text: str, fname: str, rep: Report):
    m = _FRONTMATTER.match(text.replace("\r\n", "\n"))
    if not m:
        rep.err(fname, "missing or malformed YAML frontmatter (--- ... ---)")
        return None, ""
    try:
        return yaml.safe_load(m.group(1)) or {}, m.group(2)
    except yaml.YAMLError as e:
        rep.err(fname, f"YAML parse error: {e}")
        return None, ""


def migrate(fm: dict, fname: str, rep: Report) -> dict:
    """Up-convert older authoring dialects to CURRENT_SPEC_VERSION."""
    sv = int(fm.get("spec_version", 1) or 1)
    if sv > CURRENT_SPEC_VERSION:
        rep.warn(
            fname, f"spec_version {sv} newer than supported {CURRENT_SPEC_VERSION}"
        )
    # Example future hook:
    # if sv < 2: fm = _v1_to_v2(fm); fm["spec_version"] = 2
    fm["spec_version"] = min(sv, CURRENT_SPEC_VERSION)
    return fm


def first_h1(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def h2_sections(body: str) -> set[str]:
    """Normalized H2 headings, so section matching is insensitive to case,
    trailing punctuation, and internal whitespace (`## When NOT to use.` ==
    `when not to use`). Deeper headings are structure *within* a section."""
    return {
        re.sub(r"\s+", " ", h).strip(" .:").lower()
        for h in re.findall(r"^##\s+(.+?)\s*$", body, re.MULTILINE)
    }


def _stem_ok(stem: str, fname: str, rep: Report) -> bool:
    if not KEBAB.match(stem):
        rep.err(fname, f"filename stem {stem!r} must be kebab-case")
        return False
    return True


def process(path: Path, rep: Report) -> SkillEntry | None:
    fname = path.name
    stem = path.stem
    raw = path.read_text(encoding="utf-8")
    before = len(rep.errors)  # errors raised by THIS file, not by an earlier one
    fm, body = split_frontmatter(raw, fname, rep)
    if fm is None:
        return None
    if not isinstance(fm, dict):
        rep.err(fname, "frontmatter is not a mapping")
        return None

    fm = migrate(fm, fname, rep)

    # --- inference / coercion (tolerant read) ---
    if not _stem_ok(stem, fname, rep):
        return None
    fm.setdefault("id", stem)
    if fm["id"] != stem:
        rep.err(fname, f"id {fm['id']!r} must equal filename stem {stem!r}")

    description = (fm.get("description") or "").strip()
    if not description:
        rep.err(fname, "missing required field: description")
        return None

    title = (fm.get("title") or "").strip()
    if not title:
        title = first_h1(body) or stem.replace("-", " ").title()
        rep.warn(fname, "title inferred (add an explicit `title:`)")

    tags = [t.lower() for t in coerce_list(fm.get("tags"))]

    version = str(fm.get("version") or "0.0.0")
    if not SEMVER.match(version):
        rep.err(fname, f"version must be MAJOR.MINOR.PATCH, got {version!r}")

    if "requires" in fm and "checklist" not in fm:
        # Strict side, so this is an authoring error rather than the tolerant
        # alias the runtime reader keeps for the user's own older skills.
        rep.err(fname, "`requires:` was renamed to `checklist:`")
    checklist = [str(c) for c in coerce_list(fm.get("checklist", fm.get("requires")))]

    if not body.strip():
        rep.err(fname, "empty body")
    else:
        if not first_h1(body):
            rep.warn(fname, "body has no H1 heading")
        missing = [s for s in REQUIRED_SECTIONS if s not in h2_sections(body)]
        if missing:
            rep.err(fname, "missing required H2 section(s): " + ", ".join(missing))

    if len(rep.errors) > before:  # don't emit an entry built from invalid input
        return None

    return SkillEntry(
        id=fm["id"],
        title=title,
        description=description,
        tags=tags,
        version=version,
        spec_version=int(fm["spec_version"]),
        checklist=checklist,
    )


def validate(skills_dir: Path) -> tuple[list[SkillEntry], Report]:
    """Read every skill file in `skills_dir`; return its entries and diagnostics."""
    rep = Report()
    entries: list[SkillEntry] = []
    seen: set[str] = set()
    for path in sorted(skills_dir.glob("*.md")):
        # The gate and the runtime have to agree about which files are skills
        # (`test_what_validates_is_what_the_runtime_loads`), so the rule is not
        # spelled here -- see `mcp/_skills_layout.py`.
        if not is_skill_file(path.name):
            continue
        entry = process(path, rep)
        if entry is None:
            continue
        if entry.id in seen:
            rep.err(path.name, f"duplicate id {entry.id!r}")
            continue
        seen.add(entry.id)
        entries.append(entry)
    return entries, rep
