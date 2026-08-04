"""Does this machine's fixture tree hold what the manifest says it holds.

Out-of-band on purpose, and that is the whole design decision here. The
manifest records a SHA per file; **verifying it is not part of a benchmark
run.** These are the large fixtures by construction, often on a network mount,
and hashing them would cost more than the run it guards — then do it again for
every arm.

So the split is by cost, not by importance:

- **shape and dtype, in-band** at build time (`_fixture._agrees_with_manifest`).
  A header read, and it catches the failure that matters most under
  non-decomposability: the file under this path not being the file the case was
  written against, which is the one remaining way a case name could quietly
  denote two experiments.
- **the hash, here**, run deliberately::

      uv run --no-sync pytest -m fixtures biopb-mcp/src/biopb_mcp/_tests/skills

  after syncing a tree, or when a result looks wrong. It catches altered pixels,
  which no header read can.

Everything skips when this machine has no tree, which is every machine but a
few and is not a failure.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ._fixture import (
    FIXTURE_DIR_ENV,
    MANIFEST_NAME,
    fixture_root,
    read_manifest,
)

pytestmark = pytest.mark.fixtures


def _tree() -> Path:
    root = fixture_root()
    if root is None:
        pytest.skip(f"{FIXTURE_DIR_ENV} is not set: this machine has no fixture tree")
    if not (root / MANIFEST_NAME).is_file():
        pytest.skip(f"{root} has no {MANIFEST_NAME}")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entries():
    return list(read_manifest().get("fixtures") or ())


def test_every_recorded_file_is_present_and_unaltered():
    """The check the in-band one cannot do: a file whose header still matches
    but whose pixels no longer do."""
    root = _tree()
    wrong = []
    for entry in _entries():
        here = root / entry["skill"] / entry["case_id"]
        for name, record in (entry.get("files") or {}).items():
            path = here / name
            if not path.is_file():
                wrong.append(f"{path} is recorded but missing")
                continue
            recorded = str(record.get("sha256", "")).strip()
            if not recorded:
                wrong.append(f"{path} has no sha256 in {MANIFEST_NAME}")
                continue
            if (got := _sha256(path)) != recorded:
                wrong.append(f"{path} hashes {got[:12]}…, recorded {recorded[:12]}…")
    assert not wrong, "\n".join(wrong)


def test_nothing_on_disk_is_missing_from_the_manifest():
    """How an unreviewed acquisition would otherwise reach a run: a directory
    someone copied in, with nothing anywhere saying where it came from.

    `OnDisk.available()` already refuses such a case, so this is the tree-wide
    view of the same rule — it names every one of them at once rather than one
    per skipped run.
    """
    root = _tree()
    known = {(e["skill"], e["case_id"]) for e in _entries()}
    stray = [
        f"{skill.name}/{case.name}"
        for skill in sorted(root.iterdir())
        if skill.is_dir()
        for case in sorted(skill.iterdir())
        if case.is_dir()
        and (case / "case.json").is_file()
        and (skill.name, case.name) not in known
    ]
    assert not stray, (
        f"these fixture directories are not in {MANIFEST_NAME}, so nothing "
        f"records what they are or whose they are: {stray}"
    )


def test_every_entry_says_whose_data_it_is():
    """Read here as well as at build time, so a tree can be audited without
    running anything: the obligation is the tree's, not one case's."""
    _tree()
    missing = [
        f"{e.get('skill')}/{e.get('case_id')}"
        for e in _entries()
        if not str(e.get("citation", "")).strip()
    ]
    assert not missing, f"these record no citation: {missing}"
