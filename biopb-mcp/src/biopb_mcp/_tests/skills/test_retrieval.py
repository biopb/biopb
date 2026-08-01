"""Does a real description answer the phrasings a user would type?

The other half of `docs/skill-testing.md` §4. The matcher's *semantics* are
pinned in `_tests/test_skills.py` against synthetic catalogs; this reads the
shipped descriptions and asks whether they retrieve.

It reads them directly rather than through a copy. An earlier draft kept a
checked-in snapshot, because the skills lived in another repo — a snapshot goes
stale green (it keeps passing while the real descriptions drift), and that one
had in fact been taken from an unmerged branch, so it described skills no agent
had ever seen. Now the files are right here.

A phrasing table is content-coupled by nature, so it is written to fail loudly
about the right thing: entries name a skill id, and an id that no longer ships
is reported as a stale table rather than a retrieval bug.
"""

from __future__ import annotations

import pytest

from biopb_mcp.mcp import _skills


@pytest.fixture
def shipped_only(monkeypatch, tmp_path):
    """find_skills over the shipped set alone -- no local dir, whatever the
    machine running this happens to have in ~/.config/biopb/skills."""
    monkeypatch.setattr(
        _skills,
        "_setting",
        lambda path, default=None: {
            "skills_enabled": True,
            "skills_local_dir": str(tmp_path / "none"),
        }.get(path.rsplit(".", 1)[1], default),
    )


# How a user actually asks, in the few content words the tool asks for.
RETRIEVES = [
    ("stitch", "flatfield-and-stitch-tiles"),
    ("stitch tiles", "flatfield-and-stitch-tiles"),
    ("mosaic", "flatfield-and-stitch-tiles"),
    ("flatfield", "flatfield-and-stitch-tiles"),
    ("uneven illumination", "flatfield-and-stitch-tiles"),
    ("overlapping tiles", "flatfield-and-stitch-tiles"),
    ("measure", "calibrated-measurements"),
    ("measure microns", "calibrated-measurements"),
    ("physical units", "calibrated-measurements"),
    ("voxel spacing", "calibrated-measurements"),
    ("calibrated", "calibrated-measurements"),
    ("volumes microns", "calibrated-measurements"),
    ("segmentation", "segmentation-qc-metrics"),
    ("f1 iou", "segmentation-qc-metrics"),
    ("ground truth", "segmentation-qc-metrics"),
    ("split merged", "segmentation-qc-metrics"),
    ("qc", "segmentation-qc-metrics"),
    ("write skill", "write-a-skill"),
    ("authoring", "write-a-skill"),
]

# Queries that must not drag a skill in. Over-retrieval is not harmless: the
# agent is told to prefer a curated skill over improvising, so a false hit sends
# it down a workflow written for someone else's problem.
REJECTS = [
    ("deconvolution", "flatfield-and-stitch-tiles"),
    ("stitch", "calibrated-measurements"),
    ("measure", "segmentation-qc-metrics"),
    ("segmentation", "calibrated-measurements"),
    ("tiles", "segmentation-qc-metrics"),
    ("ground truth", "flatfield-and-stitch-tiles"),
]


def _require_shipped(skill_id: str) -> None:
    if skill_id not in {s["id"] for s in _skills.find_skills("")}:
        pytest.fail(
            f"the phrasing table names {skill_id!r}, which no longer ships. "
            "Update RETRIEVES/REJECTS -- this is a stale table, not a "
            "retrieval failure."
        )


@pytest.mark.parametrize("query,expected", RETRIEVES)
def test_shipped_skill_is_retrieved_for(shipped_only, query, expected):
    _require_shipped(expected)
    got = [s["id"] for s in _skills.find_skills(query)]
    assert expected in got, f"{query!r} did not surface {expected}; got {got}"


@pytest.mark.parametrize("query,unwanted", REJECTS)
def test_shipped_skill_is_not_retrieved_for(shipped_only, query, unwanted):
    _require_shipped(unwanted)
    got = [s["id"] for s in _skills.find_skills(query)]
    assert unwanted not in got, f"{query!r} wrongly surfaced {unwanted}; got {got}"


def test_every_shipped_skill_appears_in_the_phrasing_table(shipped_only):
    """A skill added without a phrasing entry is a skill nobody checked anyone
    can find. Cheap to satisfy, and the alternative is a table that quietly
    stops covering the catalog."""
    covered = {expected for _, expected in RETRIEVES}
    shipped = {s["id"] for s in _skills.find_skills("")}
    assert shipped <= covered, f"no phrasings for: {sorted(shipped - covered)}"
