"""Does a real description answer the phrasings a user would type?

The other half of `biopb-mcp/docs/skills.md` §8. The matcher's *semantics* are
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


# The **agent's** vocabulary, not the user's. `find_skills` is called by the
# agent at the start of a task, so the query is a domain term it chose after
# reading the request -- "drift correction", not "my stage moved". Chasing user
# idiom instead pushes synonyms into the description until it stops reading as a
# request, which is its own rule (test_descriptions_are_one_sentence_...).
RETRIEVES = [
    ("foci", "count-foci-per-cell"),
    ("spots", "count-foci-per-cell"),
    ("puncta", "count-foci-per-cell"),
    ("count foci", "count-foci-per-cell"),
    ("per cell", "count-foci-per-cell"),
    ("skeleton", "skeleton-network-metrics"),
    ("branching", "skeleton-network-metrics"),
    ("network length", "skeleton-network-metrics"),
    ("physical units", "skeleton-network-metrics"),
    ("segmentation", "segmentation-qc-metrics"),
    ("f1 iou", "segmentation-qc-metrics"),
    ("ground truth", "segmentation-qc-metrics"),
    ("split merged", "segmentation-qc-metrics"),
    ("qc", "segmentation-qc-metrics"),
    ("write skill", "write-a-skill"),
    ("authoring", "write-a-skill"),
    ("drift", "drift-correction"),
    ("drift correction", "drift-correction"),
    ("registration", "drift-correction"),
    ("stage drift", "drift-correction"),
    ("register time series", "drift-correction"),
    ("time lapse registration", "drift-correction"),
    ("flatfield", "flatfield"),
    ("flat field", "flatfield"),
    ("illumination", "flatfield"),
    ("uneven illumination", "flatfield"),
    ("vignetting", "flatfield"),
    ("shading", "flatfield"),
    ("shading correction tiles", "flatfield"),
    ("track", "track-objects"),
    ("tracking", "track-objects"),
    ("cell tracking", "track-objects"),
    ("lineage", "track-objects"),
    ("follow cells over time", "track-objects"),
    ("deconvolution", "deconvolve-widefield"),
    ("deconvolve", "deconvolve-widefield"),
    ("deconvolve widefield stack", "deconvolve-widefield"),
    ("blurred stack", "deconvolve-widefield"),
    ("widefield", "deconvolve-widefield"),
    ("restoration", "deconvolve-widefield"),
    ("filament", "detect-filaments"),
    ("filaments", "detect-filaments"),
    ("trace filaments", "detect-filaments"),
    ("filament width", "detect-filaments"),
    ("ridge detection", "detect-filaments"),
    ("centreline", "detect-filaments"),
    ("fret", "ratiometric-fret"),
    ("fret ratio", "ratiometric-fret"),
    ("ratiometric", "ratiometric-fret"),
    ("biosensor", "ratiometric-fret"),
    ("bleedthrough", "ratiometric-fret"),
    ("donor acceptor", "ratiometric-fret"),
    ("stitch", "stitch-tiles"),
    ("stitch tiles", "stitch-tiles"),
    ("mosaic", "stitch-tiles"),
    ("tile grid", "stitch-tiles"),
    ("overlapping tiles", "stitch-tiles"),
    ("smlm", "measure-smlm-resolution"),
    ("storm", "measure-smlm-resolution"),
    ("palm", "measure-smlm-resolution"),
    ("single molecule localization", "measure-smlm-resolution"),
    ("super-resolution resolution", "measure-smlm-resolution"),
    ("fourier ring correlation", "measure-smlm-resolution"),
    ("localization precision", "measure-smlm-resolution"),
]

# Queries that must not drag a skill in. Over-retrieval is not harmless: the
# agent is told to prefer a curated skill over improvising, so a false hit sends
# it down a workflow written for someone else's problem.
REJECTS = [
    ("measure", "segmentation-qc-metrics"),
    ("tiles", "segmentation-qc-metrics"),
    # `flatfield` and `drift-correction` are both "fix the images before
    # measuring", and both bodies talk about correcting a collection of frames.
    # The pair that must stay apart is the one an agent would land on from a
    # single word: a moved field of view is not an uneven one.
    ("drift", "flatfield"),
    ("registration", "flatfield"),
    ("illumination", "drift-correction"),
    # `drift-correction` and `track-objects` are the pair an agent lands on
    # from a single word about a movie, and they are opposites: one cancels
    # motion, the other measures it. Each body says so in its *When NOT to
    # use*, but that only helps once the right file has been retrieved.
    ("track", "drift-correction"),
    ("drift", "track-objects"),
    ("registration", "track-objects"),
    # `deconvolve-widefield` and `flatfield` are both "the image is not what
    # the sample looked like, fix it before measuring", and both bodies talk
    # about a correction applied to a whole stack. Blur along z is not uneven
    # illumination, and neither is a moved field of view.
    ("deconvolution", "flatfield"),
    ("deconvolution", "drift-correction"),
    ("illumination", "deconvolve-widefield"),
    ("drift", "deconvolve-widefield"),
    # `detect-filaments` produces a mask-like output and reports a size in
    # microns, which puts it one word away from the skills that own those.
    # Tracing filaments is not scoring a segmentation against a truth, and
    # tracing them is not measuring a network's length and branching either.
    ("filament", "segmentation-qc-metrics"),
    ("centreline", "segmentation-qc-metrics"),
    ("segmentation", "detect-filaments"),
    ("centreline", "skeleton-network-metrics"),
    # `ratiometric-fret` aligns two detectors and fixes channel intensities
    # before it divides, which puts it one word away from all three skills that
    # own those. Registering two cameras onto the same field is not correcting a
    # stage that moved, and a channel ratio is neither an illumination field nor
    # a calibrated object measurement.
    ("registration", "ratiometric-fret"),
    ("drift", "ratiometric-fret"),
    ("illumination", "ratiometric-fret"),
    ("measure", "ratiometric-fret"),
    ("fret", "flatfield"),
    ("fret", "drift-correction"),
    # `count-foci-per-cell` takes a parent segmentation as input and reports a
    # per-object table, which puts it one word away from the skill that scores a
    # segmentation and the one that detects the parents. Counting what is inside
    # objects is not judging the objects' outlines.
    ("spots", "segmentation-qc-metrics"),
    ("foci", "segmentation-qc-metrics"),
    ("segmentation", "count-foci-per-cell"),
    # `skeleton-network-metrics` and `count-foci-per-cell` both consume a mask
    # somebody else produced and both end in a per-structure number, so the
    # words each owns must not cross: a network has a length, a cell has a count.
    ("skeleton", "count-foci-per-cell"),
    ("foci", "skeleton-network-metrics"),
    ("branching", "segmentation-qc-metrics"),
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


def test_every_shipped_skill_is_retrievable_by_its_own_name(shipped_only):
    """No skill is stranded. The one retrieval check that needs no table and
    can never go stale: each id, read as words, retrieves that skill and only
    that skill. A new skill inherits it for free."""
    for s in _skills.find_skills(""):
        query = s["id"].replace("-", " ")
        got = [r["id"] for r in _skills.find_skills(query)]
        assert got == [s["id"]], f"{query!r} -> {got}"


#: What an agent actually typed, against what would have found the skill.
#: Lifted verbatim from `.bench-outcomes/session-20260810-162211`, where the
#: catalog was on, the skill was served, and eight of eleven eligible cases
#: retrieved nothing — the agent described the task instead of naming it.
#: Every table above this one is one to three words, which is the vocabulary
#: the catalog was written for and not the one it was asked in.
OVER_SPECIFIED = [
    ("estimate illumination flatfield correction stack", "illumination", "flatfield"),
    ("count foci per nucleus", "foci", "count-foci-per-cell"),
    ("trace filament centrelines width FWHM", "filament", "detect-filaments"),
    (
        "axial resolution bead deconvolution 3D fluorescence",
        "deconvolution",
        "deconvolve-widefield",
    ),
    (
        "track segmented cells time lapse labels speed lineage",
        "tracking",
        "track-objects",
    ),
]


@pytest.mark.parametrize("typed,shorter,expected", OVER_SPECIFIED)
def test_too_many_keywords_find_nothing_and_fewer_recover(
    shipped_only, typed, shorter, expected
):
    """The trap the tool docstring now names, pinned to measured queries.

    Asserted rather than fixed, because narrowing is what a keyword filter *is*:
    every keyword can only remove results. What the docstring owes the caller is
    that an empty result means "too many", and this is the check that the
    recovery it prescribes actually works.
    """
    _require_shipped(expected)
    assert _skills.find_skills([typed]) == [], (
        f"{typed!r} now retrieves; the trap this documents is gone and the "
        "docstring's advice should be revisited"
    )
    got = [s["id"] for s in _skills.find_skills([shorter])]
    assert expected in got, f"{shorter!r} did not recover {expected}; got {got}"


def test_a_phrase_inside_one_keyword_is_split(shipped_only):
    """The list says "keywords", and a model will still pass a phrase in one.

    Matching that element whole would fail against a skill whose title carries
    one word and whose description carries the other, which is the failure the
    list-shaped parameter is meant to make unlikely — not one it should
    introduce.
    """
    assert [s["id"] for s in _skills.find_skills(["stitch tiles"])] == ["stitch-tiles"]
    assert _skills.find_skills(["stitch tiles"]) == _skills.find_skills(
        ["stitch", "tiles"]
    )


def test_no_keywords_returns_the_whole_catalog(shipped_only):
    """The escape hatch the docstring points at when a search comes back empty.
    Every spelling of "nothing" has to reach it, since that is the call an agent
    makes after being told to widen."""
    everything = _skills.find_skills(())
    assert len(everything) > 1
    for nothing in ([], "", (), [""]):
        assert _skills.find_skills(nothing) == everything


def test_every_shipped_skill_appears_in_the_phrasing_table(shipped_only):
    """A skill added without a phrasing entry is a skill nobody checked anyone
    can find. Cheap to satisfy, and the alternative is a table that quietly
    stops covering the catalog."""
    covered = {expected for _, expected in RETRIEVES}
    shipped = {s["id"] for s in _skills.find_skills("")}
    assert shipped <= covered, f"no phrasings for: {sorted(shipped - covered)}"
