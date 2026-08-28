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

**Kernel plugins get the same treatment** at the bottom of this file, because
`list_skills` returns them too. Their descriptions are module docstrings rather
than a curated `description:` field, which makes the pinning more necessary, not
less: nobody writes a docstring for a search index, and the fix for a miss is an
edit to load-bearing documentation.
"""

from __future__ import annotations

import pytest

from biopb_mcp.mcp import _skills


@pytest.fixture
def shipped_only(monkeypatch, tmp_path):
    """list_skills over the shipped set alone -- no local dir, whatever the
    machine running this happens to have in ~/.config/biopb/skills."""
    monkeypatch.setattr(
        _skills,
        "_setting",
        lambda path, default=None: {
            "skills_enabled": True,
            "skills_local_dir": str(tmp_path / "none"),
            # This table is about whether a *description* retrieves, so the
            # plugin rows are out — they would otherwise be read off whatever
            # this machine has in ~/.config/biopb/kernel, which is not a fact
            # about the shipped catalog.
            "skills_index_plugins": False,
        }.get(path.rsplit(".", 1)[1], default),
    )


# The **agent's** vocabulary, not the user's. `list_skills` is called by the
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
    ("serial sections", "align-stack-by-features"),
    ("sections", "align-stack-by-features"),
    ("align stack", "align-stack-by-features"),
    ("alignment", "align-stack-by-features"),
    ("slices", "align-stack-by-features"),
    ("scribbles", "pixel-classifier-segmentation"),
    ("pixel classifier", "pixel-classifier-segmentation"),
    ("classifier", "pixel-classifier-segmentation"),
    ("classification", "pixel-classifier-segmentation"),
    ("annotation", "pixel-classifier-segmentation"),
]

# Queries that must not drag a skill in. Over-retrieval is not harmless: the
# agent is told to prefer a curated skill over improvising, so a false hit sends
# it down a workflow written for someone else's problem.
REJECTS = [
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
    # Tracing filaments is not measuring a network's length and branching.
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
    # per-object table, which puts it one word away from the skill that detects
    # the parents. Counting what is inside objects is not producing them.
    ("segmentation", "count-foci-per-cell"),
    # `skeleton-network-metrics` and `count-foci-per-cell` both consume a mask
    # somebody else produced and both end in a per-structure number, so the
    # words each owns must not cross: a network has a length, a cell has a count.
    ("skeleton", "count-foci-per-cell"),
    ("foci", "skeleton-network-metrics"),
    # `align-stack-by-features` is the third registration skill, and the word
    # "registration" is deliberately *not* rejected for it -- it genuinely is
    # one. What must not cross is the axis being registered: serial sections cut
    # from a block are not a time-lapse that drifted, and not a tile grid
    # either. Each is a different thing to hold still.
    ("drift", "align-stack-by-features"),
    ("track", "align-stack-by-features"),
    ("stitch", "align-stack-by-features"),
    ("serial", "drift-correction"),
    ("sections", "stitch-tiles"),
    # `pixel-classifier-segmentation` produces a mask, which puts it one word
    # away from every skill that consumes one. Labelling pixels by class is not
    # tracing, counting or measuring what the classes turn out to be.
    ("filament", "pixel-classifier-segmentation"),
    ("foci", "pixel-classifier-segmentation"),
    ("skeleton", "pixel-classifier-segmentation"),
    ("track", "pixel-classifier-segmentation"),
]


def _require_shipped(skill_id: str) -> None:
    if skill_id not in {s["id"] for s in _skills.list_skills("")}:
        pytest.fail(
            f"the phrasing table names {skill_id!r}, which no longer ships. "
            "Update RETRIEVES/REJECTS -- this is a stale table, not a "
            "retrieval failure."
        )


@pytest.mark.parametrize("query,expected", RETRIEVES)
def test_shipped_skill_is_retrieved_for(shipped_only, query, expected):
    _require_shipped(expected)
    got = [s["id"] for s in _skills.list_skills(query)]
    assert expected in got, f"{query!r} did not surface {expected}; got {got}"


@pytest.mark.parametrize("query,unwanted", REJECTS)
def test_shipped_skill_is_not_retrieved_for(shipped_only, query, unwanted):
    _require_shipped(unwanted)
    got = [s["id"] for s in _skills.list_skills(query)]
    assert unwanted not in got, f"{query!r} wrongly surfaced {unwanted}; got {got}"


def test_every_shipped_skill_is_retrievable_by_its_own_name(shipped_only):
    """No skill is stranded. The one retrieval check that needs no table and
    can never go stale: each id, read as words, retrieves that skill and only
    that skill. A new skill inherits it for free."""
    for s in _skills.list_skills(""):
        query = s["id"].replace("-", " ")
        got = [r["id"] for r in _skills.list_skills(query)]
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
    assert _skills.list_skills([typed]) == [], (
        f"{typed!r} now retrieves; the trap this documents is gone and the "
        "docstring's advice should be revisited"
    )
    got = [s["id"] for s in _skills.list_skills([shorter])]
    assert expected in got, f"{shorter!r} did not recover {expected}; got {got}"


def test_a_phrase_inside_one_keyword_is_split(shipped_only):
    """The list says "keywords", and a model will still pass a phrase in one.

    Matching that element whole would fail against a skill whose title carries
    one word and whose description carries the other, which is the failure the
    list-shaped parameter is meant to make unlikely — not one it should
    introduce.
    """
    assert [s["id"] for s in _skills.list_skills(["stitch tiles"])] == ["stitch-tiles"]
    assert _skills.list_skills(["stitch tiles"]) == _skills.list_skills(
        ["stitch", "tiles"]
    )


def test_no_keywords_returns_the_whole_catalog(shipped_only):
    """The escape hatch the docstring points at when a search comes back empty.
    Every spelling of "nothing" has to reach it, since that is the call an agent
    makes after being told to widen."""
    everything = _skills.list_skills(())
    assert len(everything) > 1
    for nothing in ([], "", (), [""]):
        assert _skills.list_skills(nothing) == everything


def test_every_shipped_skill_appears_in_the_phrasing_table(shipped_only):
    """A skill added without a phrasing entry is a skill nobody checked anyone
    can find. Cheap to satisfy, and the alternative is a table that quietly
    stops covering the catalog."""
    covered = {expected for _, expected in RETRIEVES}
    shipped = {s["id"] for s in _skills.list_skills("")}
    assert shipped <= covered, f"no phrasings for: {sorted(shipped - covered)}"


# --------------------------------------------------------------------------- #
# The same question, asked of the kernel plugins
# --------------------------------------------------------------------------- #
#
# `list_skills` returns plugins too, described by their module docstring rather
# than by a curated `description:`. That makes retrieval a property of prose
# nobody wrote for a search index -- a docstring is written for someone already
# reading the file -- so it needs pinning exactly like the skills above, and for
# a sharper reason: a skill's description can be rewritten to retrieve better,
# while a plugin's is load-bearing documentation that happens to be indexed.
#
# Two of these entries are here because they *failed* when the table was first
# written, and the fix was in the docstring, not the matcher:
#
#   ("resolution", "rolling_ball")   matched "full-resolution image"
#   ("iou",        "chunked_label")  matched the "iou" inside "obvious"
#
# The second is the one to remember. Terms are substrings, not tokens, so a
# three-letter acronym is one incidental word away from a false hit anywhere in
# the catalog -- and the acronyms are exactly what an agent types.


@pytest.fixture
def shipped_with_plugins(monkeypatch, tmp_path):
    """The shipped skills plus the plugins biopb-mcp actually seeds.

    Seeded through `seed_kernel_plugins`, the installer's own path, so this
    asks about the files users get rather than a fixture's idea of them. The
    plugin dir is patched at `biopb._locations` because `mcp_plugin_dir` reads
    ``$BIOPB_CONFIG_HOME`` ahead of ``$HOME``.
    """
    import biopb._locations as locations

    from biopb_mcp.plugins._seed import seed_kernel_plugins

    kernel_dir = tmp_path / "kernel"
    seed_kernel_plugins(kernel_dir)
    monkeypatch.setattr(locations, "mcp_plugin_dir", lambda: kernel_dir)
    monkeypatch.setattr(
        _skills,
        "_setting",
        lambda path, default=None: {
            "skills_enabled": True,
            "skills_local_dir": str(tmp_path / "none"),
            "skills_index_plugins": True,
            "namespace_enabled": True,
        }.get(path.rsplit(".", 1)[1], default),
    )


PLUGIN_RETRIEVES = [
    # The acronym has to be in the opening paragraph, not just spelled out
    # further down: `blurb` stops at the first blank line after the summary.
    ("frc", "image_resolution"),
    ("fourier ring correlation", "image_resolution"),
    ("decorrelation", "image_resolution"),
    ("nanometres", "image_resolution"),
    ("smlm", "image_resolution"),
    ("rolling ball", "rolling_ball"),
    ("background subtraction", "rolling_ball"),
    ("subtract background", "rolling_ball"),
    ("imagej", "rolling_ball"),
    ("sternberg", "rolling_ball"),
    ("connected components", "chunked_label"),
    ("dask label", "chunked_label"),
    ("chunk boundaries", "chunked_label"),
    # The failure mode is the point of the plugin, so it must retrieve on it:
    # an agent that knows it has a seam problem should land here.
    ("seam", "chunked_label"),
    ("iou", "segmentation_qc"),
    ("f1", "segmentation_qc"),
    ("splits merges", "segmentation_qc"),
    ("instance segmentation", "segmentation_qc"),
    ("ground truth", "segmentation_qc"),
]

# Plugins that must not answer each other's questions. Cheaper to get wrong than
# a skill false hit -- the agent reads one line and moves on rather than
# following a whole workflow -- but these four do genuinely different things and
# the catalog should say so.
PLUGIN_REJECTS = [
    ("resolution", "rolling_ball"),
    ("segmentation", "rolling_ball"),
    ("background", "segmentation_qc"),
    ("background", "image_resolution"),
    ("background", "chunked_label"),
    ("iou", "chunked_label"),
    ("dask", "segmentation_qc"),
    ("f1", "image_resolution"),
    ("frc", "segmentation_qc"),
]


def _require_seeded(handle: str) -> None:
    seeded = {s["id"] for s in _skills.list_skills("") if s["kind"] == "plugin"}
    if handle not in seeded:
        pytest.fail(
            f"the phrasing table names the plugin {handle!r}, which is no longer "
            "seeded. Update PLUGIN_RETRIEVES/PLUGIN_REJECTS -- this is a stale "
            "table, not a retrieval failure."
        )


@pytest.mark.parametrize("query,expected", PLUGIN_RETRIEVES)
def test_seeded_plugin_is_retrieved_for(shipped_with_plugins, query, expected):
    _require_seeded(expected)
    got = [s["id"] for s in _skills.list_skills(query)]
    assert expected in got, (
        f"{query!r} did not surface the {expected} plugin; got {got}. Fix this in "
        "the module docstring's opening paragraph, which is what is indexed."
    )


@pytest.mark.parametrize("query,unwanted", PLUGIN_REJECTS)
def test_seeded_plugin_is_not_retrieved_for(shipped_with_plugins, query, unwanted):
    _require_seeded(unwanted)
    got = [s["id"] for s in _skills.list_skills(query)]
    assert unwanted not in got, f"{query!r} wrongly surfaced {unwanted}; got {got}"


def test_every_seeded_plugin_is_retrievable_by_its_own_handle(shipped_with_plugins):
    """No plugin is stranded, and the check needs no table so it cannot go
    stale: the handle read as words retrieves that plugin and nothing else. A
    new plugin inherits it by being seeded."""
    for entry in _skills.list_skills(""):
        if entry["kind"] != "plugin":
            continue
        query = entry["handle"].replace("_", " ")
        got = [r["id"] for r in _skills.list_skills(query)]
        assert got == [entry["id"]], f"{query!r} -> {got}"


def test_a_plugin_is_found_by_the_name_the_agent_would_call_it(shipped_with_plugins):
    """The handle is what goes in `execute_code`, so it is the one string an
    agent is guaranteed to have seen -- from `server_status`, or from a skill's
    `checklist: plugin:<name>`. Retrieval on it must be exact, not fuzzy."""
    got = _skills.list_skills(["segmentation_qc"])
    assert [s["id"] for s in got] == ["segmentation_qc"]
    assert got[0]["handle"] == "segmentation_qc"
