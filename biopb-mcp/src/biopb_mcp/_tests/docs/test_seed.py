"""The seed index and the shipped docs have to agree, and links have to land.

What the runtime cannot check for itself. The store is fail-open by design: a
doc the index never names is simply invisible, and a `[[dangling]]` link is
allowed in a local doc because a validator the user does not have must never be
a condition for their file to load. Neither is acceptable in the shipped seed,
where it is an authoring bug the author's own PR can catch.
"""

from __future__ import annotations

import re

from biopb_mcp.mcp import _docs

from .conftest import doc_id_of, offered_files, read_doc_file

# `[[link]]`s, ignoring any inside an inline code span -- `authoring` quotes the
# syntax itself as `[[id]]`, which is documentation, not a link.
CODE_SPAN = re.compile(r"`[^`\n]*`")
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def _named_by(index: str) -> set[str]:
    named = set(_docs._ignored_ids(index))
    named.update(
        doc_id for line in index.splitlines() if (doc_id := _docs._entry_id(line))
    )
    return named


def test_every_offered_doc_is_an_entry_or_ignored(offered_docs, seed_index):
    """Otherwise it ships and no session ever hears about it.

    An offered doc with no entry lands in the *New shipped docs* tail, which is
    the upgrade path for a doc a release adds -- not a state the release that
    adds it should ship in. Banked docs are excluded because staying out of that
    tail is exactly what banking them does.
    """
    named = _named_by(seed_index)
    missing = sorted(doc_id_of(p) for p in offered_docs if doc_id_of(p) not in named)
    assert not missing, (
        "offered docs the seed index neither lists nor ignores:\n" + "\n".join(missing)
    )


def test_every_seed_entry_names_a_doc_that_ships(shipped_docs, seed_index):
    """The other direction: an entry with no file renders as `(missing)`."""
    ships = {doc_id_of(p) for p in shipped_docs}
    dangling = sorted(
        doc_id
        for line in seed_index.splitlines()
        if (doc_id := _docs._entry_id(line))
        and not doc_id.endswith("/")
        and doc_id not in ships
    )
    assert not dangling, "seed index entries with no file:\n" + "\n".join(dangling)


def test_the_seed_index_is_within_the_cap(seed_index):
    """A seed that starts over the cap is one the agent cannot add to."""
    assert _docs.index_entry_count(seed_index) <= _docs.MAX_INDEX_ENTRIES


def test_every_shipped_doc_is_within_the_body_cap(shipped_docs):
    """`write_doc` refuses a body this long, so a shipped one must not be."""
    too_long = {
        doc_id_of(p): len(read_doc_file(p).splitlines())
        for p in shipped_docs
        if len(read_doc_file(p).splitlines()) > _docs.MAX_BODY_LINES
    }
    assert not too_long, f"over {_docs.MAX_BODY_LINES} lines: {too_long}"


def test_a_banked_doc_is_not_asserted_about():
    """The package gates read the *offered* set, not everything that ships.

    Proving the dependencies of a doc the release deliberately does not list
    would be certifying something no session is told about -- and a doc is
    usually banked because its evidence is thin, which is exactly when a green
    gate reads as an endorsement.
    """
    assert not [p for p in offered_files() if p.name.startswith("_")]


def test_every_wikilink_resolves_to_a_shipped_doc(shipped_docs):
    ships = {doc_id_of(p) for p in shipped_docs}
    dangling = []
    for path in shipped_docs:
        body = CODE_SPAN.sub("", read_doc_file(path))
        for target in WIKILINK.findall(body):
            if target not in ships:
                dangling.append(f"{doc_id_of(path)} -> [[{target}]]")
    assert not dangling, "links to docs that do not exist:\n" + "\n".join(dangling)


def test_no_doc_links_to_itself(shipped_docs):
    for path in shipped_docs:
        body = CODE_SPAN.sub("", read_doc_file(path))
        assert doc_id_of(path) not in WIKILINK.findall(body)


def test_both_display_surfaces_are_listed(seed_index):
    """napari is optional, so an index that lists only it leaves an agent on a
    headless session with no route to showing the user anything.

    Keyed to the entries, not to the heading above them: how the index is
    grouped is the agent's, and the seed's own grouping is not an invariant.
    The neighbouring test only requires a shipped doc to be listed *or ignored*,
    so this is what stops one of the two being banked.
    """
    listed = {
        doc_id for line in seed_index.splitlines() if (doc_id := _docs._entry_id(line))
    }
    assert {"napari-viewer", "web-viewer"} <= listed


def test_every_shipped_doc_declares_a_kind(shipped_docs):
    """`kind` decides what the ablation withholds, and it defaults to procedure.

    A reference doc that forgets it is silently withheld from the ablated arm,
    which moves the baseline the arm exists to establish.
    """
    missing = sorted(
        doc_id_of(p)
        for p in shipped_docs
        if _docs.parse_frontmatter(read_doc_file(p)).get("kind")
        not in (_docs.KIND_REFERENCE, _docs.KIND_PROCEDURE)
    )
    assert not missing, "shipped docs with no kind:\n" + "\n".join(missing)


def test_no_body_names_a_specific_dataset(shipped_docs):
    """A path or an id from the run that produced it makes a doc unusable."""
    offenders = []
    for path in shipped_docs:
        for line in read_doc_file(path).splitlines():
            if re.search(r"/(home|Users)/|[A-Za-z]:\\\\", line):
                offenders.append(f"{doc_id_of(path)}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders)


def test_a_banked_doc_ships_and_reads_back(shipped_docs):
    """Banked is "the release does not list it", and nothing more.

    The file ships, `read_doc` returns it by id, and one hand-written index line
    promotes it. Keeping it out of the wheel would make that impossible and buy
    nothing, since not being listed is already what makes a doc invisible.
    """
    banked = [p for p in shipped_docs if p.name.startswith("_")]
    assert banked, "nothing is banked; this test has lost its subject"
    for path in banked:
        assert _docs.describe(doc_id_of(path)) is not None, f"{path.name} is unreadable"


def test_a_banked_doc_is_not_in_the_tail(shipped_docs):
    """The point of the prefix: a fresh install is not invited to file them."""
    listed = set(_docs.shipped_ids())
    banked = {doc_id_of(p) for p in shipped_docs if p.name.startswith("_")}
    assert not (listed & banked)
