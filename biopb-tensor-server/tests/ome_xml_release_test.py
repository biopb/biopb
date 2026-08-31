"""Releasing the raw OME-XML once the catalog owns the metadata (biopb/biopb#783).

``_raw_ome_xml`` is populated at registration and, before this, never cleared --
so a serving process held one uncompressed copy per registered source forever.
On a per-plane acquisition that is tens of MB each (the OME-XML carries one
``<Plane>`` and one ``<TiffData>`` per T*C*Z), duplicating what DuckDB already
stores in ``sources.metadata_json`` in plane-stripped form.

What these pin, in the order the fix has to hold them:
  * the raw string goes and the plane-stripped one stays,
  * everything downstream of registration -- ``get_metadata``, physical scale,
    scenes built later -- still answers, and answers WITHOUT reopening the file
    (the trap: trading a memory leak for an I/O one),
  * the drop is recoverable, not lossy,
  * a source with no embedded OME-XML is untouched by any of it,
  * the release is driven by the catalog write, so a catalog-less server (the
    embedded image-base cache) never fires it.
"""

import numpy as np
import pytest
import tifffile
from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
from biopb_tensor_server.core.metadata_db import MetadataDatabase
from biopb_tensor_server.fixtures import (
    create_per_plane_ome_tiff,
    create_tiled_ome_tiff,
)

N_PLANES = 400
FIELD = "Image:0"


@pytest.fixture
def per_plane_tiff(tmp_path):
    """An OME-TIFF whose XML is dominated by per-plane elements."""
    return create_per_plane_ome_tiff(str(tmp_path / "perplane.ome.tif"), n_t=N_PLANES)


@pytest.fixture
def registered(per_plane_tiff):
    """A source adapter taken through the registration calls, before release."""
    adapter = OmeTiffAdapter(per_plane_tiff, "perplane")
    adapter.get_source_descriptor()
    adapter.get_metadata()
    return adapter


@pytest.fixture
def count_opens(monkeypatch):
    """Count real ``tifffile.TiffFile`` opens.

    A stub that raised would be swallowed by the OME-XML path's broad except, so
    this delegates to the real thing and only records.
    """
    opens = []
    real = tifffile.TiffFile

    def _counting(*a, **k):
        opens.append(a[0] if a else None)
        return real(*a, **k)

    monkeypatch.setattr(tifffile, "TiffFile", _counting)
    return opens


# --- what is dropped, what is kept ------------------------------------------


def test_release_drops_the_raw_xml_and_keeps_the_stripped_one(registered):
    raw_len = len(registered._raw_ome_xml)
    assert registered._reduced_ome_xml  # computed by get_metadata

    registered.release_registration_cache()

    assert registered._raw_ome_xml is None
    # The stripped form is O(structure), the raw one O(plane count): the whole
    # point is that the retained string does not grow with the acquisition.
    assert len(registered._reduced_ome_xml) < raw_len / 10


def test_release_marks_released_without_unprobing(registered):
    # The trap in #783: un-probing would make every later call reopen the file.
    registered.release_registration_cache()
    assert registered._raw_ome_xml_probed is True
    assert registered._raw_ome_xml_released is True


def test_release_is_idempotent(registered):
    registered.release_registration_cache()
    registered.release_registration_cache()
    assert registered._raw_ome_xml is None
    assert registered._reduced_ome_xml


# --- nothing downstream reopens the file ------------------------------------


def test_metadata_after_release_reparses_without_reopening(registered, count_opens):
    before = registered.get_metadata()
    registered.release_registration_cache()

    after = registered.get_metadata()

    assert after == before
    assert after["images"][0]["pixels"]["planes"] == []  # stripped, as always
    assert count_opens == []


def test_scene_built_after_release_keeps_physical_scale_without_reopening(
    registered, count_opens
):
    # Scenes are built lazily at serve time, i.e. after the release -- so what
    # they inherit is the stripped XML, and physical scale has to survive it.
    expected = registered.get_tensor_adapter(FIELD)._physical_scale()
    assert expected is not None
    registered._tensor_adapters.clear()
    registered.release_registration_cache()
    count_opens.clear()

    scene = registered.get_tensor_adapter(FIELD)

    assert scene._raw_ome_xml is None
    assert scene._physical_scale() == expected
    assert count_opens == []


def test_scene_built_in_the_registration_gap_is_settled_by_the_release(
    per_plane_tiff, count_opens
):
    """The window the reconciler opens: registered, not yet synced.

    ``_commit_add_claim`` calls ``register_source`` BEFORE ``sync_source_added``,
    so the source is live on the Flight server while its catalog row is still
    being written. A GetFlightInfo (or a precache warm) landing in there builds a
    scene that inherited the raw XML and no stripped one -- inheritance in
    ``get_tensor_adapter`` is a snapshot -- and that scene is cached for the
    process lifetime. Releasing it unsettled would leave it with only the file:
    its next physical-scale call would reopen AND re-cache the raw string, which
    is the leak back on an adapter nothing releases a second time.
    """
    source = OmeTiffAdapter(per_plane_tiff, "perplane")
    source.get_source_descriptor()  # descriptor discovery
    scene = source.get_tensor_adapter(FIELD)  # <-- in the gap
    assert scene._raw_ome_xml and scene._reduced_ome_xml is None

    MetadataDatabase().sync_source_added("perplane", source)  # get_metadata + release

    assert scene._raw_ome_xml is None
    assert scene._reduced_ome_xml  # settled on the way down
    count_opens.clear()
    assert scene._physical_scale() is not None
    assert scene.get_metadata()["images"][0]["pixels"]["id"] == "Pixels:0"
    assert count_opens == []
    assert scene._raw_ome_xml is None  # and never re-cached the raw string


def test_release_settles_the_stripped_form_even_if_metadata_never_ran(per_plane_tiff):
    # The release derives what it needs rather than assuming get_metadata went
    # first, so the invariant holds under any call order: no adapter is ever
    # released into a state where the file is its only remaining source.
    source = OmeTiffAdapter(per_plane_tiff, "perplane")
    source.get_source_descriptor()
    scene = source.get_tensor_adapter(FIELD)

    source.release_registration_cache()

    assert source._raw_ome_xml is None and source._reduced_ome_xml
    assert scene._raw_ome_xml is None and scene._reduced_ome_xml


def test_scene_created_before_release_is_released_too(registered):
    scene = registered.get_tensor_adapter(FIELD)
    assert scene._raw_ome_xml

    registered.release_registration_cache()

    assert scene._raw_ome_xml is None
    assert scene._raw_ome_xml_released is True


def test_stripped_and_raw_scans_agree_on_physical_scale(registered):
    # The substitution the release rests on: stripping touches <Plane>/<TiffData>
    # only, so the <Pixels> header this scan reads is byte-identical either way.
    scene = registered.get_tensor_adapter(FIELD)

    assert scene._scan_physical_scale(
        scene._reduced_ome_xml
    ) == scene._scan_physical_scale(scene._raw_ome_xml)


def test_physical_scale_does_not_force_the_strip(per_plane_tiff):
    # Reading the raw document costs an iterparse that stops at the first
    # <Pixels>; producing the stripped one costs a regex over the whole thing.
    # An adapter that has not been asked for metadata must not pay the latter.
    source = OmeTiffAdapter(per_plane_tiff, "perplane")
    source.get_source_descriptor()  # descriptors only, no get_metadata
    scene = source.get_tensor_adapter(FIELD)
    assert scene._reduced_ome_xml is None

    assert scene._physical_scale() is not None

    assert scene._reduced_ome_xml is None


# --- recoverable, not lossy --------------------------------------------------


def test_the_full_xml_can_be_read_back_after_release(registered):
    full = registered._raw_ome_xml
    registered.release_registration_cache()

    recovered = registered._local_ome_xml()

    assert recovered == full
    assert registered._raw_ome_xml_released is False  # re-read, so held again


# --- a source with no embedded OME-XML --------------------------------------


def test_source_without_ome_xml_is_untouched(tmp_path, count_opens):
    # A plain TIFF: probed once, no XML. Release must not flip the released flag,
    # or the cached "there is none" answer would cost a reopen every time.
    plain = str(tmp_path / "plain.tif")
    tifffile.imwrite(plain, np.zeros((8, 8), np.uint16))
    adapter = OmeTiffAdapter(plain, "plain")
    assert adapter._local_ome_xml() is None
    count_opens.clear()

    adapter.release_registration_cache()

    assert adapter._raw_ome_xml_released is False
    assert adapter._local_ome_xml() is None
    assert count_opens == []


# --- the catalog write is what drives it -------------------------------------


def test_sync_source_added_releases_and_the_catalog_still_has_the_metadata(registered):
    db = MetadataDatabase()

    db.sync_source_added("perplane", registered)

    assert registered._raw_ome_xml is None
    stored = db.get_metadata_json("perplane")
    assert stored and stored["images"][0]["pixels"]["id"] == "Pixels:0"


def test_resync_after_release_still_writes_the_same_row(registered):
    # An unresolved source resolving re-syncs; the second pass must not fail --
    # nor reopen the file for a string it would strip again.
    db = MetadataDatabase()
    db.sync_source_added("perplane", registered)
    first = db.get_metadata_json("perplane")

    db.sync_source_added("perplane", registered)

    assert db.get_metadata_json("perplane") == first


def test_registering_without_a_catalog_releases_nothing(per_plane_tiff):
    # The embedded image-base cache builds its TensorFlightServer with
    # metadata_db=None; a source registered there has nowhere else for its
    # metadata to live, so nothing may be dropped out from under it.
    from biopb_tensor_server import TensorFlightServer

    adapter = OmeTiffAdapter(per_plane_tiff, "perplane")
    adapter.get_source_descriptor()
    server = TensorFlightServer(location="grpc://localhost:0", writable=False)
    try:
        server.register_source("perplane", adapter)
        assert adapter._raw_ome_xml
        assert adapter._raw_ome_xml_released is False
    finally:
        server.shutdown()


# --- the delegating wrappers -------------------------------------------------


def test_normalizing_wrapper_forwards_the_release(registered):
    # SourceAdapter declares the method, so it resolves on the wrapper and never
    # reaches its __getattr__ passthrough -- it has to delegate explicitly.
    from biopb_tensor_server.core.normalize import NormalizingAdapter

    NormalizingAdapter(registered).release_registration_cache()

    assert registered._raw_ome_xml is None


def test_unresolved_proxy_forwards_the_release(registered):
    from biopb_tensor_server.adapters.unresolved import UnresolvedSourceAdapter
    from biopb_tensor_server.core.config import SourceConfig

    proxy = UnresolvedSourceAdapter(
        SourceConfig(url=registered._source_url, type="ome-tiff", source_id="perplane"),
        registry=None,
    )
    proxy.release_registration_cache()  # unresolved: a no-op, must not raise
    proxy._resolved = registered

    proxy.release_registration_cache()

    assert registered._raw_ome_xml is None


def test_every_source_adapter_answers_the_release(tmp_path):
    # It is declared on the ABC precisely so a wrapper author sees it; the
    # default is a no-op, and an adapter that holds nothing keeps it.
    path, _, _ = create_tiled_ome_tiff(str(tmp_path), shape=(2, 16, 16))
    adapter = OmeTiffAdapter(path, "tiled")
    adapter.get_metadata()
    adapter.release_registration_cache()
    assert adapter.get_metadata()["images"][0]["pixels"]["id"]
