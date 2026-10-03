"""The one record of known roots: containment, cloud, display root, overlap."""

from pathlib import Path

from biopb_tensor_server.sources.roots import (
    DND_URL_PREFIX,
    OVERLAP_MESSAGE,
    Root,
    RootKind,
    Roots,
)


def _root(kind, path, **kw):
    return Root(kind, str(path), **kw)


class TestContaining:
    def test_the_innermost_local_root_wins(self):
        outer = _root(RootKind.MONITORED, "/data")
        inner = _root(RootKind.SCAN_ONCE, "/data/sub")
        roots = Roots([outer, inner])

        assert roots.containing(Path("/data/sub/x.tif")) is inner
        assert roots.containing(Path("/data/y.tif")) is outer
        assert roots.containing(Path("/elsewhere/z.tif")) is None

    def test_an_upstream_and_a_static_cloud_source_are_not_containers(self):
        roots = Roots(
            [
                _root(RootKind.UPSTREAM, "grpc://lab:8815"),
                _root(RootKind.STATIC, "/data/one.zarr", cloud=True),
            ]
        )

        assert roots.containing(Path("/data/one.zarr")) is None


class TestCloud:
    def test_a_path_is_cloud_at_or_under_a_cloud_root(self):
        roots = Roots(
            [
                _root(RootKind.MONITORED, "/cloud", cloud=True),
                _root(RootKind.MONITORED, "/local"),
            ]
        )

        assert roots.is_cloud("/cloud")
        assert roots.is_cloud("/cloud/a/b.tif")
        assert not roots.is_cloud("/local/a.tif")
        assert not roots.is_cloud("s3://bucket/a")
        assert roots.cloud_roots() == frozenset({Path("/cloud")})

    def test_removing_a_root_takes_its_cloud_status_with_it(self):
        drop = _root(RootKind.DROPPED, "/drop", cloud=True, label="drop")
        roots = Roots([drop])
        assert roots.is_cloud("/drop/a.tif")

        roots.remove(drop)

        assert not roots.is_cloud("/drop/a.tif")
        assert roots.cloud_roots() == frozenset()


class TestDisplayUrl:
    def test_a_monitored_alias_re_roots_what_is_under_it(self):
        roots = Roots([_root(RootKind.MONITORED, "/data/exp", alias="lab")])

        assert roots.display_url("/data/exp/sub/a.tif") == "lab/sub/a.tif"
        assert roots.display_url("/data/exp") == "lab"

    def test_the_innermost_aliased_monitored_root_wins(self):
        roots = Roots(
            [
                _root(RootKind.MONITORED, "/o", alias="o"),
                _root(RootKind.MONITORED, "/o/i", alias="i"),
            ]
        )

        assert roots.display_url("/o/i/x.dat") == "i/x.dat"
        assert roots.display_url("/o/y.dat") == "o/y.dat"

    def test_no_alias_and_no_root_leave_the_plain_url(self):
        roots = Roots(
            [
                _root(RootKind.MONITORED, "/m"),
                _root(RootKind.SCAN_ONCE, "/s"),
            ]
        )

        assert roots.display_url("/m/a.tif") is None
        assert roots.display_url("/s/a.tif") is None
        assert roots.display_url("/nowhere/a.tif") is None

    def test_a_scan_once_alias_applies(self):
        roots = Roots([_root(RootKind.SCAN_ONCE, "/s", alias="once")])

        assert roots.display_url("/s/a.tif") == "once/a.tif"

    def test_a_drop_is_marked_with_its_label(self):
        roots = Roots([_root(RootKind.DROPPED, "/home/u/exp", label="exp (2)")])

        url = roots.display_url("/home/u/exp/a.tif")

        assert url == f"{DND_URL_PREFIX}exp (2)/a.tif"


class TestLabels:
    def test_a_taken_label_gets_a_counter(self):
        roots = Roots([_root(RootKind.DROPPED, "/a/exp", label="exp")])

        assert roots.unique_label(Path("/b/exp")) == "exp (2)"
        assert roots.unique_label(Path("/b/other")) == "other"
        assert roots.by_label("exp") is not None
        assert roots.by_label("nope") is None


class TestScanOnce:
    def test_each_scan_once_root_is_handed_out_once(self):
        once = _root(RootKind.SCAN_ONCE, "/s")
        roots = Roots([once, _root(RootKind.MONITORED, "/m")])

        assert roots.take_unscanned() == [once]
        assert roots.take_unscanned() == []

    def test_a_root_added_later_is_handed_out_too(self):
        roots = Roots()
        later = _root(RootKind.SCAN_ONCE, "/s")
        roots.add(later)

        assert roots.take_unscanned() == [later]


class TestOverlap:
    def test_a_registered_source_under_the_path_overlaps(self):
        roots = Roots()

        assert roots.check_overlap(Path("/d"), ["/d/a.tif"]) == OVERLAP_MESSAGE
        assert roots.check_overlap(Path("/d"), ["/other/a.tif"]) is None

    def test_a_source_already_registered_overlaps(self):
        assert (
            Roots().check_overlap(Path("/d"), [], already_registered=True)
            == OVERLAP_MESSAGE
        )

    def test_a_known_root_inside_the_path_overlaps_but_not_the_root_being_added(self):
        known = _root(RootKind.MONITORED, "/d/watched")
        new = _root(RootKind.DROPPED, "/d", label="d")
        roots = Roots([known, new])

        assert roots.check_overlap(Path("/d"), [], exclude=new) == OVERLAP_MESSAGE
        roots.remove(known)
        assert roots.check_overlap(Path("/d"), [], exclude=new) is None
