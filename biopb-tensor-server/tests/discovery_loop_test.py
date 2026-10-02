"""The walk stops on a loop the identity and symlink guards cannot see."""

import logging
import os

from biopb_tensor_server.core import discovery
from biopb_tensor_server.core.discovery import (
    WalkReport,
    walk_with_identity_tracking,
)


def _walk(root, **kwargs):
    return [str(p) for p in walk_with_identity_tracking(root, set(), **kwargs)]


def test_a_directory_that_resolves_to_an_ancestor_is_not_entered(tmp_path, monkeypatch):
    """A junction, or a mount, that ``is_symlink`` does not report."""
    top = tmp_path / "top"
    loop = top / "sub" / "loop"
    loop.mkdir(parents=True)
    (top / "sub" / "a.txt").write_text("a")
    real = discovery._real_dir
    monkeypatch.setattr(
        discovery, "_real_dir", lambda p: real(top) if p == loop else real(p)
    )
    report = WalkReport()

    walked = _walk(top, report=report)

    assert str(loop) in walked  # yielded, so it can be claimed
    assert str(top / "sub" / "a.txt") in walked
    assert str(loop) in report.declined_dirs


def test_a_directory_that_resolves_to_itself_is_not_entered(tmp_path, monkeypatch):
    top = tmp_path / "top"
    here = top / "here"
    here.mkdir(parents=True)
    real = discovery._real_dir
    monkeypatch.setattr(
        discovery, "_real_dir", lambda p: real(top) if p == here else real(p)
    )

    assert _walk(top, report=WalkReport()) == [str(here)]


def test_an_ordinary_nested_tree_is_walked_in_full(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "x.txt").write_text("x")
    report = WalkReport()

    walked = _walk(tmp_path, report=report)

    assert str(deep / "x.txt") in walked
    assert report.declined_dirs == set()


def test_the_depth_cap_stops_a_loop_that_resolves_to_fresh_paths(tmp_path, caplog):
    """No symlink, inode numbers that never repeat: only depth ends it."""
    chain = tmp_path
    for i in range(10):
        chain = chain / f"d{i}"
    chain.mkdir(parents=True)
    (chain / "x.txt").write_text("x")
    report = WalkReport()

    with caplog.at_level(logging.WARNING, logger=discovery.logger.name):
        walked = _walk(tmp_path, report=report, max_depth=4)

    assert str(chain / "x.txt") not in walked
    at_cap = tmp_path / "d0" / "d1" / "d2" / "d3" / "d4"  # d0-d3 were entered
    assert str(at_cap) in walked  # yielded, so it can be claimed
    assert str(at_cap / "d5") not in walked
    assert str(at_cap) in report.declined_dirs
    assert "levels below the root" in caplog.text


def test_the_default_cap_is_far_below_the_recursion_limit():
    import sys

    assert sys.getrecursionlimit() > discovery.MAX_WALK_DEPTH * 4


def test_a_real_symlink_loop_still_ends(tmp_path):
    top = tmp_path / "top"
    top.mkdir()
    (top / "a.txt").write_text("a")
    try:
        os.symlink(top, top / "loop")
    except OSError:
        return  # no symlinks here
    walked = _walk(top)

    assert str(top / "a.txt") in walked
    assert str(top / "loop" / "a.txt") not in walked
