"""Shared pytest fixtures for biopb-mcp tests."""

import pathlib

import pytest

from biopb_mcp._config import CONFIG


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Isolate the config singleton + config dir for every test.

    Two hazards the process-wide ``CONFIG`` singleton (issue #31) introduces for
    tests:

    1. *State leakage* -- the cache persists across tests, so a value loaded (or
       written) in one test would bleed into the next.
    2. *Non-hermeticity* -- call sites now hit ``CONFIG.get(...)``, whose first
       access reads the developer's real ``~/.config/biopb/mcp-config.json``.

    This autouse fixture points ``Path.home()`` at a per-test ``tmp_path`` (so an
    untouched config resolves to defaults) and invalidates the cache before and
    after each test. ``monkeypatch`` is function-scoped, so tests that set their
    own ``Path.home`` compose with this -- their setattr runs later and wins,
    sharing the same ``tmp_path``.
    """
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    CONFIG.reload()
    yield
    CONFIG.reload()
