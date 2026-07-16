"""Unit tests for ``biopb._kernel_plugins`` — the static kernel-plugin inspector.

The module is the control dashboard's read-only view of the "bring your own tool"
surface (biopb/biopb-mcp#92): it lists the ``~/.config/biopb/kernel/*.py`` startup
files and the installed ``biopb_mcp.namespace`` entry points **without importing
or executing** any of it. These tests pin exactly that: a docstring one-liner is
read via ``ast`` (never run), files sort and skip ``_``-prefixed names, a syntax
error degrades to an empty summary, and a missing dir is empty — all against a
per-test ``$HOME`` so the machine's real config can't leak in.
"""

from pathlib import Path

import pytest
from biopb import _kernel_plugins as kp


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate the kernel-plugin dir under a per-test home.

    ``config_dir()`` honors ``XDG_CONFIG_HOME`` ahead of ``Path.home()``, and CI
    runners set it — clear it so the default falls back to the patched home.
    """
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _plugin_dir(home: Path) -> Path:
    d = home / ".config" / "biopb" / "kernel"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestStartupFiles:
    def test_missing_dir_is_empty(self, home):
        # Default dir does not exist -> [] and no dir is created (a bare read must
        # not materialize an empty tree).
        assert kp.startup_files() == []
        assert not (home / ".config" / "biopb" / "kernel").exists()

    def test_summary_is_docstring_first_line_not_executed(self, home):
        d = _plugin_dir(home)
        # A side effect in module code proves the file is parsed, never run.
        (d / "tool.py").write_text(
            '"""Blob helpers.\n\nlonger prose ignored\n"""\n'
            'raise RuntimeError("must not run on inspect")\n',
            encoding="utf-8",
        )
        assert kp.startup_files(d) == [{"name": "tool.py", "summary": "Blob helpers."}]

    def test_sorted_skips_underscore_and_tolerates_syntax_error(self, home):
        d = _plugin_dir(home)
        (d / "b.py").write_text('"""B tool."""\n', encoding="utf-8")
        (d / "a.py").write_text("x = 1\n", encoding="utf-8")  # no docstring -> ""
        (d / "_priv.py").write_text('"""skip me"""\n', encoding="utf-8")
        (d / "bad.py").write_text("def (:\n", encoding="utf-8")  # syntax error -> ""
        assert kp.startup_files(d) == [
            {"name": "a.py", "summary": ""},
            {"name": "b.py", "summary": "B tool."},
            {"name": "bad.py", "summary": ""},
        ]


class TestSummary:
    def test_payload_shape(self, home):
        d = _plugin_dir(home)
        (d / "t.py").write_text('"""T."""\n', encoding="utf-8")
        out = kp.summary()
        assert out["dir"] == str(d)
        assert out["files"] == [{"name": "t.py", "summary": "T."}]
        # entry_points reflects whatever biopb_mcp.namespace packages the env has
        # installed (e.g. biopb-mcp's built-in rolling-ball in the workspace), so
        # assert the shape, not emptiness.
        assert isinstance(out["entry_points"], list)

    def test_entry_points_never_raise(self, home):
        # The metadata read is fail-open and returns {name, dist} rows for whatever
        # is registered (possibly nothing, depending on the install).
        rows = kp.entry_point_plugins()
        assert isinstance(rows, list)
        assert all(set(r) == {"name", "dist"} for r in rows)
