"""Where the SDK's ``__version__`` comes from.

The generated ``_version.py`` is written at build time and dist-info METADATA
at install time, so in an editable checkout they disagree and the order decides
whether ``biopb.__version__`` describes the code being imported or whenever it
was last installed. biopb/biopb#910 settled that for the product packages;
these pin the same answer for the SDK.
"""

import importlib
import sys
import types

import pytest


def _reload_biopb():
    import biopb

    return importlib.reload(biopb)


@pytest.fixture
def restore_biopb():
    """Reload the real modules afterwards: these tests stub what they import.

    ``biopb.image`` too -- it re-exports the parent's ``__version__``, which
    binds the value at import time, so a reload under the stubs leaves it
    holding one.
    """
    yield
    sys.modules.pop("biopb._version", None)
    _reload_biopb()
    image = sys.modules.get("biopb.image")
    if image is not None:
        importlib.reload(image)


class TestWhichSourceWins:
    def test_the_built_file_beats_the_installed_metadata(
        self, monkeypatch, restore_biopb
    ):
        # The whole point: an editable checkout reports what it is running, not
        # whenever `uv sync` last ran.
        stub = types.ModuleType("biopb._version")
        stub.version = "9.9.9+frombuild"
        monkeypatch.setitem(sys.modules, "biopb._version", stub)
        monkeypatch.setattr(
            importlib.metadata, "version", lambda _: "0.0.1+frominstall"
        )

        assert _reload_biopb().__version__ == "9.9.9+frombuild"

    def test_metadata_answers_when_there_is_no_built_file(
        self, monkeypatch, restore_biopb
    ):
        # An installed wheel carries both; a stripped one still reports.
        monkeypatch.setitem(sys.modules, "biopb._version", None)
        monkeypatch.setattr(
            importlib.metadata, "version", lambda _: "0.0.1+frominstall"
        )

        assert _reload_biopb().__version__ == "0.0.1+frominstall"

    def test_neither_source_is_still_not_a_crash(self, monkeypatch, restore_biopb):
        # Importing the SDK must not depend on having been packaged.
        monkeypatch.setitem(sys.modules, "biopb._version", None)

        def _absent(_):
            raise importlib.metadata.PackageNotFoundError("biopb")

        monkeypatch.setattr(importlib.metadata, "version", _absent)

        assert _reload_biopb().__version__ == "0.0.0"


class TestTheSubpackageDoesNotResolveItsOwn:
    def test_biopb_image_reports_the_same_version(self):
        # `biopb.image` is not a distribution of its own.
        import biopb
        import biopb.image

        assert biopb.image.__version__ == biopb.__version__

    def test_it_is_always_defined(self):
        # It used to be looked up separately and left *undefined* when the
        # lookup failed, so importing from an uninstalled source tree made this
        # raise AttributeError rather than report anything.
        import biopb.image

        assert isinstance(biopb.image.__version__, str)
        assert biopb.image.__version__
