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
    """Reload the real module afterwards: these tests stub what it imports."""
    yield
    sys.modules.pop("biopb._version", None)
    _reload_biopb()


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


class TestOnlyTheDistributionCarriesOne:
    """``__version__`` belongs to the distribution, so it lives on ``biopb``
    and nowhere below it.

    ``biopb.image`` carried its own until biopb/biopb#998 -- a second lookup of
    the same distribution, with no fallback, that left the attribute undefined
    when the lookup failed. Subpackages are not separately versioned, so it was
    removed rather than copied to the siblings.
    """

    @pytest.mark.parametrize("name", ["biopb.image", "biopb.tensor"])
    def test_a_subpackage_does_not_carry_one(self, name):
        module = importlib.import_module(name)

        assert not hasattr(module, "__version__"), (
            f"{name}.__version__ is a second version for one distribution"
        )

    def test_the_top_level_package_does(self):
        import biopb

        assert isinstance(biopb.__version__, str)
        assert biopb.__version__
