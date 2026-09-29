"""The Bio-Formats release is pinned unless the environment says otherwise."""

from biopb_tensor_server.adapters import bioio


def test_pins_when_the_environment_is_silent(monkeypatch):
    monkeypatch.delenv("BIOFORMATS_VERSION", raising=False)
    bioio._pin_bioformats_version()
    import os

    assert os.environ["BIOFORMATS_VERSION"] == bioio.BIOFORMATS_VERSION


def test_a_site_choice_wins(monkeypatch):
    monkeypatch.setenv("BIOFORMATS_VERSION", "7.3.1")
    bioio._pin_bioformats_version()
    import os

    assert os.environ["BIOFORMATS_VERSION"] == "7.3.1"


def test_the_pin_is_a_release_not_a_candidate():
    assert bioio.BIOFORMATS_VERSION[0].isdigit()
    assert "rc" not in bioio.BIOFORMATS_VERSION and "-m" not in bioio.BIOFORMATS_VERSION
