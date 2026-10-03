"""The launcher/kernel env contract: one set of names, one viewer value."""

import pytest

from biopb_mcp.mcp import _bootstrap, _kernel, _kernel_gate
from biopb_mcp.mcp._kernel_env import (
    ENV_HOST_SESSION,
    ENV_NO_VIEWER,
    ENV_SCRATCH,
    ENV_VIRTUAL_DISPLAY,
    ViewerMode,
)


def test_every_side_names_the_same_variable():
    assert _kernel.ENV_HOST_SESSION is _kernel_gate.ENV_HOST_SESSION
    assert _kernel.ENV_SCRATCH == ENV_SCRATCH == "BIOPB_SCRATCH_KERNEL"
    assert _kernel.ENV_NO_VIEWER == ENV_NO_VIEWER == "BIOPB_NO_VIEWER"
    assert ENV_HOST_SESSION == "BIOPB_HOST_SESSION"
    assert ENV_VIRTUAL_DISPLAY == "BIOPB_VIRTUAL_DISPLAY"


@pytest.mark.parametrize(
    "mode",
    [
        ViewerMode.real(),
        ViewerMode.virtual(":9"),
        ViewerMode.none("because"),
    ],
)
def test_a_mode_survives_the_kernel_env(mode):
    assert ViewerMode.from_env(mode.env()) == mode


def test_the_window_pipe_follows_the_mode():
    assert ViewerMode.real().has_window
    assert ViewerMode.virtual(":9").has_window
    assert not ViewerMode.none("x").has_window


def test_a_viewerless_kernel_may_still_take_the_session_display():
    assert ViewerMode.none("x", display=":9").env()["DISPLAY"] == ":9"
    assert ViewerMode.none("x", display=":9").virtual_display is None


def test_the_kernel_reads_its_reason_from_the_same_env(monkeypatch):
    monkeypatch.delenv(ENV_SCRATCH, raising=False)
    monkeypatch.setenv(ENV_NO_VIEWER, "off")
    assert _bootstrap.no_viewer_reason() == "off"
