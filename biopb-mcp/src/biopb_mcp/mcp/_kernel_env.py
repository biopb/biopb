"""The environment contract between the launcher and the kernel it starts.

A leaf module: it imports nothing from this package, so the kernel side
(``_bootstrap``, ``_kernel_gate``) and the session child (``_kernel``,
``__main__``) can all name the same variables without the kernel pulling in
``_kernel``. A name lives here once; nothing else spells one out.
"""

from dataclasses import dataclass
from typing import Optional

# The write end of the window-close pipe, inherited by the kernel.
ENV_WINDOW_CLOSE_FD = "BIOPB_WINDOW_CLOSE_FD"

# Marks the scratch kernel a verification runs in (``_scratch``).
ENV_SCRATCH = "BIOPB_SCRATCH_KERNEL"

# Why this session has no viewer. Its presence makes the bootstrap skip Qt and
# napari.
ENV_NO_VIEWER = "BIOPB_NO_VIEWER"

# The host client's session id, which arms the kernel's gate (``_kernel_gate``).
ENV_HOST_SESSION = "BIOPB_HOST_SESSION"

# Set when the kernel renders on a launcher-owned Xvfb rather than the user's
# display.
ENV_VIRTUAL_DISPLAY = "BIOPB_VIRTUAL_DISPLAY"


@dataclass(frozen=True)
class ViewerMode:
    """Where a session's napari viewer is, decided once by the launcher.

    ``kind`` is ``"real"`` (the user's display), ``"virtual"`` (a launcher-owned
    Xvfb, *display*) or ``"none"`` (*reason* says why). The kernel env and the
    window-close pipe both derive from it, so they cannot disagree; a new mode
    is a new ``kind`` here.
    """

    kind: str
    reason: Optional[str] = None
    display: Optional[str] = None

    @classmethod
    def real(cls):
        return cls("real")

    @classmethod
    def virtual(cls, display):
        return cls("virtual", display=display)

    @classmethod
    def none(cls, reason):
        return cls("none", reason=reason)

    @property
    def has_window(self) -> bool:
        return self.kind != "none"

    @property
    def virtual_display(self):
        """The Xvfb display the viewer renders on, or None."""
        return self.display if self.kind == "virtual" else None

    def env(self) -> dict:
        """The kernel env entries this mode implies."""
        out = {}
        if self.kind == "virtual":
            out["DISPLAY"] = self.display
            out[ENV_VIRTUAL_DISPLAY] = "1"
        elif self.kind == "none":
            out[ENV_NO_VIEWER] = self.reason or "no viewer"
        return out

    @classmethod
    def from_env(cls, environ):
        """The mode a kernel env says it was launched in."""
        reason = environ.get(ENV_NO_VIEWER)
        if reason:
            return cls.none(reason)
        if environ.get(ENV_VIRTUAL_DISPLAY):
            return cls.virtual(environ.get("DISPLAY"))
        return cls.real()
