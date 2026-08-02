"""`drift-correction` as benchmark data. No engine here, and none needed.

The whole per-skill contribution: what the agent is asked, who it can ask, what
the fixture is called, and where its results should land. Everything else —
arms, outcome classification, flags, the report — is `.._benchmark`.

The fixture and the verifier are the outcome layer's, reused unchanged. §5
scores `outcomes/_drift.verify` against hand-written subjects; §6 scores the
same function against whatever a model actually left in the kernel. That reuse
is deliberate: a second verifier written for §6 would be a second transcription,
and drifting from the first is exactly how a layer goes quietly decorative.
"""

from __future__ import annotations

from ...outcomes import _drift, _drift_channels  # noqa: F401 - registers the fixture
from .._benchmark import Case
from .._respondent import Persona

TASK = """\
A two-channel time-lapse is open in the napari viewer as the layer `timelapse`.
Its axes are (T, C, Y, X) = (24, 2, 192, 192), float32. There is no tensor
server in this session, so that layer is the only source of pixels.

The field of view moved during the acquisition, and I need that corrected
before anything is measured off this movie.

The microscopist who acquired it is here and can answer questions. To ask them
something, reply in plain text without calling a tool, and their answer will
come back as the next message.

When you are finished, leave two names bound in the kernel namespace:

  offsets    a (24, 2) float array of (dy, dx), relative to frame 0.
             SIGN CONVENTION: offsets[t] is the displacement frame t *carries* —
             a feature at (y, x) in frame 0 appears at (y + dy, x + dx) in
             frame t. It is NOT the shift needed to correct frame t, which is
             the negative of this.
  corrected  the drift-corrected movie, the same shape as the input

Those two names are how your result is collected, so bind them exactly.
"""

#: Appends to a kernel-level list, so "did it ask before it spent" is
#: answerable from the trace. Both libraries are wrapped because either is a
#: reasonable route to the same registration, and fail-open because a spy that
#: broke the run would be worse than no spy.
GATE_SPY = """
_expensive_calls = []


def _install_skill_spy():
    import functools

    try:
        from skimage import registration as _reg

        _orig = _reg.phase_cross_correlation

        @functools.wraps(_orig)
        def _wrapped(*a, **k):
            _expensive_calls.append("phase_cross_correlation")
            return _orig(*a, **k)

        _reg.phase_cross_correlation = _wrapped
    except Exception:
        pass

    try:
        import pystackreg

        _orig2 = pystackreg.StackReg.register_stack

        @functools.wraps(_orig2)
        def _wrapped2(self, *a, **k):
            _expensive_calls.append("register_stack")
            return _orig2(self, *a, **k)

        pystackreg.StackReg.register_stack = _wrapped2
    except Exception:
        pass


_install_skill_spy()
"""

#: The movie has two channels and no channel names; step 2 asks the user whether
#: the field or the objects moved, and which channel to register on. Both are
#: the same fact, and it is here.
#:
#: Note what is *not* here: nothing about registration, nothing about
#: `reference="previous"`, no mention of a structural channel being the right
#: one to register on. This person knows their sample, not the procedure. A
#: persona that knew the skill could answer a question the agent never asked
#: properly, and the numeric result would stop meaning what it appears to.
MICROSCOPIST = Persona(
    name="microscopist-two-channel",
    facts={
        "what channel 0 is": (
            "the vesicle reporter — bright puncta, and they move around on "
            "their own inside the cells, that motion is the thing I am "
            "studying"
        ),
        "what channel 1 is": (
            "the membrane marker — dim, but it is just the cell outlines and "
            "they do not go anywhere"
        ),
        "did the field move or the objects": (
            "both, and that is my problem: the stage drifted over the run AND "
            "the vesicles are moving. The drift is the part I want gone"
        ),
        "how the movie was acquired": (
            "24 frames, one every 30 seconds, on a spinning disk. The stage "
            "was not touched during the run"
        ),
        "why it matters": (
            "I need to track individual vesicles, so the frame has to hold still first"
        ),
    },
    background=(
        "A two-channel time-lapse of cultured cells. You are happy to answer "
        "questions about the sample and the acquisition."
    ),
)

CASE = Case(
    skill=_drift.SKILL,
    task=TASK,
    persona=MICROSCOPIST,
    layers={"timelapse": "movie"},
    collect={"offsets": "offsets", "corrected": "corrected"},
    score=_drift.verify,
    save_artifacts=_drift.save_artifacts,
    spy=GATE_SPY,
    spy_markers=("phase_cross_correlation", "register_stack", "StackReg"),
    catalog_query="drift",
    # It must be able to answer: `_drift_channels` withholds which channel is
    # structural, and this person knows both channels and that the stage moved.
    persona_must_know=("channel 0", "channel 1", "move", "drift"),
    # And it must not know the procedure — only the sample.
    persona_must_not_know=(
        "reference=",
        "stackreg",
        "register",
        "phase_cross_correlation",
        "structural channel",
    ),
)
