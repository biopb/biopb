"""The respondent fixtures: who the agent is talking to, and what they know.

Kept in Python rather than as prose files for one reason: the private facts
have to be **machine-readable**, so `test_personas` can assert that none of
them reaches the agent by any route other than asking. A prompt and a fact
table in separate files can drift apart, and that drift is invisible from the
outside — the suite would still be green while quietly testing nothing.

Reviewed like a fixture, because that is what it is. The failure mode §6 names
explicitly is a respondent that volunteers what it was not asked, which rescues
a bad agent and invalidates every row that depends on it.
"""

from __future__ import annotations

from ._respondent import Persona

#: For `outcomes/_drift_channels.py`. The movie has two channels and no channel
#: names; step 2 of `drift-correction` asks the user whether the field or the
#: objects moved, and which channel is `REF_CHANNEL`. Both are the same fact,
#: and it is here.
#:
#: Note what is *not* here: nothing about registration, nothing about
#: `reference="previous"`, no mention of a structural channel being the right
#: one to register on. This person knows their sample, not the procedure. A
#: persona that knew the skill could answer a question the agent never asked
#: properly, and the numeric result would stop meaning what it appears to.
DRIFT_CHANNELS = Persona(
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
