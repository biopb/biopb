"""The interaction layer — does a skill get asked the questions it needs?

``docs/skill-testing.md`` §6. Every other layer here reads a skill file, or runs
a procedure transcribed from one. This one puts a **model** in front of the
shipped body, against a **real session**, and scores what comes out.

Two properties make it the tier with teeth, and one makes it the hardest to
read.

**It tests what we ship, twice over.** The body arrives through the real
:mod:`biopb_mcp.mcp._skills` — ``find_skills`` and ``skill://<id>``, the same
calls the runtime makes — and the run happens against a real shim-spawned
session: real kernel, real napari, real dask, the nine real tools with their
real schemas and the server's own ``instructions``. Nothing is stood in for.
A hand-written tool surface would have put ``execute_code``'s return shape and
the ``guide://`` bodies back into a transcription, which is exactly the
property that keeps §5 out of the merge gate (§5c).

**The asking needs no separate assertion.** Fixtures are built so the ground
truth is obtainable *only by asking*: strip the fact from the data, give it to
a respondent, and a numeric verifier tests the interaction for free. That
claim is proved per fixture, deterministically and without a model, before any
model is paid to demonstrate it — see :mod:`.._tests` sibling
``outcomes/_drift_channels.py`` and §6b.

**And it is not deterministic.** A red run's cause space is the skill body, the
model, the tool schemas, the kernel, Qt, dask and the fixture. So the trace is
written before any assertion runs, and ``test_session_smoke`` exists to fail
separately when the stack rather than the skill is at fault.

Two rules that are fixtures in their own right:

- **The agent under test is not from the family that wrote the skill** (§6a).
  These bodies were co-authored with Claude, which could pass by recognising
  its own prose rather than reading it. That fact lives in
  :mod:`._models`' provider table, so it can be asserted rather than assumed.
- **The respondent is skill-blind.** It holds a persona and a few private
  facts and volunteers nothing. A respondent that has read the body can
  paraphrase step 2 back at the agent and silently invalidate the suite.
  Because of that, §6a does *not* constrain which model plays it: both sides
  are configured independently, and the authoring family is a fine respondent.

Marked ``interaction``, deselected by default, and never in CI (§10).
"""
