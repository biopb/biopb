"""The benchmark: put a model in front of a real session and score what comes out.

``biopb-mcp/docs/skills.md`` §10. The layers next door ask whether a skill file
is well-formed, whether the API it quotes still exists, and whether anyone can
retrieve it. This one runs the thing.

It answers two questions, and they were two packages until the day it became
clear they were one engine with a field set differently:

* **does *this skill* change what an agent does** — the case names a skill, and
  half the arms withhold the catalog to get a delta;
* **can an agent do *this work*** — the case names none, nothing is withheld,
  and repetition rather than a control is where the information comes from.

Keeping them apart cost two engines, two outcome vocabularies that agreed by
hand, two report writers and two answers to "where are the cases". What it
bought was a distinction that one column in the report now carries.

Two properties make this the tier with teeth, and one makes it the hardest to
read.

**It tests what we ship, twice over.** A skill body arrives through the real
:mod:`biopb_mcp.mcp._skills` — ``find_skills`` and ``skill://<id>``, the same
calls the runtime makes — and every run happens against a real shim-spawned
session: real kernel, real napari, real dask, the nine real tools with their
real schemas and the server's own ``instructions``. Nothing is stood in for.
A hand-written tool surface would have put ``execute_code``'s return shape and
the ``guide://`` bodies back into a transcription — a thing this suite once had
a whole layer of, and dropped: a hand-written procedure stays green while the
file it was transcribed from changes underneath it.

**The asking needs no separate assertion.** A skill case's fixture is built so
the ground truth is obtainable *only by asking*: strip the fact from the data,
give it to a respondent, and a numeric verifier tests the interaction for free.
Which fact to strip is the fixture's whole design problem — a *scale*, a
*unit*, a *provenance*, categorically absent from the pixels rather than merely
hard to guess (§5d). A case with no skill inverts that: its prompt is
self-sufficient and its persona holds no answer, so asking neither rescues nor
penalises a run and only makes it resemble a session.

**And it is not deterministic.** A red run's cause space is the skill body, the
model, the tool schemas, the kernel, Qt, dask and the fixture. So the trace is
written before any assertion runs, and ``test_session_smoke`` exists to fail
separately when the stack rather than the subject is at fault.

Two rules that are fixtures in their own right:

- **The agent under test is not from the family that wrote the skill** (§5a).
  These bodies were co-authored with Claude, which could pass by recognising
  its own prose rather than reading it. That fact lives in ``agentbench``'s
  provider table, so it can be asserted rather than assumed — and it binds a
  skill case only, since a task has no authored prose to recognise.
- **The respondent is skill-blind.** It holds a persona and a few private
  facts and volunteers nothing. A respondent that has read the body can
  paraphrase step 2 back at the agent and silently invalidate the suite.
  Because of that, §5a does *not* constrain which model plays it: both sides
  are configured independently, and the authoring family is a fine respondent.

**A case's contribution is data.** :mod:`._case` is the vocabulary,
:mod:`._engine` owns the grid, the outcome classification and the report and
knows no subject, :mod:`._options` is what a run may vary, and :mod:`.cases`
holds one :class:`~._case.Case` per subject. Adding one writes a single module
there and no test code, which is what has to stay true of a catalogue heading
for thirty.

Marked ``bench``, deselected by default, and never in CI (§1).
"""
