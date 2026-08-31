"""The benchmark: put a model in front of a real session and score what comes out.

``biopb-mcp/docs/skills.md`` §10. The layers next door ask whether a skill file
is well-formed, whether the API it quotes still exists, and whether anyone can
retrieve it. This one runs the thing.

It asks one question of every case — **can an agent do this work** — and two
kinds of comparison answer it. Run the same cases again with the catalog
withheld and the delta says what the catalog was worth; run them repeatedly and
the spread says what a single number is worth. Both are properties of the
invocation, and one invocation is one configuration for every case in it.

**Nothing here names a skill.** A case used to carry `skill=`, and the two
questions above read as two kinds of case. They are one kind now: this package
does not import the skills layout, glob `_skills_data`, or know which entries
ship, so promoting or banking a skill changes the catalog and no file in this
tree. What a run measured is recorded in `session.json`, and which two sessions
are worth comparing is a judgement a reader makes, not a field a case declares.

Two properties make this the tier with teeth, and one makes it the hardest to
read.

**It tests what we ship, twice over.** A skill body arrives through the real
:mod:`biopb_mcp.mcp._skills` — ``list_skills`` and ``skill://<id>``, the same
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
hard to guess (§5d). A case can also invert that: a self-sufficient prompt whose
persona holds no answer, where asking neither rescues nor penalises a run and
only makes it resemble a session. **What declares which shape a case has is
`persona_must_know`, not `skill`.** The two coincided while every case with no
skill was a task written against real data, and they stopped coinciding the day
a case could withhold a fact without naming a served skill — so
:mod:`.test_cases` reads the declaration, and a case is not assumed to withhold
nothing merely because no skill claims it.

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
:mod:`._options` is what a run may vary, :mod:`._engine` selects, runs, scores
and reports and knows no subject, and :mod:`.cases` holds one
:class:`~._case.Case` per subject. Adding one writes a single module there and
no test code, which is what has to stay true of a catalogue heading for thirty.

Marked ``bench``, deselected by default, and never in CI (§1).
"""
