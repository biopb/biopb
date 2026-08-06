"""Putting a model in front of a real biopb session, and scoring what comes out.

This package is the machinery only. It knows nothing about skills, nothing about
tasks, and nothing about what any particular run is trying to prove — it owns
the session, the two-model loop, the provider table, the fixture protocol and
the run-scoped data plane, and stops there.

That boundary is the reason it exists as its own package. The machinery grew up
inside the old ``_tests/skills/interaction/``, where it was written
skill-agnostic from the start and said so — its fixture and engine modules both
carried "knows no skill" in their own docstrings. But *living* under ``skills/``
meant a second suite could only reuse it by importing across a sibling that had
nothing to do with it, and the honest fix is the one that costs a move: the
neutral half becomes a package, and each suite keeps only what is about its own
question.

One suite consumes it today — :mod:`..bench`, which holds the cases, the
engine and the pytest surface, and which asks two questions of the same
machinery (does *this skill* change what an agent does; can an agent do *this
work*). Nothing here knows which of the two a run is about, and nothing here
knows how a run was configured: whether the catalogue was offered and who
answers the agent are arguments this package is *given*
(``live_session(skills_enabled=...)``, whichever :class:`._respondent.Respondent`
is passed to the loop), never a decision it makes.

What is here:

``_session``
    Bring-up: a real shim-spawned session, a synchronous façade over the async
    MCP client, and the environment facts that are forced rather than inherited.
``_bridge``
    MCP tool schemas to the function-calling shape a chat model expects.
``_models``
    The provider table: which model on each side, at which address, with which
    key. ``BIOPB_AGENT`` / ``BIOPB_RESPONDENT``.
``_agent``, ``_respondent``
    The two sides of the loop, each with a scripted and a live implementation.
``_conversation``
    The loop itself, the caps, and the ``Trace``.
``_fixture``
    What a run is given and what it recovers: ``Fixture``, ``Attempt``,
    ``Metric``, ``Outcome``, the fixture specs and the refs they hand out.
``_plane``
    The run-scoped tensor server a ``tensor``-presented case needs. Conditional:
    nothing starts unless a case asks.

The hermetic tests beside them run with the ordinary suite. Nothing here is
marked, because nothing here spends money — the paid run carries the ``bench``
marker over in :mod:`..bench`, and it is that suite which decides when to
spend and on what, not the machinery. Two exceptions, marked where they sit:
``test_plane``'s live checks (``bench`` — they start a real server) and
``test_fixture_tree`` (``fixtures`` — it hashes a curated tree out of band).
"""
