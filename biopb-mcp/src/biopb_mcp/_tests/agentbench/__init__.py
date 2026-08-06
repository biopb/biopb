"""Putting a model in front of a real biopb session, and scoring what comes out.

This package is the machinery only. It knows nothing about skills, nothing about
tasks, and nothing about what any particular run is trying to prove — it owns
the session, the two-model loop, the provider table, the fixture protocol and
the run-scoped data plane, and stops there.

That boundary is the reason it exists as its own package. The machinery grew up
inside :mod:`.._tests.skills.interaction`, where it was written skill-agnostic
from the start and said so — ``_fixture`` and ``_benchmark`` both carried
"knows no skill" in their own docstrings. But *living* under ``skills/`` meant a
second suite could only reuse it by importing across a sibling that had nothing
to do with it, and the honest fix is the one that costs a move: the neutral half
becomes a package, and each suite keeps only what is about its own question.

Two suites consume it, and the split between them is what each one *varies*:

- :mod:`.._tests.skills.interaction` varies **the catalogue** — a 2x2 of skill
  offered/withheld against a user who answers/stays silent, because a skill's
  claim is a behavioural delta and a delta needs a control.
- a case that names no skill varies **nothing**. It asks whether an agent can
  do a named piece of work at all, so one session is the whole measurement.
  There is no ablation because there is no claim to isolate.

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
marked, because nothing here spends money — a suite's paid run carries its own
marker (``interaction``, ``tasks``), and it is the suite that decides when to
spend, not the machinery.
"""
