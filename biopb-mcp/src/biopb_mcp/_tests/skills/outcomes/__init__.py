"""The outcome layer — does following a skill produce the right numbers?

``docs/skill-testing.md`` §5. The layers next door ask whether a skill file is
well-formed, whether the API it quotes still exists, and whether anyone can
retrieve it. This one asks the question those cannot: **run the procedure and
check the answer against a ground truth.**

Three properties shape everything here.

**A programmatic verifier, never a judged one.** These skills emit numbers with
knowable right answers, so the verifier computes a number and compares it to a
limit. Nothing reads prose.

**The fixture is substitutable.** The fixtures that ship are synthetic and
procedural — generated from a seed at test time, so nothing binary lands in git
and the truth is exact by construction. But a synthetic movie is not a
microscope, and the door is deliberately left open for a manually curated
fixture of real data to stand in its place: register a second provider and
nothing else changes. :class:`._outcome.CuratedNpz` is that path, implemented
rather than promised, and it skips when the data is not on this machine.

The cost of admitting real data is that its truth is *annotated*, not
constructed — a curated movie can carry the trajectory someone measured off
fiducials, but not the un-drifted image, because no such image exists. So a
verifier reports a metric it cannot compute as **unavailable**, never as
passing, and :attr:`._outcome.Outcome.passed` is false when nothing was scored
at all.

**The verifier is calibrated against a known-bad run.** A verifier that never
fails is indistinguishable from a working one — the failure that left the
contract layer unmanned for a release. So each case is run through the
procedure the skill body prescribes *and* through the specific mistake the body
warns about, and the suite asserts the verifier tells them apart. Every
expected-to-fail row in :mod:`.test_drift_correction` is a claim the skill file
makes in prose.

**No agent yet.** The subject under test is a reference implementation of the
procedure the body prescribes, not a model following it. That is a smaller
question than §5's, and worth answering first: it proves the fixture and the
verifier discriminate, and it catches a body whose recipe stopped working. An
agent run plugs into the same :class:`._outcome.Attempt` and the same verifier.

Marked ``outcome`` and deselected by default — the assertions are slow, and §10
places this layer outside the merge gate on purpose.
"""
