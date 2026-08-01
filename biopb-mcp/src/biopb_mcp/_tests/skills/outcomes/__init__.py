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
procedure the body prescribes, not a model following it. An agent run plugs into
the same :class:`._outcome.Attempt` and the same verifier.

**This is a diagnostic harness, not a gate.** Nothing here reads a skill file —
:mod:`._drift` is a hand transcription of what ``drift-correction.md`` says, so
a green run certifies the transcription, not the shipped catalog. That is why it
stays out of CI while the contract layer next door (whose assertions *are*
derived from the frontmatter) gates every PR. Its use is downstream of §6: an
agent run against the real skill file is the test with teeth and is
non-deterministic, and this is where one of its findings gets pinned to a
fixture, a tolerance, and a repeatable pass/fail. See ``README.md`` here for
what the fixtures deliberately do not span.

Marked ``outcome`` and deselected by default; §10 places this layer outside the
merge gate on purpose.
"""
