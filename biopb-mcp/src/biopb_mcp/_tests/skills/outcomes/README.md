# The outcome layer

[`docs/skill-testing.md`](../../../../../../docs/skill-testing.md) §5. Run the
procedure a skill prescribes against data whose answer is known, and check the
number.

```sh
# needs the skill's package -- see "what has to be installed" below
uv run --no-project --python .venv/bin/python --with pystackreg \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/skills/outcomes -m outcome
```

Deselected by default (`-m outcome`), like the satisfiability layer. The
protocol tests in `test_outcome_protocol.py` are *not* marked and run with the
ordinary suite — they are hermetic and instant, and a break in the fixture
protocol should surface as a normal test failure.

## What is here

| File | Holds |
|---|---|
| `_outcome.py` | The skill-agnostic protocol: `Fixture`, `Attempt`, `Metric`, `Outcome`, the provider registry, `CuratedNpz`, artifact writing |
| `_drift.py` | `drift-correction`: three synthetic cases, four subjects, one verifier |
| `test_drift_correction.py` | The expectation table — which subject must pass which case, and which claim in the skill body each row is |
| `test_outcome_protocol.py` | The protocol itself, including the curated path no CI machine has data for |

## The three ideas

**A programmatic verifier, never a judged one.** Nothing here reads prose. The
verifier computes a number and compares it to a limit, because these skills emit
numbers with knowable right answers. Where that stops being true — and it will,
for a skill whose output is a judgement — the fallback is binary criterion
extraction ("did it pass `spacing=`?"), not a 1–5 rating.

**The fixture is substitutable.** The shipped cases are synthetic and
procedural: generated from a seed at test time, so nothing binary lands in git
and the truth is exact by construction rather than annotated. But a synthetic
movie is not a microscope. `CuratedNpz` is the door out, implemented rather than
promised — drop a tree somewhere and point `BIOPB_SKILL_FIXTURES` at it:

```
$BIOPB_SKILL_FIXTURES/drift-correction/<case>/case.json    # provenance, which keys are data, which are truth, tolerances
$BIOPB_SKILL_FIXTURES/drift-correction/<case>/arrays.npz   # the arrays
```

Nothing in the verifier changes. What *does* change is how much truth there is:
a real movie can carry a trajectory someone measured off a bead, but not an
un-drifted reference image, because no such image was ever acquired. So a metric
the fixture cannot support reports as **unavailable**, never as passing, and an
`Outcome` that scored nothing has not passed — it has not been tested. That
distinction is the load-bearing part of the protocol and it is what
`test_outcome_protocol.py` spends most of its length on.

**A verifier is calibrated by a run it must reject.** A verifier that only ever
sees correct runs cannot be told apart from one that returns green
unconditionally — which is how the contract layer sat unmanned for a release
without anyone noticing. So every case is also run through the specific mistake
the skill body warns against, and the suite asserts the two are told apart.
Every expected-to-fail row in `EXPECTED` is a sentence of prose from the skill
file, turned into a measurement.

## What has to be installed

The subjects that follow the skill need the skill's package, and that package
deliberately does not live in the shared test env — one resolution cannot hold
every skill's dependency (see `../README.md`). So:

- rows needing `pystackreg` `importorskip` and are silent without it;
- rows on the degraded path (`skimage.registration`) always run;
- CI runs the whole layer inside `skill-contracts.yaml`'s per-package env, where
  the package is present by construction.

That last point is a change from §10's original plan of workstation-only. It
applies to the **reference-implementation** half only — deterministic, agent
free, display-free, four seconds. The agent tier stays local and advisory.

## Artifacts

Every case writes to `.skill-outcomes/<run>/<case>/<subject>/` (override with
`BIOPB_SKILL_OUTCOME_DIR`; gitignored). The number gates, the artifact explains:

- `summary.json` — provenance, every metric with its limit, and what could not
  be measured at all
- `trajectory.csv` — truth against recovered, per frame
- `raw-difference.png` / `corrected-difference.png` — last frame minus first,
  before and after. **Both are scaled to the raw difference's range**, so they
  can be read side by side; scaling each on its own would stretch a near-zero
  corrected difference to full range and make a good run look like a bad one.

A passing run's `corrected-difference.png` is black in the middle with a bright
band around the edge. That band is the invalid margin — registration slides data
off one side and invents nothing on the other, which is step 6's point. The
metrics exclude it; `summary.json` records how wide it was as `margin_px`.

## Adding a case

1. Register a provider in the skill's module (`_drift.py` is the worked
   example). A synthetic one is a dataclass with `build()`; a curated one is a
   directory.
2. Give every `(case, subject)` pair a row in `EXPECTED`, including the ones
   that must fail. `test_the_expectation_table_covers_every_case` fails
   otherwise — a case nobody decided the right answer for is not a test.
3. Set tolerances from measurement, not from taste. The ones in `_drift.py`
   carry the numbers they were derived from, and the gap between the worst
   correct run and the best failure.

## Adding a skill

A new module beside `_drift.py`, with its own provider, subjects and verifier.
The protocol in `_outcome.py` knows nothing about drift and should stay that
way: what a metric means, what truth it needs, and what the tolerance is are all
the skill's business.

Not every skill can have this layer. It needs an outcome that is a number with a
knowable right answer — `calibrated-measurements` and `segmentation-qc-metrics`
qualify (an analytic volume, a closed-form F1); `write-a-skill` does not. That
is a real limit, not a backlog.
