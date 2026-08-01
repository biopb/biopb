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

## This is a diagnostic harness, not a gate

**Nothing here tests shipped code.** `_drift.py` is a hand transcription of what
`drift-correction.md` says; it never reads that file. Edit step 3 to prescribe a
different `reference=` and this suite stays green while the body and the tested
procedure quietly part ways. Delete the skill entirely and nothing here notices.
A green result certifies the transcription — that the recipe, as written down
here, still stabilises a movie against the installed libraries.

That is why it is **not in CI**, unlike the contract layer next door, whose
assertions are derived from the shipped catalog (`skill_contracts.py` reads the
`pkg:` tokens out of the frontmatter, so deleting a skill changes the work it
does). The gate has to test the artifact users get.

What it is *for* is the tier above it: an agent run against the real skill file
(§6) is the test with teeth, and it is **non-deterministic**. When one surfaces
something — a step that reads as unambiguous and isn't, a number that comes out
wrong on one field and not another — this is where that gets pinned down: a
fixture, a subject, a tolerance, and a repeatable pass/fail. Reach for it after
§6 gives you something to isolate, not before.

## What is here

| File | Holds |
|---|---|
| `_outcome.py` | The skill-agnostic protocol: `Fixture`, `Attempt`, `Metric`, `Outcome`, the provider registry, `CuratedNpz`, artifact writing |
| `_drift.py` | `drift-correction`: three synthetic cases, four subjects, one verifier |
| `test_drift_correction.py` | The expectation table — which subject must pass which case, and which claim in the skill body each row is |
| `test_outcome_protocol.py` | The protocol itself, including the curated path almost no machine has data for |

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
- rows on the degraded path (`skimage.registration`) always run.

The `uv run --with` line at the top of this file is the whole setup: it overlays
the package on the existing `.venv` for one command without disturbing it.

## What this does not test

The scope of this layer is not the skill. It is **whatever the fixture and the
tolerances in `_drift.py` happen to span** — a judgement made by whoever wrote
them, and one the pass/fail signal does not carry. Read a green run as "correct
on these three movies, to this precision, against these four mistakes", never as
"the skill works". The ones that most narrow it:

- **The fixtures are noiseless.** Every frame is a deterministic resample of one
  base image. Registration precision is fundamentally noise-limited, so these
  numbers measure systematic correctness, not the precision a microscope gets.
- **Fixture and correction share an interpolation kernel** — both `order=3`,
  `mode="nearest"`. `residual_ratio` was designed not to share a *registration*
  error with the subject; it does share a *resampling* one. Partly self-limiting
  (cubic resampling is not exactly invertible, which is where the non-zero
  residual floor comes from), but the bias is common-mode.
- **Translation only.** Rotational or scale drift is not untested, it is
  unrepresentable: `StackReg.TRANSLATION` and `ndimage.shift` cannot express it.
- **`TOLERANCE` is a band, not a promise.** It was chosen to sit in the measured
  gap between the procedures that work and the mistakes that don't. The skill
  body states no precision, so these numbers are invented, not quoted.
- **Borders are excluded** by `_margin()` — which is exactly what step 6 of the
  body is about, so that claim is structurally outside the verifier's reach.
- **Absolute registration is out of gauge.** Both series are normalised to frame
  0, so a run that gets every relative offset right and displaces the whole
  series passes.

None of this is fixable in general — the set of things a fixture leaves out is
unbounded, and only a human can name the parts worth naming. It is written down
so a diagnostic result is read for what it is.

## Artifacts

Every case writes to `.skill-outcomes/<run>/<case>/<subject>/` (override with
`BIOPB_SKILL_OUTCOME_DIR`; gitignored). The number decides, the artifact
explains — and for a diagnostic run the artifact is usually the point:

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

**Nothing is queued.** `drift-correction` is the worked example, and the reason
to write a second module is a specific finding that needs isolating, not
coverage for its own sake. Two things bound what would be worth it:

- The outcome has to be a number with a knowable right answer.
  `calibrated-measurements` and `segmentation-qc-metrics` qualify (an analytic
  volume, a closed-form F1); `write-a-skill` emits markdown and never will.
- A **correct-by-construction reference implementation has to be able to fail**.
  Where the failure mode is an agent's *choice* — passing `spacing=` and then
  multiplying by `prod(spacing)` again, picking the wrong structural channel —
  the subject here makes the right choice by construction, and the fixture
  scores data nothing can fail. Those belong to §6.

Cost, for calibration: `drift-correction` took ~640 lines against a 157-line
skill, most of it the fixture design and the tolerance measurement rather than
the plumbing. A scalar, analytically-known truth would be a fraction of that;
drift's is a trajectory with a gauge freedom, which is the expensive shape.
