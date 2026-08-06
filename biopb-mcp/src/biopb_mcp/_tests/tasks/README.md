# The task layer — can an agent do the work?

Put a model in front of a **real** biopb session and a **real** acquisition, ask
for a named piece of work, and score what comes back.

```sh
# needs a GL-capable display (or the xvfb package), a fixture tree, and an API key
BIOPB_FIXTURES=~/biopb-fixtures \
  uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/tasks -m tasks -s
```

Deselected by default (`-m tasks`), and never in CI.

## How this differs from `skills/interaction`

They share all their machinery ([`agentbench/`](../agentbench/)) and answer
different questions, and the difference is **what each one varies**.

|  | `skills/interaction` | here |
|---|---|---|
| Question | does *this skill* change what an agent does | can an agent do *this work* |
| Arms | 2x2: skill offered/withheld x user answers/silent | one |
| Repetition | one sample per corner | N samples, `BIOPB_TASK_SAMPLES` |
| Skills at run time | the manipulation | **on**, as a user has them |
| Data | mostly synthetic, procedurally built | curated real acquisitions only |

A skill's claim is a behavioural delta, and a delta needs a control — hence the
2x2. A task has no claim to isolate, so there is one arm and nothing is
withheld. What replaces the ablation as a source of information is repetition.

**Skills stay on.** This measures the product a user actually has, not a model
in a stripped environment. The cost is that a task which later gains a covering
skill silently re-bases its own number — so every summary records which skills
the catalog offered at run time. A score compared across releases without
reading that line is a score compared against a different system.

## No synthetic data

Every case here is `OnDisk`, against a curated tree. That is a deliberate
narrowing rather than a coincidence: `docs/fixtures.md` records a
synthetic fixture that ranked two method families in the *opposite* order from
real tissue, which is exactly the failure a functional benchmark cannot survive.

What real data costs is truth, and the answer is **perturb at authoring time**.
A tool under `biopb-mcp/tools/` takes an acquisition, applies a known
transformation once, and writes the result into the tree with the
transformation recorded in the manifest's `provenance`. The run only ever
*reads* — it never perturbs anything, so `kind` stays `curated` and there is no
third provenance literal to reason about. A benchmark that re-derives its data
every run cannot notice that its data changed, and a transformation applied
during a run is a knob someone can turn between two results that later get
compared.

Where truth comes with the data instead (an annotation), a case uses it
directly and needs no authoring step at all.

## What is here

| File | Holds |
|---|---|
| `_runner.py` | `TaskCase`, `Sample`, the one-arm loop, outcome classification, the report |
| `cases/` | One module per task, each a single `CASE`. Data, not code |
| `conftest.py` | Smoke runs first, and a failed smoke *skips* the run rather than merely preceding it |
| `test_session_smoke.py` | That the stack works, with **no model in it** |
| `test_cases.py` | Every task, persona and verifier, hermetically |
| `test_tasks.py` | The paid run: every case, N samples, assert only that it reported |

## Adding a task

Two steps, and the first is usually the work.

1. **Get the data into a tree.** Either it already carries truth, or write an
   authoring tool beside
   `tools/author_align_channels_fixture.py` that perturbs a real acquisition
   once and records what it did. The tree needs `case.json` (the `data`/`truth`
   partition) and a `manifest.json` entry with a citation, per-file `sha256`
   and per-array shape/dtype. A key in both `data` and `truth` is a hard error:
   a truth the run can see is not a truth.
2. **Write one module under `cases/`** exporting a `CASE`. No registration
   step — it is found by being there, and this suite's tests start checking it
   by its arriving.

```python
CASE = TaskCase(
    case_id="align-channels-from-landmarks",   # names the run and its artifacts
    task=TASK,                                 # the prompt, incl. where results go
    persona=MICROSCOPIST,                      # answers, volunteers nothing
    fixture=OnDisk(tolerance={...}),
    layers=(Layer("moving", "moving", presentation="tensor", chunks=(256, 256)),
            Layer("moving_pts", "moving_pts", kind="points")),
    collect={"probe_mapped": "probe_mapped", "quality_px": "quality_px"},
    score=_verify,                             # (fixture, attempt) -> Outcome
)
```

**The persona holds the experimental context and no answer.** The task text is
self-sufficient, so asking neither rescues nor penalises a run — it only makes
the run resemble a session. `test_cases.py` asserts the persona does not know
what the verifier checks.

**The verifier is the part that can be wrong quietly.** A fixture that fails to
build raises and a bad prompt shows up in the transcript, but a verifier that
scores a wrong answer as a pass produces a clean green report meaning nothing.
Write its tests in `test_cases.py` first: a perfect run, a run that did nothing,
a missing deliverable, and whatever the specific way of *looking* right is for
this task.

Two rules that came out of the seed case, and generalise:

- **A deliverable that is bound but unusable is not delivered.** A name pointing
  at the wrong shape, or at `inf`, must fail rather than drop out of the score
  as "unavailable" — otherwise a half-broken run reports green on its other half.
- **A ratio metric needs a floor.** Comparing a claimed error against an actual
  one is unusable near zero: a perfect run saying "about half a pixel" divides
  by ~0 and scores as maximally dishonest.

## A benchmark, not a gate

**No run's outcome fails a test here.** A sample reports `ok`, `wrong-answer`,
`out-of-turns`, `out-of-tool-calls`, `gave-up`, `no-result`,
`unscorable-result` or `harness-error`, plus flags that change how to read it:
`cut-off-but-scored`, `stalled`, `read-harness-internals`, `fixture-overwritten`.

A provider ending a run (`max_tokens` exhausted mid-answer) looks exactly like a
model deciding to stop, so it is classified as `harness-error` before anything
else — a truncated completion is never reported as a capability.

What *is* asserted is that a report reached disk with a transcript per sample.
A poor result is informative; a missing one is not.

## Samples

`BIOPB_TASK_SAMPLES` (default 1). One sample is right for iterating on a case
and is **not a measurement** — the spread between runs of the same task is
routinely larger than the difference anyone is trying to read off it. Raise it
before quoting a number.

Reports land in `.task-outcomes/<case_id>/`, with `transcript.md` and
`trace.jsonl` per sample written *before* scoring, so the tree fills as the run
proceeds:

```sh
tail -f .task-outcomes/align-channels-from-landmarks/sample-1/transcript.md
```

Model selection, provider addresses and the fixture tree are
[`agentbench`](../agentbench/)'s: `BIOPB_AGENT`, `BIOPB_RESPONDENT`,
`BIOPB_FIXTURES`.
