# The benchmark — one runner, one case directory

[`biopb-mcp/docs/skills.md`](../../../../docs/skills.md) §10. Put a model in
front of a **real** biopb session and score what comes out.

```sh
# needs a GL-capable display, or the xvfb package -- see "what this needs" below
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/bench -m bench -s
```

Deselected by default (`-m bench`), like `satisfiability`, and never in CI (§1).

The hermetic half — the engine, the options, the trace, the personas, the
fixtures, the verifiers — is **not** marked and runs with the ordinary suite. A
break in the machinery, or a case whose persona gives its own answer away,
should surface as a normal red test rather than be found by someone
mid-diagnosis with a paid run.

A case either names a **skill** — a claim that following it changes what an
agent does, measured as two sessions either side of `--bench-skills` — or it
names none, and asks only whether the work gets done, with repetition rather
than a control behind its number. That one field is the whole difference; the
engine, the report and the switches are the same for both.

It says nothing about whether a case **withholds** something. The cases of
*banked* skills live here too — a skill behind the `_` marker, which the runtime
does not serve — and they withhold as hard as any skill case does; they simply
have no catalog entry to ablate. What declares that a case withholds is
`persona_must_know`, and that is what `test_cases.py` reads.

## Run options

Each is a pytest flag with an environment fallback, and **the flag wins**.
There is deliberately no dotenv behind them: `agentbench` reads `.env` for
credentials and model selection, which are facts about a machine, but an option
here decides what gets *spent*, and a file somebody put in the repo root a month
ago should not be the answer to "why did this take two hours".

| Flag | Environment | Values | Does |
|---|---|---|---|
| `--bench-cases` | `BIOPB_BENCH_CASES` | `all` (default), `skills`, `tasks` | which cases to pay for — `skills` is the ones with an ablation, `tasks` the complement |
| `--bench-fixtures` | `BIOPB_BENCH_FIXTURES` | `all` (default), `synthetic`, `curated` | which kind of data |
| `--bench-skills` | `BIOPB_BENCH_SKILLS` | `true` (default), `false` | whether the agent is offered the catalog |
| `--bench-responder` | `BIOPB_BENCH_RESPONDER` | `model` (default), `silent` | who answers the agent |
| `--bench-samples` | `BIOPB_BENCH_SAMPLES` | a positive integer, default 1 | how many times each case |

**The target must be at or below `_tests/`.** `pytest .../_tests`,
`pytest .../_tests/bench` and `pytest .../_tests/bench/test_report.py` all take
these flags; `pytest biopb-mcp` and a bare `pytest` from the repo root reject
them with *unrecognized arguments*. The flags are declared in
`_tests/conftest.py`, and pytest runs `pytest_addoption` only on the conftests it
loads at **startup** — the rootdir's, and those on the path down to the
arguments. A conftest that collection reaches later is too late to add a flag, so
an argument *above* `_tests/` never sees the declaration. Declaring them any
deeper (in `bench/conftest.py`) would narrow this further, to `bench/` alone.

```sh
# the shipped configuration, every case
pytest .../bench -m bench -s

# the same cases with the catalog withheld -- the other half of a skill's delta
pytest .../bench -m bench -s --bench-skills=false

# one case, three times, nobody answering
pytest .../bench -m bench -s -k drift-correction --bench-responder=silent \
  --bench-samples=3
```

Picking out a single case is pytest's own `-k`: the parametrization ids are case
labels, so `-k drift-correction`, `-k "drift or flatfield"` and the full
`-k drift-correction/two-channels-one-structural` all work. `--bench-cases` is
for the two *categories*, which `-k` cannot express.

A filter is a cap on coverage, so a filtered run **says what it did not run**,
by name, in its own terminal summary. A shorter table is otherwise
indistinguishable from a shorter catalogue.

The options narrow only what is *paid for*. Every hermetic check still runs over
every case, because they cost nothing and a filter that quietly turned off the
tests as well would be the worst of both.

### One invocation is one configuration

`--bench-skills` and `--bench-responder` are settings on the session the run
happens in, fixed for the whole invocation. A run therefore has no arms and no
grid: it runs the cases you selected, `--bench-samples` times each, in one
configuration, and writes one session directory that says which.

The 2x2 that used to be a table inside one report is four commands:

```sh
pytest ... --bench-skills=true  --bench-responder=model    # does the whole thing work
pytest ... --bench-skills=true  --bench-responder=silent   # does *asking* matter
pytest ... --bench-skills=false --bench-responder=model    # does the *skill* matter
pytest ... --bench-skills=false --bench-responder=silent   # the floor
```

**What that buys.** A run costs what you asked it for rather than what the case
kind implied. Every row in a report was configured identically, so a table can
be read without a column qualifying each row. And a case's kind stopped deciding
anything: the engine used to hand a skill case four configurations and every
other case one, which meant the same command spent four times as much depending
on data it had no reason to consult.

**What it costs.** No single report contains a delta. The delta is two session
directories, which is why the configuration is in every report header and in
`session.json` rather than in a path or a column.

**The two `silent` runs are about the fixture, not the skill.** They ask whether
the withheld fact is really unobtainable from the pixels — a property of the
construction in `cases/`, which does not change when a body is edited, and which
`test_cases.py` already checks the cheap half of by asserting no truth key
appears in `data`. A skill's own delta is the two `--bench-responder=model` runs.
So once a fixture's asymmetry is established, those two are the ones to stop
paying for.

Re-run them when the fixture changes, and when a report makes the asymmetry look
decorative. `drift-correction` is the standing reason to keep doing it: a capable
agent recovered its withheld channel anyway, by registering on both and keeping
the self-consistent one — a fixture can be built so the heuristics its author
thought of point the wrong way and still not make the fact unobtainable.

### `--bench-skills=false` on a case with no skill

Allowed, and it measures something real — whether the catalog was helping that
work at all. It is simply not the question the case was written to ask, and the
case has no entry of its own to withhold, so the two runs are answering about
the catalogue rather than about the case.

## Session directories

One invocation writes one directory, and that directory is the unit you keep,
compare or throw away:

```
.bench-outcomes/session-20260806-134501/
  session.json                       the code, the configuration, the models, the roster
  drift-correction/two-channels-one-structural/
    summary.md  summary.json
    sample-1/transcript.md  trace.jsonl  summary.json
  flatfield/offset-known-only-to-the-operator/
    ...
```

`session.json` is what makes two of these comparable — **which code produced
it**, the switches, the sample count, the case filter, both models, and every
case the session has finished with its per-sample outcomes.

The code half is `biopb_mcp`'s version plus the checkout's commit, branch and
whether the tree was **dirty**. That last one is the load-bearing field: a sha
identifies the code only if nothing was edited on top of it, so two sessions can
carry the same commit and have run different engines. A delta between sessions
built from different working trees is not a delta, and this is the only place
that is recoverable afterwards. Every per-case report additionally records the
skills the catalog *actually offered*, which is what stops a number being
compared across releases: a case that later gains a covering skill silently
re-bases itself, and the default configuration is skills-on precisely because
that is the product a user has. It is rewritten after each case rather than once
at the end, because these runs are long and the session you most want a roster
for is the one that was interrupted.

The alternative was a directory named after the configuration, under each case.
It encodes the same two switches in a path and nothing else — so two sessions
that differed by model, by sample count or by fixture tree would land on top of
each other with nothing to say they had.

Sessions are named by timestamp so they sort in the order they were run.
`BIOPB_OUTCOME_DIR` moves the root if you want a run somewhere of its own.

## Running one with real models

Both sides are `provider:model`, named separately, so they can be different
vendors — or the same compatible API at two addresses:

```sh
export BIOPB_AGENT=openai:gpt-5                  # default
export BIOPB_RESPONDENT=anthropic:claude-sonnet-5 # default
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...

uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest .../bench -m bench -s
```

Known providers: `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` — each a
`(sdk, base_url, key_env)` triple, and most of them the OpenAI-compatible API
at a different address. Override an address with
`BIOPB_AGENT_BASE_URL` / `BIOPB_RESPONDENT_BASE_URL`. A bare model
name is refused rather than guessed: which vendor serves a model is exactly the
fact §5a turns on, and inferring it from the name would make the rule depend on
vendors' naming conventions.

**§5a constrains the agent, not the respondent — and only on a skill case.** The
respondent is skill-blind and answers from a fact table, and having written the
skills does not help with that, so Anthropic is a fine respondent and is the
default, while the default *agent* is deliberately not from the authoring
family. `agentbench/test_models.py` asserts both, off the provider table rather
than off a comment. A case with no skill has no authored prose to recognise, so
every model is a legitimate subject for it.

`ollama` needs no key, which makes it the cheap way to rehearse a run end to end
before spending anything.

**A `--bench-responder=silent` session needs only the agent's key.** The silent
respondent is local and answers from a constant, so that arm reaches for no
respondent model at all — no key read, no endpoint probed, nothing spent on one.
The availability check follows the switch rather than the defaults
(`_engine.models_in_play`), so one key is enough to run the control condition.

The provider SDKs are imported lazily and are **not** dependencies of this
package — one `--with` line each. Keys are read from the environment at call
time and are never written to a trace, an artifact or a log.

## What is here

The session, the loop, the provider table, the fixture protocol and the
run-scoped data plane are **not** here — they are
[`_tests/agentbench/`](../agentbench/), which knows nothing about skills, tasks
or configurations. What stays here is what is about *scoring a subject*:

| File | Holds |
|---|---|
| `_case.py` | `Case` and `Layer` — the vocabulary, and nothing about how a run is configured |
| `_options.py` | The run options. Stdlib-only, because the tests-root conftest registers them |
| `_engine.py` | The samples, outcome classification, the session and the report |
| `cases/` | One module per case. Data, not code — **the** place a case is defined |
| `test_bench.py` | The pytest surface: run every selected case, assert only that it reported |
| `test_session_smoke.py` | That the stack works, with **no model in it** |
| `test_report.py` | That the options resolve and the engine classifies, on hand-built outcomes |
| `test_cases.py` | That every case's persona, fixture and verifier hold up — and that the catalogue is covered |
| `test_verifiers.py` | The checks one case needs and no other does |
| `conftest.py` | Which cases a run pays for; and smoke runs first, with a failed smoke *skipping* the run rather than merely preceding it |

## Adding a case

The engine is subject-agnostic, so a new case is **data**: one module under
`cases/`, exporting a module-level `CASE`. There is no registration step — the
module is discovered by being there.

```python
CASE = Case(
    case_id="twelve-nuclei-anisotropic",   # names the run and its artifacts
    skill="calibrated-measurements",  # omit for a case about the work alone
    task=TASK,                        # the prompt, incl. where results land
    persona=MICROSCOPIST,             # who holds the fact the fixture withholds
    fixture=Procedural(Ellipsoids()), # data, truth, tolerances -- and only this
    layers=(Layer("nuclei", "image"),          # in-memory numpy, client is None
            Layer("nuclei_labels", "labels", kind="labels")),
    #  ... or presentation="tensor", chunks=(1, 256, 256) for a lazy case,
    #  which brings up one data plane for the whole run.
    collect={"volumes_um3": "volumes_um3", "spacing_um": "spacing_um"},
    score=verify,                     # (fixture, attempt) -> Outcome
    save_artifacts=save_artifacts,
    plugins=("segmentation_qc",),     # kernel plugins the skill's `checklist:` names
    persona_must_know=(...), persona_must_not_know=(...),   # skill cases
)
```

A skill that cannot be benchmarked goes in `cases.NOT_BENCHMARKED` with the
reason instead; `test_cases.py` asserts the shipped catalogue is covered by one
or the other, so "what does this cover" never has to be answered by reading the
directory. Nothing equivalent constrains a case with no skill: there is no
catalogue of work to be complete against.

A **banked** skill — written but not served by the runtime, the `_` prefix on
its file — gets an ordinary case that leaves `skill` empty and sets
`namespace=` to the skill's own name. It runs the shipped corner, because there
is no catalog entry to withhold and a square over one would be four copies of a
corner. Everything else about it is a case like any other.

That is a relaxation, and what it replaced is worth knowing. The case module
used to carry the `_` prefix too and land in a `DEFERRED_CASES` tuple that was
checked hermetically and **run nowhere** — on the argument that a skill the
runtime does not serve has nothing to measure. But the *work* is real whether or
not a skill for it is served, and three of the four such cases had a complete
fixture, verifier and persona sitting behind a prefix. Now they run.

Promoting the skill is then a one-line edit: add `skill=` to the case that is
already there. `namespace` already carries the name, so nothing on disk moves,
and the coverage gate fires the moment the skill ships uncovered — which is a
better pin than the two filename prefixes that used to have to agree.

No test code either way: `test_bench.py` parametrizes over the selected cases,
so the new one brings its own report and transcripts, and `test_cases.py`
starts checking its persona, its fixture and its verifier by its arriving.

**Two rules for a curated case's data, which came out of the seed case and
generalise.** Get the data into a tree — either it already carries truth, or
write an authoring tool beside `tools/author_align_channels_fixture.py` that
perturbs a real acquisition once and records what it did, so the run only ever
*reads*. And then:

- **A deliverable that is bound but unusable is not delivered.** A name pointing
  at the wrong shape, or at `inf`, must fail rather than drop out of the score
  as "unavailable" — otherwise a half-broken run reports green on its other half.
- **A ratio metric needs a floor.** Comparing a claimed error against an actual
  one is unusable near zero: a perfect run saying "about half a pixel" divides
  by ~0 and scores as maximally dishonest.

**The verifier is the part that can be wrong quietly.** A fixture that fails to
build raises and a bad prompt shows up in the transcript, but a verifier that
scores a wrong answer as a pass produces a clean green report meaning nothing.
Write its tests in `test_verifiers.py` first: a perfect run, a run that did
nothing, a missing deliverable, and whatever the specific way of *looking* right
is for this case.

## Real data, and when a case needs it

**A case picks its fixture on the merits, not on whether it names a skill.**
Both kinds run against either kind: what decides is whether the conclusion
survives synthesis. `docs/fixtures.md` records a synthetic fixture that ranked
two method families in the *opposite* order from real tissue — so a case whose
question is "which method wins here" is written against an acquisition from the
start, and `align-stack-by-features` is. A case whose question is arithmetic on
a known truth — a voxel size, a spot count, a network length — is better served
by a procedure that constructs the answer exactly, and three of the banked
skills' cases are.

The rule used to be that a case with no skill must be `OnDisk`, and it held
only because "no skill" then meant "the one task, written against real data".
It stopped meaning that the moment the banked skills' cases became real, and a
rule that outlived its reason is worse than none: it would have forced three
procedural fixtures onto a tree nobody has.

What real data costs is truth, and the answer is **perturb at authoring time**.
A benchmark that re-derives its data every run cannot notice that its data
changed, and a transformation applied during a run is a knob someone can turn
between two results that later get compared. Where truth comes with the data
instead (an annotation), a case uses it directly and needs no authoring step.

`--bench-fixtures=curated` is how to run only the cases with real data behind
them, which on a machine with no `$BIOPB_FIXTURES` tree is also the set that
skips.

## A benchmark, not a gate

**No run's outcome fails a test here.** Each run reports an `outcome` and a
`reason` — `ok`, `wrong-answer`, `out-of-turns`, `out-of-tool-calls`,
`gave-up`, `no-result`, `unscorable-result`, `harness-error` — plus flags that
change how to read it: `cut-off-but-scored`, `over-ask-budget(n)`,
`never-asked`, `asked-but-unanswered`, `stalled`, `catalog-mismatch`,
`read-harness-internals`, `fixture-overwritten`.

Every run happens inside its own `try`, so a corner that dies becomes a row
rather than an exception that destroys the other three. The report is the
deliverable; a poor fixture is still informative, a missing report is not.

**Telling a provider failure from a behaviour.** Both sides bill their reasoning
against `max_tokens`, so a budget sized for the answer can be gone before the
answer starts, and the empty result that comes back is indistinguishable from
the agent handing off (respondent) or giving up (agent). Neither is laundered
into a reply: the backend raises `EmptyCompletion`, the run stops as
`respondent-failed` or `agent-truncated`, and both are `harness-error` rather
than a row about the subject. `asked-but-unanswered` catches the same thing from
the other end — a `--bench-responder=model` session that got no answers ran the
`silent` condition under the wrong label, and is comparable to nothing.

Two things *are* asserted, and neither judges a subject: that `summary.md`
reached disk with a transcript per sample, and that the **catalog matched the
switch**. The second is not a finding — if `--bench-skills=false` stopped
withholding the catalog, a skill's delta against the other session would read as
zero for a reason unrelated to the skill, which is a green table asserting the
opposite of the truth.

## Watching a run

A skill case is four conversations and the better part of half an hour — two and
a half of that at one sample — so the engine prints one line per sample, when
it starts and how it ended:

```
[calibrated-measurements/twelve-nuclei-anisotropic] 3 sample(s), skills=on responder=model, against `twelve-nuclei-anisotropic` -> .bench-outcomes/session-20260806-134501/calibrated-measurements/twelve-nuclei-anisotropic
[calibrated-measurements/twelve-nuclei-anisotropic] sample 1/3: running
[calibrated-measurements/twelve-nuclei-anisotropic] sample 1/3: ok in 6.2 min — within every tolerance
[calibrated-measurements/twelve-nuclei-anisotropic] sample 2/3: running
```

**That needs `-s`.** pytest discards a passing test's captured output, so
without it these lines — and the final report the fixture prints — never reach
the terminal. Nothing is lost either way (`summary.md` is on disk regardless),
but the run looks hung.

The other view is the artifact directory, from a second terminal. Every run
writes `transcript.md` and `trace.jsonl` *before* it is scored, so the tree
fills as the run proceeds:

```sh
watch -n5 'find .bench-outcomes -newermt "-1 hour" | sort'

# or follow one conversation as it happens
tail -f .bench-outcomes/<session>/<namespace>/<case>/sample-1/transcript.md
```

Reports land under `<session>/<namespace>/<case_id>/`, where the namespace is the
skill id or `tasks`. Each sample's wall-clock lands in the report as a `min`
column, so the cost of a case is legible afterwards rather than remembered.

## What a fixture has to withhold

A skill case rests on one claim: that the fact the respondent holds is **not
obtainable from the pixels**, so the numeric outcome cannot come out right
without asking.

The first fixture written here got that half right and it is worth knowing why.
`drift-correction`'s movie is built so every available heuristic — contrast,
peak intensity, feature density — points at the wrong channel, and a scripted
run that guessed was 5 px out where one that was told was 0.0006 px out. Then a
capable agent recovered the answer anyway, by registering on both channels and
keeping the self-consistent one. Defeating the heuristics *its author thought
of* is not the same as being unobtainable.

So prefer a withheld fact that is **categorically absent** from the data — a
unit, a scale, a provenance, an identity. `calibrated-measurements` withholds
µm per voxel: no amount of looking at an array of numbers yields microns.
`segmentation-qc-metrics` withholds which of two label layers a person drew,
and pays for it in precision and recall while F1 — symmetric under the swap —
stays exactly right.

**A self-sufficient case withholds nothing**, and that is a shape in its own
right: the task text is complete, so asking neither rescues nor penalises a run
— it only makes the run resemble a session. A case declares itself that shape
by naming nothing in `persona_must_know`, and `test_cases.py` then asserts its
persona does not name what the verifier reads. `align-channels-from-landmarks`
is the one; every other case in the tree withholds something, whether or not it
names a skill.

## Why this tier is the one with teeth

This suite used to carry a layer below this one that ran each skill's procedure
as a **hand transcription** of what its body said, scored against the same kind
of fixture. It was dropped, and the reason is the argument for this one: a
transcription never reads the file, so editing a step — or deleting the skill —
left it green. It also could not reach the instructions that need a *choice* in
order to be wrong, which is most of what these bodies are for.

Here the body arrives through the real `biopb_mcp.mcp._skills` — `find_skills`
and `skill://<id>`, the same calls the runtime makes — and the run happens
against a real session: real kernel, real napari, real dask, the nine real
tools with their real schemas and the server's own `instructions`.

**Nothing is stood in for, and that was a deliberate choice.** A hand-written
tool surface would have been cheaper and would have put `execute_code`'s return
shape, `server_status`'s report and the `guide://` bodies back into a
transcription — the same disease, moved from the subject into the environment.

## What this needs

**A GL-capable display.** Not just Qt: `QT_QPA_PLATFORM=offscreen` gets you a
napari `Viewer`, and then `add_image` dies inside vispy's extension probe,
because the offscreen platform provides no GL context. A desktop session's own
display works, and on a display-less box the session child spawns its own
`Xvfb` (`mcp/_xvfb.py`) — so installing the `xvfb` package is enough. Absent
both, these tests **skip with instructions** rather than run somewhere subtly
different.

A curated case additionally needs `$BIOPB_FIXTURES` pointing at a tree that has
its data; without one it skips, which is the ordinary state of a machine that
has not synced one.

Nothing else: no API key for the smoke tests, and no network beyond loopback.

## Four things the harness forces

Each of these silently changes what a run tests, so none of them is inherited
from whatever machine is running.

**A real viewer.** A session always has one — on the user's display or the
launcher's own Xvfb — but a box with neither a display nor the `xvfb` binary
cannot bring one up, and paying the slow bring-up to learn that helps nobody.
Bring-up probes for it and refuses, so a run never scores a session in which
step 2's *"show the user the first and last frames"* could not happen.

**No tensor plane, unless a case asks.** A developer box often has a data plane
up, and then `client` is live and the agent can wander into whatever catalog
that machine holds — so a finding might not reproduce anywhere else. The child
is pointed at an unreachable URL, `client` lands as `None`, and the fixture
reaches the agent as a napari layer and nothing else.

A case that presents `tensor` needs one, and gets the **run's** plane: a single
server for the whole invocation, started on the first case that asks and reaped
with the invocation (`agentbench/_plane.py`). Not one per case, because upload
is the expensive part and these are the large fixtures by construction.

**The gate is per case even so.** `run_one` hands the session a `tensor_url`
only when *that* case uploaded something, so an `array` case still meets an
unreachable address and a `None` client even in an invocation where some other
case brought a plane up. The isolation does not weaken as the run goes on.

**A config tree of our own.** `XDG_CONFIG_HOME` points at a temp dir, so the run
reads neither the developer's `mcp-config.json` nor their personal
`~/.config/biopb/skills/*.md`. The catalog under test is the shipped one.

**Only the kernel plugins a case asks for.** That same temp tree means an empty
`~/.config/biopb/kernel/`, so a skill declaring `plugin:segmentation_qc` would
be scored in a session where its own `checklist:` cannot be met. `Case.plugins`
seeds the ones it names, from the copies biopb-mcp ships, through the real
loader — and nothing else, because a plugin the skill never declared is an
environment difference nobody chose.

## Arrays cross by file, not by literal

`put_array` / `get_array` write `.npy` into a shared temp dir and have the
kernel load or save it. The session child is on this machine, a fixture movie is
several megabytes, and tool output is truncated for the agent's benefit — so
base64 inside a tool call would be both slow and lossy.

`get_array` returns `None` when the name does not evaluate to an array. A run
that left nothing behind is an ordinary outcome — the agent gave up, or never
got there — and it has to arrive as *"nothing to score"*, which `Outcome.passed`
already refuses to treat as a pass.

## Setup is not a turn

`session.call(...)` is the agent acting and is recorded with its turn number.
`session.setup(...)` is the harness talking to the kernel around the agent and
is recorded at turn `-1`. The report says whether a blocking question preceded
the expensive call, so injecting the fixture must never read as something the
agent did.

## Why the smoke tests exist

This is the least isolable tier in the suite: a red run's cause space is the
skill body, the model, the tool schemas, the kernel, Qt, dask and the fixture.
That is the real cost of testing against the full environment rather than a
stand-in, and it was taken on with open eyes.

`test_session_smoke.py` is the mitigation. When the stack is what broke, *it*
fails — separately, deterministically, and before a single token is spent. Its
last test is per case, over the same list the run will use, so a fixture that
cannot reach a viewer is found before the conversation that would have been
scored `no-result` because of it.
