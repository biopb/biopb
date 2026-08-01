# Testing the skills catalog

> **Status: in progress.** The gating layers are built. §5's machinery is built
> and exercised on one skill, but it is a **diagnostic harness rather than a
> gate** — it tests a transcription of a skill, not the shipped file (§5c), so
> it stays out of CI and nothing further is queued for it. §6 is the tier that
> actually tests the shipped bodies: its deterministic half is built for
> `drift-correction` (§6b) and the agent half is not. §7 remains a design sketch.
>
> This revision follows a structural decision that the earlier draft predates:
> **skills ship inside the biopb packages and are never fetched over the
> network** (§9). That collapses most of what the draft treated as a hard
> problem — two repositories, two release cadences, and a suite that had to be
> parameterised over which copy of the skills it was looking at.

How we find out whether a curated skill actually works. A skill is a prompt
fragment published to strangers' agents, and its claim is a **behavioural delta**:
an agent following it does better than one without it. But most of what goes wrong
with a skill is not behavioural at all — it is a sentence that stopped being true
about code in another repository. So the suite is a pyramid, and the cheap
deterministic layers carry most of the weight.

The second guiding idea, which shapes the whole functional tier: these skills are
written around **agent–human interaction** — blocking confirm-input, gates before
expensive work. That looks like an obstacle to an unattended suite. It is not.
Design the fixture so the ground truth is *obtainable only by asking*, and a
numeric assertion tests the interaction for free.

## 1. What "testing a skill" decomposes into

Five questions, very different costs. Only two of them need an agent.

| Layer | Question | Method | Deterministic |
|---|---|---|---|
| **Structure** | Is the file well-formed? | the frontmatter validator | yes |
| **Contract** | Does the API it asserts still exist — and can it be installed at all? | resolve the declared packages; import them and assert signatures | yes |
| **Retrieval** | Does `find_skills` surface it for the right request? | matcher fixtures + a phrasing table, no agent | yes |
| **Outcome** | Does following it produce the right numbers? | ground-truth fixtures + programmatic verifier | yes, but it tests a transcription, not the file (§5c) |
| **Interaction** | Does it ask at the right moments, and only then? | simulated user + trace assertions | no |

A sixth, **ablation** (§7), is not a test — it is an authoring tool. It answers
"is this content necessary", which is a question about the file, not about a run.

## 2. Local-first, display-optional

This applies to the **agent-facing layers only**. Structure, contract and
retrieval are ordinary unit tests — fast, deterministic, no session, no display —
and belong in CI like any other (§10). Nothing below changes that.

What is local-first is the functional tier: for those, the suite is an
**instrument, not a gate**. It runs on a workstation, with a real display by
default, and its failures are meant to be *looked at*.

This is an imaging project: a mosaic with every other row mirrored, a flat-field
that absorbed the specimen, a label overlay that is off by a downsample factor —
these are recognisable in a second by eye and awkward to characterise in an
assertion. So every outcome fixture emits **a number and an artifact**: the number
gates, the artifact explains. Runs write to a per-run directory of PNGs and arrays
a human can page through after the fact.

Consequences:

- Default `display_mode: auto` — a visible viewer when a display exists. Watching
  a fixture run is a supported way to use the suite, not an accident.
- `QT_QPA_PLATFORM=offscreen` is an **opt-in mode**, not the target. It exists so
  a subset can run unattended, and so the headless branch of the skills (numeric
  fallbacks where a screenshot is impossible) is itself exercised — `_config.py`
  already treats headless as a first-class production mode, so this is a real
  configuration, not a test artifact.
- CI runs the deterministic layers that test a **shipped artifact** (§10):
  structure, retrieval, and the contracts derived from the skills' own
  frontmatter. The outcome fixtures are deterministic too but score a
  transcription rather than the file, so they stay local as a diagnostic (§5c) —
  as does anything with an agent in it.

## 3. Layer 1 — contract tests

The highest-return layer, and the one that addresses what has actually bitten us.
Skill bodies are un-versioned assertions about someone else's API, across a repo
boundary. Recent breakages, none of which any agent test could have found:

- `m2stitch.stitch_images(row_col_transpose=...)` defaults to `True` — it swaps
  rows and cols for you. A body that passed them positionally produced exactly the
  "diagonal staircase" its own failure table described.
- The singleton-Z axis model, retired by #596, still described in two bodies.
- `np.prod(canvas) * itemsize` as a memory estimate, ~4× under the real footprint.

So: a test module that installs the packages a skill declares and asserts the
surface its body quotes — parameter exists, default is what the prose assumes,
return shape is what the snippet unpacks. Pin it to the `pkg:biopb-mcp>=X` bound
in the frontmatter, so the test and the declaration cannot drift apart.

Cheap, fast, and it fails in the author's PR rather than in a stranger's session.

**Manned again**, by `drift-correction` — the first skill whose package passes
§3a. The previous module was written entirely for `flatfield-and-stitch-tiles`,
which §3a rejected, and was deleted with it in #667 without anything noticing.

So the module now carries a coverage check of its own: a shipped skill that
declares a third-party package this layer says nothing about fails
`test_every_declared_package_is_covered_here`, the same shape as the
phrasing-table check in §4. The layer can go unmanned again only on purpose.

It runs in `skill-contracts.yaml` — one throwaway env per declared package, on
PRs that touch a skill. §10 has the reasoning for all three of those choices.

### 3a. Satisfiability comes before signatures

The draft assumed the packages a skill declares can simply be installed. They
can — but not necessarily at the version the body was written against, and the
resolver does not say so. Measured with `uv pip compile` (metadata only,
nothing downloaded):

| resolved together | basicpy | numpy |
|---|---|---|
| `basicpy` alone | 2.0.0 | 1.26.4 |
| `biopb[tensor]` + `basicpy` | 2.0.0 | 2.5.1 |
| `biopb-mcp[mcp]` + `basicpy` | **1.1.0** | 2.5.1 |

Three environments, three answers, and no good one. Resolved against the real
workspace venv, `basicpy` lands at 2.0.0 and takes numpy 2.3.5 → 1.26.4 with it
(the mechanism is its `scipy<1.13` pin, not a numpy pin). Nor is the older
version a fallback: 1.1.0 declares `pydantic>=1.9.1` but its source is pydantic
**v1** (`@root_validator`, `class Config`), so a resolver leaves the stack's
pydantic 2.x in place and `import basicpy` fails at class definition.

Nothing errors in either case. The install succeeds and the failure surfaces
later, somewhere else, looking like the agent's fault.

This is the layer's *first* question, and it is cheaper than the second:
resolution reads metadata, so it runs in CI, while importing `basicpy` pulls a
torch-backed solver and stays workstation-armed. It is also the layer that
cannot be run from anywhere else — resolving against PyPI answers for the last
*release*, not the branch. (Resolving `biopb-mcp[mcp]` from PyPI yields
`napari==0.8.0`, though the source pins `napari[all]==0.7.0` exactly.) Only the
repo holding the workspace can ask the question correctly, which is one of the
reasons the skills now live in it.

**And it is a rejection, not a caveat.** A skill that declares such a package is
a bug: the install succeeds, so the agent and the user both get the downgrade
silently, under a live kernel that has already imported the old versions. It
cannot be fixed in the body either — `guide://kernel` offers the agent an
install-it-for-you path that no single skill's prose overrides, and a package in
a separate environment is not importable from the kernel at all, which is the
agent's only execution surface. A package that genuinely needs its own
environment belongs behind the **algorithm plane**, as an `ops:<kind>` server
that is called rather than imported.

So the gate is unconditional — no allowlist, no xfail. `flatfield-and-stitch-tiles`
was written against `basicpy` and `m2stitch` and was dropped rather than shipped
with a workaround; it is the case the layer was built to catch, and it should
have been caught in review.

### 3b. A declared package is bounded, not floored

`pkg:<name>~=X.Y.Z` — PEP 440's compatible release, i.e. a floor plus an upper
bound at the next minor. Not `>=` alone, and not `==`.

**Why an upper bound.** Without one, §3's assertions prove the body against
whatever CI happened to resolve, which says nothing about the version in the
user's kernel. The bound is what makes the proof transferable: the assertions
hold across the declared range, and the declared range is what the agent
resolves. It is also what removes the need to re-run this layer on a schedule —
there is no drift to catch when the API cannot move under a shipped skill.

**Why not an exact pin.** The agent installs into a *live* kernel. `==0.2.8`
against a user who has 0.3.0 is satisfiable only by downgrading an
already-imported compiled extension — the precise pathology §3a rejects. A
bounded range is satisfied by anything already in range, so it never creates
downgrade pressure.

**Why not a comma pair.** `>=0.2.8,<0.3` is unrepresentable: the runtime reader
(`mcp/_skills.py`) splits a `[a, b]` frontmatter list on every comma *before* it
strips quotes, so the token reaches the agent as two broken fragments while the
strict parser here reads it correctly — a mis-parse that passes review and
appears only in the field. `~=` says the same thing in one comma-free token.

## 4. Layer 2 — retrieval tests

A skill nobody retrieves is not wrong, it is absent. `find_skills` does not rank
— it filters — so this splits in two, and both halves are hermetic unit tests.

**Matcher semantics**, against synthetic catalogs. What `query` means is a
contract in its own right, and it must not move when a description is reworded.
Writing these found the matcher comparing the *whole query* as one substring, so
every multi-word query failed unless the words happened to be adjacent in one
field — including the two examples the tool's own docstring offered the agent.
Each whitespace-separated term is now matched independently.

**A phrasing table**, against the real skills: (user phrasing → the skill it must
surface), plus **negative** cases that must *not* surface it. Catches the
description that drifted into an implementation summary, and the new skill that
cannibalises an existing one's queries. Plus one invariant that needs no table
and never goes stale: every shipped skill is retrievable by its own name.

The table was briefly written against a checked-in *snapshot* of the skills,
while they still lived in another repo. That is worth recording as a thing not
to do: a copy of someone else's content goes stale green — it keeps passing
while the real descriptions drift — and the first snapshot was in fact taken
from an unmerged release-candidate branch, so it described skills no agent had
ever seen. With the skills in this repo the table simply reads them.

## 5. Layer 3 — outcome fixtures

The principle from the wider eval literature: **programmatic verifiers beat
judged prose wherever you can construct one.** These skills are unusually well
suited to it, because they emit numbers with knowable right answers.

Built, for `drift-correction`, in `_tests/skills/outcomes/`. Its README is the
working documentation; this section is the reasoning.

Read §5c before using any of it: this is a **diagnostic harness that does not
gate**, because unlike §3 it tests no shipped artifact. `drift-correction` is
the worked example that proves the machinery, and **nothing further is queued** —
the reason to write a second module is a §6 finding that needs isolating, not
coverage for its own sake.

Fixtures are **synthetic and procedural** — generated at test time from a seed, so
nothing binary lands in git and the ground truth is exact by construction rather
than annotated. Built, and what else could be:

- **drift-correction** *(built)* — one image shifted along a chosen trajectory,
  so both the trajectory and the un-drifted image are truth to machine
  precision. Three cases: an ordinary blobby field at 1.7 px/frame, the same at
  4.0, and a smooth low-contrast field that separates the two methods.
- **calibrated-measurements** *(candidate, cheap)* — an ellipsoid of known
  semi-axes rasterised at 0.1/0.1/0.5 µm; true volume is analytic, so no
  tolerance has to be hunted for. But the failure it would target — passing
  `spacing=` to `regionprops` *and* multiplying by `prod(spacing)` again, wrong
  by µm³ squared and silent — is an agent's *choice*, and a reference
  implementation does not make it. That is a §6 question wearing a §5 costume.
- **segmentation-qc-metrics** *(candidate, duplicative)* — constructed gt/pred
  with overlaps chosen so TP/FP/FN and F1@0.5 are known in closed form. The
  metric itself is `plugin:segmentation_qc`, which should carry its own unit
  tests; scoring it through the skill re-tests the plugin at one remove.
- **write-a-skill** — never. It emits markdown; there is no number.

The bar for writing one is both of: the outcome is a number with a knowable
right answer, **and** a correct-by-construction reference implementation can
still fail it. Where only the first holds, the fixture scores data nothing can
fail. Each that is written emits its artifact alongside its number, per §2.

Where ground truth cannot be constructed, fall back to **binary criterion
extraction** — not "rate this 1–5" but "did it pass `spacing=`? did it budget
memory before allocating?" A judge model can extract booleans reliably even where
it cannot judge quality reliably.

### 5a. Synthetic by default, curated real data by substitution

A synthetic movie is not a microscope. Procedural fixtures give exact truth and
cost nothing to store, but they only contain the failure modes someone thought
to simulate — no vendor metadata, no genuine vignetting, no real stage error
(§11 has carried this as an open question from the start).

So the fixture is a **provider**, and real data is a substitution rather than a
rewrite: point `BIOPB_SKILL_FIXTURES` at a tree of `case.json` + `arrays.npz`
and the same verifier scores it. That path is implemented, not reserved — the
protocol tests build a curated tree in a temp dir and read it back, because a
door nothing has walked through is not open.

What the substitution costs is **truth**, and the protocol is shaped around
that. A curated movie can carry a trajectory someone measured off a bead; it
cannot carry the un-drifted reference image, because no such acquisition exists.
So a metric the fixture cannot support reports as **unavailable**, never as
passing — and an outcome that scored *nothing* has not passed, it has not been
tested. Without that rule a substituted fixture with a mis-declared truth turns
the layer green while measuring nothing, which is the same stale-green failure
§4 records about the skills snapshot.

The other cost is review. A seed needs none; an annotation is someone's claim
about their own data, and it is only as good as the review it got. That belongs
in the case's `provenance` string, which is why the field is free text and
required.

### 5b. A verifier is calibrated by a run it must reject

A verifier that only ever sees correct runs is indistinguishable from one that
returns green unconditionally. That is not hypothetical here: it is exactly how
§3 sat unmanned for a release.

So every case is run twice over — through the procedure the body prescribes, and
through the specific mistake the body warns about — and the suite asserts the
verifier tells them apart. Each expected-to-fail row is a sentence of prose from
the skill file turned into a measurement, which makes the table a second reader
of the body: a claim nobody can construct a failing run for is a claim worth
re-reading.

Three came out of writing the first one:

- `reference="first"` fails **deterministically** on the blob fixtures (23 px
  RMS at 1.7 px/frame, 53 px at 4.0) — sharper than the 2-of-4 the body quotes.
- `normalization="phase"` recovers ~0 drift on every case, so the failure
  table's first row reproduces without needing a special image.
- On the smooth low-contrast field the degraded path is **5.4 px** off while
  pystackreg is exact. The body calls the fallback "translation-only and less
  precise ... a real fallback, not a lesser one", and defers the check to step 4
  — but this error is *smooth*, so step 4's largest-single-frame-step test does
  not see it. The fixture records the discrepancy; whether the body should say
  more is an authoring decision, not a test fix.

That third one is the layer paying for itself: a claim that reads fine in review
and is wrong in a corner nobody would have chosen to look at.

### 5c. A diagnostic harness, not a gate

**This layer does not test shipped code, and that is what keeps it out of CI.**
Its subject is a **reference implementation** of the procedure — a hand
transcription of what the body says, which never reads the body. Edit step 3 to
prescribe a different argument and the suite stays green while the file and the
tested procedure part ways; delete the skill and nothing notices. A green run
certifies the transcription. Contrast §3, whose assertions are derived from the
shipped frontmatter: that is the property a gate needs, and this does not have it.

**The scope is a judgement, not a fact about the skill.** What gets tested is
whatever the fixtures and tolerances happen to span — three movies, a precision
nobody's prose states, four mistakes someone thought of — and the pass/fail
signal carries none of that. `drift-correction`'s fixtures, for instance, are
noiseless, and registration precision is fundamentally noise-limited. So the
outcomes README names its own blind spots explicitly; read a green run as the
narrow claim it is.

**What it is for is §6.** An agent run against the real skill file is the test
with teeth, and it is non-deterministic — a finding arrives as one bad run among
several, hard to reproduce and easy to dismiss. This is where such a finding gets
pinned to a fixture, a tolerance and a repeatable pass/fail, and where a fix can
be shown to hold. The order is §6 first, then this — not the reverse.

It also structurally cannot catch any instruction that needs a *choice* to be
wrong. `drift-correction` says to estimate on one structural channel and apply
the transforms to all of them; a reference implementation makes that choice
correctly by construction, so a second channel would be scored data no subject
could fail. Those belong to §6 too, where the information asymmetry does the work.

An agent run plugs into the same `Attempt` and the same verifier.

## 6. Layer 4 — interaction tests

A simulated user — an LLM with a persona and **private facts** — plus fixtures
with those facts deliberately stripped from the metadata:

> You are the microscopist. The grid is 6×4, snake order, 15% overlap. Answer
> truthfully and briefly. Never volunteer anything you were not asked.

An agent that assumes row-major gets a mirrored mosaic, which the §5 verifier
already catches. An agent that asks gets it right. **The asking needs no separate
assertion** — the information asymmetry makes correct interaction necessary for
the numeric outcome. Same trick for the physical scale in `calibrated-measurements`
and for which layer is ground truth in `segmentation-qc-metrics`.

Three cheaper things layer on the recorded trace:

1. **Structural assertions** — at most three blocking questions (the budget
   `write-a-skill` itself sets, so the test checks a skill against its own
   contract); a blocking question precedes the expensive call rather than
   following it; nothing destructive without a preceding ask.
2. **Uncooperative respondent variants** — real users say "I don't know".
   `calibrated-measurements` specifies that branch: report pixels and *label them
   as pixels*. Assert the columns end in `_px` and no µm claim appears. Otherwise
   that clause is never exercised.
3. **Gate spies** — make the expensive step cheap but instrumented, and assert it
   was not reached before approval. Tests validate-and-gate without needing
   anything to actually be expensive.

**The respondent prompt is a fixture and gets reviewed like one.** A chatty
respondent that volunteers the grid order rescues a bad agent and silently
invalidates the suite.

Keep this layer small. Multi-turn tests are the slowest and flakiest thing here;
point them at the checkpoint contract and get bulk coverage from §5, where the
environment already holds everything and no conversation happens.

### 6a. The agent under test is not from the family that wrote the skill

These bodies were co-authored with Claude. A Claude agent can pass by
recognising its own prose rather than by reading it, and §7 already records that
blind spots correlate within a family — so the agent under test is a hosted
**non-Anthropic** model, reached through an OpenAI-compatible tool-calling
adapter. Claude's role here is the **respondent**, which is a different job:
hold a persona and a few private facts, answer, volunteer nothing. Family
contamination is harmless there, and the respondent is deliberately kept
**skill-blind** — it never sees the body, so it cannot rescue a bad run by
paraphrasing step 2 back at the agent.

The reference agent is therefore a fixture too, and its *family* is part of what
makes the fixture valid — not just its version.

### 6b. Prove the asymmetry before paying a model to demonstrate it

The whole tier rests on one claim about each fixture: that the stripped fact is
genuinely unobtainable from the pixels, so the numeric outcome cannot come out
right without asking. That claim is easy to get wrong and expensive to test with
a model — and a fixture whose ambiguity every heuristic happens to survive
leaves a suite that certifies nothing while looking like it tests conversation.

So it gets settled first, deterministically, with **scripted subjects**: one
told the private fact, and one for each choice a run that never asked would
plausibly make. If the §5 verifier cannot separate them, there is nothing here
to test. This is §5b's calibration argument one tier up — there, a verifier no
run has ever failed is not known to work; here, a fixture nothing has ever
failed *for want of asking* is not known to test asking.

**Built for `drift-correction`** (`outcomes/_drift_channels.py`,
`test_drift_channel_choice.py`). Step 2 asks the user whether the field or the
objects moved, and which channel is `REF_CHANNEL`. The fixture is a two-channel
movie with the names stripped: channel 0 is bright puncta that carry the stage
drift *and* crawl on their own, channel 1 a dim field that moves only with the
stage. Every available heuristic points at channel 0 — five times the contrast,
an order of magnitude the peak — and registering there adds the objects' common
motion to the stage's without anything failing. Told the answer: 0.0006 px RMS
worst case. Registering on brightness: 1.95–5.28 px. Averaging the channels,
which the Parameters table forbids by name: 1.92–5.16 px. Scored by §5's
verifier unchanged.

That fixture is deterministic, needs no key and no model, so it lives under the
`outcome` marker with the rest of §5's machinery. The agent half is what needs
the harness below.

## 7. Layer 5 — ablation (authoring, not gating)

Give a small model the task **without** the skill, closed-book, and diff against
what the body says. Cut what it gets right unaided; keep what it gets wrong.

Two rules learned the hard way, both now in `write-a-skill` step 6:

- **Disclose the environment, withhold only the skill.** A first run withheld
  `basicpy`/`m2stitch` too; the models hand-rolled everything and blending came
  along for free. That manufactured evidence for one rule and destroyed the
  evidence for another, and content was cut on the strength of it.
- **Do not ask a model what is obvious.** It introspects badly, and this guidance
  fails silently by nature. Test behaviour, not self-report.

Add a **negative control**: a condition with an irrelevant skill injected. If
"+skill" wins as much when the skill is nonsense, the measurement is picking up
"more context → more effort", not content.

Cross-*family* coverage beats cross-*size* — blind spots correlate within a family.
Use a weak model to ask "is this necessary" and the strongest available to ask "is
this redundant or over-constraining".

## 8. The environment, in tiers

Test against the **agent-visible contract**, not the implementation. The agent
sees `client`, `viewer`, `ops`, and plugin names; a fixture providing those
interfaces exercises the skill faithfully, whatever is behind them.

| Tier | What it is | Serves |
|---|---|---|
| 0 | No session; environment described in the prompt | Ablation |
| 1 | Kernel + a fake `client` returning dask arrays with real `dim_labels` and physical scale | Most outcome fixtures |
| 2 | Tier 1 + napari with a genuine multiscale pyramid | Layer semantics: `layer.data` as a list, non-positional `scale`, agent-built layers with no geometry |
| 3 | Full stack, real tensor server | Upload round-trips, `add_tensor` |

Most fixtures are Tier 1. The pyramid-level trap needs Tier 2. Almost nothing
needs Tier 3.

## 9. Where the skills live, and how they ship

**A skill is documentation about a specific runtime version.** It quotes an API,
assumes a namespace handle, and depends on packages resolving a particular way.
Serving one catalog over HTTP meant every deployed version of biopb read the
same text at once: a body had to be simultaneously correct for every release in
the field, with no migration story, and a bug report could not answer "which
skill text did this session see?".

So the skills **live in this repo and ship inside the packages**, on the same
upgrade cycle as the code they describe. `biopb.org/skills/` stays served but
frozen; older clients still fetch it, and they fail open to their own bundled
copy when it goes away.

What that deletes, rather than solves:

- The network path — fetch, TTL cache, on-disk cache, atomic writes, corrupt-cache
  repair, sha verification of fetched bodies, and every test describing them.
- `catalog.json`. The frontmatter *is* the metadata and the runtime already
  parses it (the local-skills reader does exactly this), so a generated index
  only adds a second thing to disagree with the bodies — which is precisely what
  the sha check existed to catch.
- The cross-repo test problem in every form it took: a vendored matcher, a copy
  of the skills, a scheduled job diffing two repositories.
- Deploy lag as a source of confusion. At the time of writing the served catalog
  was three skills and two weeks behind the authoring branch.

**The cost, stated plainly: a skill fix now needs a release.** The escape hatch
is the one that already exists — `~/.config/biopb/skills/*.md` is merged beside
the shipped set and marked `origin: local`, so a lab can add or override a skill
without waiting for one. That path is unchanged and is now the *only* way skills
arrive out of band, which makes it load-bearing rather than a convenience.

Everything else follows: harness, fixtures, validator, and skills are in one
tree, so every deterministic layer is an ordinary hermetic unit test with no
parameterisation over where the skills came from.

## 10. What gates what

| Layer | Where | Gates a merge? |
|---|---|---|
| Structure, Retrieval | this repo's CI | yes |
| Contract — satisfiability (§3a) | this repo's CI (metadata resolution only) | yes |
| Contract — signatures (§3) | this repo's CI, on skill-touching PRs only | yes |
| Outcome (§5) | local — a diagnostic harness, run when §6 gives it something to isolate | no |
| Interaction | local | no |
| Ablation | manual, per skill edit | no |

Everything that gates is in this repo, and a skill edit and the runtime change it
depends on can land in the same PR.

**Signatures gate, but on their own trigger and in their own envs.** They were
planned as workstation-only, on the grounds that they need the package actually
installed — and a layer that is armed rather than running is how the last one
rotted. `skill-contracts.yaml` runs them, and two properties earn the change:

- **One ephemeral env per declared package.** The shared test env would force
  every skill's package to co-exist with every other's, so the first pair that
  cannot would break the whole suite — and the reflex fix would be to drop a
  skill. A package that will not share an env is not a reason not to ship a
  skill, so the harness must not make it one.
- **A `paths` filter, not every PR.** A skill's third-party dependency should
  cost nothing to a PR that does not touch skills.

**The outcome layer does not gate, and the reason is not flakiness.** The
reference-run half has every property that would qualify it — deterministic, no
agent, no display, no network, four seconds — and it was briefly armed on those
grounds. The disqualifying property is different: **it does not test anything
this repo ships.** Its subjects are a hand transcription of what a skill body
says, not a reading of the file, so a green run certifies the transcription.
Delete the skill and the suite stays green. That is the opposite of what §3's
assertions do — those are derived from the shipped frontmatter, which is what
makes them a gate worth having.

So it sits beside the merge gate rather than in it, as the harness §6's findings
get isolated in (§5c).

**And no cron**, because a skill declares a *bounded* range (§3b), so the API
cannot move under a shipped skill: what a user resolves is inside the range the
assertions were proved against. Upstream releasing a new minor is not an event
this layer needs to hear about. What is left — a body edited to call something it
never ran — is change-triggered, so it belongs on the PR that does it. A range
that stops installing is §3a's job, and §3a runs everywhere.

Stochastic gates get muted within two weeks of the first flake, and then you have
neither the gate nor the trust. Gate on the deterministic layers; treat the agent
layers as signal a human reads.

## 11. Open questions

- **Fixture overfitting.** Tuning bodies against a fixed fixture set optimises for
  the set. Hold some out, or rotate seeds — unresolved which.
- **Cost of the interaction layer.** Every run is a multi-turn conversation with
  two models. Unclear whether this is a per-edit or a per-release activity.
- **Real data.** The suite is fully synthetic and offline by design. A tier that
  fetches a real acquisition (download-on-demand, cached) would catch what
  procedural fixtures cannot — vendor metadata, genuine vignetting, real stage
  error — at the cost of offline runs.
- **Which model is the reference respondent**, and how much its phrasing changes
  outcomes. It is a fixture, so it needs a pinned version and a review process.
- **Release cadence versus skill churn** (§9). Shipping skills with the packages
  trades hot-fix for coherence. If skill edits turn out to outpace releases by
  much, the pressure valve is the local dir — but that is per-machine, so it
  does not help a broken *published* skill. Watch for it.
- **How a skill declares the runtime it needs**, now that the two ship together.
  `pkg:biopb-mcp>=X` in the frontmatter was a cross-repo version bound; within
  one release it is either redundant or a statement about back-compat, and it is
  not yet clear which.
