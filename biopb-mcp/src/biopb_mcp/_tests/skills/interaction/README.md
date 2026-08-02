# The interaction layer

[`biopb-mcp/docs/skill-testing.md`](../../../../docs/skill-testing.md) §5. Put a
model in front of the **shipped** skill body, against a **real** session, and
score what comes out.

```sh
# needs a GL-capable display -- see "what this needs" below
xvfb-run -a -s '-screen 0 1024x768x24' \
  uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills/interaction -m interaction
```

Deselected by default (`-m interaction`), like `satisfiability`, and never in
CI (§1).

The hermetic half — the loop, the trace, the personas, the fixtures, the
verifiers — is **not** marked and runs with the ordinary suite. A break in the
machinery, or a case whose persona gives its own answer away, should surface as
a normal red test rather than be found by someone mid-diagnosis with a paid
run.

## Running one with real models

Both sides are `provider:model`, named separately, so they can be different
vendors — or the same compatible API at two addresses:

```sh
export BIOPB_SKILL_AGENT=openai:gpt-5                  # default
export BIOPB_SKILL_RESPONDENT=anthropic:claude-sonnet-5 # default
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...

xvfb-run -a -s '-screen 0 1024x768x24' \
  uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest .../interaction -m interaction
```

Known providers: `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` — each a
`(sdk, base_url, key_env)` triple, and most of them the OpenAI-compatible API
at a different address. Override an address with
`BIOPB_SKILL_AGENT_BASE_URL` / `BIOPB_SKILL_RESPONDENT_BASE_URL`. A bare model
name is refused rather than guessed: which vendor serves a model is exactly the
fact §5a turns on, and inferring it from the name would make the rule depend on
vendors' naming conventions.

**§5a constrains the agent, not the respondent.** The respondent is skill-blind
and answers from a fact table, and having written the skills does not help with
that — so Anthropic is a fine respondent and is the default, while the default
*agent* is deliberately not from the authoring family. `test_models.py` asserts
both, off the provider table rather than off a comment.

`ollama` needs no key, which makes it the cheap way to rehearse a run end to end
before spending anything.

The provider SDKs are imported lazily and are **not** dependencies of this
package — one `--with` line each. Keys are read from the environment at call
time and are never written to a trace, an artifact or a log.

## What is here

| File | Holds |
|---|---|
| `_session.py` | Bring-up: a real shim-spawned session, a synchronous façade over the async MCP client, and the environment facts that are forced rather than inherited |
| `_bridge.py` | MCP tool schemas → the function-calling shape a chat model expects |
| `_models.py` | The provider table: which model on each side, at which address, with which key |
| `_agent.py` | `ChatAgent`; `ScriptedAgent`, `ReplayAgent`, and the live `ToolCallingAgent` |
| `_respondent.py` | `Persona`, `Respondent`; `ScriptedRespondent`, `SilentRespondent`, and the live `ModelRespondent` |
| `_fixture.py` | What a run is given and what it recovers: `Fixture`, `Attempt`, `Metric`, `Outcome`, the curated-data path, artifact writing. Knows no skill |
| `_benchmark.py` | The engine: `Case`, the 2x2 arms, outcome classification, the report. Knows no skill |
| `cases/` | One module per skill, each a single `CASE`. Data, not code |
| `_conversation.py` | The two-model loop, the caps, the `Trace` |
| `test_benchmark.py` | The pytest surface: run every case, assert only that it reported |
| `test_session_smoke.py` | That the stack works, with **no model in it** |
| `test_conversation.py` | That the loop works, with no model *and* no session |
| `test_report.py` | That the engine classifies and reports, on hand-built outcomes |
| `test_cases.py` | That every case's persona, fixture and verifier hold up — and that the catalogue is covered |
| `test_fixture_protocol.py` | The scoring protocol itself, including the curated path almost no machine has data for |
| `test_models.py` | That provider selection resolves, and that §5a holds of the defaults |

## Adding a skill

The engine is skill-agnostic, so a new skill is **data**: one module under
`cases/`, exporting a module-level `CASE`. There is no registration step — the
module is discovered by being there.

```python
CASE = Case(
    skill="calibrated-measurements",
    task=TASK,                        # the prompt, incl. where results land
    persona=MICROSCOPIST,             # who holds the fact the fixture withholds
    build=Ellipsoids(),               # () -> Fixture: data, truth, tolerances
    layers=(Layer("nuclei", "image"),
            Layer("nuclei_labels", "labels", kind="labels")),
    collect={"volumes_um3": "volumes_um3", "spacing_um": "spacing_um"},
    score=verify,                     # (fixture, attempt) -> Outcome
    save_artifacts=save_artifacts,
    plugins=("segmentation_qc",),     # kernel plugins the skill's `requires:` names
    persona_must_know=(...), persona_must_not_know=(...),
)
```

A skill that cannot be benchmarked goes in `cases.NOT_BENCHMARKED` with the
reason instead; `test_cases.py` asserts the shipped catalogue is covered by one
or the other, so "what does this cover" never has to be answered by reading the
directory.

No test code either way: `test_benchmark.py` parametrizes over `CASES`, so the
new case brings its own arms, report and transcripts, and `test_cases.py` starts
checking its persona, its fixture and its verifier by its arriving. Report and
transcripts land under `.skill-outcomes/interaction/<skill>/`.

## A benchmark, not a gate

**No run's outcome fails a test here.** Each arm reports an `outcome` and a
`reason` — `ok`, `wrong-answer`, `out-of-turns`, `out-of-tool-calls`,
`gave-up`, `no-result`, `unscorable-result`, `harness-error` — plus flags that
change how to read it: `cut-off-but-scored`, `over-ask-budget(n)`,
`never-asked`, `catalog-mismatch`.

Every arm runs inside its own `try`, so a corner that dies becomes a row rather
than an exception that destroys the other three. The report is the deliverable;
a poor fixture is still informative, a missing report is not.

Two things *are* asserted, and neither judges the skill: that `summary.md`
reached disk with a transcript per arm, and that the **ablation took effect**.
The second is not a finding — if `skills_enabled: false` stopped withholding
the catalog, the delta would read as zero for a reason unrelated to the skill,
which is a green table asserting the opposite of the truth.

## What the first full run showed

Agent `glm-5.1`, respondent `qwen3.5-plus`, one sample per corner:

| arm | rms px | within tol | stopped |
|---|---|---|---|
| skill + asked | 0.00033 | yes | finished |
| skill + silent | 5.28 | no | turn-cap |
| noskill + asked | 4.72 | no | finished |
| noskill + silent | 0.024 | **yes** | turn-cap |

The floor passed, and beat the arm that had a microscopist answering by 200x.
No monotonic pattern, so the spread is run-to-run variance rather than the
manipulations: **no delta can be claimed from this, in either direction.**

That is the layer's real output so far, and it is worth more than a green
light. `n=1` per corner is not a measurement — see `biopb-mcp/docs/skill-testing.md` §5c
for what follows.

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
because the offscreen platform provides no GL context. Use a desktop session's
own display, or `xvfb-run`. Without one these tests **skip with instructions**
rather than run somewhere subtly different.

Nothing else: no API key for the smoke tests, and no network beyond loopback.

## What a fixture has to withhold

The whole tier rests on one claim per case: that the fact the respondent holds
is **not obtainable from the pixels**, so the numeric outcome cannot come out
right without asking.

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

## Four things the harness forces

Each of these silently changes what a run tests, so none of them is inherited
from whatever machine is running.

**A real viewer.** `transport.display_mode` defaults to `auto`, which degrades
to a viewer-less kernel when no display is found. That is a legitimate
production mode and so nothing fails loudly — but a run that took it would be
scoring a session in which step 2's *"show the user the first and last frames"*
cannot happen at all. Bring-up probes for it and refuses.

**No tensor plane.** A developer box often has a data plane up, and then
`client` is live and the agent can wander into whatever catalog that machine
holds — so a finding might not reproduce anywhere else. The child is pointed at
an unreachable URL, `client` lands as `None`, and the fixture reaches the agent
as a napari layer and nothing else. Every skill's Parameters table already
accepts *"a layer on `viewer`"* as a source, and a session with no tensor plane
is a real configuration a user can be in, so step 1 still has something true to
resolve.

**A config tree of our own.** `XDG_CONFIG_HOME` points at a temp dir, so the run
reads neither the developer's `mcp-config.json` nor their personal
`~/.config/biopb/skills/*.md`. The catalog under test is the shipped one.

**Only the kernel plugins a case asks for.** That same temp tree means an empty
`~/.config/biopb/kernel/`, so a skill declaring `plugin:segmentation_qc` would
be scored in a session where its own `requires:` cannot be met. `Case.plugins`
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

§5 is the least isolable tier in the suite: a red run's cause space is the skill
body, the model, the tool schemas, the kernel, Qt, dask and the fixture. That is
the real cost of testing against the full environment rather than a stand-in,
and it was taken on with open eyes.

`test_session_smoke.py` is the mitigation. When the stack is what broke, *it*
fails — separately, deterministically, and before a single token is spent.
