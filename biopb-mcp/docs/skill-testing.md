# Testing the Skills Catalog

**Status:** Implemented. Four layers; three gate a merge, one is a local
benchmark.
**Component:** `biopb-mcp` — `_tests/skills/` (the suite),
`mcp/_skills_data/` (the skills under test), `mcp/_skills.py` (the runtime
reader), `.github/workflows/skill-contracts.yaml` +
`.github/scripts/skill_contracts.py` (the per-package CI job).
**Related:** [`skill-interface.md`](skill-interface.md) — what a skill *is*, how
it ships, and how `requires:` resolves at runtime.

---

A skill is a prompt fragment published to strangers' agents, and its claim is a
**behavioural delta**: an agent following it does better than one without it.
But most of what goes wrong with a skill is not behavioural — it is a sentence
that stopped being true about somebody's API. So the suite is a pyramid, and the
cheap deterministic layers carry the weight.

The layer that *is* behavioural exploits the thing that looks like an obstacle.
These skills are built around agent–human interaction — blocking confirm-input,
gates before expensive work — which seems untestable unattended. It is not:
build the fixture so the ground truth is **obtainable only by asking**, and a
numeric verifier tests the interaction for free.

## 1. The four layers, and what gates

| Layer | Question | Where it lives | Gates a merge? |
|---|---|---|---|
| **Structure** (§2) | Is the file well-formed, and does it obey the authoring rules? | `test_schema.py`, `test_validate.py`, `test_shipped_skills.py`, `test_packaging.py` | yes, in `mcp-ci` |
| **Retrieval** (§3) | Does `find_skills` surface it for the right request? | `test_retrieval.py` | yes, in `mcp-ci` |
| **Contract** (§4) | Can its packages be installed, do they import, and does the API it quotes still exist? | `test_satisfiability.py`, `test_contracts.py` | yes — satisfiability in `mcp-ci`, the rest in `skill-contracts.yaml` |
| **Interaction** (§5) | Does a model following it produce the right numbers? | `interaction/` | **no** — a benchmark; and the case *data* under it does gate |

Everything that gates is in this repo, so a skill edit and the runtime change it
depends on land in the same PR.

Two pytest markers hold work back from the default run
(`biopb-mcp/pyproject.toml` `addopts`):

- `satisfiability` — each token is a real resolver run; `mcp-ci` runs it as its
  own step on every matrix cell.
- `interaction` — needs a display, API keys and about twenty minutes.

Everything else in `_tests/skills/` — including every hermetic check on an
interaction case (§5e) — runs with the ordinary suite.

**Stochastic gates get muted within two weeks of the first flake, and then you
have neither the gate nor the trust.** That is why §5's runs report rather than
fail.

## 2. Structure

Ordinary hermetic unit tests over `mcp/_skills_data/*.md`.

- `test_schema.py` — the frontmatter contract itself: list coercion, semver,
  kebab-case ids, the shape of an entry.
- `test_validate.py` — which malformations are fatal (id disagreeing with the
  filename, missing description, non-semver version, a missing required `##`
  section, unparseable YAML, an empty body) and which only warn (a missing
  title, inferred from the H1; a future spec version, clamped).
- `test_shipped_skills.py` — the authoring rules the validator does not express
  generically: the live `requires:` vocabulary (`viewer`, `tensor`, `dask`,
  `ops:<kind>`, `plugin:<stem>`, `pkg:<spec>`), no duplicate tokens, a
  kernel-driving skill pinning a `pkg:biopb-mcp>=X` floor, a declared
  `plugin:<stem>` actually being called through that module name, `[[wiki-links]]`
  landing on a skill that exists, and the mechanically checkable guardrails from
  `write-a-skill` (one-sentence description, imperative title, at least one tag,
  a length proxy, a first step that checks requirements, no body naming a
  specific dataset).
- `test_packaging.py` — every skill reaches the wheel, none is empty, and the
  test suite does not ship.

**Two readers, on purpose.** `mcp/_skills.py` is tolerant — it is on the agent's
path, so a malformed file degrades to a skipped entry and a bare markdown file
with no frontmatter still loads. `_validate.py` here is strict — it is on the
author's path, where the same file should stop the PR.
`test_what_validates_is_what_the_runtime_loads` pins the two to the same answer
about which files are skills, so a file only the gate can read cannot pass review
and then be invisible in the field.

## 3. Retrieval

A skill nobody retrieves is not wrong, it is absent. `find_skills` filters, it
does not rank, so this splits in two — both hermetic.

**Matcher semantics**, against synthetic catalogs. What `query` means is a
contract in its own right and must not move when a description is reworded. Each
whitespace-separated term is matched independently, against
id/title/description/tags.

**A phrasing table**, against the real skills: (user phrasing → the skill it must
surface), plus **negative** cases that must not surface it. That catches a
description drifting into an implementation summary, and a new skill
cannibalising an existing one's queries. Two invariants need no table and never
go stale: every shipped skill is retrievable by its own name, and every shipped
skill appears in the phrasing table.

## 4. Contract

A skill body is an un-versioned assertion about someone else's API, and this is
where that gets checked. Recent breakages it exists for: a stitching call whose
`row_col_transpose=` defaults to `True` and silently swaps axes; a retired
singleton-Z axis model still described in two bodies; `np.prod(canvas) *
itemsize` as a memory estimate, ~4× under the real footprint.

The layer asks three questions in order, and the first is cheapest.

### 4a. Satisfiability — may this package be installed here at all?

`test_satisfiability.py`, marker `satisfiability`, metadata only (`uv pip
install --dry-run`); nothing is downloaded.

A skill may not declare a `pkg:` token that installs only by **moving something
already in the environment**. `uv pip install --dry-run basicpy` succeeds — and
takes numpy 2.3.5 down to 1.26.4, plus pandas and scipy, under a live kernel that
has already imported numpy 2. Nothing errors; the failure surfaces later,
somewhere else, looking like the agent's fault.

**It is a rejection, not a caveat.** It cannot be fixed in the body — the kernel
is the agent's only execution surface, so "install it in a separate venv" is not
a resolution. A package that genuinely needs its own environment belongs behind
the algorithm plane as an `ops:<kind>` server, called rather than imported. The
gate is unconditional: no allowlist, no xfail, since either would be a place to
record that a known-bad skill ships anyway.

**It runs on every `mcp-ci` matrix cell** (ubuntu 3.10/3.11/3.12, macos 3.12,
windows 3.12), because the answer depends on the interpreter and the platform —
wheel availability differs per cell. One red cell rejects the skill: the catalog
ships to every platform, so "resolves on Linux" is not good enough.

Only this repo can ask the question correctly. Resolving against PyPI answers for
the last *release*, not the branch — `biopb-mcp[mcp]` from PyPI yields
`napari==0.8.0` where the source pins `napari[all]==0.7.0`.

The workspace's own distributions (`biopb`, `biopb-mcp`, `biopb-tensor-server`,
`biopb-control`) are skipped: a floor on one is a statement about this repo's
release history, not about a third party.

### 4b. Import — does the installed package work at all?

`test_contracts.py::test_every_installed_declared_package_actually_imports`, one
check over whatever the catalog declares. §4a asks whether a package can be
installed without damage, and a package can pass that and still be useless:
`uv pip install --dry-run stardist` resolves clean and moves nothing, because
`csbdeep` declares TensorFlow only under a `[tf1]` extra, and then `import
stardist` raises. The skill dead-ends at step 1 for every user.

Unlike §4c this needs no per-package authoring — it is not a claim about anyone's
API — so it runs over every declared package the env has installed. Which env
that is does not matter: `skill_contracts.py` gives each package its own, so an
absent distribution is legitimate, and a present one that will not import is
fatal on every platform.

### 4c. Signatures — is the API still what the body quotes?

`test_contracts.py`: parameter exists, default is what the prose assumes, return
shape is what the snippet unpacks. Currently manned by `drift-correction` —
`pystackreg`'s modes and `reference="previous"` default, where step 4 reads the
translation out of the transform matrix, and `skimage`'s
`phase_cross_correlation` normalization default for the degraded path.

Three properties keep it honest:

- **The work is derived from the shipped frontmatter.** The packages come out of
  the skills' own `requires:`, so deleting a skill changes what this layer does.
- **Coverage is checked in both directions.** A shipped skill declaring a
  third-party package with nothing asserting its surface fails
  `test_every_declared_package_is_covered_here`; a `COVERED` entry naming a
  package no skill declares fails `test_covered_is_not_stale`. This layer once
  sat unmanned for a release because its only skill had been dropped, and that
  is what the pair prevents.
- **The installed version must be inside every declared range**
  (`test_the_installed_version_is_inside_every_declared_range`), which ties the
  assertions to the frontmatter in both directions.

**A third-party token is bounded, not floored** — `pkg:<name>~=X.Y.Z`, PEP 440's
compatible release: a floor plus an upper bound at the next minor. The bound is
what makes the proof transferable — the assertions hold across the declared
range, and the declared range is what the agent resolves — and it is why this
layer needs no cron: the API cannot move under a shipped skill. An exact `==`
pin is wrong for the opposite reason: the agent installs into a *live* kernel, so
a pin against a user who already has a newer version is satisfiable only by the
downgrade §4a rejects. `pkg:biopb-mcp>=X` stays a bare floor; it is a statement
about this repo's own release history.

The grammar accepts no comma pair. `>=0.2.8,<0.3` is unrepresentable: the runtime
reader splits a `[a, b]` frontmatter list on every comma *before* it strips
quotes, so the token would reach the agent as two broken fragments while the
strict parser here reads it correctly — a mis-parse that passes review and
appears only in the field. `~=` says the same thing in one comma-free token.

**Its own workflow, not a step in `mcp-ci`** (`skill-contracts.yaml` +
`skill_contracts.py`): one throwaway env per declared package, over the same
five-cell matrix, behind a `paths` filter. A shared env would force every skill's
package to co-exist with every other's, so the first pair that cannot would break
the whole suite — and the reflex fix would be to drop a skill. Co-resolving one
skill package *with the workspace* is exactly what §4a certifies is safe;
co-resolving skill packages *with each other* is what the per-package envs avoid.

## 5. Interaction

A simulated user — an LLM with a persona and **private facts** — plus a fixture
with those facts stripped from the data. The agent under test drives a **real
biopb-mcp session** and its result is read back out of the kernel and scored by a
programmatic verifier.

> You are the microscopist. One pixel is 0.1 µm across and the z-step is 0.5 µm.
> Answer truthfully and briefly. Never volunteer anything you were not asked.

An agent that assumes cubic 1 µm voxels reports volumes wrong by 200×. An agent
that asks gets it right. **The asking needs no separate assertion** — the
information asymmetry makes correct interaction necessary for the numeric
outcome.

The verifier is **programmatic, never judged**: these skills emit numbers with
knowable right answers, so it computes a number and compares it to a limit, and
nothing reads prose.

Skills listed in `cases.NOT_BENCHMARKED` are skipped, e.g. `write-a-skill`. It
emits markdown, and there is no number with a knowable right answer.

### 5a. The agent matrix

```
BIOPB_SKILL_AGENT=openai:gpt-5                    # default
BIOPB_SKILL_RESPONDENT=anthropic:claude-sonnet-5  # default
```

Both sides are `provider:model` and are configured independently, with separate
base-URL overrides (`BIOPB_SKILL_{AGENT,RESPONDENT}_BASE_URL`). Known
providers: `openai`, `anthropic`, `gemini`, `deepseek`, `ollama` — each a
`(sdk, base_url, key_env)` triple.
Keys may come from the environment or a `.env` (`BIOPB_SKILL_ENV_FILE`, the repo
root, then the biopb config dir), and are never written to a trace or an
artifact. Anthropic agents were involved in the authoring of the skills and can
pass by recognising their own prose rather than by reading it, so the agent under
test is explicitly gated to be non-Anthropic. That fact lives in the provider
table rather than in prose, so `test_models.py` asserts it.

**The rule constrains the agent and only the agent.** The respondent holds a
persona and answers from a fact table; it is deliberately **skill-blind**, so it
cannot rescue a bad run by paraphrasing step 2 back at the agent, and having
written the skills does not help it do that job. Claude is therefore a fine
respondent, and is the default.

### 5b. A real session, not a stand-in tool surface

The run happens against a real shim-spawned session child: a real IPython
kernel, a real napari viewer, real dask, and the nine real tools reached over
real MCP with their own schemas and the server's own `instructions`. The body
arrives through the real `mcp/_skills.py` — `find_skills` and `skill://<id>`,
the same calls the runtime makes — so editing or deleting a skill changes what a
run is scored against.

The cost is that a red run's cause space includes the kernel, Qt, dask and the
tool schemas. Two things bound it: the trace is written before any assertion
runs, and `test_session_smoke.py` — no model, no key — fails separately, first,
and for free when the stack is what broke. A failed smoke test *skips* the
benchmark rather than merely preceding it, because a run on a broken stack does
not produce a weak result, it produces a meaningless one that reads like a weak
one.

Four environment facts are **forced rather than inherited**, because each
silently changes what a run tests:

- **A GL-capable display.** `display_mode: auto` degrades to a viewer-less
  kernel, and `QT_QPA_PLATFORM=offscreen` is not enough either — napari builds
  and then `add_image` dies in vispy's extension probe. Either way a step that
  says *"show the user the first and last frames"* could not happen. Bring-up
  probes and refuses; use a desktop session or `xvfb-run`.
- **No tensor plane.** `BIOPB_TENSOR_URL` points at an unreachable address, so
  `client` lands as `None` and the agent cannot wander into whatever catalog the
  developer's machine happens to hold. The fixture reaches it as a napari layer,
  which every skill's Parameters table accepts as a source.
- **A config tree of its own.** `XDG_CONFIG_HOME` points at a temp dir, so the
  catalog under test is the shipped set and not the developer's personal
  `~/.config/biopb/skills/*.md`.
- **Only the kernel plugins the case declares.** That same private tree means an
  empty `~/.config/biopb/kernel/`, so a skill requiring `plugin:segmentation_qc`
  would otherwise be scored where its own `requires:` cannot be met.
  `Case.plugins` seeds the ones it names, from the copies biopb-mcp ships,
  through the real loader — and nothing else.

The fixture is injected through `session.setup()`, recorded at turn `-1`, so it
never reads as something the agent did. Arrays cross as `.npy` files in a shared
temp dir rather than as base64 in a tool call.

### 5c. A benchmark, not a gate

A skill's claim is a behavioural delta, so measuring it needs a baseline. Each
case runs a 2x2 and reports the corners:

| | respondent answers | respondent silent |
|---|---|---|
| **skill offered** | does the whole thing work | does *asking* matter |
| **skill withheld** | does the *skill* matter | the floor |

Withholding is `services.skills_enabled: false` — a real shipped configuration,
so the kernel, napari, dask and every library stay as they are and only the
curated procedure goes. The ablation is checked on what the catalog *returns*,
not on whether `find_skills` was called: the tool stays registered either way and
`load_catalog()` is what gates.

**No run's outcome fails a test.** Each arm becomes a row with an outcome and a
reason — `ok`, `wrong-answer`, `out-of-turns`, `out-of-tool-calls`, `gave-up`,
`no-result`, `unscorable-result`, `harness-error` — plus flags that change how to
read it: `cut-off-but-scored`, `over-ask-budget(n)`, `never-asked`,
`catalog-mismatch`. Ordering matters: a cap beats a bad number, so a run severed
mid-workflow is not reported as a wrong answer. Every arm runs inside its own
`try`, so a corner that dies becomes a row instead of an exception that destroys
the other three.

Two things *are* asserted, and neither judges a skill: that the report reached
disk with a transcript per arm, and that the ablation took effect. The second is
not a finding — if `skills_enabled: false` stopped withholding the catalog, the
delta would read as zero for a reason unrelated to the skill.

`asked` counts blocking questions against the budget `write-a-skill` step 4 sets
(at most three), and the trace records whether a question preceded the first
expensive call. Both are reported, not asserted.

**Outputs.** Per case, under `.skill-outcomes/interaction/<skill>/` (override
with `BIOPB_SKILL_OUTCOME_DIR`, gitignored): `summary.md` and `summary.json`,
and per arm a `transcript.md`, a `trace.jsonl`, the verifier's `summary.json`,
and the case's artifacts — PNGs and CSVs. The number is the result; the artifact
explains it, which in an imaging project is usually what a person needs.

A run is bounded at 90 turns and 200 tool calls. The caps are generous on
purpose: these workflows promote compute to background jobs, and a cap that stops
a working run only produces unreadable results.

### 5d. The fixture

Each case ships a builder that returns a `Fixture`: `data` (what the run is
given), `truth` (what it has to recover, withheld), `tolerance`, and a
`provenance` string. Keeping `data` and `truth` as separate mappings makes the
leak the whole layer depends on not happening — a truth key appearing in `data` —
a thing a test can assert about any fixture without knowing the skill.

**Withhold something categorically absent from the data** — a unit, a scale, a
provenance, an identity. Defeating the heuristics the fixture's author thought of
is weaker: `drift-correction`'s movie is built so contrast, peak intensity and
feature density all point at the wrong channel, and a capable agent still
recovered the answer by registering on both and keeping the self-consistent one.
A µm-per-voxel figure has no such back door.

**Truth is data, not a formula**, so real data can replace a synthetic case
without touching the verifier: point `BIOPB_SKILL_FIXTURES` at a tree of
`case.json` + `arrays.npz` per skill and `curated_for()` uses it instead. A
curated movie carries whatever a human annotated — a trajectory measured off a
bead, but never the un-drifted reference image, because no such acquisition
exists.

**A metric that cannot be computed is `unavailable`, never passing**, and an
`Outcome` that scored *nothing* has not passed — it has not been tested. That
rule covers both halves of the same problem: a curated fixture whose truth does
not support a measurement, and an agent that left nothing behind or bound a name
to the wrong shape. Without it the silent arm of every 2x2 would read as a clean
run.

Fixtures are otherwise **synthetic and procedural** — generated at test time from
a seed, so nothing binary lands in git and the truth is exact by construction.
Where a second derivation is cheap, the builder asserts the two agree before
handing the fixture over: `segmentation-qc-metrics` checks its closed-form
TP/FP/FN against what `plugin:segmentation_qc` actually matches, and
`calibrated-measurements` checks its voxel-count truth against
`regionprops(spacing=)`. A fixture whose truth is wrong makes every arm scored
against it meaningless, so that fails at build time rather than reporting a quiet
zero later.

### 5e. One file per skill, and what gates about it

The engine (`_benchmark.py`) owns the arms, the outcome classification and the
report, and knows no skill. The scoring vocabulary (`_fixture.py`) knows no skill
either. A skill contributes exactly one module under `cases/` exporting a
module-level `CASE`:

```python
CASE = Case(
    skill="calibrated-measurements",
    task=TASK,                        # the prompt, incl. where results land
    persona=MICROSCOPIST,             # who holds the withheld fact
    build=Ellipsoids(),               # () -> Fixture
    layers=(Layer("nuclei", "image"),
            Layer("nuclei_labels", "labels", kind="labels")),
    collect={"volumes_um3": "volumes_um3", "spacing_um": "spacing_um"},
    score=verify,                     # (fixture, attempt) -> Outcome
    save_artifacts=save_artifacts,
    plugins=(),                       # kernel plugins the skill's requires: names
    persona_must_know=(...), persona_must_not_know=(...),
)
```

Modules are discovered by being there — no registration line, no engine change,
no test code. `test_benchmark.py` parametrizes over them.

The hermetic checks in `test_cases.py` run in CI over every case, so a case is
checked by arriving rather than by someone remembering to write a test for it:

- **The catalogue is covered** — every shipped skill is either benchmarked or in
  `NOT_BENCHMARKED` with a reason, and nothing names a skill that no longer
  ships. Otherwise "what does this layer cover" is answerable only by reading a
  directory.
- **The persona gives nothing away** — every fact reaches the rendered prompt,
  none leaks into the freely-shared `background`, the "never volunteer"
  instruction survives, and the persona knows what the fixture strips
  (`persona_must_know`) but not the skill's own vocabulary
  (`persona_must_not_know`). A chatty respondent rescues a bad agent and a green
  run looks identical, which is why this is asserted rather than reviewed.
- **The fixture keeps its truth out of its data**, provides every layer the case
  loads, and says where it came from.
- **The verifier refuses an empty attempt** — it must report metrics, all
  unscored, each with a reason. A verifier that passed a run that left nothing
  behind would make every arm look fine.
- **The task names what the harness scrapes**, since the collect names are a
  harness convention rather than something the skill asks for.

`test_conversation.py`, `test_report.py` and `test_fixture_protocol.py` cover the
loop, the report writer and the scoring protocol with no model and no session.

## 6. Ablation — an authoring tool, not a test

Give a model the task **without** the skill, closed-book, and diff against what
the body says. Cut what it gets right unaided; keep what it gets wrong. It
answers "is this content necessary", which is a question about the file rather
than about a run, so it is manual and per skill edit. The procedure is
`write-a-skill` step 6; three rules earned in practice:

- **Disclose the environment, withhold only the skill.** A run that withheld the
  third-party packages too had the models hand-roll everything, manufacturing
  evidence for one rule and destroying it for another. §5c's ablation arm follows
  the same rule mechanically.
- **Do not ask a model what is obvious.** It introspects badly. Test behaviour,
  not self-report.
- **Use a negative control** — a condition with an irrelevant skill injected. If
  "+skill" wins as much when the skill is nonsense, the measurement is picking up
  "more context → more effort", not content.

Cross-*family* coverage beats cross-*size*: blind spots correlate within a
family. Use a weak model to ask "is this necessary" and the strongest available
to ask "is this redundant or over-constraining".

## 7. Running it

```sh
# everything that gates, and every hermetic check (~1.5 s)
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills

# the resolver layer (§4a); CI runs this as its own step
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills -m satisfiability

# the import + signature layers (§4b, §4c), which need the skill's package
uv run --no-project --python .venv/bin/python --with pystackreg \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/skills/test_contracts.py

# the benchmark (§5): a GL display, two API keys, ~20 min per skill
xvfb-run -a -s '-screen 0 1024x768x24' \
  uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/skills/interaction -m interaction -s
```

`-s` is not optional in practice: pytest discards a *passing* test's captured
output, so without it the engine's per-arm progress lines never appear and the
terminal sits blank for the whole run. From a second terminal, the artifact
directory is the other progress view — every arm writes its transcript before
it is scored:

```sh
watch -n5 'find .skill-outcomes/interaction -newermt "-1 hour" | sort'
```

**Adding a skill.** Drop the `.md` in `mcp/_skills_data/` — the suite discovers
the directory and applies every rule. It will ask for: a `requires:` token inside
the vocabulary, a `[[link]]` that resolves, a one-sentence description, a
phrasing-table entry (§3), a contract test for any third-party package (§4c), and
either a benchmark case or a `NOT_BENCHMARKED` reason (§5e).
