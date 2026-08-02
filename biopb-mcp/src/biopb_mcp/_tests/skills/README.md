# The skills authoring gate

The deterministic layers of [`biopb-mcp/docs/skill-testing.md`](../../../../docs/skill-testing.md)
— no agent, no session, no display — run against the skills this package ships
(`biopb_mcp/mcp/_skills_data/*.md`).

```sh
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills
```

Two layers are held back from that default run, each by a marker.
`satisfiability` needs a real resolver run (below). `interaction` needs a
display, two API keys and about twenty minutes, and — unlike everything else
here — is **not a gate at all**: it is a benchmark that reports what four
conversations did (`interaction/README.md`).

## What is covered

| File | Layer | Asks |
|---|---|---|
| `test_schema.py` | Structure | Does the frontmatter contract behave — coercion, semver, kebab-case, the resulting shape? |
| `test_validate.py` | Structure | Which malformations are fatal versus tolerable, and is a valid tree read correctly? |
| `test_shipped_skills.py` | Structure+ | Do the *real* skill files obey the rules the validator does not express generically? |
| `test_retrieval.py` | Retrieval | Do the real descriptions answer the phrasings a user would type? |
| `test_satisfiability.py` | Contract | Can a skill's declared packages be installed here at all? |
| `test_contracts.py` | Contract | Does the third-party API a body quotes still look like that? |
| `interaction/` | Interaction | A model in front of the shipped body, against a real session, scored on numbers *(§5; a benchmark, not a gate)* |
| `test_packaging.py` | — | Do the skills actually reach the wheel? |

`test_shipped_skills.py` is where the authoring rules live: the `requires:`
grammar the agent resolves at runtime, `[[wiki-links]]` landing on a skill that
exists, a declared `plugin:<stem>` actually being called through its module
name, and the guardrails from `write-a-skill` that are mechanically checkable.

**Two readers, on purpose.** `mcp/_skills.py` is tolerant — it is on the agent's
path, so a malformed file degrades to a skipped entry. `_validate.py` here is
strict — it is on the author's path, where the same file should stop the PR.
`test_what_validates_is_what_the_runtime_loads` pins them to the same answer
about which files are skills, so a file only the gate can read cannot pass
review and then be invisible.

## Adding a skill

Drop the `.md` in `mcp/_skills_data/`. Nothing to register: the suite discovers
the directory and applies every rule. Expect to be told about a `requires:`
token outside the vocabulary, a link to a skill that does not exist, a
description that runs to two sentences — and, from `test_retrieval.py`, that you
owe the new skill a phrasing entry.

You also owe it either a benchmark case (`interaction/cases/<skill>.py`) or a
line in `interaction/cases.NOT_BENCHMARKED` saying why it cannot have one;
`interaction/test_cases.py` fails until one of the two exists.

## The `satisfiability` marker

A skill may not declare a `pkg:` token that installs only by moving something
already in the environment. `uv pip install --dry-run basicpy` *succeeds* — and
takes numpy 2.3.5 down to 1.26.4, plus pandas and scipy, under a live kernel
that has already imported numpy 2.

```sh
pytest biopb-mcp/src/biopb_mcp/_tests/skills                     # deselected
pytest biopb-mcp/src/biopb_mcp/_tests/skills -m satisfiability   # just this layer
```

Deselected by default only because each token is a real resolver run; CI runs it
as its own step. Metadata only — nothing is downloaded or installed.

**CI runs it on every matrix cell** (ubuntu 3.10/3.11/3.12, macos 3.12, windows
3.12), because the answer is environment-dependent: a package can co-install on
one interpreter and not another. That is not hypothetical for this class of
dependency — m2stitch pins pandas 1.5.3 and has no 3.12 wheel. **One red cell
rejects the skill.** A shipped catalog goes to every platform, so a skill that
only resolves on Linux is one most users would get a silent downgrade from.

The gate is unconditional: no allowlist, no xfail. Those would be somewhere to
record that a known-bad skill ships anyway, which is what it exists to prevent.
A package that genuinely needs its own environment belongs behind the algorithm
plane as an `ops:<kind>` server, called rather than imported — the kernel's
interpreter is the agent's only execution surface, so "install it in a separate
venv" is not a resolution.

`test_the_extractor_finds_pkg_tokens` keeps the layer from going vacuously green
if the shipped catalog ever stops declaring a third-party package again.

## The `interaction` marker

A **real** biopb-mcp session — shim-spawned child, real kernel, real napari,
real dask, the nine real tools over real MCP — with the skill body arriving
through the real `_skills.py`, and a model in front of it talking to a simulated
user who holds a fact the fixture withheld. `biopb-mcp/docs/skill-testing.md` §5 and
`interaction/README.md`.

```sh
xvfb-run -a -s '-screen 0 1024x768x24' \
  uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills/interaction -m interaction
```

**It needs a GL-capable display**, and not merely offscreen Qt: napari builds
under `QT_QPA_PLATFORM=offscreen` and then `add_image` dies in vispy's GL probe.
Without one the tests skip with instructions. It also needs two API keys, and
costs four conversations per skill.

**It is a benchmark, not a gate, and no run's outcome fails a test.** Each of
the four arms — skill offered or withheld, respondent answering or silent —
reports an outcome and a reason. The report is the deliverable.

Nothing here is stood in for, deliberately — a hand-written tool surface would
put `execute_code`'s return shape and the `guide://` bodies back into a
transcription, which is the disease that got the layer below this one deleted.
The cost is that a red run's cause space includes the kernel, Qt and dask, so
`test_session_smoke.py` exists to fail separately when the stack rather than the
skill is at fault — and it runs without a model or a key.

Most of that directory is *not* marked and runs with the ordinary suite: the
conversation loop, the report writer, the fixture protocol, and — per skill —
its persona, its fixture and its verifier (`interaction/test_cases.py`). A case
whose persona volunteers the answer, or whose verifier passes a run that left
nothing behind, is a normal red test and never a surprise mid-run.

## There used to be an `outcome` marker

A layer that ran each skill's procedure against a ground-truth fixture and
checked the number, with the procedure written **by hand** from what the body
said. It was deleted, and the reasoning is worth keeping:

- **It tested nothing this repo ships.** The subject was a transcription that
  never read the file, so editing a step — or deleting the skill — left it
  green. That is the opposite of `test_contracts.py` next door, whose
  assertions come out of the shipped frontmatter.
- **It could not reach the instructions that matter.** Anything that needs a
  *choice* in order to be wrong — which channel is structural, whether to pass
  `spacing=` — is made correctly by construction in a reference implementation,
  so the fixture scored data no subject could fail.
- **It did not scale.** `drift-correction` cost ~640 lines against a 157-line
  skill, most of it reference implementations and tolerance measurement, and
  every skill would have owed the same.

What was worth keeping moved into `interaction/`: the fixture protocol, the
substitutable curated-data path (`BIOPB_SKILL_FIXTURES`), the tolerances, and
the verifiers themselves — which now score what a model actually left in the
kernel instead of what a transcription computed.
