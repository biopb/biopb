# The skills authoring gate

The deterministic layers of [`docs/skill-testing.md`](../../../../../docs/skill-testing.md)
— no agent, no session, no display — run against the skills this package ships
(`biopb_mcp/mcp/_skills_data/*.md`).

```sh
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills
```

Everything runs except the `contract` layer, which skips (see below).

## What is covered

| File | Layer | Asks |
|---|---|---|
| `test_schema.py` | Structure | Does the frontmatter contract behave — coercion, semver, kebab-case, the resulting shape? |
| `test_validate.py` | Structure | Which malformations are fatal versus tolerable, and is a valid tree read correctly? |
| `test_shipped_skills.py` | Structure+ | Do the *real* skill files obey the rules the validator does not express generically? |
| `test_retrieval.py` | Retrieval | Do the real descriptions answer the phrasings a user would type? |
| `test_satisfiability.py` | Contract | Can a skill's declared packages be installed here at all? |
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

No shipped skill declares a third-party package today, so the parametrized half
is empty and `test_the_extractor_finds_pkg_tokens` is what keeps the layer from
going vacuously green.

> A **contract** layer used to live here too (`test_contracts.py`), asserting
> that the third-party APIs a body quotes still look that way — `importorskip`ed,
> armed on a workstation. It was written entirely for `flatfield-and-stitch-tiles`
> and went with it. Bring it back when a skill declares a package that passes the
> gate above; the shape is in git and in `docs/skill-testing.md` §3.
