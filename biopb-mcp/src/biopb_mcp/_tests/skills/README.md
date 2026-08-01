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
| `test_contracts.py` | Contract | Do the third-party APIs the bodies quote still look like that? |

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

## The `contract` marker

`test_contracts.py` asserts the surface of packages a *skill* depends on
(`basicpy`, `m2stitch`) — not packages this one depends on. They are
`importorskip`ed:

```sh
pytest biopb-mcp/src/biopb_mcp/_tests/skills              # they skip
pytest biopb-mcp/src/biopb_mcp/_tests/skills -m contract  # just this layer, once armed
```

To arm it, python **3.11** — not 3.12 — because m2stitch pins pandas 1.5.3 and
there is no 3.12 wheel for it:

```sh
uv venv .venv-contract --python 3.11
uv pip install --python .venv-contract/bin/python pytest pyyaml basicpy m2stitch
```

Budget ~5 GB: `basicpy` is torch-backed and pulls CUDA. That cost is why CI does
not arm this layer — pulling a solver in to read a function signature is a bad
trade, and the layer's job is to fail on a workstation before an edit ships.

It asserts against **code fences only**, never the prose. The failure-modes
tables quote the same call signatures in English, so a whole-body match would
pass even after the call itself lost the argument.

Note what this layer does *not* answer: whether those packages can be installed
alongside biopb at the version the body was written against. That is
`test_satisfiability.py`, which needs no install and does run in CI.
