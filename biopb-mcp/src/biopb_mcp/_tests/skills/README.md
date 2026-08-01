# The skills authoring gate

The deterministic layers of [`docs/skill-testing.md`](../../../../../docs/skill-testing.md)
— no agent, no session, no display — run against the skills this package ships
(`biopb_mcp/mcp/_skills_data/*.md`).

```sh
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/skills
```

Two layers are held back from that default run, each by a marker and for the
same reason — they need something the shared env does not have. `satisfiability`
needs a real resolver run (below); `outcome` needs each skill's own package, so
it runs in the per-package envs `skill-contracts.yaml` builds
(`outcomes/README.md`).

## What is covered

| File | Layer | Asks |
|---|---|---|
| `test_schema.py` | Structure | Does the frontmatter contract behave — coercion, semver, kebab-case, the resulting shape? |
| `test_validate.py` | Structure | Which malformations are fatal versus tolerable, and is a valid tree read correctly? |
| `test_shipped_skills.py` | Structure+ | Do the *real* skill files obey the rules the validator does not express generically? |
| `test_retrieval.py` | Retrieval | Do the real descriptions answer the phrasings a user would type? |
| `test_satisfiability.py` | Contract | Can a skill's declared packages be installed here at all? |
| `test_contracts.py` | Contract | Does the third-party API a body quotes still look like that? |
| `outcomes/` | Outcome | Does following a skill's procedure produce the right numbers? |
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

`test_the_extractor_finds_pkg_tokens` keeps the layer from going vacuously green
if the shipped catalog ever stops declaring a third-party package again.

## The `outcome` marker

Fixtures with a known answer, and a verifier that scores a run against them —
`docs/skill-testing.md` §5. Deselected by default, and unlike `satisfiability`
it is not a matter of cost: the subjects import the skill's package, and that
package is deliberately not in this environment. One shared resolution would
force every skill's dependency to co-exist with every other's.

```sh
pytest biopb-mcp/src/biopb_mcp/_tests/skills -m outcome   # needs the package
```

See `outcomes/README.md`. Two things there are worth knowing from here: real
data can be substituted for a synthetic fixture without touching a verifier
(`BIOPB_SKILL_FIXTURES`), and every case is deliberately also run through the
mistake its skill body warns about, because a verifier nothing has ever failed
is not known to work.
