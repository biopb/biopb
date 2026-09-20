# The knowledge-store gate

The deterministic layers of the store described in
[`biopb-mcp/docs/knowledge.md`](../../../../docs/knowledge.md) — no agent, no
session, no display — run against the docs this package ships
(`biopb_mcp/mcp/_docs_data/*.md`).

```sh
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/docs
```

Two layers are held back from that default run, each by a marker, because both
need real resolver runs (below). A third, `bench`, is not in this directory at
all: the interaction layer lives in [`_tests/bench/`](../bench/), since one
engine serves both a doc's delta and an unaided task. It needs a display, two
API keys and about twenty minutes, and — unlike everything here — is **not a
gate**.

There is no structure layer any more, and that is the redesign's point. A doc is
a markdown file with optional frontmatter, and `mcp/_docs.py`'s tolerant reader
is the only reader; a schema, a strict validator and a shared layout module were
guarding fifteen small files against malformations that now simply degrade. The
store's own unit tests are [`../test_docs.py`](../test_docs.py).

## What is covered

| File | Asks |
|---|---|
| `test_seed.py` | Do the seed index and the shipped set agree, do `[[links]]` land, is every doc within the caps? |
| `test_packaging.py` | Does every doc reach the wheel? |
| `test_satisfiability.py` | Would installing a declared package move something already in this environment? |
| `test_availability.py` | Can a declared package be installed on every interpreter and platform we ship to? |
| `test_contracts.py` | Does the third-party API a body quotes still look like that? |
| `../bench/` | A model in front of the shipped body, against a real session, scored on numbers *(a benchmark, not a gate)* |

## Running the held-back layers

```sh
# the damage gate; CI runs it as its own step, on every matrix cell
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/docs -m satisfiability

# the availability grid: 9 cells per declared package, ~1 s each
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/docs -m availability

# the import + signature layers, which need the doc's own package. CI gives
# each package an env of its own (.github/scripts/doc_contracts.py), because
# one shared resolution would force every doc's package to co-exist with every
# other's and the first pair that cannot would break the suite rather than the
# doc.
uv run --no-project --python .venv/bin/python --with pystackreg \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/docs/test_contracts.py

# the benchmark, and the other half of a doc's delta. One invocation is one
# configuration, so a delta is two of these.
uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/bench -m bench -s
uv run --no-project --python .venv/bin/python --with openai --with anthropic \
  python -m pytest biopb-mcp/src/biopb_mcp/_tests/bench -m bench -s \
  --bench-docs=false
```

**Stochastic gates get muted within two weeks of the first flake, and then you
have neither the gate nor the trust.** That is why the benchmark reports rather
than fails.

## Adding a doc to the shipped seed

Drop the `.md` in `mcp/_docs_data/` and add its line to `index.md` — the suite
discovers the directory and applies every rule. It will ask for a `kind:`, an
index entry (or a place on the `ignored:` line), `[[links]]` that resolve, a
body inside the cap, and a contract test for any third-party package.

**Banking one** is prefixing its filename with `_`. It ships and `read_doc`
returns it by id, so one index line promotes it; it is simply never listed, and
the package gates leave it alone. The index's `ignored:` line is the other half
of that idea and belongs to the agent, not to the release.

What the suite *cannot* ask for is the part that happens before the file exists:
if the body states a method, measure it first and cite the measurement in the
PR. Nothing goes red when that is skipped, which is why it is written here and
why a reviewer should ask for it.
