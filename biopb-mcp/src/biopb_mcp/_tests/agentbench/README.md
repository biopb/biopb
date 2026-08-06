# agentbench — the machinery, and nothing it is used for

Everything needed to put a model in front of a **real** biopb session and score
what comes out, with no opinion about what is being scored. Why it is a package
rather than a directory inside the suite that grew it is in `__init__.py`.

Nothing here is marked and nothing here spends money, so it all runs with the
ordinary suite:

```sh
uv run --no-sync pytest biopb-mcp/src/biopb_mcp/_tests/agentbench
```

A suite's *paid* run carries its own marker — `-m interaction` for
[`skills/interaction/`](../skills/interaction/), `-m tasks` for `tasks/` — and
it is the suite that decides when to spend, not the machinery.

| File | Holds |
|---|---|
| `_session.py` | Bring-up: a real shim-spawned session, a synchronous façade over the async MCP client, and the environment facts that are forced rather than inherited |
| `_bridge.py` | MCP tool schemas → the function-calling shape a chat model expects |
| `_models.py` | The provider table: which model on each side, at which address, with which key |
| `_agent.py` | `ChatAgent`; `ScriptedAgent`, `ReplayAgent`, and the live `ToolCallingAgent` |
| `_respondent.py` | `Persona`, `Respondent`; `ScriptedRespondent`, `SilentRespondent`, and the live `ModelRespondent` |
| `_conversation.py` | The two-model loop, the caps, the `Trace` |
| `_fixture.py` | What a run is given and what it recovers: `Fixture`, `Attempt`, `Metric`, `Outcome`, the fixture specs (`Procedural`/`OnDisk`) and the refs they hand out, artifact writing |
| `_plane.py` | The run-scoped tensor server a `tensor`-presented case needs. Conditional: nothing starts unless a case asks |
| `test_conversation.py` | That the loop works, with no model *and* no session |
| `test_fixture_protocol.py` | The scoring protocol itself, including the on-disk path almost no machine has data for |
| `test_fixture_tree.py` | `-m fixtures`: hashes a curated tree against its manifest. Out-of-band, never inside a run |
| `test_models.py` | That provider selection resolves, and that §5a holds of the defaults |
| `test_layout.py` | That nobody reaches these modules as if they were siblings — the import that merges clean and then stops a whole package collecting |
| `test_plane.py` | That plane — hermetic checks on its config, plus live ones against a real server |

## Environment

Both sides are `provider:model`, named separately, so they can be different
vendors — or the same compatible API at two addresses:

| Variable | What it sets |
|---|---|
| `BIOPB_AGENT` / `BIOPB_RESPONDENT` | Which model plays each side |
| `BIOPB_AGENT_BASE_URL` / `BIOPB_RESPONDENT_BASE_URL` | Override a provider's address |
| `BIOPB_ENV_FILE` | Read the above from a file. Falls back to `.env` at the repo root, then `~/.config/biopb/harness.env` |
| `BIOPB_FIXTURES` | Root of the curated fixture tree. Unset means this machine has none, and `OnDisk` cases skip |
| `BIOPB_OUTCOME_DIR` | Where artifacts land, overriding each suite's default |
| `BIOPB_GUARD_LOG` / `BIOPB_GUARD_MARKERS` | The agent filesystem guardrail — see `docs/agent-fs-guardrail.md` |

These were `BIOPB_SKILL_*` while the machinery lived under `skills/`. They were
renamed with the move and **no aliases were kept**: a name that lies about its
scope is worse than one that breaks loudly once.
