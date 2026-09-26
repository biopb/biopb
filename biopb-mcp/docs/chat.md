# Chat: the built-in loop

`mcp/_chat.py` is the chat pane's agent, and the only one. It exists for a user
with no MCP harness of their own: a session an agent is already driving does not
get a pane at all (`_chat_api.configure`'s *agentless* term), because two agents
on one kernel means one of them holds the claim and the other answers questions
and then refuses to run anything.

**Chat is off whenever the control runs `--remote`.** Its turns are reachable
only while the control is loopback-bound, on a separate proxy root (`chat`),
never on a public bind.

## How it works

**Calls biopb's own tools as Python functions**, from the same FastMCP tool
registry an external MCP client uses -- not over the wire -- so it needs no
MCP client SDK.

**Does not reuse `execute_code`'s `promote_after` window.** That wait saves
an external client a round trip; in-process a poll is a function call, so
the loop submits with no promote window and streams a job's partial stdout
as it accumulates instead of blocking on it.

**Runs in the session child process, not the napari kernel.** The kernel is
a grandchild the loop must survive restarting.

**No third-party agent framework is vendored** -- the loop calls tools
in-process and needs no provider SDK.

**A screenshot is refused rather than silently sent to a non-multimodal
model** -- capability is checked before an image reaches the provider, with
automatic recovery for a model that can't take it.

## Why there is no second engine

There was one: `mcp/_chat_acp.py` handed the pane to a coding harness the user
already ran, over the [Agent Client Protocol](https://agentclientprotocol.com/),
so it drove this viewer on the user's own subscription. It was retired for two
reasons, neither of which has since changed:

**Almost nothing could be on the other end of it.** Only opencode shipped a
native `acp` subcommand *and* honoured the `mcpServers` array in `session/new`,
which is what made the harness attach to the session already in front of the
user rather than spawning a second viewer. Claude Code needed a third-party
adapter, Codex CLI offered only the inverse (Codex as an MCP server), Cursor's
`cursor-agent acp` ignored `mcpServers` outright, and Claude Desktop has no
headless agent at all. One supported harness is not an ecosystem.

**It cost a seam everywhere.** Two engines meant a runtime switch
(`POST /chat/engine`), a readiness probe per engine, two thread shapes (an
append cursor for the loop, a revision watermark for items that update in
place), two history payloads, two command namespaces, a permission-question
item type the loop can never produce, and a `chat.acp_*` block in the config.
Every one of those was a branch in a file that otherwise has one path.

If a second engine is ever worth having, the shape to bring back is the
adapter -- `chatThread.ts` still translates the wire thread into what the pane
renders, which is the part that was worth keeping -- not the runtime switch.

## What a harness does instead

Register biopb as an MCP server in the harness's own config and drive the
session that way. That is a different question from the one the ACP engine
answered (`biopb._agents.status()` checks exactly this), it is what the
installer already sets up, and it works for every client, not just the one.
