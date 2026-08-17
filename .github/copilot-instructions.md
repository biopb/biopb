# Copilot Instructions for biopb PR Review

## Scope

These instructions define how GitHub Copilot agent reviews pull requests in this repository.

Primary goal: catch correctness, compatibility, and performance regressions early.

## Review Mode

When asked to review a PR, default to a code-review mindset.

- Prioritize findings over summaries.
- Report issues ordered by severity: Critical, High, Medium, Low.
- Include concrete file and line references.
- Focus on behavior changes, regressions, and missing tests.
- If no defects are found, explicitly say so and list residual risks/testing gaps.

## Architecture-Aware Checks (biopb)

Review with the repo architecture in mind.

### 1. Data plane vs compute plane boundaries

- `biopb-tensor-server/` is the data plane (Arrow Flight, discovery, metadata, lazy chunked tensor access).
- `biopb-image-runtime/` is the compute-plane base for gRPC algorithm servers.
- `proto/` is the protocol source of truth.

Flag changes that blur these responsibilities without strong justification.

### 2. Lazy/eager contract correctness

For compute plane changes, verify:

- large outputs are handled via lazy references when appropriate;
- eager responses are still correct for small payloads;
- source IDs, tensor references, and metadata propagation remain valid;
- no accidental full-materialization of large arrays in client/control paths.

### 3. Protocol compatibility and polyglot impact

For `proto/` edits, check for backward-compatibility risks:

- field renumbering or removals;
- behavior changes not reflected in comments/docs/tests;
- implications for Python, Java, and JS/TS generated clients.

Flag breaking changes unless they are explicitly versioned and documented.

### 4. Performance and scalability regressions

Pay special attention to:

- changes that increase data transfer size or read amplification;
- loss of chunked/lazy behavior;
- repeated serialization/deserialization of large tensors;
- unnecessary copies or conversions in hot paths.

### 5. Reliability and isolation

For runtime/server changes, check:

- failure handling and error propagation;
- concurrency safety/thread safety;
- process/kernel isolation assumptions (where relevant);
- cleanup of cache/temp resources.

### 6. Security posture (local/remote two-mode model)

Deployment has two knobs. `--base-port` (default 8810) places all three
listeners: control base+3, sidecar base+4, flight base+5 — the container's
convention. `--grpc-bind` (default `127.0.0.1`) decides exposure: loopback is
tokenless by default, a public address *requires* a token and defaults TLS **on**
(the bind drives TLS, never the reverse). `--remote` is a deprecated alias for
`--grpc-bind 0.0.0.0`.

The bind address *is* the mode, read once through
`biopb._web_auth.host_is_public_bind` — shared by the core CLI, the control's
bind guard, and the tensor `launch` — so "public + unauthenticated" is
unrepresentable and the three cannot drift. There is no dev-mode token bypass.
Only the flight plane is ever published: the control (browser UI) and the sidecar
stay on loopback with either bind, because the control is plaintext HTTP with no
TLS support; a remote browser tunnels in with `ssh -L 8813:localhost:8813 <host>`
(biopb/biopb#614).

Still flag:

- any change that could bind the flight server or control UI publicly without a
  required token, or that weakens the local/remote fail-closed split;
- insecure defaults introduced in deployment docs/scripts;
- leaks of credentials/tokens in logs.

## Testing Expectations

Require tests that match the change type.

- Bug fix: regression test required.
- Protocol or wire behavior change: integration/contract test required.
- Performance-sensitive path: at least a guardrail test or benchmark note.

If tests are missing, call this out as a finding (usually Medium unless risk is high).

## What to Read During Review

Prefer reading:

- changed files in the PR;
- nearby tests;
- related protocol definitions in `proto/`;
- relevant docs (`README.md`, `CLAUDE.md`, subproject READMEs) when behavior contracts are involved.

Do not nitpick style unless it hides a correctness/maintainability issue.

## Output Format for PR Reviews

Use this structure:

1. Findings
2. Open Questions / Assumptions
3. Brief Change Summary

Each finding should include:

- severity;
- concise description of the risk/bug;
- evidence (file/line reference);
- recommended fix.

## Repository Conventions for Agent Edits

- Keep edits minimal and focused on the reported issue.
- Do not revert unrelated local changes.
- Preserve existing public APIs unless change is intentional and documented.
- Prefer targeted tests over broad refactors during review-driven fixes.
