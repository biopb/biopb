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

Deployment is two fail-closed modes: **local** (every listener binds loopback, no
token — the default) and **remote** (`biopb control start --remote`: public bind
behind a *required* token). The bind address *is* the mode, so "public +
unauthenticated" is unrepresentable; there is no dev-mode token bypass (a `None`
token is local mode).

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
