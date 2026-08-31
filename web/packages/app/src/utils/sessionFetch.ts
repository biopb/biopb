// One way to call a session's API through the control front.
//
// The observe page was ported from the buildless page each MCP session child
// used to serve on its own loopback port, where there was no token to send. The
// token requirement arrived when it moved behind the control
// (`_ControlAuthMiddleware._guarded` covers `/session/<id>/api/*` and every
// other proxied root) and the page never caught up: every call went out bare
// and 401'd whenever a token was configured — including on a *loopback* control,
// since `--token` is optional there rather than absent (biopb#468, biopb#730).
//
// It failed silently, which is why it survived. A 401 body is valid JSON and
// each caller read straight through it: `data.jobs` undefined rendered as "no
// jobs yet", `s.alive` undefined rendered a healthy kernel as `dead · starting`,
// and interrupt/restart/notebook did nothing at all. A token-mode observe page
// looked like an idle session with a dead kernel.
//
// So the token and the 401 are handled here rather than at six call sites: the
// page cannot acquire a seventh that forgets.

import { authHeaders, redirectToUnlock } from "../auth";

/** Raised when the control refuses a call for want of a usable token.
 *
 * A distinct type because it is not a failure of the thing being asked for —
 * the caller is already navigating away, and there is nothing to report or
 * retry. Callers on this page all swallow it into their existing "keep the last
 * good render" path. */
export class SessionLocked extends Error {
  constructor() {
    super("Session locked — re-enter the access token.");
    this.name = "SessionLocked";
  }
}

/** What a session-API reply says about the *session*, as opposed to the kernel
 * inside it.
 *
 * The control answers 404 for a session id it cannot resolve, and prunes the
 * record as it does, so that answer never comes back: the child exited. A 502
 * is the opposite — a session still registered but not answering right now.
 * Worth separating because the page's response differs in kind: a blip is
 * waited out, an ended session means stop, and say so.
 *
 * Every non-ok reply used to be read straight through instead. A 404 body is
 * valid JSON, so `data.jobs` came back undefined and rendered as "no jobs yet"
 * and `s.alive` undefined rendered as `dead · starting` — an ended session
 * looked exactly like a fresh idle one, the same way a token-gated one did
 * before `sessionFetch` caught the 401. */
export function sessionVerdict(
  status: number,
): "live" | "ended" | "unreachable" {
  if (status === 404) return "ended";
  if (!(status >= 200 && status < 300)) return "unreachable";
  return "live";
}

/**
 * `fetch`, carrying the stored token and treating a 401 as a locked session.
 *
 * The redirect matches what every other page already does with a 401
 * (`AdminPage`, `LogsPage`, `McpAdminPage`). It also settles a gate the observe
 * page could not otherwise ask about: the console and the chat pane decide
 * *whether to render at all* from two advertised halves — the control is
 * loopback-bound, and the child serves the root — and a token gate is a third
 * the deployment has but neither half reports. Both halves say yes under
 * `--token`, so an editor appeared and every submit 401'd. Leaving the page
 * closes that gap without the gate having to grow a third term.
 */
export async function sessionFetch(
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  const r = await fetch(url, {
    ...init,
    headers: authHeaders(init.headers as Record<string, string> | undefined),
  });
  if (r.status === 401) {
    redirectToUnlock();
    throw new SessionLocked();
  }
  return r;
}
