import { beforeEach, describe, expect, it, vi } from "vitest";

// `authHeaders` reads sessionStorage and `redirectToUnlock` navigates, neither
// of which exists in the node test environment — and the point of the test is
// what this module does with them, not what they do.
vi.mock("../auth", () => ({
  authHeaders: (extra?: Record<string, string>) => ({
    ...(extra || {}),
    Authorization: "Bearer tkn",
  }),
  redirectToUnlock: vi.fn(),
}));

import { redirectToUnlock } from "../auth";
import { SessionLocked, sessionFetch } from "./sessionFetch";

const unlock = redirectToUnlock as unknown as ReturnType<typeof vi.fn>;

type FetchSpy = ReturnType<typeof answering>;

function answering(status: number) {
  const spy = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(
    async () => new Response("{}", { status }),
  );
  vi.stubGlobal("fetch", spy);
  return spy;
}

const sentInit = (spy: FetchSpy): RequestInit => spy.mock.calls[0]![1] ?? {};

const sentHeaders = (spy: FetchSpy) =>
  sentInit(spy).headers as Record<string, string>;

beforeEach(() => {
  unlock.mockClear();
  vi.unstubAllGlobals();
});

describe("sessionFetch", () => {
  it("carries the token", () => {
    // The whole bug: the observe page called the control bare, so every one of
    // its calls 401'd whenever a token was configured — `--token` on a loopback
    // control included.
    const spy = answering(200);
    return sessionFetch("/session/s1/api/jobs").then(() => {
      expect(sentHeaders(spy).Authorization).toBe("Bearer tkn");
    });
  });

  it("keeps the caller's own headers", async () => {
    // The console POST needs its JSON content-type: it is what a cross-site
    // form cannot set, and the child requires it on that route for that reason.
    const spy = answering(200);
    await sessionFetch("/session/s1/console/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const headers = sentHeaders(spy);
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer tkn");
    expect(sentInit(spy).method).toBe("POST");
  });

  it("treats a 401 as a locked session, not as an answer", async () => {
    // A 401 body is valid JSON, so a caller reading through it saw `jobs`
    // undefined and rendered "no jobs yet" — a healthy session looking idle.
    // It has to raise rather than return.
    answering(401);
    await expect(sessionFetch("/session/s1/api/jobs")).rejects.toBeInstanceOf(
      SessionLocked,
    );
    expect(unlock).toHaveBeenCalledOnce();
  });

  it("leaves other failures to the caller", async () => {
    // A busy kernel answers 409 and a wedged session 502. Neither is a locked
    // session, and redirecting on them would throw the user off a working page.
    answering(409);
    const r = await sessionFetch("/session/s1/console/execute", {
      method: "POST",
    });
    expect(r.status).toBe(409);
    expect(unlock).not.toHaveBeenCalled();
  });
});
