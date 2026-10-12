import { beforeEach, describe, expect, it, vi } from "vitest";

// Neither sessionStorage nor a base path exists in the node environment; the
// point is what the probe does with them.
let token: string | null = null;
vi.mock("../auth", () => ({
  getToken: () => token,
  authHeaders: () => (token ? { Authorization: "Bearer " + token } : {}),
  redirectToUnlock: vi.fn(),
}));

import { redirectToUnlock } from "../auth";
import { fetchCanExec } from "./useCanExec";

function answering(status: number, body: unknown = {}) {
  const spy = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(
    async () => new Response(JSON.stringify(body), { status }),
  );
  vi.stubGlobal("fetch", spy);
  return spy;
}

beforeEach(() => {
  token = "tkn";
  vi.unstubAllGlobals();
  (redirectToUnlock as unknown as ReturnType<typeof vi.fn>).mockClear();
});

describe("fetchCanExec", () => {
  it("asks with the token and reads the scope", async () => {
    const spy = answering(200, { scope: "view" });
    expect(await fetchCanExec()).toBe(false);
    expect(
      (spy.mock.calls[0]![1]!.headers as Record<string, string>).Authorization,
    ).toBe("Bearer tkn");
    answering(200, { scope: "exec" });
    expect(await fetchCanExec()).toBe(true);
  });

  it("does not ask at all without a token", async () => {
    token = null;
    const spy = answering(200, { scope: "view" });
    expect(await fetchCanExec()).toBe(true);
    expect(spy).not.toHaveBeenCalled();
  });

  it("leaves the controls alone, and the user on the page, when refused", async () => {
    // A 401 here must not become a redirect to /unlock: this probe only greys out
    // buttons.
    answering(401);
    expect(await fetchCanExec()).toBe(true);
    expect(redirectToUnlock).not.toHaveBeenCalled();
  });

  it("leaves the controls alone when the read fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
    expect(await fetchCanExec()).toBe(true);
  });
});
