import { describe, it, expect, afterEach, vi } from "vitest";

/**
 * `auth.ts` reaches `base.ts`, which reads `window.__BIOPB_BASE__` and
 * `location.pathname` once at module load -- so each case installs a window and
 * re-imports, the same way `base.test.ts` does.
 */
async function loadWith(href: string, base = "/") {
  vi.resetModules();
  const url = new URL(href);
  const assign = vi.fn();
  (globalThis as { window?: unknown }).window = {
    __BIOPB_BASE__: base,
    location: { href, pathname: url.pathname, search: url.search, assign },
  };
  const auth = await import("./auth.js");
  return { auth, assign };
}

/** The path `/unlock` was told to return to, decoded. */
function nextOf(target: string): string | null {
  const q = target.indexOf("?");
  if (q < 0) return null;
  return new URLSearchParams(target.slice(q + 1)).get("next");
}

afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});

describe("redirectToUnlock", () => {
  it("carries the query back, so a shared viewer link survives unlocking", async () => {
    // Without this the link arrives at /viewer stripped of the very state it
    // was sent to show.
    const href = "http://localhost:8813/viewer?id=nikon_0944c8d850bc/A3&t=5&g=1.5";
    const { auth, assign } = await loadWith(href);
    auth.redirectToUnlock();
    expect(nextOf(assign.mock.calls[0]![0] as string)).toBe(
      "/viewer?id=nikon_0944c8d850bc%2FA3&t=5&g=1.5",
    );
  });

  it("keeps the token out of the path it returns to", async () => {
    // `next` is handed to the router and lands in history; a token riding along
    // in it would outlive the one-shot hand-off ClientBootstrap does.
    const { auth, assign } = await loadWith("http://localhost:8813/viewer?token=s3cret&t=2");
    auth.redirectToUnlock();
    const target = assign.mock.calls[0]![0] as string;
    expect(target).not.toContain("s3cret");
    expect(nextOf(target)).toBe("/viewer?t=2");
  });

  it("sends a bare path with no query", async () => {
    const { auth, assign } = await loadWith("http://localhost:8813/admin");
    auth.redirectToUnlock();
    expect(nextOf(assign.mock.calls[0]![0] as string)).toBe("/admin");
  });

  it("does not ask /unlock to return to itself", async () => {
    const { auth, assign } = await loadWith("http://localhost:8813/unlock");
    auth.redirectToUnlock();
    expect(assign.mock.calls[0]![0]).toBe("/unlock");
  });

  it("strips the url prefix from next but keeps it on the assign target", async () => {
    const { auth, assign } = await loadWith("http://localhost:8813/node/h/p/viewer?t=1", "/node/h/p");
    auth.redirectToUnlock();
    const target = assign.mock.calls[0]![0] as string;
    expect(target.startsWith("/node/h/p/unlock")).toBe(true);
    expect(nextOf(target)).toBe("/viewer?t=1");
  });
});
