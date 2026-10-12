import { afterEach, describe, expect, it, vi } from "vitest";
import { captureUrlToken, getToken } from "./auth";

// A tiny browser: the URL, sessionStorage and history are all this reads.
function browser(href: string) {
  const store = new Map<string, string>();
  const replaceState = vi.fn();
  vi.stubGlobal("sessionStorage", {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
  });
  vi.stubGlobal("window", { location: { href }, history: { replaceState } });
  return { replaceState };
}

afterEach(() => vi.unstubAllGlobals());

describe("captureUrlToken", () => {
  it("stores the token from the access URL and strips it from the address bar", () => {
    const { replaceState } = browser("https://h/node/n/1/?token=abc123&x=1");
    expect(captureUrlToken()).toBe(true);
    expect(getToken()).toBe("abc123");
    expect(replaceState).toHaveBeenCalledWith(null, "", "/node/n/1/?x=1");
  });

  it("is idempotent: a second call finds nothing and keeps the token", () => {
    browser("https://h/?token=abc123");
    captureUrlToken();
    vi.stubGlobal("window", {
      location: { href: "https://h/" },
      history: { replaceState: vi.fn() },
    });
    expect(captureUrlToken()).toBe(false);
    expect(getToken()).toBe("abc123");
  });
});
