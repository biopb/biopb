import { describe, it, expect, afterEach, vi } from "vitest";

// base.ts reads window.__BIOPB_BASE__ once, at module load — the value is fixed
// for the life of the page — so each case installs a window and re-imports.
//
// It also reads location.pathname, because the prefix only applies when the
// document was served under it. `pathname` defaults to a path *inside* whatever
// prefix is declared (the portal case); pass it explicitly to model a document
// served somewhere else, as the unprefixed-root cases below do.
async function loadWith(value: unknown, pathname?: string) {
  vi.resetModules();
  const declared =
    typeof value === "string"
      ? "/" + value.trim().replace(/^\/+/, "").replace(/\/+$/, "")
      : "/";
  (globalThis as { window?: unknown }).window = {
    __BIOPB_BASE__: value,
    location: { pathname: pathname ?? declared },
  };
  return await import("./base.js");
}

afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});

describe("BASE", () => {
  it("is empty at the origin root", async () => {
    for (const raw of [undefined, "", "/", "   ", null, 42]) {
      const { BASE } = await loadWith(raw);
      expect(BASE, String(raw)).toBe("");
    }
  });

  it("normalizes what the control injects", async () => {
    for (const [raw, want] of [
      ["/node/mantis-051/29847", "/node/mantis-051/29847"],
      ["/node/h/29847/", "/node/h/29847"], // trailing slash dropped
      ["node/h/29847", "/node/h/29847"], // leading slash supplied
      ["  /biopb  ", "/biopb"],
      ["/apps/biopb_v2", "/apps/biopb_v2"],
    ] as const) {
      const { BASE } = await loadWith(raw);
      expect(BASE, raw).toBe(want);
    }
  });

  it("degrades to no prefix rather than send the app off-origin", async () => {
    // The control validates before injecting, so these should never arrive --
    // but a base that is not a plain same-origin path would turn every link,
    // fetch and asset in the app into an off-origin request, so it is re-checked
    // here. "" (serve at the root) is the safe reading of a value we distrust.
    for (const hostile of [
      "//evil.com",
      "https://evil.com/x",
      "/\\evil.com", // the URL parser reads the backslash as an authority
      "/a\t\\evil.com",
      "/a b",
      "/a?next=x",
      "/a#x",
      "/a%2fb",
      '/a"onerror=x',
      "/a<script>",
    ]) {
      const { BASE } = await loadWith(hostile);
      expect(BASE, hostile).toBe("");
    }
  });

  it("ignores a prefix the document was not served under", async () => {
    // A prefixed control serves its one rewritten shell to *every* request, so a
    // direct http://127.0.0.1:8813/ gets __BIOPB_BASE__ too. Honouring it there
    // would set the router basename to /node/h/29847 while location.pathname is
    // "/", and react-router renders null — a blank page, not a broken-looking
    // one. That root is what `biopb ui` opens and what the ssh -L hint targets.
    for (const here of ["/", "/viewer", "/node/h/29847x/admin", "/node/h"]) {
      const { BASE } = await loadWith("/node/h/29847", here);
      expect(BASE, here).toBe("");
    }
  });

  it("applies the prefix on the paths actually served under it", async () => {
    for (const here of [
      "/node/h/29847",
      "/node/h/29847/",
      "/node/h/29847/viewer",
      "/node/h/29847/session/s1/observe",
    ]) {
      const { BASE } = await loadWith("/node/h/29847", here);
      expect(BASE, here).toBe("/node/h/29847");
    }
  });

  it("is inert at the unprefixed root, not merely un-routed", async () => {
    // The whole module degrades together: a half-applied prefix would keep
    // sending fetches and links into /node/h/29847 from a page at the root.
    const { BASE, withBase, appPath } = await loadWith("/node/h/29847", "/");
    expect(BASE).toBe("");
    expect(withBase("/api/status")).toBe("/api/status");
    expect(appPath("/admin")).toBe("/admin");
  });
});

describe("withBase", () => {
  it("is the identity at the origin root", async () => {
    const { withBase } = await loadWith("");
    expect(withBase("/api/status")).toBe("/api/status");
    expect(withBase("/biopb-logo.png")).toBe("/biopb-logo.png");
  });

  it("places root-relative paths under the prefix", async () => {
    const { withBase } = await loadWith("/node/h/29847");
    expect(withBase("/api/status")).toBe("/node/h/29847/api/status");
    expect(withBase("/data_plane")).toBe("/node/h/29847/data_plane");
    expect(withBase("/session/s1")).toBe("/node/h/29847/session/s1");
    expect(withBase("/biopb-logo.png")).toBe("/node/h/29847/biopb-logo.png");
  });
});

describe("appPath", () => {
  it("is the identity at the origin root", async () => {
    const { appPath } = await loadWith("");
    expect(appPath("/admin")).toBe("/admin");
    expect(appPath("/")).toBe("/");
  });

  it("strips the prefix so router paths are not doubled", async () => {
    // The bug this exists to prevent: handing a raw pathname to navigate()
    // under a basename would land at /node/h/29847/node/h/29847/admin.
    const { appPath } = await loadWith("/node/h/29847");
    expect(appPath("/node/h/29847/admin")).toBe("/admin");
    expect(appPath("/node/h/29847/session/s1/observe")).toBe(
      "/session/s1/observe",
    );
    expect(appPath("/node/h/29847")).toBe("/"); // the app root itself
    expect(appPath("/node/h/29847/")).toBe("/");
  });

  it("leaves a path that only looks prefixed alone", async () => {
    const { appPath } = await loadWith("/node/h/29847");
    expect(appPath("/node/h/29847x/admin")).toBe("/node/h/29847x/admin");
    expect(appPath("/admin")).toBe("/admin");
  });

  it("round-trips with withBase", async () => {
    const { withBase, appPath } = await loadWith("/node/h/29847");
    for (const p of ["/admin", "/viewer", "/session/s1/observe", "/"]) {
      expect(appPath(withBase(p))).toBe(p);
    }
  });
});
