import { describe, expect, it, vi } from "vitest";
import type { AddSourceJobStatus, AddSourceResult } from "@biopb/tensor-flight-client";
import { addCaution, addNotice, describeAdd, removeNotice, runAdd } from "./addSource";

const status = (
  state: AddSourceJobStatus["state"],
  extra: Partial<AddSourceJobStatus> = {},
): AddSourceJobStatus => ({
  kind: "add",
  source_id: "/data",
  state,
  progress: {},
  error: null,
  elapsed_seconds: 0,
  cancel_requested: false,
  ...extra,
});

const result = (over: Partial<AddSourceResult> = {}): AddSourceResult => ({
  added: 0,
  already_present: 0,
  refreshed: 0,
  removed: 0,
  skipped_offline: 0,
  skipped_cloud_dirs: 0,
  failed: [],
  failed_total: 0,
  ...over,
});

describe("runAdd", () => {
  it("polls a running add until it ends, reporting each read", async () => {
    const api = {
      addSource: vi.fn().mockResolvedValue(status("running")),
      addSourceStatus: vi
        .fn()
        .mockResolvedValueOnce(status("running", { progress: { added_count: 3 } }))
        .mockResolvedValueOnce(status("done", { result: result({ added: 5 }) })),
    };
    const seen: string[] = [];
    const wait = vi.fn().mockResolvedValue(undefined);
    const last = await runAdd(api, "/data", (s) => seen.push(s.state), { wait });
    expect(last.state).toBe("done");
    expect(seen).toEqual(["running", "running", "done"]);
    expect(wait).toHaveBeenCalledTimes(2);
    expect(api.addSourceStatus).toHaveBeenCalledWith("/data");
  });

  it("does not poll an add that has already ended", async () => {
    const api = {
      addSource: vi.fn().mockResolvedValue(status("error", { error: "path not found" })),
      addSourceStatus: vi.fn(),
    };
    const last = await runAdd(api, "/nope", () => {}, { wait: vi.fn() });
    expect(last.error).toBe("path not found");
    expect(api.addSourceStatus).not.toHaveBeenCalled();
  });

  it("stops following when told to, leaving the add running", async () => {
    let stop = false;
    const api = {
      addSource: vi.fn().mockResolvedValue(status("running")),
      addSourceStatus: vi.fn().mockResolvedValue(status("running")),
    };
    const wait = vi.fn(async () => {
      stop = true;
    });
    const last = await runAdd(api, "/data", () => {}, { wait, stopped: () => stop });
    expect(last.state).toBe("running");
    expect(api.addSourceStatus).toHaveBeenCalledTimes(1);
  });

  it("polls by the name the server gave the job, not the path as typed", async () => {
    const api = {
      addSource: vi.fn().mockResolvedValue(status("running", { source_id: "/data" })),
      addSourceStatus: vi.fn().mockResolvedValue(status("done", { source_id: "/data" })),
    };
    await runAdd(api, "/data/", () => {}, { wait: vi.fn().mockResolvedValue(undefined) });
    expect(api.addSource).toHaveBeenCalledWith("/data/");
    expect(api.addSourceStatus).toHaveBeenCalledWith("/data");
  });
});

describe("describeAdd", () => {
  it("says what was added, and what was not new", () => {
    expect(describeAdd(result({ added: 12, already_present: 3 }))).toBe(
      "Added 12 sources, 3 already served",
    );
    expect(describeAdd(result({ added: 1 }))).toBe("Added 1 source");
  });

  it("counts a rebuilt source apart from one merely already served", () => {
    expect(describeAdd(result({ already_present: 4, refreshed: 1 }))).toBe(
      "1 rebuilt, 3 already served",
    );
  });

  it("reports failures and an empty result", () => {
    expect(describeAdd(result({ added: 2, failed_total: 1 }))).toBe("Added 2 sources, 1 failed");
    expect(describeAdd(result())).toBe("Nothing found to add");
  });
});

describe("addNotice", () => {
  it("is nothing while the add still runs", () => {
    expect(addNotice(status("running"))).toBeNull();
  });

  it("tells a finished add in one info line", () => {
    const n = addNotice(status("done", { result: result({ added: 3, already_present: 1 }) }));
    expect(n).toEqual({ kind: "info", text: "Added 3 sources, 1 already served", details: [] });
  });

  it("says when it was stopped, and keeps what it registered", () => {
    const n = addNotice(status("cancelled", { result: result({ added: 2 }) }));
    expect(n?.text).toBe("Stopped. Added 2 sources");
  });

  it("lists the first failed paths, as info while something worked", () => {
    const failed = ["a", "b", "c", "d"].map((p) => ({ path: `/x/${p}`, reason: "unreadable" }));
    const n = addNotice(status("done", { result: result({ added: 1, failed, failed_total: 4 }) }));
    expect(n?.kind).toBe("info");
    expect(n?.details).toEqual(["/x/a: unreadable", "/x/b: unreadable", "/x/c: unreadable"]);
  });

  it("is an error when the add failed, or came to nothing but failures", () => {
    const failed = [{ path: "/x/a", reason: "unreadable" }];
    expect(addNotice(status("error", { error: "path not found" }))).toEqual({
      kind: "error",
      text: "Could not add /data: path not found",
      details: [],
    });
    expect(addNotice(status("done", { result: result({ failed, failed_total: 1 }) }))?.kind).toBe(
      "error",
    );
  });
});

describe("removeNotice", () => {
  it("confirms a clean removal with the count", () => {
    expect(removeNotice("scratch", { removed: 3, failed: [] })).toEqual({
      kind: "info",
      text: "Removed scratch (3 sources)",
      details: [],
    });
  });

  it("is an error naming the failures when any remain", () => {
    const n = removeNotice("scratch", {
      removed: 2,
      failed: [{ path: "zarr_a", reason: "not present (already removed?)" }],
    });
    expect(n.kind).toBe("error");
    expect(n.text).toBe("Removed 2 of scratch; 1 failed");
    expect(n.details).toEqual(["zarr_a: not present (already removed?)"]);
  });
});

describe("addCaution", () => {
  it("warns only when cloud files were skipped", () => {
    expect(addCaution(result())).toBeNull();
    expect(addCaution(result({ skipped_offline: 2 }))).toMatch(/cloud/);
    expect(addCaution(result({ skipped_cloud_dirs: 1 }))).toMatch(/cloud/);
  });
});
