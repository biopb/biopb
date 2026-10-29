import { describe, expect, it, vi } from "vitest";
import type { AddSourceJobStatus, AddSourceResult } from "@biopb/tensor-flight-client";
import { addCaution, describeAdd, runAdd } from "./addSource";

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
    const last = await runAdd(api, "/data", (s) => seen.push(s.state), wait);
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
    const last = await runAdd(api, "/nope", () => {}, vi.fn());
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
    const last = await runAdd(api, "/data", () => {}, wait, () => stop);
    expect(last.state).toBe("running");
    expect(api.addSourceStatus).toHaveBeenCalledTimes(1);
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

describe("addCaution", () => {
  it("warns only when cloud files were skipped", () => {
    expect(addCaution(result())).toBeNull();
    expect(addCaution(result({ skipped_offline: 2 }))).toMatch(/cloud/);
    expect(addCaution(result({ skipped_cloud_dirs: 1 }))).toMatch(/cloud/);
  });
});
