import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { SourceJobStatus } from "@biopb/tensor-flight-client";
import { WarmTrayRow } from "./WarmTray";

// The view takes props, so it server-renders; `WarmTray` itself reads the store
// and cannot (zustand's server snapshot is getInitialState()) -- the same split
// as TreeRow/SourceTree.

const job = (over: Partial<SourceJobStatus> = {}): SourceJobStatus => ({
  kind: "warm",
  source_id: "cloud/plate1.zarr",
  state: "running",
  progress: { files_total: 10, files_done: 4, bytes_total: 2048, bytes_done: 512 },
  error: null,
  elapsed_seconds: 2,
  cancel_requested: false,
  ...over,
});

const render = (j: SourceJobStatus) =>
  renderToStaticMarkup(
    <WarmTrayRow job={j} onCancel={() => {}} onDismiss={() => {}} />,
  );

describe("WarmTrayRow", () => {
  it("measures progress in bytes, not files", () => {
    // File sizes inside one zarr vary by orders of magnitude, so a file count
    // jumps and stalls where bytes move steadily. 512/2048 = 25%, not 4/10.
    expect(render(job())).toContain("width:25%");
  });

  it("names the source and the counts", () => {
    const html = render(job());
    expect(html).toContain("plate1.zarr");
    expect(html).toContain("4/10");
  });

  it("offers cancel while running and says so once asked", () => {
    expect(render(job())).toContain("Cancel");
    // cancel_requested, not state: the flag flips immediately, the state only
    // once the server-side worker unwinds.
    expect(render(job({ cancel_requested: true }))).toContain("Stopping…");
  });

  it("swaps cancel for dismiss once it has settled", () => {
    const html = render(job({ state: "done" }));
    expect(html).not.toContain("Cancel");
    expect(html).toContain("Dismiss");
  });

  it("reports a failure inline rather than blocking", () => {
    // Warming is background work -- a failure downgrades reads to lazy recall,
    // it does not stop anything, so it must not take over the screen.
    const html = render(job({ state: "error", error: "recall timed out" }));
    expect(html).toContain("recall timed out");
    expect(html).toContain("warm-row error");
  });

  it("stays honest while still counting files", () => {
    const html = render(job({ progress: {} }));
    expect(html).toContain("Counting files…");
    // No aria-valuenow: a bar reading 0% for a minute looks stuck, where an
    // absent value is announced as "in progress".
    expect(html).not.toContain("aria-valuenow");
    expect(html).toContain("indeterminate");
  });

  it("says what a cancel actually kept", () => {
    expect(render(job({ state: "cancelled" }))).toContain("Stopped after 4 files");
  });
});
