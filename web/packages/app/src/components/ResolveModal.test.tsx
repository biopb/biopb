import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { SourceJobStatus } from "@biopb/tensor-flight-client";
import { ResolveModalView } from "./ResolveModal";

const job = (over: Partial<SourceJobStatus> = {}): SourceJobStatus => ({
  kind: "resolve",
  source_id: "cloud/timelapse.zarr",
  state: "running",
  progress: { elapsed_seconds: 3, target_name: "timelapse.zarr", target_bytes: 4096 },
  error: null,
  elapsed_seconds: 3,
  cancel_requested: false,
  ...over,
});

const render = (j: SourceJobStatus) =>
  renderToStaticMarkup(
    <ResolveModalView job={j} onCancel={() => {}} onDismiss={() => {}} />,
  );

describe("ResolveModalView", () => {
  it("says what is being downloaded and that it can be stopped", () => {
    const html = render(job());
    expect(html).toContain("timelapse.zarr");
    expect(html).toContain("Stop");
  });

  it("claims no percentage, because the server reports none", () => {
    // Resolve heartbeats carry elapsed time and the target's size -- never how
    // much of it has landed. Any number here would be invented.
    const html = render(job());
    expect(html).not.toContain("aria-valuenow");
    expect(html).toContain("indeterminate");
  });

  it("reports a failure prominently, with the reason", () => {
    const html = render(job({ state: "error", error: "host unreachable" }));
    expect(html).toContain("Could not resolve");
    expect(html).toContain("host unreachable");
    // No Stop on a job that already stopped.
    expect(html).toContain("Close");
  });

  it("disables stop once the cancel is in flight", () => {
    expect(render(job({ cancel_requested: true }))).toContain("Stopping…");
  });
});
