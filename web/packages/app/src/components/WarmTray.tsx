"use client";

import type { SourceJobStatus } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";

/**
 * Hydrate-ahead progress, pinned to the foot of the catalog pane.
 *
 * A tray rather than a bar drawn into each source row: warms are started
 * automatically after every resolve, so several run at once, and rows scroll.
 * Progress you have to hunt for is progress you cannot cancel, and cancelling
 * is the whole point of showing it -- a warm is a bulk recall the user may want
 * to stop once they see its size.
 *
 * Resolve deliberately does not appear here. There is one at a time and it
 * blocks opening the source, so it owns a modal instead.
 */

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

/** Short label for a source, which here is only ever an id. */
function shortId(sourceId: string): string {
  const parts = sourceId.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? sourceId;
}

export function WarmTrayRow({
  job,
  onCancel,
  onDismiss,
}: {
  job: SourceJobStatus;
  onCancel: (sourceId: string) => void;
  onDismiss: (sourceId: string) => void;
}) {
  const { files_total = 0, files_done = 0, bytes_total = 0, bytes_done = 0 } =
    job.progress;
  // Percent off bytes, not files: file sizes vary by orders of magnitude inside
  // one zarr, so a file count jumps and stalls while bytes move steadily.
  const pct = bytes_total > 0 ? Math.min(100, (bytes_done / bytes_total) * 100) : 0;
  const running = job.state === "running";
  const determinate = running && bytes_total > 0;

  let detail: string;
  if (job.state === "error") {
    detail = job.error ?? "Failed";
  } else if (job.state === "cancelled") {
    detail = `Stopped after ${files_done} file${files_done === 1 ? "" : "s"}`;
  } else if (job.state === "done") {
    detail = `${files_done} file${files_done === 1 ? "" : "s"} ready`;
  } else if (files_total > 0) {
    detail = `${files_done}/${files_total} · ${formatBytes(bytes_done)} of ${formatBytes(bytes_total)}`;
  } else {
    detail = "Counting files…";
  }

  return (
    <li className={`warm-row ${job.state}`}>
      <div className="warm-row-head">
        <span className="warm-row-name" title={job.source_id}>
          {shortId(job.source_id)}
        </span>
        {running ? (
          <button
            className="warm-row-action"
            onClick={() => onCancel(job.source_id)}
            disabled={job.cancel_requested}
            title="Stop recalling this source's files"
          >
            {job.cancel_requested ? "Stopping…" : "Cancel"}
          </button>
        ) : (
          <button
            className="warm-row-action"
            onClick={() => onDismiss(job.source_id)}
            title="Dismiss"
            aria-label={`Dismiss ${shortId(job.source_id)}`}
          >
            ✕
          </button>
        )}
      </div>
      <div
        className={`warm-bar ${determinate ? "" : "indeterminate"}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        // Omitted while enumerating: a bar that reads 0% for a minute looks
        // stuck, where an absent value is announced as "in progress".
        aria-valuenow={determinate ? Math.round(pct) : undefined}
        aria-label={`Warming ${shortId(job.source_id)}`}
      >
        <div className="warm-bar-fill" style={{ width: `${determinate ? pct : 100}%` }} />
      </div>
      <div className="warm-row-detail">{detail}</div>
    </li>
  );
}

export function WarmTray() {
  const sourceJobs = useAppStore((s) => s.sourceJobs);
  const cancelSourceJob = useAppStore((s) => s.cancelSourceJob);
  const dismissSourceJob = useAppStore((s) => s.dismissSourceJob);

  const warms = Object.values(sourceJobs).filter(
    (j) =>
      j.kind === "warm" &&
      // A source with nothing to warm finishes instantly with files_total 0.
      // Showing a completed bar for work that never existed is noise, and it
      // is the single-file case -- i.e. most sources.
      !(j.state === "done" && (j.progress.files_total ?? 0) === 0),
  );
  if (warms.length === 0) return null;

  return (
    <div className="warm-tray">
      <div className="warm-tray-title">
        Hydrating {warms.length === 1 ? "1 source" : `${warms.length} sources`}
      </div>
      <ul className="warm-tray-list">
        {warms.map((job) => (
          <WarmTrayRow
            key={job.source_id}
            job={job}
            onCancel={(id) => void cancelSourceJob("warm", id)}
            onDismiss={(id) => dismissSourceJob("warm", id)}
          />
        ))}
      </ul>
    </div>
  );
}
