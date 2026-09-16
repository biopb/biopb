"use client";

import type { SourceJobStatus } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";

/**
 * Modal progress for an in-flight resolve, and the one place a failed resolve
 * is reported.
 *
 * Modal because resolving is the blocking, consenting step: it downloads the
 * whole source and nothing can be opened until it lands, so there is no useful
 * work to do behind it. Warming is the opposite -- background, concurrent,
 * several at once -- and lives in the tray at the foot of the catalog.
 *
 * Shown for `running` and `error` only. A cancel is the user's own doing and
 * closes quietly; a completed resolve closes too, because the result is the
 * source becoming openable in the tree, which says it better than an alert.
 */

function shortId(sourceId: string): string {
  const parts = sourceId.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? sourceId;
}

export function ResolveModalView({
  job,
  onCancel,
  onDismiss,
}: {
  job: SourceJobStatus;
  onCancel: (sourceId: string) => void;
  onDismiss: (sourceId: string) => void;
}) {
  const failed = job.state === "error";
  const name = job.progress.target_name || shortId(job.source_id);

  return (
    <div className="admin-modal-backdrop">
      <div className="admin-modal" role="dialog" aria-modal="true">
        <h2>{failed ? "Could not resolve" : "Resolving source"}</h2>
        <p>
          {failed ? (
            <>
              <strong>{shortId(job.source_id)}</strong> was not resolved.{" "}
              {job.error}
            </>
          ) : (
            <>
              Downloading <strong>{name}</strong>. Cloud sources are fetched in
              full before they can be opened, which can take minutes. You can
              stop and pick it up later — whatever has already arrived is kept.
            </>
          )}
        </p>
        {!failed && (
          <div
            className="warm-bar indeterminate"
            role="progressbar"
            // No aria-valuenow at all: the server reports elapsed time and the
            // target's size, never how much of it has landed, so any percentage
            // here would be invented.
            aria-label={`Resolving ${name}`}
          >
            <div className="warm-bar-fill" style={{ width: "100%" }} />
          </div>
        )}
        <div className="admin-modal-actions">
          {failed ? (
            <button
              className="submit-btn"
              onClick={() => onDismiss(job.source_id)}
            >
              Close
            </button>
          ) : (
            <button
              className="submit-btn"
              onClick={() => onCancel(job.source_id)}
              disabled={job.cancel_requested}
            >
              {job.cancel_requested ? "Stopping…" : "Stop"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function ResolveModal() {
  const sourceJobs = useAppStore((s) => s.sourceJobs);
  const cancelSourceJob = useAppStore((s) => s.cancelSourceJob);
  const dismissSourceJob = useAppStore((s) => s.dismissSourceJob);

  // First match, not all: resolves are started one at a time from the tree, and
  // stacking modals would be unusable even if two somehow overlapped.
  const job = Object.values(sourceJobs).find(
    (j) => j.kind === "resolve" && (j.state === "running" || j.state === "error"),
  );
  if (!job) return null;

  return (
    <ResolveModalView
      job={job}
      onCancel={(id) => void cancelSourceJob("resolve", id)}
      onDismiss={(id) => dismissSourceJob("resolve", id)}
    />
  );
}
