import { splitArrayVersion } from "@biopb/tensor-flight-client";

/** True when *a* and *b* name the same tensor, ignoring a token either one
 * may carry (a pinned link, or the server's own auto-attached content
 * version -- see `isResolutionUpgrade`'s and `currentArrayId`'s comments). */
export function sameStableAddress(a: string, b: string): boolean {
  return splitArrayVersion(a).arrayId === splitArrayVersion(b).arrayId;
}

/**
 * True when *next* is *prev* resolved to a specific field by the server --
 * the same tensor, more precisely spelled, addressed by `prev/field` (a
 * token on either address, from a pinned link, does not change this: both
 * are stripped before comparing). False for a genuine switch, including one
 * field to another under the same already-resolved source, since *prev*
 * itself names a field then and nothing can be "resolved" further.
 */
export function isResolutionUpgrade(prev: string, next: string): boolean {
  const { arrayId: prevStable } = splitArrayVersion(prev);
  const { arrayId: nextStable } = splitArrayVersion(next);
  return !prevStable.includes("/") && nextStable.startsWith(`${prevStable}/`);
}

/** The two values `ViewerPane` carries across renders in its remount key. */
export interface ViewerKeyState {
  /** What the remount key is actually built from. */
  keyedTensorId: string;
  /** The raw tensor id prop as of the last render, whatever it was keyed on. */
  lastTensorId: string;
}

/**
 * *state* advanced to *tensorId*.
 *
 * `lastTensorId` always becomes *tensorId*, so a later prop is compared
 * against what actually arrived, not against `keyedTensorId` -- which a
 * resolution leaves behind on the bare source_id. Comparing a *second*
 * change against that stale bare id would read it as *another* resolution of
 * an already-bare source, when it is really a switch to a different field
 * with no bare hop in front of it (e.g. one tensor row clicked right after
 * another, both under the same source). `keyedTensorId` only follows when
 * the change is not that resolution, which is what keeps the pane mounted
 * across it and remounted across everything else.
 */
export function advanceViewerKey(state: ViewerKeyState, tensorId: string): ViewerKeyState {
  if (tensorId === state.lastTensorId) return state;
  return {
    keyedTensorId: isResolutionUpgrade(state.lastTensorId, tensorId)
      ? state.keyedTensorId
      : tensorId,
    lastTensorId: tensorId,
  };
}
