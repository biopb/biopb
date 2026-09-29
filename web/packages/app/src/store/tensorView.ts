import type { RoiAnnotation, RoiSetInfo } from "@biopb/tensor-flight-client";
import type { RoiDraft } from "../utils/roiDraft";
import type { AppState } from "./types";

/** What one landed fetch scope said about itself. */
export interface RoiScopeState {
  /** The server's per-tensor cap clipped this fetch: it holds more than was returned. */
  truncated: boolean;
  /** Rows whose geometry this client could not read -- see `decodeRoiListResult`. */
  skipped: number;
}

/**
 * What belongs to one tensor, held under its `ViewTarget.key`.
 */
export interface TensorView {
  /**
   * The rows held, across every scope that has landed. Filtered to the plane
   * and to the visible sets at render time.
   */
  rois: RoiAnnotation[];
  /**
   * Every set on the tensor, server-owned ones included, with its stored row
   * count. The discovery half of the listing: `rois` holds a server-owned set
   * only once it was asked for by name.
   */
  roiSets: RoiSetInfo[];
  /**
   * The fetch scopes that have landed: `""` for the unqualified listing (the
   * client-owned sets, all at once), a set name for a server-owned set fetched
   * on its own. Presence is what makes `loadRois` idempotent, and that is
   * load-bearing rather than an optimisation: switching to the 3-D viewer and
   * back remounts the whole 2-D subtree (`ViewerPane` keys on the render mode),
   * so without it a round trip through 3-D would refetch a set that can reach
   * megabytes.
   *
   * A scope is dropped again when a later listing's counts disagree with the
   * rows held for it -- the server rebuilds a reserved set on every
   * registration -- so the next `loadRois` fetches it afresh. Opening the
   * tensor drops them all, keeping the rows on screen while they are re-listed:
   * someone else may have annotated it in the meantime.
   */
  roiScopes: Record<string, RoiScopeState>;
  /** Scopes in flight. */
  roisPending: string[];
  roisError: string | null;
  /**
   * The sets on screen, by name, or `null` for this tensor's default: the
   * client-owned sets shown, the server-owned ones not.
   *
   * A positive list rather than a hidden one because the default is not "all":
   * a link has to be able to say *show* `@ome`, and a hidden list could only
   * say the opposite. The default is materialised on the first toggle, as
   * `broadcastAxes` is. A server-owned set on the list is what `loadRois`
   * fetches by name -- visibility is what drives the lazy fetch.
   */
  visibleSets: string[] | null;
  /**
   * Axes a new annotation should NOT pin, i.e. broadcast across. `null` means
   * "the default for this tensor" (see `selectBroadcastAxes`), which is not the
   * same as "none" -- an empty array is a deliberate choice to pin everything.
   */
  broadcastAxes: number[] | null;
  /** The shape being placed, and the slice position it is being drawn at. */
  draft: { shape: RoiDraft; sliceKey: string } | null;
  /**
   * The widest window the data has shown so far, per channel.
   *
   * A float dtype names no range of its own, so the track a fixed window is
   * chosen on is the levels that have actually appeared. Taking that from the
   * plane in view alone would shrink the track on a dim plane -- exactly when a
   * fixed window needs to reach the levels of a bright one. Per channel because
   * two channels of one tensor need not share a scale.
   */
  observedLimits: Record<number, [number, number]>;
  /** Logical clock of the last write, for eviction. */
  used: number;
}

/** What a tensor with no record yet reads as. Stable, so selectors do not churn. */
export const EMPTY_VIEW: TensorView = {
  rois: [],
  roiSets: [],
  roiScopes: {},
  roisPending: [],
  roisError: null,
  visibleSets: null,
  broadcastAxes: null,
  draft: null,
  observedLimits: {},
  used: 0,
};

/** Tensors whose record is kept. The one in view is never evicted. */
const VIEW_LIMIT = 8;
let viewClock = 0;

/**
 * A patch to `views` that updates `key`'s record, or nothing when there is no
 * tensor to write to. Evicts the least recently written record past the limit.
 */
export function withView(
  s: AppState,
  key: string | null,
  patch: Partial<TensorView> | ((v: TensorView) => Partial<TensorView>),
): Partial<AppState> {
  if (!key) return {};
  const current = s.views[key] ?? EMPTY_VIEW;
  const views = {
    ...s.views,
    [key]: { ...current, ...(typeof patch === "function" ? patch(current) : patch), used: ++viewClock },
  };
  const keys = Object.keys(views);
  if (keys.length > VIEW_LIMIT) {
    const victim = keys
      .filter((k) => k !== key && k !== s.target.key)
      .sort((a, b) => (views[a]?.used ?? 0) - (views[b]?.used ?? 0))[0];
    if (victim !== undefined) delete views[victim];
  }
  return { views };
}

/**
 * The record of the tensor in view, or an empty one while none has resolved.
 *
 * Every per-tensor read goes through here, so nothing compares an id: a tensor
 * that is not in view simply is not the one this returns.
 */
export function selectView(s: AppState): TensorView {
  return (s.target.key ? s.views[s.target.key] : undefined) ?? EMPTY_VIEW;
}
