import type {
  RoiAnnotation,
  RoiListResult,
  RoiSetInfo,
} from "@biopb/tensor-flight-client";
import { isReservedSetName, splitLabelArrayId } from "@biopb/tensor-flight-client";
import type { RoiDraft } from "../utils/roiDraft";
import { roiSetCounts } from "../utils/roiSets";
import { sliceKey } from "./slice";
import { type RoiScopeState, type TensorView, selectView } from "./tensorView";
import type { AppState } from "./types";

/**
 * The label set to draw over the tensor in view, or null.
 *
 * Scoped by the id itself rather than by a per-tensor record: a set's
 * `array_id` names its image, so the question "is this overlay about what is on
 * screen?" is answered by the two ids and nothing has to be kept in step.
 *
 * Compared against the *stable* address on both sides, because a pinned link
 * (`id@token`) is the same tensor as the id the tree offered sets for.
 */
export function selectLabelOverlay(s: AppState): string | null {
  const shown = s.target.key;
  if (!shown || !s.labelOverlay) return null;
  const address = splitLabelArrayId(s.labelOverlay);
  if (!address) return null;
  return address.imageArrayId === shown ? s.labelOverlay : null;
}

// --- ROI annotations -------------------------------------------------------
//
// Every one of these hides state belonging to another tensor rather than
// relying on someone having reset it: an async writer can land after the tensor
// changed, and a reset cannot catch that.
//
// Stable empty constants: a selector returning a fresh [] on every call would
// re-render its subscriber on every unrelated store write.
/**
 * Prefix for a row drawn before the server has stored it.
 *
 * Namespaced with a character the wire cannot produce, so it can never collide
 * with a server id: the delete route splits ids on "," and the Flight action
 * rejects one containing it, which is the same reasoning that lets ids be
 * joined into a query string.
 */
const PROVISIONAL_PREFIX = "pending,";
let provisionalSeq = 0;

export function provisionalRoiId(): string {
  return `${PROVISIONAL_PREFIX}${++provisionalSeq}`;
}

/** A row that exists only in this client, because its write is still in flight. */
export function isProvisionalRoiId(roiId: string): boolean {
  return roiId.startsWith(PROVISIONAL_PREFIX);
}

/**
 * What the server substitutes for an empty set name before storing.
 *
 * Repeated here so a provisional row lands in the set row its stored form will,
 * rather than appearing under a blank name for the length of a round trip.
 */
export const DEFAULT_ROI_SET = "default";

/** The scope of the unqualified listing: every client-owned set at once. */
export const CLIENT_OWNED_SCOPE = "";

export function selectRois(s: AppState): RoiAnnotation[] {
  return selectView(s).rois;
}

/** Every set on the tensor in view, server-owned ones included. */
export function selectRoiSets(s: AppState): RoiSetInfo[] {
  return selectView(s).roiSets;
}

/** The scopes landed for the tensor in view, with what each fetch said. */
export function selectRoiScopes(s: AppState): Record<string, RoiScopeState> {
  return selectView(s).roiScopes;
}

/** The scopes in flight for the tensor in view -- not one looked at earlier. */
export function selectRoisPending(s: AppState): string[] {
  return selectView(s).roisPending;
}

export function selectRoisError(s: AppState): string | null {
  return selectView(s).roisError;
}

/**
 * The sets this tensor shows by default, spelled out: every client-owned one
 * the listing knows or the rows hold. What a toggle edits when nothing has
 * been chosen yet.
 */
export function defaultVisibleSets(s: AppState): string[] {
  const names = [
    ...selectRoiSets(s).filter((set) => !set.reserved).map((set) => set.setName),
    ...roiSetCounts(selectRois(s)).map((set) => set.setName),
  ];
  return [...new Set(names)].filter((name) => !isReservedSetName(name));
}

/** `roiSets` with one set's stored count moved by `delta`, added if it is new. */
export function withSetCount(sets: RoiSetInfo[], setName: string, delta: number): RoiSetInfo[] {
  if (!sets.some((set) => set.setName === setName)) {
    return delta > 0
      ? [...sets, { setName, count: delta, reserved: isReservedSetName(setName) }]
      : sets;
  }
  return sets.map((set) =>
    set.setName === setName ? { ...set, count: Math.max(0, set.count + delta) } : set,
  );
}

/** The rows a scope's fetch answers for. */
function inScope(roi: RoiAnnotation, scope: string): boolean {
  return scope === CLIENT_OWNED_SCOPE ? !isReservedSetName(roi.setName) : roi.setName === scope;
}

/**
 * Fold a landed fetch into the store.
 *
 * The first landing for a tensor replaces everything held for the last one --
 * the tensor is the unit of eviction -- and a later one merges: its rows
 * replace the rows of its own scope, and a provisional row is kept wherever it
 * is, since its write is still in flight.
 *
 * Every response enumerates the whole tensor's sets with their stored counts,
 * so it also says whether the rows held for *other* scopes are still current.
 * A scope whose counts disagree is un-landed, and the next `loadRois` fetches
 * it again -- which is how a reserved set the server rebuilt on a
 * re-registration, or a set another client wrote to, catches up without a
 * reload. A clipped scope is exempt: it disagrees by construction.
 *
 * Counts are a proxy: a per-set version in `RoiSetInfo` would make this an
 * exact comparison and let the writes stop shadowing the counts.
 */
export function landRoiScope(
  v: TensorView,
  scope: string,
  result: RoiListResult,
): Partial<TensorView> {
  const kept = v.rois.filter((roi) => !inScope(roi, scope) || isProvisionalRoiId(roi.roiId));
  const rois = [...kept, ...result.rois];
  const scopes: Record<string, RoiScopeState> = { ...v.roiScopes };
  scopes[scope] = { truncated: result.truncated, skipped: result.skipped };

  const stored = new Map(result.sets.map((set) => [set.setName, set.count]));
  const held = new Map(
    roiSetCounts(rois.filter((roi) => !isProvisionalRoiId(roi.roiId))).map((set) => [
      set.setName,
      set.count,
    ]),
  );
  const total = (counts: Map<string, number>, names: string[]) =>
    names.reduce((sum, name) => sum + (counts.get(name) ?? 0), 0);
  for (const [other, state] of Object.entries(scopes)) {
    if (other === scope || state.truncated) continue;
    const names =
      other === CLIENT_OWNED_SCOPE
        ? [...new Set([...stored.keys(), ...held.keys()])].filter((name) => !isReservedSetName(name))
        : [other];
    if (total(stored, names) !== total(held, names) + state.skipped) delete scopes[other];
  }

  return {
    rois,
    roiSets: result.sets,
    roiScopes: scopes,
    roisPending: v.roisPending.filter((p) => p !== scope),
  };
}

/**
 * A link's own set names while its target is not ready, or the resolved ones.
 *
 * For the URL writer. The scoped selectors answer null both for "the tensor's
 * default" and for "no tensor resolved yet", and the writer reads null as
 * "drop the param" -- so a link whose `tile_info` is slow or failed would have
 * its `rs=` rewritten away, and a reload could no longer retry it. Until the
 * target is ready, a linked open still says what the link named (`[]` being an
 * explicit "none", null "no value").
 */
export function selectUrlVisibleSets(s: AppState): string[] | null {
  const { target } = s;
  return target.linked && target.status !== "ready" ? target.seedSets : selectVisibleSets(s);
}

/** A link's own label overlay while its target is not ready. See {@link selectUrlVisibleSets}. */
export function selectUrlLabelOverlay(s: AppState): string | null {
  const { target } = s;
  return target.linked && target.status !== "ready" ? s.labelOverlay : selectLabelOverlay(s);
}

/**
 * The sets chosen for the tensor in view, or null for its default. Names do not
 * carry across tensors.
 */
export function selectVisibleSets(s: AppState): string[] | null {
  return selectView(s).visibleSets;
}

/**
 * The shape being placed, if it belongs here.
 *
 * Scoped to the tensor AND to 2-D: the volume viewer has no drawing surface, and
 * a half-placed polygon reappearing after a trip through 3-D is only confusing.
 * Both conditions answered at the read, so no writer has to remember either.
 */
export function selectDraft(s: AppState): RoiDraft | null {
  // `showRois` off means the annotation surface is off, not just that stored
  // shapes are hidden -- otherwise a draft keeps drawing over an overlay the
  // user switched off, and finishing writes something they cannot see. To thin
  // clutter while drawing, hide the noisy set instead; that is what the per-set
  // toggles are for.
  if (!s.showRois) return null;
  if (s.render3d) return null;
  const draft = selectView(s).draft;
  // Vertices were traced against the pixels of one plane. Navigating away --
  // the slider, a keyboard scroll, or play stepping an axis -- makes them a
  // shape drawn on an image nobody is looking at any more, and finishing there
  // would pin it to the plane it was NOT drawn on. Answered at the read, so
  // every route that moves the slice is covered without naming any of them.
  return draft && draft.sliceKey === sliceKey(s.position) ? draft.shape : null;
}

/**
 * The selected annotation, resolved against the set actually in view.
 *
 * Resolving rather than storing it means a selection cannot outlive the
 * annotation: deleting it, switching tensor, or a refetch that no longer
 * contains it all leave this null with nothing to clean up.
 */
export function selectSelectedRoi(s: AppState): RoiAnnotation | null {
  if (!s.selectedRoiId) return null;
  return selectRois(s).find((roi) => roi.roiId === s.selectedRoiId) ?? null;
}

/**
 * The selected annotation's id, or null when the selection is not in view.
 *
 * A primitive, so a subscriber re-renders on a change of selection rather than
 * on every change to the set it lives in.
 */
export function selectSelectedRoiId(s: AppState): string | null {
  return selectSelectedRoi(s)?.roiId ?? null;
}

/**
 * Axes a new annotation will broadcast across, given this tensor's defaults.
 *
 * `defaults` is computed from the grid by the caller (channel, normally -- see
 * `defaultBroadcastAxes`), because the store does not read `TileInfo`.
 */
export function selectBroadcastAxes(s: AppState, defaults: number[]): number[] {
  return selectView(s).broadcastAxes ?? defaults;
}
