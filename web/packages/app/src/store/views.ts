import type { StateCreator } from "zustand";
import type { RoiAnnotation, RoiGeometry } from "@biopb/tensor-flight-client";
import { TensorApiError, isReservedSetName } from "@biopb/tensor-flight-client";
import { DEFAULT_POLYLINE_WIDTH, clampPolylineWidth } from "../utils/roiDraft";
import type { RoiDraft, RoiTool } from "../utils/roiDraft";
import { DEFAULT_LABEL_OPACITY, clampLabelOpacity } from "../utils/vivUtils";
import { sliceKey } from "./slice";
import { EMPTY_VIEW, type TensorView, withView } from "./tensorView";
import {
  CLIENT_OWNED_SCOPE,
  DEFAULT_ROI_SET,
  defaultVisibleSets,
  isProvisionalRoiId,
  landRoiScope,
  provisionalRoiId,
  selectBroadcastAxes,
  selectVisibleSets,
  withSetCount,
} from "./roi";
import type { AppState } from "./types";

/** ROI annotations, the label overlay, and the per-tensor record they live in. */
export interface ViewsSlice {
  // --- per-tensor state ---------------------------------------------------
  /**
   * Everything that belongs to "the tensor in view", one record per tensor, keyed
   * by `target.key`. A small LRU: leaving a tensor keeps its record, so coming
   * back finds its annotation rows warm.
   *
   * An async writer captures the key when it starts and writes into that
   * record, so a landing for a tensor the user has left is harmless rather than
   * something to guard against. Read through the selectors below, which resolve
   * the tensor in view -- never `views` directly.
   */
  views: Record<string, TensorView>;
  /**
   * The server does not offer annotations at all (501: disabled, or no metadata
   * DB). Distinct from an error, because it is a fact about the deployment
   * rather than a failure -- the UI hides itself instead of reporting a fault.
   */
  roisUnavailable: boolean;
  /** Overlay on/off. A view preference, so it outlives a tensor change. */
  showRois: boolean;

  // --- authoring ----------------------------------------------------------
  /** Which tool the pointer carries. A preference: it outlives a tensor change. */
  tool: RoiTool;
  /** Selected annotation. Read through `selectSelectedRoi`, which validates it. */
  selectedRoiId: string | null;
  /** Label and set the next new annotation gets. Preferences, kept across tensors. */
  newLabel: string;
  newSetName: string;
  /**
   * Width a new polyline gets, in image pixels. A preference like the two
   * above, and geometry rather than styling -- it is the band of pixels the
   * stroke claims, so it is stored on the annotation and scales with the image.
   */
  newPolylineWidth: number;
  /** A write that failed or lost a conditional put, for the panel to report. */
  roiWriteError: string | null;

  // --- label overlay ------------------------------------------------------
  /**
   * The label set drawn over the image, as its whole `array_id`, or null.
   *
   * One at a time. Two categorical overlays stacked would be two palettes
   * fighting for the same pixel, with no blend of them that says which object
   * is which -- so picking a second set replaces the first.
   *
   * Scoped like the annotation state rather than reset on a source change: the
   * id names its own image, so `selectLabelOverlay` can tell whether the set
   * belongs to the tensor in view, and coming back to that tensor finds its
   * overlay still on.
   */
  labelOverlay: string | null;
  /**
   * The overlay's alpha, 0-1.
   *
   * Never 1 by default: the point of an overlay is the image under it, and a
   * fully opaque one is just a different image.
   */
  labelOpacity: number;
  /**
   * Fetch what the tensor in view should hold and does not yet: the client-owned
   * sets, and every server-owned set that is visible. Idempotent on what has
   * landed and what is in flight, so callers fire it freely -- on mount, and
   * whenever the visible sets change.
   */
  loadRois: () => Promise<void>;
  setShowRois: (value: boolean) => void;
  toggleSetVisible: (setName: string) => void;
  setTool: (tool: RoiTool) => void;
  setDraft: (draft: RoiDraft | null) => void;
  setSelectedRoi: (roiId: string | null) => void;
  setNewLabel: (label: string) => void;
  setNewSetName: (setName: string) => void;
  setNewPolylineWidth: (width: number) => void;
  toggleBroadcastAxis: (axis: number, defaults: number[]) => void;
  createRoi: (geometry: RoiGeometry, plane: Record<number, number>) => Promise<void>;
  deleteRoi: (roiId: string) => Promise<void>;
  clearRoiSet: (setName: string) => Promise<void>;
  /** Draw this label set over its image, or nothing. See `labelOverlay`. */
  setLabelOverlay: (arrayId: string | null) => void;
  setLabelOpacity: (value: number) => void;
}

export const createViewsSlice: StateCreator<AppState, [], [], ViewsSlice> = (set, get) => ({
  views: {},
  roisUnavailable: false,
  showRois: true,

  tool: "select",
  selectedRoiId: null,
  newLabel: "",
  newSetName: "",
  newPolylineWidth: DEFAULT_POLYLINE_WIDTH,
  roiWriteError: null,

  labelOverlay: null,
  labelOpacity: DEFAULT_LABEL_OPACITY,

  async loadRois() {
    const { client, target } = get();
    if (!client || target.status !== "ready" || !target.info || !target.key) return;
    if (get().roisUnavailable) return;
    // Held under the token-free key, fetched under the exact address, which
    // carries the server's current version token.
    const key = target.key;
    const arrayId = target.info.array_id;

    const { roiScopes: landed, roiSets, roisPending: pending } = get().views[key] ?? EMPTY_VIEW;
    // Any scope landing brings the whole listing of set names with it.
    const listed = Object.keys(landed).length > 0;
    // The unqualified listing always; a server-owned set only while it is on
    // screen, which is what makes a set that can reach megabytes lazy. Once a
    // listing has landed it says which names are server-owned, so a link
    // naming a set that is not there costs nothing; before that, the prefix
    // is the only guess there is.
    const wanted = [CLIENT_OWNED_SCOPE];
    for (const name of selectVisibleSets(get()) ?? []) {
      const reserved = listed
        ? roiSets.find((set) => set.setName === name)?.reserved
        : isReservedSetName(name);
      if (reserved) wanted.push(name);
    }
    // Idempotent: already held, or already asked for. Callers fire this from a
    // mount effect, and the 2-D subtree remounts on every render-mode flip.
    const scopes = wanted.filter((scope) => !(scope in landed) && !pending.includes(scope));
    if (scopes.length === 0) return;
    set((s) =>
      withView(s, key, (v) => ({ roisPending: [...v.roisPending, ...scopes], roisError: null })),
    );

    const fetchScope = async (scope: string) => {
      try {
        const result = await client.http.listRois(arrayId, scope || undefined);
        set((s) => withView(s, key, (v) => landRoiScope(v, scope, result)));
      } catch (err) {
        // 501 is the server saying it does not do annotations. Latched for the
        // session so every later tensor skips the round trip.
        if (err instanceof TensorApiError && err.status === 501) {
          set((s) => ({ roisUnavailable: true, ...withView(s, key, { roisPending: [] }) }));
          return;
        }
        // The scope deliberately stays un-landed so a remount or a tensor switch
        // can try again; the effect's own deps keep that from becoming a retry
        // loop.
        set((s) =>
          withView(s, key, (v) => ({
            roisPending: v.roisPending.filter((p) => p !== scope),
            roisError: err instanceof Error ? err.message : String(err),
          })),
        );
      }
    };
    await Promise.all(scopes.map(fetchScope));
  },

  setTool(tool) {
    // A draft belongs to the tool that started it; switching abandons it rather
    // than reinterpreting placed vertices under different rules.
    set((s) =>
      s.tool === tool
        ? s
        : {
            tool,
            views: Object.fromEntries(
              Object.entries(s.views).map(([k, v]) => [k, v.draft ? { ...v, draft: null } : v]),
            ),
          },
    );
  },

  setDraft(draft) {
    set((s) =>
      withView(s, s.target.key, {
        draft: draft ? { shape: draft, sliceKey: sliceKey(s.position) } : null,
      }),
    );
  },

  setSelectedRoi(roiId) {
    set({ selectedRoiId: roiId });
  },

  setNewLabel(label) {
    set({ newLabel: label });
  },

  setNewSetName(setName) {
    set({ newSetName: setName });
  },

  setNewPolylineWidth(width) {
    set({ newPolylineWidth: clampPolylineWidth(width) });
  },

  toggleBroadcastAxis(axis, defaults) {
    set((s) => {
      // `null` means "this tensor's default", so materialise that before
      // editing -- otherwise the first toggle would silently also pin whatever
      // the default was broadcasting.
      const current = selectBroadcastAxes(s, defaults);
      return withView(s, s.target.key, {
        broadcastAxes: current.includes(axis)
          ? current.filter((a) => a !== axis)
          : [...current, axis],
      });
    });
  },

  async createRoi(geometry, plane) {
    const { client, newLabel, newSetName } = get();
    const arrayId = get().target.info?.array_id;
    const key = get().target.key;
    if (!client || !arrayId) return;

    // Drawn before it is stored. The click that finishes a shape also clears
    // the draft, so without this the shape the user just traced disappears for
    // the length of a round trip and comes back when the server answers --
    // which is the whole of the felt latency of drawing.
    //
    // Not selected yet: the id is this client's invention until the server
    // answers with its own, and selecting a row whose identity is about to
    // change puts that invention in front of the user.
    const provisional: RoiAnnotation = {
      roiId: provisionalRoiId(),
      arrayId,
      setName: newSetName || DEFAULT_ROI_SET,
      label: newLabel,
      geometry,
      plane,
      props: {},
      rev: 0,
      createdAtMs: Date.now(),
      updatedAtMs: Date.now(),
    };
    set((s) => ({
      roiWriteError: null,
      ...withView(s, key, (v) => ({ rois: [...v.rois, provisional] })),
    }));

    /** Take the provisional row back out of the list it was added to. */
    const withoutProvisional = (rois: RoiAnnotation[]) =>
      rois.filter((roi) => roi.roiId !== provisional.roiId);

    try {
      const result = await client.http.putRois(
        arrayId,
        [{ geometry, plane, label: newLabel, setName: newSetName || undefined }],
        { checkRev: true },
      );
      const stored = result.stored[0];
      if (!stored) {
        set((s) => ({
          roiWriteError: "The server stored no annotation",
          ...withView(s, key, (v) => ({ rois: withoutProvisional(v.rois) })),
        }));
        return;
      }
      // Swapped in place rather than appended, so the annotation does not jump
      // to the end of a list it was already drawn in. Written into this tensor's
      // record even if the user has left it: the annotation is stored, and its
      // own record is the right place for it.
      set((s) => ({
        // Only selected while its tensor is the one in view.
        ...(s.target.key === key ? { selectedRoiId: stored.roiId } : {}),
        ...withView(s, key, (v) => {
          const visible = v.visibleSets;
          return {
            rois: v.rois.map((roi) => (roi.roiId === provisional.roiId ? stored : roi)),
            // The listing's count follows the write, so a later listing does not
            // read this row as someone else's change (see landRoiScope).
            roiSets: withSetCount(v.roiSets, stored.setName, 1),
            // Onto the screen it was drawn on. A materialised list is exactly
            // the sets shown, so a new name -- or one switched off -- would
            // otherwise swallow the shape the user just traced.
            ...(visible !== null && !visible.includes(stored.setName)
              ? { visibleSets: [...visible, stored.setName] }
              : {}),
          };
        }),
      }));
    } catch (err) {
      set((s) => ({
        roiWriteError: err instanceof Error ? err.message : String(err),
        ...withView(s, key, (v) => ({ rois: withoutProvisional(v.rois) })),
      }));
    }
  },

  async deleteRoi(roiId) {
    const { client } = get();
    const arrayId = get().target.info?.array_id;
    const key = get().target.key;
    if (!client || !arrayId) return;
    // Nothing to ask the server about: this row is a local placeholder for a
    // write still in flight, and its id is one this client invented. Left for
    // the write to resolve, which is a round trip away.
    if (isProvisionalRoiId(roiId)) return;

    // Gone on the keystroke, not on the answer. The row is under the pointer
    // when Delete is pressed, so leaving it there for a round trip reads as the
    // key having missed -- and the far commoner outcome by far is that the
    // delete succeeds.
    const held = (get().views[key ?? ""] ?? EMPTY_VIEW).rois;
    const index = held.findIndex((roi) => roi.roiId === roiId);
    const removed = held[index];
    set((s) => ({
      roiWriteError: null,
      selectedRoiId: s.selectedRoiId === roiId ? null : s.selectedRoiId,
      ...withView(s, key, (v) => ({
        rois: v.rois.filter((roi) => roi.roiId !== roiId),
        roiSets: removed ? withSetCount(v.roiSets, removed.setName, -1) : v.roiSets,
      })),
    }));

    try {
      await client.http.deleteRois(arrayId, [roiId]);
      // An id the server did not have is already gone as far as the UI is
      // concerned, so an empty answer is not a failure: the row stays removed
      // rather than coming back as one that cannot be got rid of.
    } catch (err) {
      set((s) => {
        const failed = { roiWriteError: err instanceof Error ? err.message : String(err) };
        if (!removed) return failed;
        return {
          ...failed,
          ...withView(s, key, (v) => {
            const rois = [...v.rois];
            rois.splice(Math.min(index, rois.length), 0, removed);
            return { rois, roiSets: withSetCount(v.roiSets, removed.setName, 1) };
          }),
        };
      });
    }
  },

  async clearRoiSet(setName) {
    const { client } = get();
    const arrayId = get().target.info?.array_id;
    const key = get().target.key;
    if (!client || !arrayId) return;
    set({ roiWriteError: null });
    try {
      // No ids: the server drops the whole set in one transaction, so this is
      // not limited to the rows the cap let this client see.
      await client.http.deleteRois(arrayId, undefined, { setName });
      // Filtered by name rather than by the ids that came back, for the same
      // reason: what was deleted is the set, and the response enumerates only
      // what the server chose to list.
      set((s) => {
        const held = (s.views[key ?? ""] ?? EMPTY_VIEW).rois;
        return {
          selectedRoiId:
            held.find((roi) => roi.roiId === s.selectedRoiId)?.setName === setName
              ? null
              : s.selectedRoiId,
          ...withView(s, key, (v) => ({
            rois: v.rois.filter((roi) => roi.setName !== setName),
            // Gone from the listing too, as the server's own next listing would
            // have it: it enumerates stored rows, and there are none.
            roiSets: v.roiSets.filter((set) => set.setName !== setName),
            // The name means nothing once the set is gone, and leaving it on the
            // list would pin a later set of the same name to whatever this one
            // was toggled to.
            ...(v.visibleSets?.includes(setName)
              ? { visibleSets: v.visibleSets.filter((name) => name !== setName) }
              : {}),
          })),
        };
      });
    } catch (err) {
      set({ roiWriteError: err instanceof Error ? err.message : String(err) });
    }
  },

  setShowRois(value) {
    set((s) => (s.showRois === value ? s : { showRois: value }));
  },

  setLabelOverlay(arrayId) {
    set((s) => (s.labelOverlay === arrayId ? s : { labelOverlay: arrayId }));
  },

  setLabelOpacity(value) {
    const opacity = clampLabelOpacity(value);
    set((s) => (s.labelOpacity === opacity ? s : { labelOpacity: opacity }));
  },

  toggleSetVisible(setName) {
    set((s) => {
      // `null` means the default, so materialise that before editing --
      // otherwise the first toggle would silently also hide every other
      // client-owned set.
      const current = selectVisibleSets(s) ?? defaultVisibleSets(s);
      return withView(s, s.target.key, {
        visibleSets: current.includes(setName)
          ? current.filter((name) => name !== setName)
          : [...current, setName],
      });
    });
  },
});
