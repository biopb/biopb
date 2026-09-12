import { useEffect, useMemo } from "react";
import { vivDtype, type TileInfo } from "@biopb/tensor-flight-client";
import { selectObservedLimits, useAppStore, type SliceState } from "../store";
import {
  clampContrastLimits,
  contrastLimitsFrom,
  contrastTrack,
  percentileBounds,
} from "../utils/vivUtils";

/**
 * The contrast window a viewer hands its shader, and the state it publishes.
 *
 * One hook rather than a copy in each viewer (biopb/biopb#955). The two were
 * identical but for whether their samples arrived wrapped, and the panel had a
 * third derivation of the track alone -- three call sites of a *variadic*
 * `contrastTrack`, where no argument added, dropped or mistyped at one of them
 * could be caught by the type checker.
 *
 * The viewer is the deriver and publishes; the panel reads
 * `selectContrastTrack`. That ordering is not just deduplication: the track is
 * derived from `planeLimits` as computed *here*, which reaches the store one
 * effect later, so a panel deriving its own was drawing the bar on a track the
 * shader was not clamping into for a render after every plane change.
 *
 * `samples` is the plane's values, already sorted by `contrastSamples`. Null
 * before one has been taken -- the window then falls back to the track, which
 * is what shows an image at all while the histogram is in flight.
 */
export function useContrastWindow(
  info: TileInfo | null,
  samples: Float64Array | null,
  arrayId: string,
  slice: SliceState,
): [number, number] {
  // The plane's own extremes, which the window no longer reports once it is
  // fixed. Deliberately not keyed to the current selection: the last plane
  // sampled is what Min/Max should reset onto, and holding it across a read
  // keeps the button from going dead for the length of one.
  const planeLimits = useMemo(
    () => (samples ? contrastLimitsFrom(samples, 0, 100) : null),
    [samples],
  );
  const setPlaneLimits = useAppStore((s) => s.setPlaneLimits);
  const noteObservedLimits = useAppStore((s) => s.noteObservedLimits);
  useEffect(() => {
    if (!planeLimits) return;
    setPlaneLimits(planeLimits);
    noteObservedLimits(planeLimits, arrayId, slice.c);
  }, [planeLimits, arrayId, slice.c, setPlaneLimits, noteObservedLimits]);

  // Every level the tensor has shown, which is what a float track is drawn on:
  // a window fixed against a bright plane has to stay reachable from a dim one.
  const observedLimits = useAppStore(selectObservedLimits);

  const dtype = info ? vivDtype(info.dtype) : null;
  // `planeLimits` as well as the union: on a tensor's first plane the union has
  // not been written back to the store yet.
  const track = useMemo<[number, number]>(
    () => contrastTrack(dtype, observedLimits, planeLimits, slice.fixedLimits),
    [dtype, observedLimits, planeLimits, slice.fixedLimits],
  );

  const setContrastTrack = useAppStore((s) => s.setContrastTrack);
  useEffect(() => {
    // Only once the tensor's real dtype is known. Before that this track is the
    // 0-1 fallback, and the panel's own dtype -- which it can get from the
    // catalog without waiting for a read -- gives a better one.
    if (!info) return;
    setContrastTrack(track, arrayId);
  }, [info, track, arrayId, setContrastTrack]);

  const contrastLimits = useMemo<[number, number]>(() => {
    if (!info) return [0, 1];
    // A fixed window is the user's, not the plane's: it is not re-derived per
    // plane, only brought inside the track it is being applied to -- the
    // dtype's range, or on a float tensor the levels the data has shown.
    if (slice.contrastMode === "fixed") {
      return slice.fixedLimits
        ? clampContrastLimits(slice.fixedLimits, track, dtype)
        : track;
    }
    if (!samples) return track;
    const [lo, hi] = percentileBounds(slice.percentileScale);
    return contrastLimitsFrom(samples, lo, hi);
  }, [
    info,
    dtype,
    track,
    samples,
    slice.contrastMode,
    slice.fixedLimits,
    slice.percentileScale,
  ]);

  // Published so the panel can seed a fixed window from what is on screen.
  const setAppliedLimits = useAppStore((s) => s.setAppliedLimits);
  useEffect(() => {
    setAppliedLimits(contrastLimits);
  }, [contrastLimits, setAppliedLimits]);

  return contrastLimits;
}
