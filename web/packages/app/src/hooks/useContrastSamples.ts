import { useEffect } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import type { PixelSources } from "../components/VivStage";
import { contrastSamples } from "../utils/vivUtils";
import { useMountEpoch } from "./useMountEpoch";

/** The single grey level a plane holds, when `samples` are its own and uniform. */
function uniformValue(
  samples: { plane: object; values: Float64Array } | null,
  selection: object,
): number | null {
  if (!samples || samples.plane !== selection) return null;
  const v = samples.values;
  const first = v[0];
  return first !== undefined && first === v[v.length - 1] ? first : null;
}

/**
 * Sample the plane asked for, and say whether it is featureless.
 *
 * Reads the coarsest level once per selection and hands the sorted values to the
 * store (`notePlaneSamples`), which keeps them and notes their extremes as
 * levels the tensor has shown. The window and the track are selectors over that,
 * so the intensity slider re-derives limits locally instead of refetching.
 *
 * Returns the plane's single value when the sample says it is uniform, else
 * null. A featureless plane renders black, and so does one whose tiles have not
 * arrived and one whose contrast window excludes everything; black is the right
 * rendering for an all-zero plane -- what was missing is saying so.
 *
 * Keyed to the selection because the stored samples are deliberately kept across
 * a plane change so the contrast does not flash: unkeyed, the label would
 * describe the plane before last.
 */
export function useContrastSamples(
  sources: PixelSources | null,
  info: TileInfo,
  selection: Record<string, number>,
): number | null {
  const epoch = useMountEpoch();
  const notePlaneSamples = useAppStore((s) => s.notePlaneSamples);
  const channel = useAppStore((s) => s.position.c);

  useEffect(() => {
    if (!sources) return;
    // Interleaved RGB is rendered as colour, not through a contrast ramp.
    if (info.plane.s !== null) return;
    const overview = sources[sources.length - 1];
    if (!overview) return;
    const controller = new AbortController();
    let live = true;
    overview
      .getRaster({ selection, signal: controller.signal })
      .then((raster) => {
        if (live) {
          notePlaneSamples(
            { plane: selection, values: contrastSamples(raster.data) },
            channel,
            epoch,
          );
        }
      })
      .catch(() => {
        // Keep the previous limits: a failed histogram is a worse reason to
        // blank the image than to show it with slightly stale contrast.
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [sources, info, selection, channel, epoch, notePlaneSamples]);

  // A selector returning a number, so this re-renders when the answer changes
  // and not on every sample landing (which is every plane, and every frame of a
  // play).
  return useAppStore((s) => uniformValue(s.runtime.samples, selection));
}
