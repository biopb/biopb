import { useEffect, useMemo } from "react";
import { sliderAxes } from "@biopb/tensor-flight-client";
import { selectTileInfo, useAppStore } from "../store";
import {
  PLAY_FRAME_MS,
  PLAY_READY_POLL_MS,
  PLAY_STALL_MS,
  axisIndexOf,
  nextPlayIndex,
  orderSliderAxes,
} from "../utils/sliceUi";

/**
 * Automatic scrubbing, mounted once by the page.
 *
 * Paced to the data plane rather than to the timer alone: the next frame is
 * asked for once the last one is actually on the canvas (`runtime.planeReady`,
 * published by whichever viewer is mounted), so a source that cannot deliver
 * 10/s plays slower instead of queueing reads it will never catch up on.
 * PLAY_STALL_MS is the escape -- a plane that never loads must not stop the
 * sequence.
 *
 * Lives outside the panel so that play does not depend on the panel staying
 * mounted; `SliceControls` only toggles `playAxis`. The axes it may step are the
 * ones the sliders offer, taken from the resolved grid.
 */
export function usePlayback(): void {
  const playAxis = useAppStore((s) => s.playAxis);
  const setPlayAxis = useAppStore((s) => s.setPlayAxis);
  const info = useAppStore(selectTileInfo);
  const render3d = useAppStore((s) => s.render3d);

  // Z is the volume's depth in 3-D, read whole -- there is no plane to step
  // through, so it is not playable there.
  const playable = useMemo(
    () =>
      info
        ? orderSliderAxes(
            sliderAxes(info.dim_labels, info.shape).filter(
              (axis) => axis.extent > 1 && !(render3d && axis.named === "z"),
            ),
          )
        : [],
    [info, render3d],
  );
  const axis = playAxis ? playable.find((a) => a.key === playAxis) : undefined;

  // A play cannot outlive the axis driving it: switching to 3-D takes Z away,
  // and a tensor swap replaces the axes entirely.
  useEffect(() => {
    if (playAxis && !axis) setPlayAxis(null);
  }, [playAxis, axis, setPlayAxis]);

  useEffect(() => {
    if (!axis) return;
    let timer: ReturnType<typeof setTimeout>;
    let asked = Date.now();

    const step = () => {
      const store = useAppStore.getState();
      if (!store.runtime.planeReady && Date.now() - asked < PLAY_STALL_MS) {
        timer = setTimeout(step, PLAY_READY_POLL_MS);
        return;
      }
      store.setAxisIndex(axis, nextPlayIndex(axisIndexOf(store.position, axis), axis.extent));
      // Said here rather than waited for: the viewer publishes the same fact
      // from an effect, and this timer is armed before that effect runs.
      store.setPlaneReady(false, store.runtime.epoch);
      asked = Date.now();
      timer = setTimeout(step, PLAY_FRAME_MS);
    };

    timer = setTimeout(step, PLAY_FRAME_MS);
    return () => clearTimeout(timer);
  }, [axis]);
}
