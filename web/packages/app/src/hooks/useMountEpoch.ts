import { useEffect, useState } from "react";
import { useAppStore } from "../store";

/**
 * The epoch this viewer was mounted under, which it tags every publication with.
 * A viewer lives for exactly one epoch (`ViewerPane` keys on it), so a write
 * that outlives its tensor is dropped by the store rather than leaking into the
 * next one.
 */
export function useMountEpoch(): number {
  const [epoch] = useState(() => useAppStore.getState().target.epoch);
  return epoch;
}

/**
 * Publish whether what is on the canvas is the plane that was asked for.
 *
 * Play reads it to pace itself to the data plane rather than to a timer.
 */
export function usePublishPlaneReady(ready: boolean, epoch: number): void {
  const setPlaneReady = useAppStore((s) => s.setPlaneReady);
  useEffect(() => {
    setPlaneReady(ready, epoch);
  }, [ready, epoch, setPlaneReady]);
}
