import { useCallback } from "react";
import { useAppStore } from "../store";
import type { Camera2DState, Camera3DState } from "../store";
import { CAMERA_MIRROR_MS } from "../utils/vivUtils";
import { useDebouncedCommit } from "./useDebouncedCommit";

interface Cameras {
  "2d": Camera2DState;
  "3d": Camera3DState;
}

/**
 * Mirrors a viewer's camera into the store, on the trailing edge only.
 *
 * deck.gl owns the camera; the store's copy is what a link should carry and what
 * seeds the next mount, so the resting viewport is what matters and the frames
 * of a drag are noise that would otherwise reach the URL at pointer rate. The
 * returned callback is stable, and never feeds anything back into the viewer:
 * this stays a mirror.
 */
export function useCameraMirror<K extends keyof Cameras>(kind: K): (camera: Cameras[K]) => void {
  const setCamera2d = useAppStore((s) => s.setCamera2d);
  const setCamera3d = useAppStore((s) => s.setCamera3d);
  const schedule = useDebouncedCommit(CAMERA_MIRROR_MS);
  return useCallback(
    (camera: Cameras[K]) =>
      schedule("camera", () => {
        if (kind === "2d") setCamera2d(camera as Camera2DState);
        else setCamera3d(camera as Camera3DState);
      }),
    [kind, schedule, setCamera2d, setCamera3d],
  );
}
