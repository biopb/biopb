import { useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { useAppStore } from "../store";
import { DEFAULT_VIEWER_URL_STATE, encodeViewerState } from "../utils/viewerUrl";

/**
 * Keeps the query string and the viewing state in step, in that order.
 *
 * Hydration runs once, and only once the catalog has arrived: `applyViewerState`
 * resolves an `array_id` against `sources`, which is empty until `loadSources`
 * resolves. Re-running it later would fight the user, so it is latched rather
 * than re-derived from `searchParams`.
 *
 * Writing back is gated on that latch for the same reason in reverse -- the
 * write effect fires on mount with the store still at its defaults, and an
 * ungated write would erase the very link it was asked to open.
 */
export function useViewerUrlSync() {
  const [searchParams, setSearchParams] = useSearchParams();
  const connectionState = useAppStore((s) => s.connectionState);
  const sources = useAppStore((s) => s.sources);
  const applyViewerState = useAppStore((s) => s.applyViewerState);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  const slice = useAppStore((s) => s.slice);
  const render3d = useAppStore((s) => s.render3d);
  const volumeRenderMode = useAppStore((s) => s.volumeRenderMode);

  const hydrated = useRef(false);
  // The effects below must not re-run when the URL changes -- they are what
  // changes it -- so the live params are read through a ref instead of a dep.
  const paramsRef = useRef(searchParams);
  paramsRef.current = searchParams;

  useEffect(() => {
    if (hydrated.current) return;
    if (connectionState !== "connected" || sources.length === 0) return;
    hydrated.current = true;
    applyViewerState(paramsRef.current);
  }, [connectionState, sources, applyViewerState]);

  useEffect(() => {
    if (!hydrated.current) return;
    // Nothing selected is not a state worth pinning: it would rewrite a link
    // someone just pasted into an empty `/viewer` before hydration can use it.
    if (!activeTensorId) return;
    const next = encodeViewerState(
      paramsRef.current,
      { arrayId: activeTensorId, slice, render3d, volumeRenderMode },
      { arrayId: activeTensorId, ...DEFAULT_VIEWER_URL_STATE },
    );
    // Both scrub paths debounce before reaching the store, so this is already
    // rate-limited; `replace` is what keeps a drag out of the back button.
    if (next.toString() !== paramsRef.current.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [activeTensorId, slice, render3d, volumeRenderMode, setSearchParams]);
}
