import { useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { useAppStore } from "../store";
import { DEFAULT_VIEWER_URL_STATE, encodeViewerState } from "../utils/viewerUrl";

/**
 * Keeps the query string and the viewing state in step, in that order.
 *
 * Hydration runs once, as soon as there is a client to fetch with. It needs no
 * catalog: `applyViewerState` takes the selection out of the id itself, so the
 * link opens whether or not a listing has arrived, or arrived at all.
 * Re-running it later would fight the user, so it is latched rather than
 * re-derived from `searchParams`.
 *
 * Writing back is gated on that latch for the same reason in reverse -- the
 * write effect fires on mount with the store still at its defaults, and an
 * ungated write would erase the very link it was asked to open.
 */
export function useViewerUrlSync() {
  const [searchParams, setSearchParams] = useSearchParams();
  const client = useAppStore((s) => s.client);
  const applyViewerState = useAppStore((s) => s.applyViewerState);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  const slice = useAppStore((s) => s.slice);
  const render3d = useAppStore((s) => s.render3d);
  const volumeRenderMode = useAppStore((s) => s.volumeRenderMode);
  const tileInfo = useAppStore((s) => s.tileInfo);
  const requestedArrayId = useAppStore((s) => s.requestedArrayId);
  const camera3d = useAppStore((s) => s.camera3d);
  const camera2d = useAppStore((s) => s.camera2d);

  const hydrated = useRef(false);
  // The effects below must not re-run when the URL changes -- they are what
  // changes it -- so the live params are read through a ref instead of a dep.
  const paramsRef = useRef(searchParams);
  paramsRef.current = searchParams;

  useEffect(() => {
    if (hydrated.current) return;
    // Gated on the client, not on the catalog. `applyViewerState` reads the
    // selection out of the id itself, so waiting for a listing would hold a
    // shared link behind a scan it does not need -- and behind a *failed*
    // listing forever, where the link would otherwise have opened fine.
    if (!client) return;
    hydrated.current = true;
    applyViewerState(paramsRef.current);
  }, [client, applyViewerState]);

  useEffect(() => {
    if (!hydrated.current) return;
    // Nothing selected is not a state worth pinning: it would rewrite a link
    // someone just pasted into an empty `/viewer` before hydration can use it.
    if (!activeTensorId) return;
    // The pinned address, so a link names the content it was made from -- and
    // an unpinned link is upgraded in place once the grid answers, which is the
    // rewrite that makes a hand-written id legal input.
    //
    // `requestedArrayId` sits between the two so a link whose version is gone
    // keeps its token in the bar: the viewer's 404 is the answer, and quietly
    // rewriting the URL to the current version would hide which one failed.
    const arrayId = tileInfo?.array_id ?? requestedArrayId ?? activeTensorId;
    const next = encodeViewerState(
      paramsRef.current,
      { arrayId, slice, render3d, volumeRenderMode, camera3d, camera2d },
      { arrayId, ...DEFAULT_VIEWER_URL_STATE },
    );
    // Both scrub paths debounce before reaching the store, so this is already
    // rate-limited; `replace` is what keeps a drag out of the back button.
    if (next.toString() !== paramsRef.current.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [
    activeTensorId,
    requestedArrayId,
    tileInfo,
    slice,
    render3d,
    volumeRenderMode,
    camera3d,
    camera2d,
    setSearchParams,
  ]);
}
