import { useCallback, useEffect, useRef, useState } from "react";
import { withBase } from "../base";
import { selectRoisPending, useAppStore } from "../store";
import { SessionLocked, sessionFetch } from "../utils/sessionFetch";
import {
  LABEL_SETTLE_MS,
  READY_TIMEOUT_MS,
  readCanvas,
  readiness,
  viewParams,
  waitFor,
} from "../utils/viewCapture";

/** One id per page load; the control tells tabs apart by it. */
const TAB_ID = crypto.randomUUID();

/** Pause after a failed poll, so a control that is down is not hammered. */
const RETRY_MS = 3000;

interface CaptureJob {
  req: string;
  view: string;
  max_edge: number;
}

/**
 * Answers an agent's request for a picture of the viewer.
 *
 * A session with no napari window cannot see its own results, so it asks the
 * control, which hands the request to a page that is polling here. The page
 * applies the requested state as it would a shared link -- and leaves it
 * applied, as napari's does -- waits for the image and overlays to land,
 * and posts the canvas back.
 *
 * It polls only while the tab is visible: a hidden tab does not repaint, so a
 * capture sent to one would return the frame from before the request. The parked
 * poll is also the control's evidence that this tab exists.
 *
 * Returns whether a capture is running, which is when the page locks input: the
 * state must not move between applying it and reading it.
 */
export function useViewerCapture(): { capturing: boolean; cancel: () => void } {
  const [capturing, setCapturing] = useState(false);
  const cancelled = useRef(false);

  useEffect(() => {
    let visible = document.visibilityState === "visible";
    let abort: AbortController | null = null;
    let stopped = false;

    const answer = (req: string, body: Blob | null, query: URLSearchParams) =>
      sessionFetch(withBase(`/api/viewer/answer/${req}?${query}`), {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body,
      });

    const run = async (job: CaptureJob) => {
      cancelled.current = false;
      setCapturing(true);
      const query = new URLSearchParams();
      try {
        const params = viewParams(job.view);
        if (!useAppStore.getState().applyViewerState(params)) {
          throw new Error("the view has no id");
        }
        const verdict = await waitFor(
          () => {
            const s = useAppStore.getState();
            return readiness({
              status: s.target.status,
              error: s.target.error?.reason ?? null,
              planeReady: s.runtime.planeReady,
              roisPending: selectRoisPending(s).length,
            });
          },
          READY_TIMEOUT_MS,
          () => cancelled.current,
        );
        if (verdict.kind === "cancelled") throw new Error("cancelled by the user");
        if (verdict.kind === "failed") throw new Error(verdict.error);
        if (verdict.kind === "timeout") {
          query.set("partial", "1");
          query.append("note", "timed out waiting for tiles");
        }
        if (params.get("lb")) await new Promise((r) => setTimeout(r, LABEL_SETTLE_MS));
        if (cancelled.current) throw new Error("cancelled by the user");
        // A hidden tab does not repaint: what it holds is the frame from before.
        if (document.visibilityState !== "visible") throw new Error("the viewer tab was hidden");
        const png = await readCanvas(job.max_edge);
        await answer(job.req, png, query);
      } catch (err) {
        const q = new URLSearchParams({ error: err instanceof Error ? err.message : String(err) });
        await answer(job.req, null, q).catch(() => undefined);
      } finally {
        setCapturing(false);
      }
    };

    const loop = async () => {
      while (!stopped && visible) {
        abort = new AbortController();
        try {
          const resp = await sessionFetch(withBase(`/api/viewer/next?client=${TAB_ID}`), {
            signal: abort.signal,
          });
          if (resp.status === 200) await run((await resp.json()) as CaptureJob);
          else if (resp.status !== 204) await new Promise((r) => setTimeout(r, RETRY_MS));
        } catch (err) {
          // Locked: `sessionFetch` is already sending the page to unlock.
          if (err instanceof SessionLocked || stopped || !visible) return;
          await new Promise((r) => setTimeout(r, RETRY_MS));
        }
      }
    };

    // Two loops must never overlap: the control keeps one poll per tab.
    let running = false;
    const start = () => {
      if (running || stopped || !visible) return;
      running = true;
      void loop().finally(() => {
        running = false;
        if (visible && !stopped) start();
      });
    };
    const onVisibility = () => {
      visible = document.visibilityState === "visible";
      if (visible) start();
      else abort?.abort();
    };
    document.addEventListener("visibilitychange", onVisibility);
    start();
    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", onVisibility);
      abort?.abort();
    };
  }, []);

  const cancel = useCallback(() => {
    cancelled.current = true;
  }, []);
  return { capturing, cancel };
}
