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

interface ShowJob {
  req: string;
  view: string;
  image: boolean;
  max_edge: number;
}

/**
 * Answers an agent's request to show the user a view.
 *
 * A session with no napari window cannot present its results, so it asks the
 * control, which hands the request to a page that is polling here. The page
 * applies the requested state as it would a shared link -- and leaves it
 * applied, as napari's does -- and acknowledges at once. If the agent also wants
 * an image and the page is on screen, it waits for the image and overlays to
 * land and posts the canvas back.
 *
 * It polls whether or not the tab is visible, and says which on every poll and
 * answer: a hidden tab moves its state but does not repaint, so it never draws
 * and the agent is told. The parked poll is also the control's evidence that
 * this tab exists.
 *
 * Returns whether an image capture is running, which is when the page locks
 * input: the state must not move between applying it and reading it.
 */
export function useViewerShow(): { capturing: boolean; cancel: () => void } {
  const [capturing, setCapturing] = useState(false);
  const cancelled = useRef(false);

  useEffect(() => {
    let abort: AbortController | null = null;
    let stopped = false;
    const isVisible = () => document.visibilityState === "visible";

    const answer = (req: string, body: Blob | null, query: URLSearchParams) =>
      sessionFetch(withBase(`/api/viewer/answer/${req}?${query}`), {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body,
      });

    const run = async (job: ShowJob) => {
      cancelled.current = false;
      const query = new URLSearchParams();
      try {
        const params = viewParams(job.view);
        if (!useAppStore.getState().applyViewerState(params)) {
          throw new Error("the view has no id");
        }
        let png: Blob | null = null;
        // A hidden tab does not repaint: it has moved, and says it cannot draw.
        if (job.image && isVisible()) {
          setCapturing(true);
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
            () => cancelled.current || !isVisible(),
          );
          if (verdict.kind === "failed") throw new Error(verdict.error);
          if (verdict.kind === "timeout") {
            query.set("partial", "1");
            query.append("note", "timed out waiting for tiles");
          }
          if (verdict.kind !== "cancelled" && params.get("lb")) {
            await new Promise((r) => setTimeout(r, LABEL_SETTLE_MS));
          }
          if (cancelled.current) throw new Error("cancelled by the user");
          if (isVisible()) png = await readCanvas(job.max_edge);
        }
        // No frames while an image was wanted counts as hidden, whatever
        // `visibilityState` says.
        query.set("visible", isVisible() && (png || !job.image) ? "1" : "0");
        await answer(job.req, png, query);
      } catch (err) {
        const q = new URLSearchParams({ error: err instanceof Error ? err.message : String(err) });
        await answer(job.req, null, q).catch(() => undefined);
      } finally {
        setCapturing(false);
      }
    };

    const loop = async () => {
      while (!stopped) {
        const ctl = new AbortController();
        abort = ctl;
        try {
          const resp = await sessionFetch(
            withBase(`/api/viewer/next?client=${TAB_ID}&visible=${isVisible() ? 1 : 0}`),
            { signal: ctl.signal },
          );
          // Past this point the job is ours: a visibility change must not
          // cancel reading it.
          abort = null;
          if (resp.status === 200) await run((await resp.json()) as ShowJob);
          else if (resp.status !== 204) await new Promise((r) => setTimeout(r, RETRY_MS));
        } catch (err) {
          // Locked: `sessionFetch` is already sending the page to unlock.
          if (err instanceof SessionLocked || stopped) return;
          // Aborted to re-park with the new visibility: no pause.
          if (ctl.signal.aborted) continue;
          await new Promise((r) => setTimeout(r, RETRY_MS));
        }
      }
    };

    // The control must learn when this tab hides or shows, so it can prefer a
    // visible one: end the parked poll and park again with the new state.
    const onVisibility = () => abort?.abort();
    document.addEventListener("visibilitychange", onVisibility);
    void loop();
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
