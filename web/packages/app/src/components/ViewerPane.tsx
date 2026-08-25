"use client";

/**
 * Hosts the tiled viewer and says what happened when it cannot start.
 *
 * The tiled Viv viewer is the only viewer. It needs WebGL2, a dtype with a GPU
 * equivalent, and a server new enough to serve `/api/tile_info` — so there are
 * tensors and browsers it cannot show, and this pane's job is to name the reason
 * rather than to substitute a second viewer for it.
 *
 * That reason splits two ways, and the split is what {@link ViewerErrorKind}
 * carries. A fact about the tensor or the browser stays decided: re-running the
 * same load would fail the same way. A server that did not answer says nothing
 * about whether the tensor can be tiled, so it is offered again.
 */

import { Component, Suspense, lazy, useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";

// deck.gl + luma.gl are the app's largest dependency by a wide margin, and only
// this pane uses them. Splitting them out keeps them off the admin and observe
// routes entirely.
const TileViewer = lazy(() => import("./TileViewer"));

interface ViewerPaneProps {
  sourceId: string;
  tensorId: string;
}

/**
 * Viv renders 16-bit data through integer textures, which are WebGL2-only —
 * there is no WebGL1 path to degrade to. Probed once per page: the answer
 * follows the GPU and driver, not the document.
 */
let webgl2Support: boolean | null = null;
function hasWebGL2(): boolean {
  if (webgl2Support === null) {
    try {
      webgl2Support = !!document.createElement("canvas").getContext("webgl2");
    } catch {
      webgl2Support = false;
    }
  }
  return webgl2Support;
}

/**
 * Keeps a throw inside deck.gl from taking the app down with it.
 *
 * React unmounts the whole tree on an uncaught render error, so without this a
 * single bad tensor blanks the page rather than the pane. There is no reset
 * method on purpose: the `key`-driven remount below discards this boundary along
 * with its child.
 */
class TileViewerBoundary extends Component<
  { children: ReactNode; onError: (reason: string, kind: ViewerErrorKind) => void },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    // Retryable, because this boundary cannot tell a deck.gl throw about the
    // tensor from a rejected `import()` of the viewer chunk. Only the first
    // deserves a dead end, and a retry costs one remount. Verified refusals
    // arrive through `onUnsupported` already classified.
    this.props.onError(error.message || "it failed to start", "transport");
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

/**
 * Why the viewer could not start.
 *
 * `"capability"` is a settled fact about this browser or tensor, not worth
 * re-testing. `"transport"` is anything that might go the other way next time,
 * so it gets a retry.
 */
export type ViewerErrorKind = "capability" | "transport";

interface ViewerError {
  reason: string;
  kind: ViewerErrorKind;
}

const noWebGL2: ViewerError = {
  reason: "WebGL2 is unavailable in this browser",
  kind: "capability",
};

export function ViewerPane({ sourceId, tensorId }: ViewerPaneProps) {
  const [failure, setFailure] = useState<ViewerError | null>(
    hasWebGL2() ? null : noWebGL2,
  );
  // Bumped to remount the tiled viewer on a manual retry. The tensor has not
  // changed, so `key={tensorId}` alone would hand back the same instance.
  const [attempt, setAttempt] = useState(0);

  // A new tensor gets a fresh verdict: the last one may have failed for a
  // reason specific to it.
  useEffect(() => {
    setFailure(hasWebGL2() ? null : noWebGL2);
  }, [sourceId, tensorId]);

  const onUnsupported = useCallback(
    (reason: string, kind: ViewerErrorKind) => setFailure({ reason, kind }),
    [],
  );

  const retry = useCallback(() => {
    setFailure(null);
    setAttempt((n) => n + 1);
  }, []);

  if (failure !== null) {
    return (
      <div className="viewer-unavailable">
        {failure.kind === "transport" ? (
          <>
            The viewer could not start — {failure.reason}{" "}
            <button type="button" className="viewer-retry" onClick={retry}>
              Try again
            </button>
          </>
        ) : (
          <>This tensor cannot be displayed — {failure.reason}</>
        )}
      </div>
    );
  }

  return (
    <TileViewerBoundary key={`${tensorId}#${attempt}`} onError={onUnsupported}>
      <Suspense fallback={<div className="loading-overlay">Loading viewer…</div>}>
        <TileViewer
          // Remount per tensor: view state, contrast samples and the tile cache
          // all belong to one image, and resetting them by hand is the kind of
          // bookkeeping that goes stale.
          key={`${tensorId}#${attempt}`}
          sourceId={sourceId}
          arrayId={tensorId}
          onUnsupported={onUnsupported}
        />
      </Suspense>
    </TileViewerBoundary>
  );
}
