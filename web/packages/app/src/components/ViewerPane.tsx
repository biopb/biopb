"use client";

/**
 * Picks which viewer the pane gets.
 *
 * The tiled Viv viewer is the target and handles everything that reaches it, but
 * it cannot cover every case: it needs WebGL2, a dtype with a GPU equivalent, a
 * canonical `[..., Y, X, S]` axis order, and a server new enough to serve
 * `/api/tile_info`. The server-rendered viewer has none of those requirements,
 * so it stays as the fallback rather than being replaced outright — the failure
 * modes are all "this tensor", not "this deployment".
 */

import { Component, Suspense, lazy, useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { ImageViewer } from "./ImageViewer";

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
  { children: ReactNode; onError: (reason: string) => void },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    this.props.onError(error.message || "the tiled viewer failed to start");
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function ViewerPane({ sourceId, tensorId }: ViewerPaneProps) {
  const [fallback, setFallback] = useState<string | null>(
    hasWebGL2() ? null : "WebGL2 is unavailable in this browser",
  );

  // A new tensor gets a fresh verdict: the previous one may have failed for a
  // reason that is specific to it (dtype, axis order).
  useEffect(() => {
    setFallback(hasWebGL2() ? null : "WebGL2 is unavailable in this browser");
  }, [sourceId, tensorId]);

  const onUnsupported = useCallback((reason: string) => setFallback(reason), []);

  if (fallback !== null) {
    return (
      <>
        <ImageViewer sourceId={sourceId} tensorId={tensorId} />
        <div className="viewer-fallback-note">server-rendered — {fallback}</div>
      </>
    );
  }

  return (
    <TileViewerBoundary key={tensorId} onError={onUnsupported}>
      <Suspense fallback={<div className="loading-overlay">Loading viewer…</div>}>
        <TileViewer
          // Remount per tensor: view state, contrast samples and the tile cache
          // all belong to one image, and resetting them by hand is the kind of
          // bookkeeping that goes stale.
          key={tensorId}
          sourceId={sourceId}
          arrayId={tensorId}
          onUnsupported={onUnsupported}
        />
      </Suspense>
    </TileViewerBoundary>
  );
}
