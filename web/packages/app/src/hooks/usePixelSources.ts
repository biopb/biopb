import { useCallback, useEffect, useRef, useState } from "react";
import { pixelSourcesFromInfo, type TileInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import type { ViewerErrorKind } from "../store";
import type { PixelSources } from "../components/VivStage";

/**
 * The Viv pixel sources for a resolved grid, and the last tile failure.
 *
 * `tile_info` was fetched (and retried) by `openTensor`, so this is a pure
 * mapping and cannot itself fail for a transient reason; it refuses only a
 * tensor whose axes this server's tile route cannot select, which is a fact
 * about the tensor and reported as such.
 *
 * `clearTileError` is for the caller to run when the plane changes: a failed
 * read is scoped to the plane that produced it, so it must not go on labelling
 * later planes that loaded perfectly well.
 */
export function usePixelSources(
  info: TileInfo,
  onUnsupported: (reason: string, kind: ViewerErrorKind) => void,
): { sources: PixelSources | null; tileError: string | null; clearTileError: () => void } {
  const client = useAppStore((s) => s.client);
  const [sources, setSources] = useState<PixelSources | null>(null);
  const [tileError, setTileError] = useState<string | null>(null);

  // Report upward through a ref: `onUnsupported` comes from the parent's render,
  // and listing it as a dependency would re-run the source build on every
  // parent re-render (i.e. on every slider move).
  const onUnsupportedRef = useRef(onUnsupported);
  onUnsupportedRef.current = onUnsupported;

  useEffect(() => {
    if (!client) return;
    let live = true;
    setTileError(null);
    try {
      setSources(
        pixelSourcesFromInfo(client.http, info, {
          onTileError: (err) => {
            if (live) setTileError(err.message);
          },
        }),
      );
    } catch (err) {
      setSources(null);
      onUnsupportedRef.current(err instanceof Error ? err.message : String(err), "capability");
    }
    return () => {
      live = false;
    };
  }, [client, info]);

  const clearTileError = useCallback(() => setTileError(null), []);
  return { sources, tileError, clearTileError };
}
