import { useEffect, useState } from "react";
import {
  TensorAbortError,
  createTensorPixelSources,
  splitLabelArrayId,
  type TensorFlightClient,
  type TileInfo,
} from "@biopb/tensor-flight-client";

/** The label set's own pixel sources and grid, once both have landed. */
export interface LabelOverlay {
  /** The set's whole address, so a stale load can be told from a current one. */
  arrayId: string;
  /** The set's name, for the layer id and for the panel. */
  name: string;
  sources: Awaited<ReturnType<typeof createTensorPixelSources>>["data"];
  info: TileInfo;
}

/**
 * The label set to draw over the image, loaded exactly as the image is.
 *
 * A set is an ordinary tensor, so this is the same `tile_info` round trip and
 * the same tile route -- no second wire format and no second cache. What it is
 * not is part of the image's own load: a set that 404s, or one uploaded against
 * a server that cannot serve it, must cost the overlay and not the image. So
 * the failure is returned rather than raised, and `TileViewer` keeps rendering.
 *
 * Deliberately no retry. The image's own load re-asks once through
 * `TILE_INFO_RETRY_MS` because a tensor with no viewer is a dead pane; an
 * overlay that did not start leaves the image on screen, and the row that
 * switched it on is still there to click again.
 */
export function useLabelOverlay(
  client: TensorFlightClient | null,
  arrayId: string | null,
): { overlay: LabelOverlay | null; error: string | null } {
  const [overlay, setOverlay] = useState<LabelOverlay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Cleared here rather than only on success: the previous set's sources must
    // not be drawn over the next image for the length of a fetch.
    setOverlay(null);
    setError(null);
    if (!client || !arrayId) return;
    const name = splitLabelArrayId(arrayId)?.name;
    if (name === undefined) {
      setError(`${arrayId} does not name a label set`);
      return;
    }

    const controller = new AbortController();
    let live = true;
    createTensorPixelSources(client.http, arrayId, {
      signal: controller.signal,
      onTileError: (err) => {
        if (live) setError(err.message);
      },
    })
      .then(({ data, info }) => {
        if (live) setOverlay({ arrayId, name, sources: data, info });
      })
      .catch((err: unknown) => {
        if (!live || err instanceof TensorAbortError) return;
        setError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      live = false;
      controller.abort();
    };
  }, [client, arrayId]);

  return { overlay, error };
}
