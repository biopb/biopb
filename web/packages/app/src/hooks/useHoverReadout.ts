import { useCallback, useMemo, useRef } from "react";
import type { MutableRefObject } from "react";
import type { HoverSample } from "../components/HoverReadout";
import type { HoverInfo } from "../components/VivStage";
import type { XY } from "../utils/roiLayers";

/**
 * The value under the pointer, fed to a badge without re-rendering the viewer.
 *
 * Costs no read. Viv picks the value out of the tile deck.gl already has
 * (`info.tile.content.data`) and gives up when there is none, so hovering over a
 * tile that has not arrived reports nothing rather than fetching it.
 *
 * Fed to the badge through a sink ref instead of through the caller's state: a
 * pointer move at 60/s that re-rendered the viewer would rebuild `VivStage` and,
 * with it, every deck.gl layer. `draftCursorSinkRef` is the authoring hook's
 * sink for the trailing segment of a shape being placed; it is null unless a
 * draft is open, which is what keeps this handler from re-rendering anything
 * the rest of the time.
 */
export function useHoverReadout(
  draftCursorSinkRef: MutableRefObject<((at: XY | null) => void) | null>,
) {
  const hoverSinkRef = useRef<((sample: HoverSample | null) => void) | null>(null);
  const bindHover = useCallback((sink: ((sample: HoverSample | null) => void) | null) => {
    hoverSinkRef.current = sink;
  }, []);

  const hover = useMemo(() => {
    // Where the pointer is comes from deck.gl's own hover, which fires for
    // every move; the value comes from Viv's hook, which fires only when there
    // is a tile to read. Keeping them apart is what lets the readout go blank
    // off-image instead of holding the last value it saw.
    let at: HoverSample | null = null;
    return {
      onHover: (info: HoverInfo) => {
        const c = info?.coordinate;
        if (!c || !info.sourceLayer || c[0] === undefined || c[1] === undefined) {
          at = null;
          hoverSinkRef.current?.(null);
          draftCursorSinkRef.current?.(null);
          return;
        }
        // Viv reads `2 ** round(-z)` as the level's scale, so this is the same
        // number the badge would need to explain a downsampled value.
        const z = info.tile?.index?.z;
        at = {
          x: Math.floor(c[0]),
          y: Math.floor(c[1]),
          value: null,
          scale: typeof z === "number" ? Math.max(1, 2 ** Math.round(-z)) : 1,
        };
        hoverSinkRef.current?.(at);
        // The one place the viewer re-renders at pointer rate, and only while a
        // shape is being placed: the trailing segment has to follow the pointer
        // to be worth anything.
        if (draftCursorSinkRef.current) draftCursorSinkRef.current([c[0], c[1]]);
      },
      hooks: {
        handleValue: (values: number[]) => {
          const v = values?.[0];
          if (at && typeof v === "number" && Number.isFinite(v)) {
            hoverSinkRef.current?.({ ...at, value: v });
          }
        },
        // Never called: Viv 0.22 destructures this hook as `handleCoordnate`,
        // so the coordinate comes from `onHover` above. Present because the
        // prop's type requires it.
        handleCoordinate: () => {},
      },
    };
  }, [draftCursorSinkRef]);

  return { hover, bindHover };
}
