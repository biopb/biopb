"use client";

/**
 * Client-side rendered viewer: Viv over the tile API.
 *
 * Pixels arrive as raw tiles and contrast is applied in the shader, so panning
 * refetches only the tiles that came into view and a contrast drag costs no
 * round trip at all. That is the difference from {@link ImageViewer}, which asks
 * the server to re-render the whole region on every interaction — fine on
 * loopback, unusable across a WAN.
 *
 * Default-exported so the route can `lazy()` it: deck.gl and luma.gl are by far
 * the largest thing the app depends on and no other page needs them.
 *
 * See biopb-tensor-server/docs/remote-viewer-tiles.md.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import { DETAIL_VIEW_ID, DetailView, VivViewer, getDefaultInitialViewState } from "@hms-dbmi/viv";
import {
  TensorAbortError,
  createTensorPixelSources,
  vivDtype,
  type TileInfo,
} from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import {
  contrastLimitsFrom,
  contrastSamples,
  dtypeContrastLimits,
  percentileBounds,
  pinnedNotice,
  samplesPerPixel,
  tileCacheSize,
  vivColor,
  vivSelection,
} from "../utils/vivUtils";

type PixelSources = Awaited<ReturnType<typeof createTensorPixelSources>>["data"];

interface TileViewerProps {
  sourceId: string;
  /** The tensor's whole address; `source_id` for a single-tensor source. */
  arrayId: string;
  /**
   * The tensor cannot be shown this way — no tile route, an unsupported dtype,
   * a non-canonical axis order. The caller falls back to the rendered-image
   * viewer rather than leaving the pane empty.
   */
  onUnsupported: (reason: string) => void;
}

/** Slice navigation: hold one of these and scroll. Matches {@link ImageViewer}. */
const SLICE_KEYS = ["t", "z", "c"] as const;
const SLICE_WHEEL_QUIET_MS = 120;

export default function TileViewer({ sourceId, arrayId, onUnsupported }: TileViewerProps) {
  const client = useAppStore((s) => s.client);
  const slice = useAppStore((s) => s.slice);
  const channelNames = useAppStore((s) => s.channelNames);
  const channelColors = useAppStore((s) => s.channelColors);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const size = useElementSize(hostRef);

  const [loaded, setLoaded] = useState<{ sources: PixelSources; info: TileInfo } | null>(null);
  const [tileError, setTileError] = useState<string | null>(null);

  // Report upward through a ref: onUnsupported comes from the parent's render,
  // and listing it as a dependency would re-run the whole load on every parent
  // re-render (i.e. on every slider move).
  const onUnsupportedRef = useRef(onUnsupported);
  onUnsupportedRef.current = onUnsupported;

  // --- pixel sources ------------------------------------------------------
  useEffect(() => {
    if (!client) return;
    const controller = new AbortController();
    let live = true;
    setLoaded(null);
    setTileError(null);
    createTensorPixelSources(client.http, arrayId, {
      signal: controller.signal,
      onTileError: (err) => {
        if (live) setTileError(err.message);
      },
    })
      .then(({ data, info }) => {
        if (live) setLoaded({ sources: data, info });
      })
      .catch((err: unknown) => {
        if (!live || err instanceof TensorAbortError) return;
        onUnsupportedRef.current(err instanceof Error ? err.message : String(err));
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [client, arrayId]);

  const info = loaded?.info ?? null;
  // Identity has to follow content, not the store write that produced it. Viv's
  // ImageLayer refetches whenever `selections` is a new *reference*, and zustand
  // hands out a fresh slice object on every `setSlice` — so deriving the
  // selection straight from `slice` makes a contrast drag refetch the overview.
  const selectionKey = useMemo(
    () => (info ? JSON.stringify(vivSelection(info, slice)) : ""),
    [info, slice],
  );
  const selection = useMemo<Record<string, number> | null>(
    () => (selectionKey ? (JSON.parse(selectionKey) as Record<string, number>) : null),
    [selectionKey],
  );

  // --- is what is on screen the plane that was asked for? ------------------
  // Both Viv layers keep their previous raster until a new read resolves, so a
  // t/c/z change leaves the old plane painted for exactly as long as the read
  // takes -- with nothing on screen to say so. That is worse than a blank
  // frame: a stale plane is indistinguishable from the right one, and a plane
  // that never changed reads as a hung viewer rather than a slow one.
  //
  // deck.gl's TileLayer reports when the viewport's tiles have all landed.
  // Reached through Viv, which forwards unknown props down to it and pins the
  // background ImageLayer's own callback to null, so this fires once per
  // completed viewport and not twice.
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  // Read through a ref: the callback's identity has to stay stable or every
  // layerProps rebuild would look like a prop change to deck.gl.
  const selectionKeyRef = useRef(selectionKey);
  selectionKeyRef.current = selectionKey;
  const onViewportLoad = useCallback(() => setLoadedKey(selectionKeyRef.current), []);
  // Zoom and pan never invalidate: they change which tiles are wanted, not
  // which plane, so their partial state is legitimate progressive refinement.
  const dataValid = loadedKey !== null && loadedKey === selectionKey;

  // --- contrast limits ----------------------------------------------------
  // Read the coarsest level once per selection and keep the sorted samples, so
  // the intensity slider re-derives limits locally instead of refetching.
  const [samples, setSamples] = useState<Float64Array | null>(null);
  useEffect(() => {
    if (!loaded || !selection) return;
    const { sources, info: grid } = loaded;
    // Interleaved RGB is rendered as colour, not through a contrast ramp.
    if (grid.plane.s !== null) return;
    const overview = sources[sources.length - 1];
    if (!overview) return;
    const controller = new AbortController();
    let live = true;
    overview
      .getRaster({ selection, signal: controller.signal })
      .then((raster) => {
        if (live) setSamples(contrastSamples(raster.data));
      })
      .catch(() => {
        // Keep the previous limits: a failed histogram is a worse reason to
        // blank the image than to show it with slightly stale contrast.
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [loaded, selection]);

  const contrastLimits = useMemo<[number, number]>(() => {
    if (!info) return [0, 1];
    const [lo, hi] = percentileBounds(slice.useMinMax, slice.percentileScale);
    if (!samples) return dtypeContrastLimits(vivDtype(info.dtype));
    return contrastLimitsFrom(samples, lo, hi);
  }, [info, samples, slice.useMinMax, slice.percentileScale]);

  // --- colour -------------------------------------------------------------
  const color = useMemo(() => {
    const stored = channelColors[sourceId]?.[slice.c] ?? "auto";
    // `channelNames` is filled asynchronously, so this runs at least once with
    // the name still unknown. `resolveAutoColor` answers grey then and grey
    // again once an unrecognised name lands, which is what keeps the first
    // frames from being a different colour than the settled one.
    return vivColor(stored, channelNames[sourceId]?.[slice.c]);
  }, [channelColors, channelNames, sourceId, slice.c]);

  const maxCacheSize = useMemo(
    () =>
      info
        ? tileCacheSize(info.tile_size, vivDtype(info.dtype), samplesPerPixel(info), 1)
        : 0,
    [info],
  );

  useSliceWheelNavigation(hostRef, info);

  const pinned = info ? pinnedNotice(info) : null;

  return (
    <div
      ref={hostRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "#1a1a2e",
      }}
    >
      {loaded && selection && size ? (
        <VivStage
          sources={loaded.sources}
          selection={selection}
          contrastLimits={contrastLimits}
          color={color}
          maxCacheSize={maxCacheSize}
          onViewportLoad={onViewportLoad}
          width={size.width}
          height={size.height}
        />
      ) : (
        <div style={OVERLAY_TEXT}>Loading tiles…</div>
      )}
      {loaded && selection && size && !dataValid && (
        // Opaque, not a scrim: the point is that the stale plane stops being
        // visible, which a translucent overlay would not achieve.
        <div style={{ ...OVERLAY_TEXT, background: "#1a1a2e", zIndex: 1 }}>Reading plane…</div>
      )}
      {pinned && <div style={{ ...BADGE, bottom: 10, left: 10 }}>{pinned}</div>}
      {tileError && (
        <div style={{ ...BADGE, bottom: 10, right: 10, color: "#ff6b6b" }}>{tileError}</div>
      )}
    </div>
  );
}

/**
 * The deck.gl half, mounted only once both the sources and the pane size exist.
 *
 * Separate because the initial view state must be computed exactly once: it is
 * derived from the pane size, and recomputing it on a resize would snap the
 * user's pan and zoom back to fit. Mounting with both values already known makes
 * "once" the natural thing to write.
 */
function VivStage({
  sources,
  selection,
  contrastLimits,
  color,
  maxCacheSize,
  onViewportLoad,
  width,
  height,
}: {
  sources: PixelSources;
  selection: Record<string, number>;
  contrastLimits: [number, number];
  color: [number, number, number];
  maxCacheSize: number;
  onViewportLoad: () => void;
  width: number;
  height: number;
}) {
  const sizeRef = useRef({ width, height });
  const viewStates = useMemo(
    () => [
      {
        ...getDefaultInitialViewState(sources, sizeRef.current, 0.5),
        id: DETAIL_VIEW_ID,
      },
    ],
    // Deliberately not [width, height]: a resize must move the viewport, not
    // reset it. VivViewer carries the current pan/zoom through the new size.
    [sources],
  );

  const views = useMemo(
    () => [new DetailView({ id: DETAIL_VIEW_ID, height, width })],
    [height, width],
  );

  // Its own memo: this array's identity is what Viv's ImageLayer diffs on, so it
  // must survive a contrast or colour change untouched.
  const selections = useMemo(() => [selection], [selection]);

  const layerProps = useMemo(
    () => [
      {
        loader: sources,
        selections,
        contrastLimits: [contrastLimits],
        colors: [color],
        channelsVisible: [true],
        // Reaches deck.gl's TileLayer: DetailView spreads these into the
        // MultiscaleImageLayer, which spreads its own props into the TileLayer.
        maxCacheSize,
        onViewportLoad,
      },
    ],
    [sources, selections, contrastLimits, color, maxCacheSize, onViewportLoad],
  );

  return <VivViewer views={views} layerProps={layerProps} viewStates={viewStates} />;
}

/** The pane's pixel size, or null before the first measurement. */
function useElementSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const { clientWidth, clientHeight } = el;
      if (clientWidth > 0 && clientHeight > 0) {
        setSize((prev) =>
          prev && prev.width === clientWidth && prev.height === clientHeight
            ? prev
            : { width: clientWidth, height: clientHeight },
        );
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

/**
 * Hold t/z/c and scroll to step that axis, as the rendered-image viewer does.
 *
 * Capture phase on the pane: deck.gl listens on the canvas below, so stopping
 * the event here is what keeps a slice scroll from also zooming. The whole
 * gesture is accumulated and applied once it stops, rather than one store write
 * per wheel notch — each write costs a tile refetch.
 */
function useSliceWheelNavigation(
  ref: RefObject<HTMLElement | null>,
  info: TileInfo | null,
) {
  useEffect(() => {
    const el = ref.current;
    if (!el || !info) return;

    const held = new Set<string>();
    const pending = { axis: null as (typeof SLICE_KEYS)[number] | null, steps: 0 };
    let timer: ReturnType<typeof setTimeout> | null = null;

    const flush = () => {
      timer = null;
      const { axis, steps } = pending;
      pending.axis = null;
      pending.steps = 0;
      const wireAxis = axis === null ? null : info.selectable[axis];
      if (axis === null || wireAxis === null || steps === 0) return;
      const max = Math.max(0, (info.shape[wireAxis] ?? 1) - 1);
      const current = useAppStore.getState().slice[axis];
      useAppStore
        .getState()
        .setSlice({ [axis]: Math.min(max, Math.max(0, current + steps)) });
    };

    const onWheel = (e: WheelEvent) => {
      const axis = SLICE_KEYS.find((k) => held.has(k));
      if (!axis || info.selectable[axis] === null) return;
      e.preventDefault();
      e.stopPropagation();
      if (pending.axis !== axis) {
        pending.axis = axis;
        pending.steps = 0;
      }
      pending.steps += e.deltaY > 0 ? -1 : 1;
      if (timer) clearTimeout(timer);
      timer = setTimeout(flush, SLICE_WHEEL_QUIET_MS);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if ((SLICE_KEYS as readonly string[]).includes(key)) held.add(key);
    };
    const onKeyUp = (e: KeyboardEvent) => held.delete(e.key.toLowerCase());
    const onBlur = () => held.clear();

    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      el.removeEventListener("wheel", onWheel, { capture: true });
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      if (timer) clearTimeout(timer);
    };
  }, [ref, info]);
}

const OVERLAY_TEXT: CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "#94a3b8",
  fontSize: 13,
};

const BADGE: CSSProperties = {
  position: "absolute",
  padding: "4px 8px",
  borderRadius: 4,
  background: "rgba(0, 0, 0, 0.65)",
  color: "#cbd5e1",
  fontSize: 11,
  pointerEvents: "none",
  zIndex: 2,
};
