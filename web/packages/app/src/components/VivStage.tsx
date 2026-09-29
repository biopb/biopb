/**
 * The deck.gl half of the tiled viewer, mounted only once both the sources and
 * the pane size exist.
 *
 * Separate because the initial view state must be computed exactly once: it is
 * derived from the pane size, and recomputing it on a resize would snap the
 * user's pan and zoom back to fit. Mounting with both values already known makes
 * "once" the natural thing to write.
 */

import { useCallback, useMemo, useRef } from "react";
import { OrthographicView } from "@deck.gl/core";
import {
  ColorPaletteExtension,
  DETAIL_VIEW_ID,
  DetailView,
  VivViewer,
  getDefaultInitialViewState,
} from "@hms-dbmi/viv";
import type { pixelSourcesFromInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { useCameraMirror } from "../hooks/useCameraMirror";
import { GammaExtension } from "../utils/vivGamma";

export type PixelSources = ReturnType<typeof pixelSourcesFromInfo>;

/** The subset of deck.gl's picking info the hover handler reads. */
export interface HoverInfo {
  coordinate?: number[];
  sourceLayer?: unknown;
  tile?: { index?: { z?: number } };
}

/**
 * Stops deck.gl holding every click for a third of a second.
 *
 * deck wires its click recognizer as `requireFailure(['dblclick'])`, and
 * mjolnir answers that by deferring the emit by the recognizer's `interval`
 * (300 ms by default) -- then cancelling that pending emit outright if another
 * press arrives first, because `TapRecognizer.process` opens with a `reset()`
 * that clears the timer and overwrites the input it would have reported. Three
 * vertices placed a tenth of a second apart therefore arrive as one click, at
 * the last position: the first two are cancelled, not queued.
 *
 * Zero here means the emit lands on the next task instead. Nothing else pushes
 * the recognizer through `process` in between -- mouse moves reach it only
 * while a button is down -- so the only thing that ever cancelled a click was
 * the next click. `dblclick` keeps its own recognizer and its own interval, so
 * double-click zoom is untouched.
 */
const CLICK_OPTIONS = { click: { interval: 0 } };

/**
 * Viv's detail view, with a say over whether a double click zooms.
 *
 * `VivView.getDeckGlView` hard-codes `controller: true`, which is deck's whole
 * default gesture set. That is the wrong set while a click places a vertex: the
 * two taps of a double click are two vertices, and zooming out from under them
 * on the same gesture moves everything already placed relative to what is on
 * screen. Suppressing the vertices instead would mean waiting to find out
 * whether a second tap is coming, which is the 300 ms {@link CLICK_OPTIONS}
 * exists to get rid of.
 *
 * Only the double click goes: scroll still zooms, and drag still pans, so the
 * gesture that is actually used to navigate while drawing is untouched.
 */
class BiopbDetailView extends DetailView {
  private readonly doubleClickZoom: boolean;

  constructor(props: {
    id: string;
    height: number;
    width: number;
    doubleClickZoom: boolean;
  }) {
    super(props);
    this.doubleClickZoom = props.doubleClickZoom;
  }

  getDeckGlView() {
    return new OrthographicView({
      controller: { doubleClickZoom: this.doubleClickZoom },
      id: this.id,
      height: this.height,
      width: this.width,
      x: this.x,
      y: this.y,
    });
  }
}

/**
 * Viv's default palette plus gamma. Module-level because deck.gl treats a change
 * of this array as a change of extensions, which rebuilds every layer's shader:
 * a fresh array per render would recompile on every slider move.
 *
 * ColorPaletteExtension has to be listed explicitly — naming `extensions` at all
 * replaces Viv's default rather than adding to it, and dropping it would leave
 * the channel colour unapplied.
 */
const VIV_EXTENSIONS = [new ColorPaletteExtension(), new GammaExtension()];

export function VivStage({
  sources,
  selection,
  contrastLimits,
  gamma,
  color,
  maxCacheSize,
  onViewportLoad,
  onHover,
  hoverHooks,
  overlayLayers,
  onDeckClick,
  width,
  height,
}: {
  sources: PixelSources;
  selection: Record<string, number>;
  contrastLimits: [number, number];
  gamma: number;
  color: [number, number, number];
  maxCacheSize: number;
  onViewportLoad: (loaded?: unknown) => void;
  onHover: (info: HoverInfo) => void;
  hoverHooks: { handleValue: (values: number[]) => void; handleCoordinate: () => void };
  /** Drawn over the image; see utils/roiLayers.ts for the id constraint. */
  overlayLayers: unknown[];
  /** Reaches DeckGL's root `onClick`, which Viv does not override. */
  onDeckClick: (info: { coordinate?: number[]; viewport?: { zoom?: number | number[] } }) => void;
  width: number;
  height: number;
}) {
  const sizeRef = useRef({ width, height });
  const mirrorCamera = useCameraMirror("2d");

  const viewStates = useMemo(
    () => {
      // Read once, not subscribed. `VivViewer.componentDidUpdate` diffs this
      // prop and overwrites its own view state when it differs, so a
      // subscription would push every pan back at the viewport mid-gesture.
      // Reading it is still right: a link that named a camera must open at it.
      const seed = useAppStore.getState().camera2d;
      const base = seed
        // The stored target is [x, y]; an orthographic view wants the z back.
        ? { target: [seed.target[0], seed.target[1], 0], zoom: seed.zoom }
        : getDefaultInitialViewState(sources, sizeRef.current, 0.5);
      return [{ ...base, id: DETAIL_VIEW_ID }];
    },
    // Deliberately not [width, height]: a resize must move the viewport, not
    // reset it. VivViewer carries the current pan/zoom through the new size.
    [sources],
  );

  const onViewStateChange = useCallback(
    // `object`, not a narrower shape: that is how Viv types the callback.
    ({ viewState }: { viewState: object }) => {
      const { target, zoom } = viewState as { target: number[]; zoom: number | number[] };
      const [x = 0, y = 0] = target ?? [];
      // deck.gl may report an orthographic zoom per axis; Viv's own initial
      // state is scalar, and the views here are never anisotropic.
      const z = Array.isArray(zoom) ? (zoom[0] ?? 0) : zoom;
      mirrorCamera({ target: [x, y], zoom: z });
      // Returns nothing on purpose: VivViewer falls back to the view state it
      // already computed (`onViewStateChange?.(...) || viewState`), so the
      // viewport stays Viv's to drive and this stays a mirror.
    },
    [mirrorCamera],
  );

  // Only while a click would place something. With the overlay off, with no
  // annotation support on the server, or with the select tool held, a double
  // click has nothing to collide with and keeps deck's zoom.
  const placing = useAppStore((s) => s.showRois && !s.roisUnavailable && s.tool !== "select");

  // A new instance, but the same id, height and width -- which is all
  // `VivViewer.getDerivedStateFromProps` looks at, so switching tools
  // reconfigures the controller without touching the camera.
  const views = useMemo(
    () => [
      new BiopbDetailView({
        id: DETAIL_VIEW_ID,
        height,
        width,
        doubleClickZoom: !placing,
      }),
    ],
    [height, width, placing],
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
        // Costs no fetch: gamma is a uniform, so moving it recolours the tiles
        // already on the GPU. Same for contrastLimits.
        extensions: VIV_EXTENSIONS,
        gamma,
        // Reaches deck.gl's TileLayer: DetailView spreads these into the
        // MultiscaleImageLayer, which spreads its own props into the TileLayer.
        maxCacheSize,
        onViewportLoad,
      },
    ],
    [sources, selections, contrastLimits, gamma, color, maxCacheSize, onViewportLoad],
  );

  // VivViewer concatenates these after its own layers -- but only draws the
  // ones whose id contains its view id, which is why they are built through
  // `roiLayerId`. Memoised on the array itself: `deckProps` is spread into
  // DeckGL, and a fresh object every render would be a prop change every frame.
  //
  // `onClick` rides the same object. Viv overrides `layerFilter`, `layers`,
  // `onViewStateChange`, `views`, `viewState`, `useDevicePixels` and `getCursor`
  // AFTER spreading deckProps -- but not the pointer callbacks, so this one
  // survives. The overlay is unpickable, so it arrives with no picked layer and
  // `coordinate` set, which is exactly what placing a vertex needs.
  const deckProps = useMemo(
    () => ({ layers: overlayLayers, onClick: onDeckClick, eventRecognizerOptions: CLICK_OPTIONS }),
    [overlayLayers, onDeckClick],
  );

  return (
    <VivViewer
      views={views}
      layerProps={layerProps}
      viewStates={viewStates}
      onViewStateChange={onViewStateChange}
      onHover={onHover}
      hoverHooks={hoverHooks}
      deckProps={deckProps}
    />
  );
}
