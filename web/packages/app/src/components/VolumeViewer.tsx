"use client";

/**
 * Client-side 3-D rendering: Viv's `XR3DLayer` over one server-scaled volume.
 *
 * This leaves the tile ladder entirely. `XR3DLayer` has no `loader` prop — it
 * takes a single 3-D texture — so there is nothing to tile, nothing to zoom
 * between, and Viv's dyadic/isotropic level constraints do not apply. What it
 * needs is the one scale the server keeps a whole volume warm at, which is why
 * the read delegates that decision (`scale_policy: "volume"`) rather than
 * naming a `scale_hint`.
 *
 * `VolumeViewer`/`VolumeLayer`, Viv's packaged 3-D path, is deliberately not
 * used: it reads `loader[resolution]` and issues `Z / 2**resolution` separate
 * raster requests, decimating Z in the browser. Driven directly, this consumes
 * the same coarsest level napari's 3-D mode reads, in one request.
 *
 * Default-exported so the route can `lazy()` it, for the same reason
 * {@link TileViewer} is.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { OrbitView } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { Matrix4 } from "@math.gl/core";
import { XR3DLayer } from "@hms-dbmi/viv";
import {
  TensorAbortError,
  asTypedArray,
  vivDtype,
  type TileInfo,
  type VolumeAvailable,
} from "@biopb/tensor-flight-client";
import { useShallow } from "zustand/react/shallow";
import { useCameraMirror } from "../hooks/useCameraMirror";
import { BADGE, OVERLAY_TEXT } from "./viewerStyles";
import { useElementSize } from "../hooks/useElementSize";
import { useMountEpoch, usePublishPlaneReady } from "../hooks/useMountEpoch";
import { selectContrastWindow, useAppStore } from "../store";
import type { ViewerErrorKind } from "../store";
import { gamma3dExtensions } from "../utils/vivGamma";
import {
  clampGamma,
  contrastSamples,
  vivColor,
} from "../utils/vivUtils";
import { compositeChannels, compositeColors, compositeWindows } from "../utils/composite";
import {
  compositeVolumeFits,
  volumeCentre,
  volumeKey,
  volumeRefusal,
  volumeRequest,
  volumeScaleRatio,
  volumeZoom,
  type VolumeRenderMode,
} from "../utils/volumeUtils";

interface VolumeViewerProps {
  sourceId: string;
  /** The resolved `tile_info`, fetched once by `openTensor`. */
  info: TileInfo;
  /** Same contract as {@link TileViewer}: a settled fact vs. a bad moment. */
  onUnsupported: (reason: string, kind: ViewerErrorKind) => void;
}

/**
 * One frozen array per mode, built once at module scope.
 *
 * Both halves matter. deck.gl treats a *new* `extensions` array as an extension
 * change and rebuilds the shader, so building these per render would recompile
 * on every slider move — the same reason {@link TileViewer} hoists its
 * `VIV_EXTENSIONS`. And switching mode is *supposed* to recompile, which is
 * exactly what a different frozen array gets for free.
 *
 * The extension is the mechanism, not `XR3DLayer.renderingMode`: the layer
 * reads `_BEFORE_RENDER` / `_RENDER` / `_AFTER_RENDER` off `extensions`
 * (`getRenderingFromExtensions`) and throws if nothing defines them.
 */
const VOLUME_EXTENSIONS: Record<VolumeRenderMode, unknown[]> = {
  mip: gamma3dExtensions("mip"),
  additive: gamma3dExtensions("additive"),
  minip: gamma3dExtensions("minip"),
};

/** One entry, so `XR3DLayer.getNumChannels()` compiles a single-channel shader. */
const ONE_CHANNEL = [{}];

const VOLUME_VIEW_ID = "volume";

/** The window before there is a grid to derive one from. */
const FALLBACK_WINDOW: [number, number] = [0, 1];

type XR3DLayerProps = ConstructorParameters<typeof XR3DLayer>[0];

export default function VolumeViewer({ sourceId, info: tileInfo, onUnsupported }: VolumeViewerProps) {
  const client = useAppStore((s) => s.client);
  const position = useAppStore((s) => s.position);
  const epoch = useMountEpoch();
  const notePlaneSamples = useAppStore((s) => s.notePlaneSamples);
  const composite = useAppStore((s) => s.composite);
  // This source's own entries, not the whole maps: an edit on another source
  // must not rebuild this one's layer (the same split TileViewer makes).
  const names = useAppStore((s) => s.channelNames[sourceId]);
  const sourceColors = useAppStore((s) => s.channelColors[sourceId]);
  const renderMode = useAppStore((s) => s.volumeRenderMode);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const size = useElementSize(hostRef);

  // Reported through a ref for the reason TileViewer does it: this comes from
  // the parent's render, and depending on it would re-run the check on every
  // slider move.
  const onUnsupportedRef = useRef(onUnsupported);
  onUnsupportedRef.current = onUnsupported;

  // --- the tensor's volume plan -------------------------------------------
  // A fact about the tensor, not a bad moment: no z axis and an oversized
  // volume both fail the same way on a retry. A refused volume is not drawn,
  // but its grid stays in the store for the sliders.
  const refusal = useMemo(() => volumeRefusal(tileInfo), [tileInfo]);
  useEffect(() => {
    if (refusal !== null) onUnsupportedRef.current(refusal, "capability");
  }, [refusal]);
  const info = refusal === null ? tileInfo : null;

  // `volumeRefusal` already established this is the available branch; the cast
  // is what lets the rest of the component read the plan without re-narrowing.
  const plan = (info?.volume ?? null) as VolumeAvailable | null;

  // --- which channels are drawn -------------------------------------------
  // A composite draws each channel as its own volume. It applies in the same
  // cases as the 2-D path (`compositeChannels`), and then only if the channels
  // fit in VRAM together (`compositeVolumeFits`): N channels multiply a read
  // already up to a gigabyte. Over budget it falls back to the current channel
  // alone, so 3-D still renders rather than refusing.
  const wantedChannels = useMemo(
    () => (info ? compositeChannels(info, composite) : null),
    [info, composite],
  );
  const channels = useMemo(
    () =>
      wantedChannels && plan && compositeVolumeFits(plan, wantedChannels.length)
        ? wantedChannels
        : null,
    [wantedChannels, plan],
  );
  // The channels actually read: the composite set, or the single current one.
  const readChannels = useMemo(() => channels ?? [position.c], [channels, position.c]);

  // --- the volume(s) -------------------------------------------------------
  // Keyed on the *requests*, so a contrast drag or a colour change — neither of
  // which alters a byte of them — cannot re-issue a read of hundreds of MB. A
  // composite's selections override `c`, so a C move leaves the key unchanged.
  const requests = useMemo(
    () =>
      info && plan ? readChannels.map((c) => volumeRequest(info, plan, { ...position, c })) : null,
    [info, plan, position, readChannels],
  );
  const requestKey = requests ? requests.map(volumeKey).join("~") : "";

  type VolumeData = ReturnType<typeof asTypedArray>;
  const [volume, setVolume] = useState<{ key: string; data: VolumeData[] } | null>(null);
  const [readError, setReadError] = useState<string | null>(null);

  useEffect(() => {
    if (!client || !info || !requests) return;
    const controller = new AbortController();
    let live = true;
    setReadError(null);
    Promise.all(
      requests.map((req) =>
        client.http
          // Not the tile budget: a volume is up to three orders of magnitude
          // larger, and the deadline covers the body download as well as the read.
          .slice(req, { signal: controller.signal, timeoutMs: client.http.volumeTimeoutMs })
          .then((arr) => asTypedArray(arr.buffer, vivDtype(arr.dtype))),
      ),
    )
      .then((datas) => {
        if (!live) return;
        // Each channel is windowed on its own samples, noted under its own index
        // so `channelSamples[c]` feeds `compositeWindows` the way the 2-D path
        // does. Sampled from the volume itself: this is already the coarsest
        // level, so a strided subsample costs nothing beyond the sort.
        datas.forEach((data, i) =>
          notePlaneSamples({ plane: data, values: contrastSamples(data) }, readChannels[i]!, epoch),
        );
        setVolume({ key: requestKey, data: datas });
      })
      .catch((err: unknown) => {
        if (!live || err instanceof TensorAbortError) return;
        setReadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      live = false;
      controller.abort();
    };
    // `requestKey` and not `requests`: the object identity changes on every
    // store write, its content only when the pixels would.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, info, requestKey, epoch, notePlaneSamples]);

  const current = volume && volume.key === requestKey ? volume.data : null;

  // Published for the play driver; see TileViewer.
  const playing = useAppStore((s) => s.playAxis !== null);
  usePublishPlaneReady(current !== null, epoch);

  // Under play, keep the last volume on the canvas while the next read is in
  // flight. Unmounting the stage between frames -- which is what a null here
  // does -- makes a scrub through T a strobe of empty panes. Only when the
  // channel count still matches, so toggling composite mid-play does not feed
  // the layer a stale volume of the wrong channel count.
  const shown =
    current ??
    (playing && volume && volume.data.length === readChannels.length ? volume.data : null);

  // --- contrast and colour -------------------------------------------------
  // The single-channel window: the same selector the panel's bar reads
  // (biopb/biopb#955). A composite replaces it with a per-channel window.
  const singleWindow = useAppStore(useShallow(selectContrastWindow)) ?? FALLBACK_WINDOW;
  const gamma = useAppStore((s) => clampGamma(s.display.gamma));
  const channelSamples = useAppStore((s) => (channels ? s.runtime.channelSamples : null));
  const percentileScale = useAppStore((s) => s.display.percentileScale);

  // A composite windows each channel on its own samples; a channel not sampled
  // yet is held back (hidden) rather than drawn through a guessed window.
  const windows = useMemo(
    () =>
      channels && channelSamples
        ? compositeWindows(channels, channelSamples, percentileScale)
        : null,
    [channels, channelSamples, percentileScale],
  );
  const contrastLimits = useMemo<[number, number][]>(
    () => (windows ? windows.map((w) => w ?? FALLBACK_WINDOW) : [singleWindow]),
    [windows, singleWindow],
  );
  const channelsVisible = useMemo(
    () => (windows ? windows.map((w) => w !== null) : [true]),
    [windows],
  );
  const colors = useMemo<[number, number, number][]>(() => {
    if (channels) {
      // `compositeColors` resolves every channel to a colour itself (names
      // included), so `vivColor` is given only the answer.
      return compositeColors(
        channels.map((c) => sourceColors?.[c] ?? "auto"),
        channels.map((c) => names?.[c]),
      ).map((color) => vivColor(color));
    }
    return [vivColor(sourceColors?.[position.c] ?? "auto", names?.[position.c])];
  }, [channels, sourceColors, names, position.c]);
  // One entry per drawn channel, so `XR3DLayer.getNumChannels` sizes the shader.
  const selections = useMemo(() => (channels ? channels.map(() => ({})) : ONE_CHANNEL), [channels]);

  return (
    <div ref={hostRef} style={HOST}>
      {info && plan && shown && size ? (
        <VolumeStage
          plan={plan}
          data={shown}
          dtype={vivDtype(info.dtype)}
          contrastLimits={contrastLimits}
          gamma={gamma}
          colors={colors}
          channelsVisible={channelsVisible}
          selections={selections}
          renderMode={renderMode}
          width={size.width}
          height={size.height}
        />
      ) : (
        <div style={OVERLAY_TEXT}>
          {readError
            ? `Volume unavailable — ${readError}`
            : plan
              ? `Reading volume (${((plan.bytes * readChannels.length) / 1024 / 1024).toFixed(0)} MB)…`
              : "Loading volume…"}
        </div>
      )}
      {plan && shown && size && (
        <div style={{ ...BADGE, bottom: 10, left: 10 }} title="The scale the server keeps this volume warm at.">
          {plan.width}×{plan.height}×{plan.depth} at 1/{plan.scale_hint[plan.axes.x]}
          {plan.spacing === null && " · isotropic (no physical scale)"}
        </div>
      )}
    </div>
  );
}

/**
 * The deck.gl half, mounted only once both the volume and the pane size exist.
 *
 * Separate for the reason `VivStage` is: the initial camera is derived from the
 * pane size and must be computed exactly once, or a resize would snap the
 * user's orbit back to the opening view.
 */
function VolumeStage({
  plan,
  data,
  dtype,
  contrastLimits,
  gamma,
  colors,
  channelsVisible,
  selections,
  renderMode,
  width,
  height,
}: {
  plan: VolumeAvailable;
  /** One volume per drawn channel, aligned with the arrays below. */
  data: ReturnType<typeof asTypedArray>[];
  dtype: ReturnType<typeof vivDtype>;
  contrastLimits: [number, number][];
  gamma: number;
  colors: [number, number, number][];
  channelsVisible: boolean[];
  selections: unknown[];
  renderMode: VolumeRenderMode;
  width: number;
  height: number;
}) {
  const sizeRef = useRef({ width, height });
  const mirrorCamera = useCameraMirror("3d");

  // The anisotropy, and the only place it enters: `XR3DLayer` scales its unit
  // cube by `physicalSizeScalingMatrix.transformPoint([w, h, d])`, so this
  // matrix is what makes a 0.5 µm z-step render half as thick as it is wide.
  const physicalSizeScalingMatrix = useMemo(
    () => new Matrix4().scale(volumeScaleRatio(plan)),
    [plan],
  );

  // Identity, deliberately. Viv's own VolumeLayer sets `scale(2 ** resolution)`
  // because its world is level-0 pixels and its texture is a decimated one;
  // here the world *is* this volume's voxels, so there is nothing to correct.
  const resolutionMatrix = useMemo(() => new Matrix4(), []);

  const initialViewState = useMemo(
    () => {
      // Read once per plan rather than subscribed. After mount deck.gl owns the
      // camera and the store only mirrors it, so a subscription here would feed
      // every orbit back into `initialViewState` -- which `Deck#setProps` treats
      // as an instruction to overwrite its own view state, i.e. a fight with the
      // user's pointer. Reading it is still right: a link that named a camera
      // has to open at it.
      const seed = useAppStore.getState().camera3d;
      if (seed) return seed;
      return {
        target: volumeCentre(plan),
        zoom: volumeZoom(plan, sizeRef.current),
        rotationX: 0,
        rotationOrbit: 0,
      };
    },
    // Deliberately not [width, height]: a resize must move the viewport, not
    // reset the camera.
    [plan],
  );

  const onViewStateChange = useCallback(
    ({ viewState }: { viewState: Record<string, unknown> }) => {
      const { target, zoom, rotationX, rotationOrbit } = viewState as {
        target: number[];
        zoom: number;
        rotationX: number;
        rotationOrbit: number;
      };
      const [x = 0, y = 0, z = 0] = target;
      mirrorCamera({ target: [x, y, z], zoom, rotationX, rotationOrbit });
      // Returns nothing on purpose: `Deck#_onViewStateChange` falls back to the
      // view state it already computed, so the camera stays deck.gl's to drive
      // and this stays a mirror.
    },
    [mirrorCamera],
  );

  const views = useMemo(
    // orbitAxis "Y" matches Viv's own VolumeView: the stack's depth stays
    // upright as the camera orbits, which is what makes a z-stack legible.
    () => new OrbitView({ id: VOLUME_VIEW_ID, controller: true, orbitAxis: "Y" }),
    [],
  );

  const layers = useMemo(
    () => [
      // Cast because Viv's `XR3DLayer` typedef is narrower than its
      // implementation: `selections` (which `getNumChannels` reads to size the
      // shader) and `physicalSizeScalingMatrix` (which `loadTexture`
      // dereferences unguarded, with no default) are both absent from the
      // declared props, and `id` is a plain deck.gl one the typedef drops.
      // Checked against @vivjs/layers@0.22.1 — this reaches over a gap in the
      // types, not over props the layer ignores.
      new XR3DLayer({
        // The channel count is in the id: `getNumChannels` sizes the shader from
        // `selections.length`, so a change there must build a new layer.
        id: `volume-${plan.width}-${plan.height}-${plan.depth}-${selections.length}`,
        channelData: {
          data,
          width: plan.width,
          height: plan.height,
          depth: plan.depth,
        },
        selections,
        dtype,
        contrastLimits,
        colors,
        channelsVisible,
        physicalSizeScalingMatrix,
        resolutionMatrix,
        extensions: VOLUME_EXTENSIONS[renderMode],
        // A prop of `Gamma3DExtension`, not of the layer: a uniform, so moving
        // it re-renders without reloading the texture.
        gamma,
      } as unknown as XR3DLayerProps),
    ],
    [
      plan,
      data,
      dtype,
      contrastLimits,
      gamma,
      colors,
      channelsVisible,
      selections,
      renderMode,
      physicalSizeScalingMatrix,
      resolutionMatrix,
    ],
  );

  return (
    <DeckGL
      views={views}
      layers={layers}
      initialViewState={initialViewState}
      onViewStateChange={onViewStateChange}
      width={width}
      height={height}
      // Black rather than the 2-D pane's slate: additive blending sums the
      // volume's intensity onto whatever is behind it, so a non-black ground
      // is added to every voxel and washes the whole render out.
      style={{ background: "#000" }}
    />
  );
}

const HOST: CSSProperties = {
  position: "relative",
  width: "100%",
  height: "100%",
  overflow: "hidden",
  background: "#000",
};
