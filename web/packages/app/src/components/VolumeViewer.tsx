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
 * the same coarsest level napari's 3-D mode reads, in one request. See
 * biopb-tensor-server/docs/precache-policy.md §3.1.
 *
 * Default-exported so the route can `lazy()` it, for the same reason
 * {@link TileViewer} is.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import { OrbitView } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { Matrix4 } from "@math.gl/core";
import { ColorPalette3DExtensions, XR3DLayer } from "@hms-dbmi/viv";
import {
  TensorAbortError,
  asTypedArray,
  isTransportError,
  vivDtype,
  type TileInfo,
  type VolumeAvailable,
} from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import type { ViewerErrorKind } from "./ViewerPane";
import {
  clampContrastLimits,
  contrastLimitsFrom,
  contrastSamples,
  dtypeContrastLimits,
  percentileBounds,
  vivColor,
  CAMERA_MIRROR_MS,
} from "../utils/vivUtils";
import {
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
  arrayId: string;
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
  mip: [new ColorPalette3DExtensions.MaximumIntensityProjectionExtension({})],
  additive: [new ColorPalette3DExtensions.AdditiveBlendExtension({})],
  minip: [new ColorPalette3DExtensions.MinimumIntensityProjectionExtension({})],
};

/** One entry, so `XR3DLayer.getNumChannels()` compiles a single-channel shader. */
const ONE_CHANNEL = [{}];

const VOLUME_VIEW_ID = "volume";

type XR3DLayerProps = ConstructorParameters<typeof XR3DLayer>[0];

export default function VolumeViewer({ sourceId, arrayId, onUnsupported }: VolumeViewerProps) {
  const client = useAppStore((s) => s.client);
  const slice = useAppStore((s) => s.slice);
  const channelNames = useAppStore((s) => s.channelNames);
  const channelColors = useAppStore((s) => s.channelColors);
  const renderMode = useAppStore((s) => s.volumeRenderMode);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const size = useElementSize(hostRef);

  const [info, setInfo] = useState<TileInfo | null>(null);

  // Reported through a ref for the reason TileViewer does it: this comes from
  // the parent's render, and depending on it would re-run the load on every
  // slider move.
  const onUnsupportedRef = useRef(onUnsupported);
  onUnsupportedRef.current = onUnsupported;

  // --- the tensor's volume plan -------------------------------------------
  useEffect(() => {
    if (!client) return;
    const controller = new AbortController();
    let live = true;
    setInfo(null);
    client.http
      .tileInfo(arrayId, { signal: controller.signal })
      .then((loaded) => {
        if (!live) return;
        const refusal = volumeRefusal(loaded);
        if (refusal !== null) {
          // A fact about the tensor, not a bad moment: no z axis and an
          // oversized volume both fail the same way on a retry.
          onUnsupportedRef.current(refusal, "capability");
          return;
        }
        setInfo(loaded);
      })
      .catch((err: unknown) => {
        if (!live || err instanceof TensorAbortError) return;
        const message = err instanceof Error ? err.message : String(err);
        onUnsupportedRef.current(message, isTransportError(err) ? "transport" : "capability");
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [client, arrayId]);

  // Published for SliceControls, as in TileViewer. A refused volume never sets
  // `info`, so the sliders fall back to the catalog rather than being bounded by
  // a grid this viewer could not use.
  const setTileInfo = useAppStore((s) => s.setTileInfo);
  useEffect(() => {
    setTileInfo(info, arrayId);
  }, [info, arrayId, setTileInfo]);

  // `volumeRefusal` already established this is the available branch; the cast
  // is what lets the rest of the component read the plan without re-narrowing.
  const plan = (info?.volume ?? null) as VolumeAvailable | null;

  // --- the volume ----------------------------------------------------------
  // Keyed on the *request*, so a contrast drag or a colour change — neither of
  // which alters a byte of it — cannot re-issue a read of hundreds of MB.
  const request = useMemo(
    () => (info && plan ? volumeRequest(info, plan, slice) : null),
    [info, plan, slice],
  );
  const requestKey = request ? volumeKey(request) : "";

  type VolumeData = ReturnType<typeof asTypedArray>;
  const [volume, setVolume] = useState<{ key: string; data: VolumeData } | null>(null);
  const [readError, setReadError] = useState<string | null>(null);

  useEffect(() => {
    if (!client || !info || !request) return;
    const controller = new AbortController();
    let live = true;
    setReadError(null);
    client.http
      // Not the tile budget: a volume is up to three orders of magnitude
      // larger, and the deadline covers the body download as well as the read.
      .slice(request, {
        signal: controller.signal,
        timeoutMs: client.http.volumeTimeoutMs,
      })
      .then((arr) => {
        if (!live) return;
        setVolume({ key: requestKey, data: asTypedArray(arr.buffer, vivDtype(arr.dtype)) });
      })
      .catch((err: unknown) => {
        if (!live || err instanceof TensorAbortError) return;
        setReadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      live = false;
      controller.abort();
    };
    // `requestKey` and not `request`: the object identity changes on every
    // store write, its content only when the pixels would.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, info, requestKey]);

  const current = volume && volume.key === requestKey ? volume.data : null;

  // Published for the play driver; see TileViewer.
  const playing = useAppStore((s) => s.playAxis !== null);
  const setPlaneReady = useAppStore((s) => s.setPlaneReady);
  useEffect(() => {
    setPlaneReady(current !== null);
  }, [current, setPlaneReady]);

  // Under play, keep the last volume on the canvas while the next read is in
  // flight. Unmounting the stage between frames -- which is what a null here
  // does -- makes a scrub through T a strobe of empty panes.
  const shown = current ?? (playing ? (volume?.data ?? null) : null);

  // --- contrast ------------------------------------------------------------
  // Sampled from the volume itself. There is no coarser level to sample here
  // the way the tiled viewer samples its overview — this *is* the coarsest —
  // and the whole volume is already in memory, so a strided subsample of it
  // costs nothing beyond the sort.
  const samples = useMemo(() => (current ? contrastSamples(current) : null), [current]);

  const contrastLimits = useMemo<[number, number]>(() => {
    if (!info) return [0, 1];
    const range = dtypeContrastLimits(vivDtype(info.dtype));
    // A fixed window is the user's, not the plane's: it is not re-derived per
    // plane, only brought inside the dtype it is being applied to.
    if (slice.contrastMode === "fixed") {
      return slice.fixedLimits ? clampContrastLimits(slice.fixedLimits, range) : range;
    }
    if (!samples) return range;
    const [lo, hi] = percentileBounds(slice.percentileScale);
    return contrastLimitsFrom(samples, lo, hi);
  }, [info, samples, slice.contrastMode, slice.fixedLimits, slice.percentileScale]);

  // Published so the panel can seed a fixed window from what is on screen.
  const setAppliedLimits = useAppStore((s) => s.setAppliedLimits);
  useEffect(() => {
    setAppliedLimits(contrastLimits);
  }, [contrastLimits, setAppliedLimits]);

  // And the plane's own extremes, which the window no longer reports once it
  // is fixed. Deliberately not keyed to the current selection: the last plane
  // sampled is what Min/Max should reset onto, and holding it across a read
  // keeps the button from going dead for the length of one.
  const planeLimits = useMemo(() => samples ? contrastLimitsFrom(samples, 0, 100) : null, [samples]);
  const setPlaneLimits = useAppStore((s) => s.setPlaneLimits);
  useEffect(() => {
    if (planeLimits) setPlaneLimits(planeLimits);
  }, [planeLimits, setPlaneLimits]);

  const color = useMemo(() => {
    const stored = channelColors[sourceId]?.[slice.c] ?? "auto";
    return vivColor(stored, channelNames[sourceId]?.[slice.c]);
  }, [channelColors, channelNames, sourceId, slice.c]);

  return (
    <div ref={hostRef} style={HOST}>
      {info && plan && shown && size ? (
        <VolumeStage
          plan={plan}
          data={shown}
          dtype={vivDtype(info.dtype)}
          contrastLimits={contrastLimits}
          color={color}
          renderMode={renderMode}
          width={size.width}
          height={size.height}
        />
      ) : (
        <div style={OVERLAY_TEXT}>
          {readError
            ? `Volume unavailable — ${readError}`
            : plan
              ? `Reading volume (${(plan.bytes / 1024 / 1024).toFixed(0)} MB)…`
              : "Loading volume…"}
        </div>
      )}
      {plan && shown && size && (
        <div style={BADGE} title="The scale the server keeps this volume warm at.">
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
  color,
  renderMode,
  width,
  height,
}: {
  plan: VolumeAvailable;
  data: ReturnType<typeof asTypedArray>;
  dtype: ReturnType<typeof vivDtype>;
  contrastLimits: [number, number];
  color: [number, number, number];
  renderMode: VolumeRenderMode;
  width: number;
  height: number;
}) {
  const sizeRef = useRef({ width, height });
  const setCamera3d = useAppStore((s) => s.setCamera3d);

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

  // Trailing edge only: the resting camera is what a link should carry, and the
  // intermediate frames of a drag are noise that would otherwise reach the URL
  // at pointer rate.
  const mirrorRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (mirrorRef.current) clearTimeout(mirrorRef.current);
  }, []);

  const onViewStateChange = useCallback(
    ({ viewState }: { viewState: Record<string, unknown> }) => {
      if (mirrorRef.current) clearTimeout(mirrorRef.current);
      const { target, zoom, rotationX, rotationOrbit } = viewState as {
        target: number[];
        zoom: number;
        rotationX: number;
        rotationOrbit: number;
      };
      const [x = 0, y = 0, z = 0] = target;
      mirrorRef.current = setTimeout(() => {
        setCamera3d({ target: [x, y, z], zoom, rotationX, rotationOrbit });
      }, CAMERA_MIRROR_MS);
      // Returns nothing on purpose: `Deck#_onViewStateChange` falls back to the
      // view state it already computed, so the camera stays deck.gl's to drive
      // and this stays a mirror.
    },
    [setCamera3d],
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
        id: `volume-${plan.width}-${plan.height}-${plan.depth}`,
        channelData: {
          data: [data],
          width: plan.width,
          height: plan.height,
          depth: plan.depth,
        },
        selections: ONE_CHANNEL,
        dtype,
        contrastLimits: [contrastLimits],
        colors: [color],
        channelsVisible: [true],
        physicalSizeScalingMatrix,
        resolutionMatrix,
        extensions: VOLUME_EXTENSIONS[renderMode],
      } as unknown as XR3DLayerProps),
    ],
    [
      plan,
      data,
      dtype,
      contrastLimits,
      color,
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

const HOST: CSSProperties = {
  position: "relative",
  width: "100%",
  height: "100%",
  overflow: "hidden",
  background: "#000",
};

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
  bottom: 10,
  left: 10,
  padding: "4px 8px",
  borderRadius: 4,
  background: "rgba(0, 0, 0, 0.65)",
  color: "#cbd5e1",
  fontSize: 11,
  pointerEvents: "none",
  zIndex: 2,
};
