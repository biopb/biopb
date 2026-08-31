/**
 * Store state -> a 3-D volume read, and the geometry that renders it.
 *
 * The arithmetic behind {@link VolumeViewer}, kept out of the component so it
 * can be tested without a WebGL context.
 *
 * The scale is deliberately absent from all of this. It is the server's, from
 * `TileInfo.volume` — the one scale it keeps a whole volume warm at. A client
 * that computed its own would have to reimplement the server's pyramid planner,
 * and landing one rung away misses every warmed chunk and pays a cold decode of
 * the source. See biopb-tensor-server/docs/precache-policy.md §5.
 */

import { sliderAxes } from "@biopb/tensor-flight-client";
import type { SliceRequest, TileInfo, VolumeAvailable } from "@biopb/tensor-flight-client";
import type { SliceIndices } from "./vivUtils";

/**
 * How the ray-cast combines the voxels it passes through.
 *
 * Viv's own three (`RENDERING_MODES` in `@vivjs/constants`), keyed short because
 * these are button labels and a store field, not display strings. The mechanism
 * is the layer's `extensions` array — `XR3DLayer` reads `_BEFORE_RENDER` /
 * `_RENDER` / `_AFTER_RENDER` off it and rebuilds the shader; its
 * `renderingMode` prop only triggers that rebuild and decides nothing.
 *
 * MIP first because it is the right default for most of what this serves.
 * Fluorescence is sparse signal on a dark ground: the brightest voxel along a
 * ray *is* the structure, and it survives the projection at full contrast.
 * Additive sums the whole ray instead, so depth accumulates into a haze that
 * washes out exactly the thin structures a z-stack was acquired to show, and it
 * makes brightness a function of stack depth rather than of the sample.
 */
export const VOLUME_RENDER_MODES = [
  { key: "mip", label: "MIP", title: "Maximum intensity projection" },
  { key: "additive", label: "Add", title: "Additive blending" },
  // Niche but not free to rebuild later: transmitted-light and absorbance
  // stacks are dark signal on a bright ground, where max picks the background.
  { key: "minip", label: "MinIP", title: "Minimum intensity projection" },
] as const;

export type VolumeRenderMode = (typeof VOLUME_RENDER_MODES)[number]["key"];

export const DEFAULT_VOLUME_RENDER_MODE: VolumeRenderMode = "mip";

/**
 * Wire bytes this viewer will accept in one volume.
 *
 * Bounds the *download and the transient copy*: the body lands as an
 * ArrayBuffer, is viewed as its own dtype and then copied into a Float32Array,
 * so a float64 volume at the server's voxel budget would be ~720 MiB on the
 * wire and 1.4 GiB of live JS heap before the upload. That kills the tab.
 *
 * This used to be the only size check here, on the reasoning that the server
 * already bounded the voxel count. It does not always: a source shipping a
 * native pyramid has that ladder advertised *instead of* the computed plan, so
 * one that downsamples only Y/X leaves a full-depth volume with no 3-D budget
 * applied (biopb/biopb#891). Bytes are a poor proxy for that — at uint8 a
 * volume can be six times the voxel budget and still land under this cap — so
 * {@link VOLUME_MAX_VOXELS} checks the quantity that actually decides VRAM.
 */
export const VOLUME_MAX_BYTES = 512 * 1024 * 1024;

/** Bytes per voxel once uploaded: Viv casts every volume to Float32. */
const GPU_BYTES_PER_VOXEL = 4;

/**
 * Voxels this viewer will upload, expressed as a VRAM ceiling.
 *
 * Viv casts to Float32 regardless of source dtype, so VRAM follows the voxel
 * count alone and nothing else about the volume changes it. Stated as a memory
 * budget rather than as the server's 448³ so it is this viewer's own limit
 * rather than a copy of a server constant — 448³ is 343 MiB at this width, so
 * the ceiling clears the server's plan with headroom while still refusing the
 * unbudgeted ones, which start at ~1.5 GiB.
 */
export const VOLUME_MAX_VRAM_BYTES = 512 * 1024 * 1024;
export const VOLUME_MAX_VOXELS = VOLUME_MAX_VRAM_BYTES / GPU_BYTES_PER_VOXEL;

/** The volume plan, if this tensor has one this viewer will render. */
export function volumeRefusal(info: TileInfo): string | null {
  const volume = info.volume;
  // Absent: a server predating the field. Same outcome as unavailable, and the
  // reason has to be ours because that server has none to give.
  if (!volume) return "this server does not serve volumes";
  if (!volume.available) return volume.reason;
  if (volume.bytes > VOLUME_MAX_BYTES) {
    return (
      `the volume is ${(volume.bytes / 1024 / 1024).toFixed(0)} MB at the ` +
      `server's coarsest scale, over this viewer's ${VOLUME_MAX_BYTES / 1024 / 1024} MB limit`
    );
  }
  const voxels = volume.depth * volume.height * volume.width;
  if (voxels > VOLUME_MAX_VOXELS) {
    return (
      `the volume is ${voxels.toLocaleString()} voxels at the server's coarsest ` +
      `scale — ${((voxels * GPU_BYTES_PER_VOXEL) / 1024 / 1024).toFixed(0)} MB of GPU ` +
      `memory once uploaded, over this viewer's ${VOLUME_MAX_VRAM_BYTES / 1024 / 1024} MB limit`
    );
  }
  return null;
}

/**
 * The read for one volume: whole in z/y/x, pinned on every other axis.
 *
 * `slice_start`/`slice_stop` are full-resolution world coordinates — the units
 * `slice_hint` is applied in, before the server's scale — so the z/y/x span is
 * the tensor's own extent, not the volume's.
 *
 * Indices are clamped for the same reason {@link vivSelection} clamps them: the
 * store carries a slice position across a source change, and an out-of-range
 * index is a 422 rather than a nearby plane.
 */
export function volumeRequest(
  info: TileInfo,
  volume: VolumeAvailable,
  slice: SliceIndices,
): SliceRequest {
  const whole = new Set([volume.axes.z, volume.axes.y, volume.axes.x]);
  const pinned = new Map<number, number>();
  for (const axis of sliderAxes(info.dim_labels, info.shape)) {
    if (whole.has(axis.axis)) continue;
    const want = axis.named ? slice[axis.named] : slice.axes[axis.key] ?? 0;
    pinned.set(axis.axis, Math.min(Math.max(0, want), Math.max(0, axis.extent - 1)));
  }
  const slice_start: number[] = [];
  const slice_stop: number[] = [];
  info.shape.forEach((extent, i) => {
    const at = pinned.get(i);
    // An axis that is neither a volume axis nor a slider axis has extent 1 by
    // construction (`sliderAxes` drops nothing else), so 0 is its only index.
    slice_start.push(whole.has(i) ? 0 : at ?? 0);
    slice_stop.push(whole.has(i) ? extent : (at ?? 0) + 1);
  });
  return { array_id: info.array_id, slice_start, slice_stop, scale_policy: "volume" };
}

/**
 * A key that changes exactly when the volume's *pixels* would.
 *
 * The fetch is seconds and hundreds of megabytes, so it must not re-run on a
 * contrast drag or a colour change. Deriving the key from the request rather
 * than from the store slice is what guarantees that: two store states that
 * produce the same read produce the same key.
 */
export function volumeKey(req: SliceRequest): string {
  return `${req.array_id}|${(req.slice_start ?? []).join(",")}|${(req.slice_stop ?? []).join(",")}`;
}

/**
 * Per-axis scaling that makes an anisotropic volume render at its true shape.
 *
 * Returned as deck.gl's `[x, y, z]`, normalised so the finest axis is 1 — the
 * same convention as Viv's own `getPhysicalSizeScalingMatrix`, which scales by
 * `size / min(sizes)`. Normalising rather than using the physical numbers
 * directly keeps the model in voxel-ish units, so the camera distance a
 * `getDefaultInitialViewState`-style fit computes stays meaningful.
 *
 * Isotropic (`[1, 1, 1]`) when the source declares no physical size. That is a
 * real limitation, not a default worth hiding: a confocal stack with 0.1 µm
 * pixels and 0.5 µm z-steps renders five times too thin, and the only fix is
 * for the source to carry its physical scale.
 */
export function volumeScaleRatio(volume: VolumeAvailable): [number, number, number] {
  const { spacing } = volume;
  if (!spacing) return [1, 1, 1];
  const { x, y, z } = spacing;
  if (!(x > 0) || !(y > 0) || !(z > 0)) return [1, 1, 1];
  const min = Math.min(x, y, z);
  return [x / min, y / min, z / min];
}

/**
 * Where the camera looks: the centre of the volume, in the model's own units.
 *
 * `volumeScaleRatio` is applied here rather than left to the layer's model
 * matrix because deck.gl's `OrbitView` target is in world space — an
 * unscaled centre would put the camera off-axis on any anisotropic stack.
 */
export function volumeCentre(volume: VolumeAvailable): [number, number, number] {
  const [sx, sy, sz] = volumeScaleRatio(volume);
  return [(volume.width * sx) / 2, (volume.height * sy) / 2, (volume.depth * sz) / 2];
}

/**
 * Camera distance that fits the volume in view.
 *
 * `OrbitView`'s zoom is log2 of world-units-per-pixel, like the orthographic
 * views: at zoom `z` one world unit is `2**z` pixels. Fitting the volume's
 * longest scaled diagonal-ish extent into the smaller of the pane's dimensions
 * is what "the whole thing is visible from any orbit angle" means, and the
 * backoff leaves the corners inside the frame as it rotates.
 */
export function volumeZoom(
  volume: VolumeAvailable,
  view: { width: number; height: number },
  backoff = 0.5,
): number {
  const [sx, sy, sz] = volumeScaleRatio(volume);
  const extent = Math.max(volume.width * sx, volume.height * sy, volume.depth * sz);
  if (!(extent > 0) || !(view.width > 0) || !(view.height > 0)) return 0;
  return Math.log2(Math.min(view.width, view.height) / extent) - backoff;
}
