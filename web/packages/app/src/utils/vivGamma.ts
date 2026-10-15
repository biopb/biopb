/**
 * A gamma control for Viv's shader, as a deck.gl layer extension.
 *
 * Viv applies contrast in `DECKGL_PROCESS_INTENSITY`, a shader hook it registers
 * and `XRLayer` fills with a linear ramp *unless an extension claims it*.
 * Claiming it is therefore all-or-nothing: this extension has to re-state the
 * contrast ramp as well as the gamma curve, because the layer's own injection is
 * skipped once an extension defines the hook. `apply_contrast_limits` is still
 * in scope — it lives in the always-included `channelIntensity` module — so the
 * ramp is reused rather than reimplemented.
 *
 * Doing it at the hook, rather than after colouring, is what keeps hue stable as
 * gamma moves: the exponent lands on the *intensity* and only then multiplies by
 * the channel colour. Applying it to the final RGB instead would shift hue,
 * since pow(i*c) != pow(i)*c. This is now the only implementation of the curve —
 * the server-side one went with `renderer.py`.
 *
 * The 3-D layer is the other way round: it applies contrast itself and has no
 * per-sample hook worth claiming, so {@link gamma3dExtensions} bends the one
 * value the ray march ends on -- see there.
 *
 * Interleaved RGB is drawn by a `BitmapLayer`, which Viv constructs with
 * `extensions: []` — gamma does not reach it, exactly as contrast limits do not.
 */

import type { Layer, UpdateParameters } from "@deck.gl/core";
import { ColorPalette3DExtensions, VivLayerExtension } from "@hms-dbmi/viv";

/**
 * Names the uniform block, its GLSL accessor and the `shaderInputs` key — luma
 * derives all three from the module name, so they cannot be chosen separately.
 */
const MODULE = "gammaModule";

const gammaModule = {
  name: MODULE,
  uniformTypes: { gamma: "f32" },
  fs: `uniform ${MODULE}Uniforms {
  float gamma;
} ${MODULE};`,
  inject: {
    // Clamped before pow, not after: a negative base is undefined in GLSL, and
    // past 1 the curve would keep climbing where the colour step clamps anyway.
    "fs:DECKGL_PROCESS_INTENSITY": `
  intensity = apply_contrast_limits(intensity, contrastLimits);
  intensity = pow(clamp(intensity, 0.0, 1.0), ${MODULE}.gamma);
`,
  },
};

/** The prop this extension adds to whatever layer it is attached to. */
export interface GammaExtensionProps {
  gamma?: number;
}

export class GammaExtension extends VivLayerExtension {
  static override extensionName = "GammaExtension";

  // Declaring the prop here is what makes it a real layer prop: deck.gl merges
  // an extension's defaultProps into the layer's, so a change to `gamma` counts
  // as a prop change and reaches updateState. Undeclared, it would be dropped
  // before any layer saw it.
  static defaultProps = {
    gamma: { type: "number", value: 1, compare: true },
  };

  getVivShaderTemplates() {
    return { modules: [gammaModule] };
  }

  // deck.gl calls this with `this` bound to the layer, not to the extension.
  override updateState(
    this: Layer,
    params: UpdateParameters<Layer>,
    extension: GammaExtension,
  ) {
    // Not `super.updateState`: the base declares its `extension` parameter as
    // the polymorphic `this`, which a fixed subclass cannot satisfy. Reaching
    // the same implementation through the prototype keeps the call (a no-op
    // today, but the base is free to grow one) without the type gymnastics.
    VivLayerExtension.prototype.updateState.call(this, params, extension);
    pushGamma(this);
  }
}

function pushGamma(layer: Layer) {
  const { gamma = 1 } = layer.props as GammaExtensionProps;
  for (const model of layer.getModels()) {
    model.shaderInputs.setProps({ [MODULE]: { gamma } });
  }
}

// ---------------------------------------------------------------------------
// 3-D
// ---------------------------------------------------------------------------

type Rendering = { _BEFORE_RENDER: string; _RENDER: string; _AFTER_RENDER: string };

/** The projection modes' final step: the held max/min, clamped into the colormap. */
const PROJECTED = /min\(1\.0, ((?:max|min)Val\w*)\)/;

/**
 * A copy of a 3-D rendering with gamma applied to the value it ends on.
 *
 * Max and min projection pick one sample per ray, and gamma is monotonic, so
 * bending the picked value is the same as bending every sample first -- the
 * result is exact, and the curve costs one `pow` per pixel instead of one per
 * step. Additive blending sums the samples, which a curve does not commute
 * with, so there it bends the finished picture instead: an approximation,
 * applied to the brightest channel so the hue stays put (per-channel `pow`
 * would shift it, for the reason given at the top of this file).
 *
 * Rewrites Viv's shader text, so it does not guess: a Viv whose projection
 * step no longer matches is warned about and rendered without gamma, which
 * costs the slider and nothing else.
 */
export function withGamma(mode: "mip" | "minip" | "additive", rendering: Rendering): Rendering {
  const g = `${MODULE}.gamma`;
  if (mode === "additive") {
    return {
      ...rendering,
      _AFTER_RENDER: `${rendering._AFTER_RENDER}
  float peak = max(color.r, max(color.g, color.b));
  if (peak > 0.0) {
    color.rgb *= pow(min(peak, 1.0), ${g}) / peak;
  }
`,
    };
  }
  if (!PROJECTED.test(rendering._AFTER_RENDER)) {
    console.warn(`gamma: Viv's ${mode} projection no longer ends in a clamped max/min; rendering without gamma`);
    return rendering;
  }
  return {
    ...rendering,
    _AFTER_RENDER: rendering._AFTER_RENDER.replace(
      PROJECTED,
      (_, v: string) => `min(1.0, pow(clamp(${v}, 0.0, 1.0), ${g}))`,
    ),
  };
}

/**
 * Carries the gamma uniform into the 3-D layer and, being the last extension
 * with a `rendering`, is the one whose strings the layer compiles.
 *
 * Unlike {@link GammaExtension} it claims no `DECKGL_PROCESS_INTENSITY`: the 3-D
 * layer then keeps its own contrast ramp, which is what the samples must go
 * through before anything is picked.
 */
class Gamma3DExtension extends VivLayerExtension {
  static override extensionName = "Gamma3DExtension";
  static defaultProps = GammaExtension.defaultProps;

  rendering: Rendering;

  constructor(rendering: Rendering) {
    super();
    this.rendering = rendering;
  }

  getVivShaderTemplates() {
    return { modules: [{ name: MODULE, uniformTypes: gammaModule.uniformTypes, fs: gammaModule.fs }] };
  }

  override updateState(this: Layer, params: UpdateParameters<Layer>, extension: Gamma3DExtension) {
    VivLayerExtension.prototype.updateState.call(this, params, extension);
    pushGamma(this);
  }
}

/**
 * The extensions that render `mode` with gamma: Viv's own, then the one that
 * bends their result. Build once per mode at module scope -- a new array is a
 * recompile (see `VOLUME_EXTENSIONS`).
 */
export function gamma3dExtensions(mode: "mip" | "minip" | "additive"): unknown[] {
  const base = {
    mip: new ColorPalette3DExtensions.MaximumIntensityProjectionExtension({}),
    minip: new ColorPalette3DExtensions.MinimumIntensityProjectionExtension({}),
    additive: new ColorPalette3DExtensions.AdditiveBlendExtension({}),
  }[mode];
  return [base, new Gamma3DExtension(withGamma(mode, (base as unknown as { rendering: Rendering }).rendering))];
}
