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
 * Doing it at the hook, rather than after colouring, is what makes the tiled
 * viewer agree with the server-rendered one: both raise the *intensity* to the
 * exponent and only then multiply by the channel colour. Applying it to the
 * final RGB instead would shift hue as gamma moved, since pow(i*c) != pow(i)*c.
 *
 * Interleaved RGB is drawn by a `BitmapLayer`, which Viv constructs with
 * `extensions: []` — gamma does not reach it, exactly as contrast limits do not.
 */

import type { Layer, UpdateParameters } from "@deck.gl/core";
import { VivLayerExtension } from "@hms-dbmi/viv";

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
    const { gamma = 1 } = this.props as GammaExtensionProps;
    for (const model of this.getModels()) {
      model.shaderInputs.setProps({ [MODULE]: { gamma } });
    }
  }
}
