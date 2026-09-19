/**
 * Categorical colour for a label tensor, as a Viv layer extension.
 *
 * A label set holds instance *ids*, not intensities: 7 is not "brighter" than 6,
 * it is a different object. So the overlay replaces Viv's colour step entirely
 * rather than tinting a contrast ramp -- `DECKGL_MUTATE_COLOR` is the hook the
 * palette lives in, and naming this extension replaces the
 * `ColorPaletteExtension` the layer would otherwise default to.
 *
 * **The ramp is left in place, set to identity.** `XRLayer` injects its own
 * `DECKGL_PROCESS_INTENSITY` unless an extension claims the hook, and with
 * `contrastLimits = [0, 1]` that injection is `(id - 0) / 1` -- so the raw id
 * reaches this module without a second claim on the shader. See
 * {@link LABEL_CONTRAST_LIMITS}: it is the identity ramp, not a window.
 *
 * Sibling of `vivGamma.ts`, which claims the other hook; the two are never used
 * together, because one draws an image and the other draws labels.
 */

import type { Layer, UpdateParameters } from "@deck.gl/core";
import { VivLayerExtension } from "@hms-dbmi/viv";

/**
 * What the label layer must pass as its contrast window.
 *
 * Not a display choice: it is what turns Viv's ramp into the identity, so the
 * id the server stored is the id this palette colours. Anything else silently
 * renames every object.
 */
export const LABEL_CONTRAST_LIMITS: [number, number] = [0, 1];

/**
 * Names the uniform block, its GLSL accessor and the `shaderInputs` key -- luma
 * derives all three from the module name, so they cannot be chosen separately.
 */
const MODULE = "labelPaletteModule";

const labelPaletteModule = {
  name: MODULE,
  uniformTypes: { opacity: "f32" },
  fs: `uniform ${MODULE}Uniforms {
  float opacity;
} ${MODULE};

vec3 label_hsv_to_rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// Hue rotated by the golden-ratio conjugate, so consecutive ids -- which is
// exactly how a segmentation numbers its objects -- land a third of the circle
// apart instead of running a gradient. Saturation and value are rotated by two
// other irrationals, which separates ids that happen to collide in hue.
//
// A pure function of the id, so one object keeps its colour across tiles,
// pyramid levels and planes. Ids above 2^24 lose precision here (the shader
// reads the texture as a float), which shifts their colour but not their
// silhouette.
vec3 label_color(float id) {
  float h = fract(id * 0.6180339887498949);
  float s = 0.55 + 0.35 * fract(id * 0.7548776662466927);
  float v = 0.70 + 0.30 * fract(id * 0.5698402909980532);
  return label_hsv_to_rgb(vec3(h, s, v));
}
`,
  inject: {
    // 0 is background, and background is nothing -- fully transparent, not a
    // dark pixel, so the image underneath shows through unaltered. The compare
    // is against 0.5 rather than 0.0 because the id arrives as a float.
    "fs:DECKGL_MUTATE_COLOR": `
  float labelId = intensity[0];
  rgba = labelId < 0.5
    ? vec4(0.0)
    : vec4(label_color(labelId), ${MODULE}.opacity);
`,
  },
};

/** The props this extension reads off whatever layer it is attached to. */
export interface LabelPaletteExtensionProps {
  opacity?: number;
}

export class LabelPaletteExtension extends VivLayerExtension {
  static override extensionName = "LabelPaletteExtension";

  // `opacity` is a deck.gl layer prop already, but it reaches the shader only
  // where a layer's own code uses it -- Viv's fragment shader does not. So the
  // extension that writes the alpha is the one that has to carry it, exactly as
  // `ColorPaletteExtension` does.
  static defaultProps = {
    opacity: { type: "number", value: 1, compare: true },
  };

  getVivShaderTemplates() {
    return { modules: [labelPaletteModule] };
  }

  // deck.gl calls this with `this` bound to the layer, not to the extension.
  override updateState(
    this: Layer,
    params: UpdateParameters<Layer>,
    extension: LabelPaletteExtension,
  ) {
    // Through the prototype for the reason `GammaExtension` does it: the base
    // declares its `extension` parameter as the polymorphic `this`, which a
    // fixed subclass cannot satisfy.
    VivLayerExtension.prototype.updateState.call(this, params, extension);
    const { opacity = 1 } = this.props as LabelPaletteExtensionProps;
    for (const model of this.getModels()) {
      model.shaderInputs.setProps({ [MODULE]: { opacity } });
    }
  }
}
