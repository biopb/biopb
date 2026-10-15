import { describe, expect, it } from "vitest";
import { gamma3dExtensions } from "./vivGamma";

type Rendered = { rendering: { _RENDER: string; _AFTER_RENDER: string } };

/** The strings the layer compiles: those of the last extension that has any. */
function compiled(mode: "mip" | "minip" | "additive") {
  const exts = gamma3dExtensions(mode) as Rendered[];
  return (exts[exts.length - 1] as Rendered).rendering;
}

describe("gamma3dExtensions", () => {
  it.each([
    ["mip", "maxVal"],
    ["minip", "minVal"],
  ] as const)("%s bends the picked value, not the per-step samples", (mode, held) => {
    const { _RENDER, _AFTER_RENDER } = compiled(mode);
    expect(_AFTER_RENDER).toMatch(
      new RegExp(`min\\(1\\.0, pow\\(clamp\\(${held}\\w*, 0\\.0, 1\\.0\\), gammaModule\\.gamma\\)\\)`),
    );
    // The march itself is Viv's: one pow per pixel, none per step.
    expect(_RENDER).not.toContain("pow");
  });

  it("additive bends the finished colour and keeps its hue", () => {
    const { _AFTER_RENDER } = compiled("additive");
    expect(_AFTER_RENDER).toContain("pow(min(peak, 1.0), gammaModule.gamma) / peak");
  });

  it("puts Viv's own rendering extension in front of the gamma one", () => {
    expect(gamma3dExtensions("mip")).toHaveLength(2);
  });
});
