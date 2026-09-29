import { describe, expect, it } from "vitest";
import {
  buildLabelLayers,
  labelLayerId,
  type LabelLayerOptions,
} from "./labelLayers";

describe("labelLayerId", () => {
  it("carries Viv's view id, without which the layer is never drawn", () => {
    expect(labelLayerId("nuclei")).toContain("-#detail#");
  });

  it("is distinct per set, so two sets are two layers", () => {
    expect(labelLayerId("nuclei")).not.toBe(labelLayerId("cells"));
  });
});

describe("buildLabelLayers", () => {
  const sources = [{ one: true }, { two: true }];

  /** The layer as built, with the two gate props at their ordinary values. */
  const build = (over: Partial<LabelLayerOptions> = {}) =>
    buildLabelLayers({
      name: "nuclei",
      sources,
      selection: {},
      opacity: 0.5,
      showing: true,
      onViewportLoad: () => {},
      ...over,
    }) as Array<{ id: string; props: Record<string, unknown> }>;

  it("draws nothing without a set", () => {
    expect(buildLabelLayers(null)).toEqual([]);
  });

  it("draws nothing when the sources are empty", () => {
    expect(build({ sources: [] })).toEqual([]);
  });

  it("builds one layer, under an id Viv will draw", () => {
    const layers = build({ selection: { t: 3 } });
    expect(layers).toHaveLength(1);
    expect(layers[0]!.id).toBe(labelLayerId("nuclei"));
    expect(layers[0]!.props.opacity).toBe(0.5);
    expect(layers[0]!.props.selections).toEqual([{ t: 3 }]);
    // The identity ramp: anything else renames every object.
    expect(layers[0]!.props.contrastLimits).toEqual([[0, 1]]);
    expect(layers[0]!.props.interpolation).toBe("nearest");
    expect(layers[0]!.props.pickable).toBe(false);
  });

  it("hands a single level the one source rather than the pyramid", () => {
    expect(build({ sources: [sources[0]!] })[0]!.props.loader).toBe(sources[0]);
  });

  it("holds an out-of-step overlay back, without unmounting it", () => {
    // Transparent rather than gone: the layer is out of step *because* it is
    // still loading, and it has to stay mounted to finish and catch up.
    const layers = build({ showing: false, opacity: 0.5 });
    expect(layers).toHaveLength(1);
    expect(layers[0]!.props.opacity).toBe(0);
    expect(build({ showing: true, opacity: 0.5 })[0]!.props.opacity).toBe(0.5);
  });

  it("passes the load report through, which is what decides that", () => {
    const onViewportLoad = () => {};
    expect(build({ onViewportLoad })[0]!.props.onViewportLoad).toBe(onViewportLoad);
  });

  it("keeps one extensions array, so a slider move recompiles no shader", () => {
    const one = build({ opacity: 1 });
    const two = build({ opacity: 0.2 });
    const props = (l: unknown) => (l as { props: { extensions: unknown } }).props.extensions;
    expect(props(one[0])).toBe(props(two[0]));
  });
});
