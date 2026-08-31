import { describe, expect, it } from "vitest";
import {
  colorToHex,
  extractChannelNames,
  getColorMultipliers,
  guessDefaultColor,
  resolveAutoColor,
} from "./colorUtils";

describe("resolveAutoColor", () => {
  it("leaves an explicit choice alone", () => {
    expect(resolveAutoColor("red", "DAPI")).toBe("red");
    expect(resolveAutoColor("#804000")).toBe("#804000");
  });

  it("guesses from a marker name", () => {
    expect(resolveAutoColor("auto", "DAPI")).toBe("blue");
    expect(resolveAutoColor("auto", "GFP")).toBe("green");
  });

  it("is grey when there is no name to guess from", () => {
    expect(resolveAutoColor("auto")).toBe("gray");
    expect(resolveAutoColor("auto", "")).toBe("gray");
  });
});

describe("the answer before and after channel names load", () => {
  // The bug this covers: names arrive asynchronously, so every viewer renders
  // at least one frame with the name unknown. If the two answers differ, that
  // frame is a colour the viewer then contradicts.
  const settled = (name: string) => getColorMultipliers("auto", name);
  const transient = () => getColorMultipliers("auto", undefined);

  it("agrees for a channel whose name carries no marker", () => {
    // What extractChannelNames fabricates for an unnamed OME channel.
    expect(settled("Channel 0")).toEqual(transient());
    expect(settled("Channel 3")).toEqual(transient());
    // And for a real name that matches none of the patterns.
    expect(settled("STORM")).toEqual(transient());
  });

  it("agrees whatever the channel index would have been", () => {
    // The old fallback cycled green/red/blue/magenta/cyan by index, so every
    // index disagreed with the settled answer.
    for (const c of [0, 1, 2, 3, 4, 5]) {
      expect(settled(`Channel ${c}`)).toEqual(transient());
    }
  });

  it("still upgrades grey to a marker's colour when the name lands", () => {
    expect(settled("DAPI")).not.toEqual(transient());
    expect(colorToHex("auto", "DAPI")).toBe("#0000ff");
  });
});

describe("guessDefaultColor", () => {
  it("falls back to grey rather than picking something", () => {
    expect(guessDefaultColor("Channel 0")).toBe("gray");
    expect(guessDefaultColor("STORM")).toBe("gray");
  });

  it("still reads the conventional markers and wavelengths", () => {
    expect(guessDefaultColor("AF647")).toBe("magenta");
    expect(guessDefaultColor("mCherry")).toBe("red");
    expect(guessDefaultColor("laser 405")).toBe("blue");
  });
});

describe("extractChannelNames", () => {
  it("reads all three metadata shapes", () => {
    expect(extractChannelNames({ omero: { channels: [{ label: "DAPI" }] } })).toEqual(["DAPI"]);
    expect(
      extractChannelNames({ images: [{ pixels: { channels: [{ name: "GFP" }] } }] }),
    ).toEqual(["GFP"]);
    expect(extractChannelNames({ Summary: { ChNames: ["STORM"] } })).toEqual(["STORM"]);
  });

  it("fabricates a name for an unnamed channel, which then resolves to grey", () => {
    const names = extractChannelNames({ omero: { channels: [{}, {}] } });
    expect(names).toEqual(["Channel 0", "Channel 1"]);
    expect(names.map((n) => resolveAutoColor("auto", n))).toEqual(["gray", "gray"]);
  });

  it("returns nothing when the metadata has no channels at all", () => {
    expect(extractChannelNames({})).toEqual([]);
  });
});
