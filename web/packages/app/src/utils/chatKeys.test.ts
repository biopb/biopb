import { describe, expect, it } from "vitest";
import { escAction, sendsOnEnter } from "./chatKeys";

const key = (over: Partial<Parameters<typeof sendsOnEnter>[0]> = {}) => ({
  key: "Enter",
  shiftKey: false,
  ...over,
});

describe("sendsOnEnter", () => {
  it("sends on a plain Enter", () => {
    expect(sendsOnEnter(key())).toBe(true);
  });

  it("leaves Shift+Enter to insert a newline", () => {
    expect(sendsOnEnter(key({ shiftKey: true }))).toBe(false);
  });

  it("ignores any other key", () => {
    expect(sendsOnEnter(key({ key: "a" }))).toBe(false);
  });

  it("does not send the Enter that commits an IME candidate", () => {
    // The one this guard exists for: typing CJK, the first Enter of a phrase
    // commits the candidate the IME is showing. Sending there would post half
    // a sentence and drop what was still in the IME buffer.
    expect(sendsOnEnter(key({ isComposing: true }))).toBe(false);
  });
});

describe("escAction", () => {
  const state = (over: Partial<Parameters<typeof escAction>[0]> = {}) => ({
    composing: false,
    imageOpen: false,
    inConsole: false,
    busy: false,
    ...over,
  });

  it("cancels the running turn", () => {
    expect(escAction(state({ busy: true }))).toBe("cancel-turn");
  });

  it("does nothing when no turn is running", () => {
    expect(escAction(state())).toBe("none");
  });

  it("closes the zoomed image first", () => {
    // Innermost dismissible thing wins, even mid-turn: the reader who opened it
    // is asking to close it, not to stop the work behind it.
    expect(escAction(state({ imageOpen: true, busy: true }))).toBe("close-image");
  });

  it("leaves Escape to the console when the console has focus", () => {
    expect(escAction(state({ inConsole: true, busy: true }))).toBe("none");
  });

  it("still closes the image from the console", () => {
    // The overlay covers the page, so focus being in the console behind it is
    // an accident of where the reader last clicked, not a claim on the key.
    expect(escAction(state({ inConsole: true, imageOpen: true }))).toBe(
      "close-image",
    );
  });

  it("gives Escape to the IME before anything else", () => {
    // An IME candidate window takes Escape to dismiss itself. Killing a turn
    // because someone reconsidered a word would be the worst of the four.
    expect(escAction(state({ composing: true, busy: true }))).toBe("none");
    expect(escAction(state({ composing: true, imageOpen: true }))).toBe("none");
  });
});
