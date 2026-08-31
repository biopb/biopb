import { describe, expect, it } from "vitest";
import { escAction, recallAction, sendsOnEnter } from "./chatKeys";

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

  it("refuses the agent's question before cancelling the turn", () => {
    // Dismiss-innermost, and the kinder reading of the key: refusing one action
    // is not the same as throwing away the turn that led to it.
    expect(escAction(state({ permissionOpen: true, busy: true }))).toBe(
      "refuse-permission",
    );
  });

  it("still gives the image overlay and the IME precedence over it", () => {
    expect(escAction(state({ permissionOpen: true, imageOpen: true }))).toBe(
      "close-image",
    );
    expect(escAction(state({ permissionOpen: true, composing: true }))).toBe(
      "none",
    );
  });

  it("answers a question even when no turn is running", () => {
    // The turn is what is waiting *on* the answer, so a pane that only bound
    // Escape while busy would leave the question unanswerable exactly when the
    // agent is blocked on it.
    expect(escAction(state({ permissionOpen: true, busy: false }))).toBe(
      "refuse-permission",
    );
  });
});

describe("recallAction", () => {
  // A one-line draft with the caret at its end: the shape almost every prompt
  // has, and where both arrows have to walk.
  const press = (
    over: Partial<Parameters<typeof recallAction>[0]> = {},
  ): Parameters<typeof recallAction>[0] => {
    const value = over.value ?? "count the cells";
    return {
      key: "ArrowUp",
      shiftKey: false,
      altKey: false,
      ctrlKey: false,
      metaKey: false,
      selectionStart: value.length,
      selectionEnd: value.length,
      ...over,
      value,
    };
  };

  it("walks back on Up and forward on Down", () => {
    expect(recallAction(press())).toBe("older");
    expect(recallAction(press({ key: "ArrowDown" }))).toBe("newer");
  });

  it("walks from an empty composer", () => {
    expect(recallAction(press({ value: "", selectionStart: 0, selectionEnd: 0 }))).toBe(
      "older",
    );
  });

  it("ignores any other key", () => {
    expect(recallAction(press({ key: "ArrowLeft" }))).toBe("none");
    expect(recallAction(press({ key: "a" }))).toBe("none");
  });

  it("leaves a multi-line draft its caret", () => {
    // The whole reason the rule is positional: taking the arrows outright would
    // make a two-line prompt uneditable.
    const value = "first line\nsecond line";
    const onSecond = { value, selectionStart: value.length, selectionEnd: value.length };
    expect(recallAction(press({ key: "ArrowUp", ...onSecond }))).toBe("none");
    const onFirst = { value, selectionStart: 3, selectionEnd: 3 };
    expect(recallAction(press({ key: "ArrowDown", ...onFirst }))).toBe("none");
  });

  it("still walks off the near edge of a multi-line draft", () => {
    const value = "first line\nsecond line";
    expect(
      recallAction(press({ key: "ArrowUp", value, selectionStart: 3, selectionEnd: 3 })),
    ).toBe("older");
    expect(
      recallAction(
        press({
          key: "ArrowDown",
          value,
          selectionStart: value.length,
          selectionEnd: value.length,
        }),
      ),
    ).toBe("newer");
  });

  it("leaves the modified arrows to the platform", () => {
    // Shift extends a selection; Alt/Ctrl/Cmd are word- and line-wise motions.
    expect(recallAction(press({ shiftKey: true }))).toBe("none");
    expect(recallAction(press({ altKey: true }))).toBe("none");
    expect(recallAction(press({ ctrlKey: true }))).toBe("none");
    expect(recallAction(press({ metaKey: true }))).toBe("none");
  });

  it("does not walk over a selection", () => {
    expect(recallAction(press({ selectionStart: 0, selectionEnd: 5 }))).toBe("none");
  });

  it("leaves the arrows to an IME candidate list", () => {
    // Same guard as Enter, for the same reason: the candidate list is driven by
    // these keys, and walking there would replace the composer mid-word.
    expect(recallAction(press({ isComposing: true }))).toBe("none");
  });
});
