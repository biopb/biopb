import { describe, expect, it } from "vitest";
import {
  MIN_CHAT_WIDTH,
  MIN_WORK_WIDTH,
  clampChatWidth,
  defaultChatWidth,
} from "./chatPaneWidth";

const LAPTOP = 1280;
const WIDE = 2560;

describe("defaultChatWidth", () => {
  it("matches the CSS fallback that sizes an untouched pane", () => {
    // clamp(340px, 34%, 520px) -- the two must agree, or the pane jumps the
    // first time someone nudges the splitter with the keyboard.
    expect(defaultChatWidth(1000)).toBe(340); // 34% below the floor
    expect(defaultChatWidth(LAPTOP)).toBe(435); // 34%, in range
    expect(defaultChatWidth(WIDE)).toBe(520); // 34% above the ceiling
  });
});

describe("clampChatWidth", () => {
  it("keeps a dragged width as given", () => {
    expect(clampChatWidth(500, LAPTOP)).toBe(500);
  });

  it("will not starve the composer", () => {
    expect(clampChatWidth(40, LAPTOP)).toBe(MIN_CHAT_WIDTH);
  });

  it("will not starve the job list", () => {
    expect(clampChatWidth(9999, LAPTOP)).toBe(LAPTOP - MIN_WORK_WIDTH);
  });

  it("re-fits a width remembered from a wider monitor", () => {
    // The reason the bound follows the viewport instead of being a constant:
    // 1100px is fine on the monitor it was chosen on and leaves no job list on
    // a laptop, and the person would have no way to see that a remembered
    // preference was the cause.
    const chosen = clampChatWidth(1100, WIDE);
    expect(chosen).toBe(1100);
    expect(clampChatWidth(chosen, LAPTOP)).toBe(LAPTOP - MIN_WORK_WIDTH);
  });

  it("still yields a usable pane on a window too small for both", () => {
    // Both minimums cannot be honoured at once here; the composer wins, and the
    // stacked layout takes over below 900px anyway.
    expect(clampChatWidth(400, 500)).toBe(MIN_CHAT_WIDTH);
  });

  it("rejects a corrupt stored value rather than pinning to the minimum", () => {
    // 0 is the caller's signal to fall back to the default. Returning the
    // minimum instead would look like a working preference set very narrow.
    expect(clampChatWidth(Number("nonsense"), LAPTOP)).toBe(0);
    expect(clampChatWidth(Infinity, LAPTOP)).toBe(0);
  });
});
