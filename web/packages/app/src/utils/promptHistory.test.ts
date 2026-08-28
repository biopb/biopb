import { describe, expect, it } from "vitest";
import {
  abandon,
  MAX_PROMPTS,
  noHistory,
  pastPrompts,
  recall,
  remember,
  seed,
  type PromptHistory,
} from "./promptHistory";
import type { ThreadItem } from "./chatThread";

const sent = (...prompts: string[]): PromptHistory =>
  prompts.reduce((h, p) => remember(h, p), noHistory);

/** The text a walk in `dir` lands on, or null when the keystroke does nothing. */
const step = (h: PromptHistory, dir: "older" | "newer", current = "") =>
  recall(h, dir, current);

describe("remember", () => {
  it("keeps prompts oldest first", () => {
    expect(sent("one", "two").entries).toEqual(["one", "two"]);
  });

  it("trims, and ignores an empty prompt", () => {
    expect(sent("  hi  ").entries).toEqual(["hi"]);
    expect(sent("hi", "   ").entries).toEqual(["hi"]);
  });

  it("does not stack a repeat of the newest prompt", () => {
    expect(sent("/context", "/context", "/context").entries).toEqual(["/context"]);
  });

  it("keeps a duplicate that is not consecutive, in the order it was used", () => {
    // Walking back should reach prompts where they were said. Collapsing every
    // duplicate would silently reorder the history around it.
    expect(sent("a", "b", "a").entries).toEqual(["a", "b", "a"]);
  });

  it("drops the oldest past the cap", () => {
    const many = Array.from({ length: MAX_PROMPTS + 5 }, (_, i) => `p${i}`);
    const h = many.reduce((acc, p) => remember(acc, p), noHistory);
    expect(h.entries).toHaveLength(MAX_PROMPTS);
    expect(h.entries[0]).toBe("p5");
  });

  it("ends a walk, so the next one starts from the newest", () => {
    const walked = step(sent("a", "b"), "older")!.history;
    expect(walked.at).toBe(0);
    expect(remember(walked, "c").at).toBeNull();
  });
});

describe("recall", () => {
  it("does nothing with an empty buffer", () => {
    expect(step(noHistory, "older", "half typed")).toBeNull();
  });

  it("walks back newest first", () => {
    const h = sent("one", "two", "three");
    const first = step(h, "older")!;
    expect(first.text).toBe("three");
    const second = step(first.history, "older")!;
    expect(second.text).toBe("two");
    expect(step(second.history, "older")!.text).toBe("one");
  });

  it("stops at the oldest rather than wrapping", () => {
    // Null, not the newest entry: wrapping past the end would put a walk back
    // where it started with nothing on screen to say it had.
    const oldest = step(step(sent("one", "two"), "older")!.history, "older")!;
    expect(oldest.text).toBe("one");
    expect(step(oldest.history, "older")).toBeNull();
  });

  it("gives the half-typed draft back at the near end", () => {
    const walked = step(sent("one", "two"), "older", "what I was typing")!;
    expect(walked.text).toBe("two");
    const back = step(walked.history, "newer")!;
    expect(back.text).toBe("what I was typing");
    expect(back.history.at).toBeNull();
  });

  it("ignores a forward step while the draft is showing", () => {
    expect(step(sent("one"), "newer", "mine")).toBeNull();
  });

  it("walks forward through the middle of the buffer", () => {
    const back = step(step(sent("a", "b", "c"), "older")!.history, "older")!;
    expect(back.text).toBe("b");
    expect(step(back.history, "newer")!.text).toBe("c");
  });
});

describe("abandon", () => {
  it("ends the walk and forgets the stashed draft", () => {
    const walked = step(sent("one"), "older", "draft")!.history;
    const edited = abandon(walked);
    expect(edited.at).toBeNull();
    expect(edited.draft).toBe("");
    // The next backward step starts again from the newest, and stashes what is
    // on screen now -- the edit, which is what the reader would lose otherwise.
    const again = step(edited, "older", "one, edited")!;
    expect(again.text).toBe("one");
    expect(step(again.history, "newer")!.text).toBe("one, edited");
  });

  it("leaves a buffer that is not walking alone", () => {
    const h = sent("one");
    expect(abandon(h)).toBe(h);
  });
});

describe("seed", () => {
  it("fills an empty buffer, applying the same rules", () => {
    expect(seed(noHistory, ["one", "one", " two "]).entries).toEqual(["one", "two"]);
  });

  it("leaves a buffer that has anything in it", () => {
    // The guard that makes seeding safe on every keystroke: the thread is
    // already folded in, and doing it twice would double every prompt.
    const h = sent("mine");
    expect(seed(h, ["one", "two"])).toBe(h);
  });
});

describe("pastPrompts", () => {
  const say = (id: string, role: "user" | "assistant", text: string): ThreadItem => ({
    kind: "message",
    id,
    role,
    blocks: [{ type: "text", text }],
  });

  it("takes the user's messages, in order", () => {
    const thread: ThreadItem[] = [
      say("1", "user", "count the cells"),
      say("2", "assistant", "42"),
      say("3", "user", "now the nuclei"),
    ];
    expect(pastPrompts(thread)).toEqual(["count the cells", "now the nuclei"]);
  });

  it("skips everything that is not speech, and messages with no words", () => {
    const thread: ThreadItem[] = [
      {
        kind: "tool_call",
        id: "t1",
        title: "execute_code",
        status: "completed",
        blocks: [],
      },
      { kind: "message", id: "2", role: "user", blocks: [] },
      say("3", "user", "  "),
      say("4", "user", "still here"),
    ];
    expect(pastPrompts(thread)).toEqual(["still here"]);
  });

  it("reads the text out of a message that also carries a picture", () => {
    const thread: ThreadItem[] = [
      {
        kind: "message",
        id: "1",
        role: "user",
        blocks: [
          { type: "image", data: "AAAA", mime: "image/png" },
          { type: "text", text: "what is this" },
        ],
      },
    ];
    expect(pastPrompts(thread)).toEqual(["what is this"]);
  });
});
