import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import ChatPane from "./ChatPane";
import type { ChatStatus } from "../utils/chatClient";

// Server-rendered, because the workspace has no DOM test environment. Effects
// do not run, so this covers the first paint only — enough to catch a pane that
// throws on mount, and to pin what an unconfigured session is told.

const status = (over: Partial<ChatStatus> = {}): ChatStatus => ({
  enabled: true,
  ready: true,
  reason: null,
  engine: "builtin",
  engines: [
    { engine: "builtin", ready: true, reason: null },
    { engine: "acp", ready: false, reason: "opencode is not installed" },
  ],
  model: "claude-sonnet-5",
  compacted: 0,
  ...over,
});

const render = (s: ChatStatus) =>
  renderToStaticMarkup(<ChatPane base="/session/abc" status={s} pollMs={3000} />);

// The pane keeps the thread scrolled to the newest message, which is a layout
// effect: measuring in a passive effect shows the reader one frame at the old
// position. React warns that a layout effect cannot run on the server, which is
// true and irrelevant — this app only ever renders in a browser, and the server
// renderer is this file's device for reaching the component at all. Silenced
// narrowly so a real warning still fails loudly.
const realError = console.error;
beforeAll(() => {
  console.error = (...args: unknown[]) => {
    if (typeof args[0] === "string" && args[0].includes("useLayoutEffect")) return;
    realError(...args);
  };
});
afterAll(() => {
  console.error = realError;
});

describe("ChatPane", () => {
  it("mounts an empty thread", () => {
    const html = render(status());
    expect(html).toContain("claude-sonnet-5");
    expect(html).toContain("<textarea");
  });

  it("says why chat cannot run, before anyone types", () => {
    // The reason is reported rather than raised precisely so it lands here,
    // instead of the user discovering it by sending a message.
    const html = render(
      status({ ready: false, reason: "chat.api_key is not set" }),
    );
    expect(html).toContain("chat.api_key is not set");
    expect(html).toContain("disabled");
  });

  it("advertises the keys, and still offers a control to click", () => {
    // The Send button is gone in favour of Enter, so the composer has to say
    // so -- and a submit control still has to exist for the pointer and for a
    // screen reader, which is what puts it in the corner of the box.
    const html = render(status());
    expect(html).toContain("to send");
    expect(html).toContain('aria-label="Send message"');
  });

  it("offers a way out of a thread that has grown too long", () => {
    // The conversation is re-projected to the provider whole on every turn and
    // has no other bound; without this the only escape was restarting the
    // session child, which takes the kernel and the viewer with it.
    expect(render(status())).toContain("chat-new");
  });

  it("tells the reader the commands are there", () => {
    // A slash command nobody knows about is not a feature. The list itself
    // appears once a slash is typed, which a static render cannot reach, so
    // the standing hint is the part that has to be here.
    expect(render(status())).toContain("for commands");
  });

  it("leaves the composer usable when chat is ready", () => {
    expect(render(status())).not.toContain("<textarea disabled");
  });
});
