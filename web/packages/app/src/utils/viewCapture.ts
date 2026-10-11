/**
 * Taking a picture of the viewer for an agent: pure helpers.
 *
 * The page applies a requested view exactly as it would a shared link, waits for
 * what it asked for to land, and reads the deck.gl canvas back. Reading is
 * possible without `preserveDrawingBuffer` because deck draws on demand: after
 * the last frame the buffer stays as it was, and a read taken two animation
 * frames after the load report sees the finished tiles.
 */

/** How long the page waits for tiles and overlays before answering anyway. */
export const READY_TIMEOUT_MS = 8000;

/** Extra wait once a label overlay was asked for: its tiles have no ready signal. */
export const LABEL_SETTLE_MS = 600;

/** `size` scaled down, never up, so the longest edge is at most `maxEdge`. */
export function fitWithin(
  size: { width: number; height: number },
  maxEdge: number,
): { width: number; height: number } {
  const longest = Math.max(size.width, size.height);
  if (!(maxEdge > 0) || longest <= maxEdge) return { width: size.width, height: size.height };
  const k = maxEdge / longest;
  return {
    width: Math.max(1, Math.round(size.width * k)),
    height: Math.max(1, Math.round(size.height * k)),
  };
}

/** The query string of a request's `view`, which may be a whole address. */
export function viewParams(view: string): URLSearchParams {
  const noFragment = view.split("#", 1)[0] ?? "";
  const q = noFragment.includes("?") ? noFragment.slice(noFragment.indexOf("?") + 1) : noFragment;
  return new URLSearchParams(q);
}

export type Readiness =
  | { kind: "ready" }
  | { kind: "failed"; error: string }
  | { kind: "waiting" };

export interface ReadinessInputs {
  status: string;
  error: string | null;
  planeReady: boolean;
  roisPending: number;
}

/** Whether what the request asked for has landed. */
export function readiness(s: ReadinessInputs): Readiness {
  if (s.status === "failed") return { kind: "failed", error: s.error ?? "the tensor failed to open" };
  if (s.status === "ready" && s.planeReady && s.roisPending === 0) return { kind: "ready" };
  return { kind: "waiting" };
}

/**
 * Poll `check` until it answers something other than "waiting", or `timeoutMs`
 * passes (answering "timeout"). `cancelled` is read each step.
 */
export async function waitFor(
  check: () => Readiness,
  timeoutMs: number,
  cancelled: () => boolean,
  stepMs = 100,
): Promise<Readiness | { kind: "timeout" } | { kind: "cancelled" }> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (cancelled()) return { kind: "cancelled" };
    const r = check();
    if (r.kind !== "waiting") return r;
    if (Date.now() >= deadline) return { kind: "timeout" };
    await new Promise((res) => setTimeout(res, stepMs));
  }
}

/** How long a frame may take before the tab is taken to be hidden: a hidden page runs none. */
const FRAME_TIMEOUT_MS = 1000;

/** Resolves true on the next frame, false if none comes in time. */
const nextFrame = () =>
  new Promise<boolean>((res) => {
    const timer = setTimeout(() => res(false), FRAME_TIMEOUT_MS);
    requestAnimationFrame(() => {
      clearTimeout(timer);
      res(true);
    });
  });

/**
 * The deck canvas as a PNG at most `maxEdge` on a side, composited onto the
 * viewer's own background: the canvas is transparent outside the image, which
 * an agent's image reader would show as white or black at random.
 *
 * Null when the page stopped running frames, which is what hiding it does: the
 * canvas would hold the picture from before.
 */
export async function readCanvas(maxEdge: number): Promise<Blob | null> {
  // Two frames: deck redraws after the load report, and the read must see it.
  if (!(await nextFrame()) || !(await nextFrame())) return null;
  const canvas =
    document.querySelector<HTMLCanvasElement>("canvas#deckgl-overlay") ??
    document.querySelector<HTMLCanvasElement>("canvas");
  if (!canvas || canvas.width === 0 || canvas.height === 0) {
    throw new Error("the viewer has no canvas to read");
  }
  const { width, height } = fitWithin(canvas, maxEdge);
  const out = document.createElement("canvas");
  out.width = width;
  out.height = height;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("could not create a 2-D canvas");
  ctx.fillStyle = backgroundOf(canvas);
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(canvas, 0, 0, width, height);
  return new Promise((res, rej) =>
    out.toBlob((b) => (b ? res(b) : rej(new Error("PNG encoding failed"))), "image/png"),
  );
}

/** The first opaque background up the canvas's ancestors, else black. */
function backgroundOf(el: HTMLElement): string {
  for (let n: HTMLElement | null = el; n; n = n.parentElement) {
    const c = getComputedStyle(n).backgroundColor;
    if (c && c !== "transparent" && !/rgba\(.*,\s*0\)$/.test(c)) return c;
  }
  return "#000";
}
