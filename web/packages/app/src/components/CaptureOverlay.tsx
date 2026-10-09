import { useEffect } from "react";
import { BADGE } from "./viewerStyles";

/**
 * Over the whole page while an agent's capture runs, so the view cannot move
 * between being set and being read. Escape cancels; the state stays as the
 * agent set it.
 */
export function CaptureOverlay({ cancel }: { cancel: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") cancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cancel]);
  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 10000,
        cursor: "progress",
        // Barely there: the user is meant to watch the view being set.
        background: "rgba(0,0,0,0.12)",
      }}
    >
      <div style={{ ...BADGE, top: 12, left: "50%", transform: "translateX(-50%)", fontSize: 13 }}>
        An agent is capturing this view — Esc to cancel
      </div>
    </div>
  );
}
