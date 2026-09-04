import { Link } from "react-router-dom";
import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { useAppStore } from "../store";
import { ViewerPane } from "../components/ViewerPane";
import { MetaPanel } from "../components/MetaPanel";
import { SliceControls } from "../components/SliceControls";
import { SourceTree } from "../components/SourceTree";
import { TipBar } from "../components/TipBar";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import { useViewerUrlSync } from "../hooks/useViewerUrlSync";
import { withBase } from "../base";
import {
  DEFAULT_CONTROL_WIDTH,
  DEFAULT_SIDEBAR_WIDTH,
  MIN_CONTROL_WIDTH,
  MIN_SIDEBAR_WIDTH,
  clampPaneWidth,
} from "../utils/paneWidth";

/**
 * Remembered per browser, as the chat pane's width is: how much room the tree
 * and the controls deserve depends on the monitor and on what is being looked
 * at, and neither is state anyone else needs.
 */
const SIDEBAR_WIDTH_KEY = "biopb.viewer.sidebarWidth";
const CONTROL_WIDTH_KEY = "biopb.viewer.controlWidth";

/** A stored width, or null for "never dragged" -- which leaves CSS its default. */
function storedWidth(key: string): number | null {
  try {
    const raw = localStorage.getItem(key);
    const n = raw === null ? Number.NaN : Number(raw);
    return Number.isFinite(n) ? n : null;
  } catch {
    // A private window, or site data blocked: an unremembered width is still a
    // working viewer.
    return null;
  }
}

export function HomePage() {
  useDocumentTitle("BioPB tensor - viewer");
  useViewerUrlSync();
  const connectionState = useAppStore((s) => s.connectionState);
  const connectionError = useAppStore((s) => s.connectionError);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  const activeTensorId = useAppStore((s) => s.activeTensorId);
  // What the render path fetches: the exact address a link asked for, which may
  // be content-pinned, falling back to the selection for an ordinary click.
  const requestedArrayId = useAppStore((s) => s.requestedArrayId);

  // --- pane widths ---------------------------------------------------------
  const shellRef = useRef<HTMLDivElement | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState<number | null>(() =>
    storedWidth(SIDEBAR_WIDTH_KEY),
  );
  const [controlWidth, setControlWidth] = useState<number | null>(() =>
    storedWidth(CONTROL_WIDTH_KEY),
  );
  useEffect(() => {
    if (sidebarWidth === null) return;
    try {
      localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
    } catch {
      /* a preference that cannot be saved is still a working session */
    }
  }, [sidebarWidth]);
  useEffect(() => {
    if (controlWidth === null) return;
    try {
      localStorage.setItem(CONTROL_WIDTH_KEY, String(controlWidth));
    } catch {
      /* as above */
    }
  }, [controlWidth]);

  const dragging = useRef<"sidebar" | "control" | null>(null);

  // Measured off the shell rather than the window: the two panes are its edges,
  // and a width taken from `clientX` alone would be wrong the moment the shell
  // is not the whole page.
  const resizeTo = useCallback(
    (edge: "sidebar" | "control", clientX: number) => {
      const shell = shellRef.current;
      if (!shell) return;
      const rect = shell.getBoundingClientRect();
      if (edge === "sidebar") {
        const width = clampPaneWidth(
          clientX - rect.left,
          MIN_SIDEBAR_WIDTH,
          controlWidth ?? DEFAULT_CONTROL_WIDTH,
          rect.width,
        );
        if (width) setSidebarWidth(width);
        return;
      }
      const width = clampPaneWidth(
        rect.right - clientX,
        MIN_CONTROL_WIDTH,
        sidebarWidth ?? DEFAULT_SIDEBAR_WIDTH,
        rect.width,
      );
      if (width) setControlWidth(width);
    },
    [sidebarWidth, controlWidth],
  );

  /** The drag handle between two panes. Pointer events, so capture keeps the
   *  drag alive when the cursor outruns the 8px handle -- which it will. */
  const splitter = (edge: "sidebar" | "control", label: string) => (
    <div
      className="pane-splitter"
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      tabIndex={0}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        dragging.current = edge;
      }}
      onPointerMove={(e) => {
        if (dragging.current === edge) resizeTo(edge, e.clientX);
      }}
      onPointerUp={(e) => {
        dragging.current = null;
        e.currentTarget.releasePointerCapture(e.pointerId);
      }}
      onKeyDown={(e) => {
        // Usable without a pointer, and the only way to nudge it exactly.
        const step = e.key === "ArrowLeft" ? -16 : e.key === "ArrowRight" ? 16 : 0;
        if (!step) return;
        e.preventDefault();
        const shell = shellRef.current;
        if (!shell) return;
        const rect = shell.getBoundingClientRect();
        if (edge === "sidebar") {
          resizeTo(edge, rect.left + (sidebarWidth ?? DEFAULT_SIDEBAR_WIDTH) + step);
        } else {
          // The control pane grows leftwards, so a right arrow narrows it.
          resizeTo(edge, rect.right - (controlWidth ?? DEFAULT_CONTROL_WIDTH) + step);
        }
      }}
    />
  );

  const paneStyle: CSSProperties = {
    // Unset until dragged, so an untouched pane is sized by the CSS default
    // rather than by a number this component would have to keep in step.
    ...(sidebarWidth === null ? {} : { ["--sidebar-w" as string]: `${sidebarWidth}px` }),
    ...(controlWidth === null ? {} : { ["--control-w" as string]: `${controlWidth}px` }),
  };

  return (
    <div className="app-shell" ref={shellRef} style={paneStyle}>
      <header className="app-topbar">
        <img
          className="topbar-logo"
          src={withBase("/biopb-logo.png")}
          alt=""
          aria-hidden="true"
        />
        <h1>BioPB tensor - viewer</h1>
        <span className={`status-pill ${connectionState}`}>
          {connectionState === "connected"
            ? "Connected"
            : connectionState === "connecting"
              ? "Connecting..."
              : connectionState === "error"
                ? "Error"
                : "Idle"}
        </span>
        <div className="topbar-spacer" />
        <Link className="icon-btn" to="/admin" title="Server admin">
          ⚙ Admin
        </Link>
      </header>

      <aside className="app-sidebar">
        <SourceTree />
      </aside>

      <main className="app-main">
        {splitter("sidebar", "Resize the source list")}
        {activeSourceId && activeTensorId ? (
          <>
            <div className="viewer-column">
              <div className="viewer-canvas-wrap">
                <ViewerPane
                  sourceId={activeSourceId}
                  tensorId={requestedArrayId ?? activeTensorId}
                />
              </div>
            </div>
            {splitter("control", "Resize the control panel")}
            <div className="control-column">
              <SliceControls sourceId={activeSourceId} tensorId={activeTensorId} />
              {/*
                Remount per source. Without the key, a switch re-renders the
                whole tree against the *previous* source's metadata — the new
                fetch has not resolved yet — and then reconciles it away once it
                does. It also carries every node's expanded/collapsed state over
                to a tree that has nothing to do with it.
              */}
              <MetaPanel key={activeSourceId} sourceId={activeSourceId} />
            </div>
          </>
        ) : (
          <div
            className="loading-overlay"
            style={{ position: "static", flex: 1, background: "transparent" }}
          >
            {connectionState === "connected"
              ? "Select a source from the sidebar"
              : connectionState === "error"
                ? `Connection error: ${connectionError ?? "unknown"}`
                : "Connecting to server..."}
          </div>
        )}
      </main>

      <TipBar />

      {connectionError && connectionState === "error" && (
        <div className="error-toast">
          <strong>Connection error</strong>
          <br />
          {connectionError}
        </div>
      )}
    </div>
  );
}
