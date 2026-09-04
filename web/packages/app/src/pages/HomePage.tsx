import { Link } from "react-router-dom";
import { useAppStore } from "../store";
import { ViewerPane } from "../components/ViewerPane";
import { MetaPanel } from "../components/MetaPanel";
import { SliceControls } from "../components/SliceControls";
import { SourceTree } from "../components/SourceTree";
import { TipBar } from "../components/TipBar";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import { useViewerUrlSync } from "../hooks/useViewerUrlSync";
import { withBase } from "../base";

export function HomePage() {
  useDocumentTitle("BioPB tensor - viewer");
  useViewerUrlSync();
  const connectionState = useAppStore((s) => s.connectionState);
  const connectionError = useAppStore((s) => s.connectionError);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  const activeTensorId = useAppStore((s) => s.activeTensorId);

  return (
    <div className="app-shell">
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
        {activeSourceId && activeTensorId ? (
          <>
            <div className="viewer-column">
              <div className="viewer-canvas-wrap">
                <ViewerPane
                  sourceId={activeSourceId}
                  tensorId={activeTensorId}
                />
              </div>
            </div>
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
