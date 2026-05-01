"use client";

import { useAppStore } from "./store";
import { ImageViewer } from "./components/ImageViewer";
import { MetaPanel } from "./components/MetaPanel";
import { SliceControls } from "./components/SliceControls";
import { SourceTree } from "./components/SourceTree";

export default function HomePage() {
  const connectionState = useAppStore((s) => s.connectionState);
  const connectionError = useAppStore((s) => s.connectionError);
  const clearSession = useAppStore((s) => s.clearSession);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  const activeTensorId = useAppStore((s) => s.activeTensorId);

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <h1>BioPB Tensor Viewer</h1>
        <span className={`status-pill ${connectionState}`}>
          {connectionState === "connected"
            ? "Connected"
            : connectionState === "connecting"
              ? "Connecting..."
              : connectionState === "error"
                ? "Error"
                : "Idle"}
        </span>
        <button className="icon-btn" onClick={clearSession} title="Lock session">
          Lock
        </button>
      </header>

      <aside className="app-sidebar">
        <SourceTree />
      </aside>

      <main className="app-main">
        {activeSourceId && activeTensorId ? (
          <>
            <div className="viewer-canvas-wrap">
              <ImageViewer sourceId={activeSourceId} tensorId={activeTensorId} />
            </div>
            <SliceControls sourceId={activeSourceId} tensorId={activeTensorId} />
            <MetaPanel sourceId={activeSourceId} />
          </>
        ) : (
          <div className="loading-overlay" style={{ position: "static", flex: 1, background: "transparent" }}>
            {connectionState === "connected"
              ? "Select a source from the sidebar"
              : connectionState === "error"
                ? `Connection error: ${connectionError ?? "unknown"}`
                : "Connecting to server..."}
          </div>
        )}
      </main>

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
