import { Link } from "react-router-dom";
import { useAppStore } from "../store";
import { ImageViewer } from "../components/ImageViewer";
import { MetaPanel } from "../components/MetaPanel";
import { SliceControls } from "../components/SliceControls";
import { SourceTree } from "../components/SourceTree";

export function HomePage() {
  const connectionState = useAppStore((s) => s.connectionState);
  const connectionError = useAppStore((s) => s.connectionError);
  const clearSession = useAppStore((s) => s.clearSession);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  const activeTensorId = useAppStore((s) => s.activeTensorId);

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <img
          className="topbar-logo"
          src={`${import.meta.env.BASE_URL}biopb-logo.png`}
          alt=""
          aria-hidden="true"
        />
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
        <div className="topbar-spacer" />
        <Link className="icon-btn" to="/admin" title="Server admin">
          ⚙ Admin
        </Link>
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
            <div className="viewer-column">
              <div className="viewer-canvas-wrap">
                <ImageViewer
                  sourceId={activeSourceId}
                  tensorId={activeTensorId}
                />
              </div>
            </div>
            <div className="control-column">
              <SliceControls sourceId={activeSourceId} tensorId={activeTensorId} />
              <MetaPanel sourceId={activeSourceId} />
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
