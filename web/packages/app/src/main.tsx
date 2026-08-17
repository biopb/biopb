import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { BASE } from "./base";
import { ViewerLayout } from "./ViewerLayout";
import { HomePage } from "./pages/HomePage";
import { UnlockPage } from "./pages/UnlockPage";
import { AdminPage } from "./pages/AdminPage";
import "./index.css";

// The dashboard and per-session observe surfaces are their own routes with no
// shared code with the (heavy) viewer, so lazy-load them: the observe shell then
// pulls only its own chunk plus the shared runtime, not the Pixi/Arrow viewer.
const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const ObservePage = lazy(() => import("./pages/ObservePage"));
// The biopb-mcp settings page is control-owned (it edits the global mcp config
// via the control's /api/mcp_config), not part of the tensor viewer, so it is a
// standalone route like the dashboard — outside ViewerLayout (no tensor client).
const McpAdminPage = lazy(() => import("./pages/McpAdminPage"));
// The data-plane log monitor: a standalone dashboard surface polling the
// control's /api/data_plane/logs. No tensor client, so outside ViewerLayout.
const LogsPage = lazy(() => import("./pages/LogsPage"));

const root = document.getElementById("root")!;

// The control front serves this bundle at its root, so the basename is "/" —
// unless the control was published below the root (`--url-prefix`), which it
// tells the app at runtime via BASE. Not import.meta.env.BASE_URL: the build is
// always base "/" because the prefix is a per-job value with no build-time
// answer (see base.ts).
const basename = BASE || "/";

createRoot(root).render(
  <StrictMode>
    <BrowserRouter basename={basename}>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/mcp/admin" element={<McpAdminPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route
            path="/session/:sessionId/observe"
            element={<ObservePage />}
          />
          <Route element={<ViewerLayout />}>
            <Route path="/viewer" element={<HomePage />} />
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/unlock" element={<UnlockPage />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  </StrictMode>,
);
