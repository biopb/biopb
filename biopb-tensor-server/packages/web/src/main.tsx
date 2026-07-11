import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ClientBootstrap } from "./ClientBootstrap";
import { HomePage } from "./pages/HomePage";
import { UnlockPage } from "./pages/UnlockPage";
import { AdminPage } from "./pages/AdminPage";
import "./index.css";

const root = document.getElementById("root")!;

// Router basename tracks the Vite base (import.meta.env.BASE_URL, e.g.
// "/data_plane/viewer/" when the control front serves us under a namespace, "/"
// when root-served). Strip the trailing slash React Router does not want; "" ->
// "/" (its default) for the root-served case.
const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

createRoot(root).render(
  <StrictMode>
    <BrowserRouter basename={basename}>
      <ClientBootstrap />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="/unlock" element={<UnlockPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
