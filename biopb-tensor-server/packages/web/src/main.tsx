import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ClientBootstrap } from "./ClientBootstrap";
import { HomePage } from "./pages/HomePage";
import { UnlockPage } from "./pages/UnlockPage";
import "./index.css";

const root = document.getElementById("root")!;

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <ClientBootstrap />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/unlock" element={<UnlockPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
