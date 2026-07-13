// Shared token/auth helpers for the control-served SPA.
//
// biopb has two deployment modes: **local** (default — every listener binds
// loopback, no token) and **remote** (`biopb control start --remote` — the
// control's browser UI + the flight server bind publicly behind a required
// token). The same bundle ships to both, so the app can't know its mode at build
// time; it learns it at runtime from the control's public `/health` probe
// (`auth_required`). The bundle itself, `/health`, and `/unlock` are always
// served unauthenticated so the app can bootstrap far enough to ask for a token.

const TOKEN_KEY = "biopb_token";

export function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(t: string): void {
  sessionStorage.setItem(TOKEN_KEY, t.trim());
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

/** Bearer header for the stored token, or nothing when there is none (local
 * mode). Spread into a fetch `headers` object. */
export function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const t = getToken();
  return { ...(extra || {}), ...(t ? { Authorization: "Bearer " + t } : {}) };
}

/** Capture a `?token=…` from the current URL into sessionStorage and strip it
 * from the visible URL (so it doesn't linger in browser history). This is how
 * the one-time access URL (`http://host:8813/?token=…`) hands the token to the
 * app. Returns true if a token was captured. */
export function captureUrlToken(): boolean {
  const url = new URL(window.location.href);
  const t = url.searchParams.get("token");
  if (!t) return false;
  setToken(t);
  url.searchParams.delete("token");
  window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  return true;
}

/** Whether the control requires a token (remote mode), from the public `/health`
 * probe. Defaults to false (local mode) if the probe can't be read, so a
 * transient failure never traps the user on the unlock page. */
export async function authRequired(): Promise<boolean> {
  try {
    const r = await fetch("/health");
    if (!r.ok) return false;
    const j = await r.json();
    return !!j.auth_required;
  } catch {
    return false;
  }
}

/** Send the browser to the unlock page, returning here afterwards. */
export function redirectToUnlock(): void {
  const here = window.location.pathname;
  const next =
    here && here !== "/unlock" ? "?next=" + encodeURIComponent(here) : "";
  window.location.assign("/unlock" + next);
}
