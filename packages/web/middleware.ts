/**
 * Next.js middleware — website token gate.
 *
 * Rules:
 *  - /livez, /readyz, /healthz → always pass through (health probes)
 *  - /_next/*, /favicon.ico, /unlock → always pass through (static + unlock UI)
 *  - All other routes: require a valid session cookie (biopb_session)
 *    or a ?token= query parameter (sets the cookie and redirects clean URL)
 *
 * Dev-mode bypass:
 *  When the server is running with BIOPB_WEB_DEV_BYPASS and is bound to
 *  localhost, NEXT_PUBLIC_DEV_MODE=true is injected at build time and the
 *  token gate is skipped entirely.  Non-localhost bindings always enforce
 *  the token regardless of the flag.
 */

import { NextRequest, NextResponse } from "next/server";

const COOKIE_NAME = "biopb_session";
const COOKIE_MAX_AGE = 0; // session cookie (until browser close)
const TOKEN_PARAM = "token";

/** Routes that bypass auth entirely. */
const PUBLIC_PREFIXES = ["/_next/", "/favicon.ico", "/unlock", "/livez", "/readyz", "/healthz"];

export function middleware(req: NextRequest): NextResponse {
  const { pathname, searchParams, origin } = req.nextUrl;

  // 1. Public routes — pass through unconditionally
  if (PUBLIC_PREFIXES.some((p) => pathname.startsWith(p) || pathname === p)) {
    return NextResponse.next();
  }

  // 2. Dev-mode bypass (localhost only, enforced server-side too)
  const devMode = process.env.NEXT_PUBLIC_DEV_MODE === "true";
  if (devMode) {
    // Confirmed localhost-only at server startup; no extra check needed here
    return NextResponse.next();
  }

  const expectedToken = process.env.BIOPB_WEB_TOKEN ?? "";
  if (!expectedToken) {
    // No token configured — allow access with a warning
    console.warn("[biopb] BIOPB_WEB_TOKEN is not set; access is unrestricted");
    return NextResponse.next();
  }

  // 3. Accept token from URL param, set cookie, redirect to clean URL
  const urlToken = searchParams.get(TOKEN_PARAM);
  if (urlToken && urlToken === expectedToken) {
    const cleanUrl = req.nextUrl.clone();
    cleanUrl.searchParams.delete(TOKEN_PARAM);
    const res = NextResponse.redirect(cleanUrl);
    res.cookies.set(COOKIE_NAME, expectedToken, {
      httpOnly: true,
      sameSite: "lax",
      secure: req.nextUrl.protocol === "https:",
      path: "/",
      // maxAge omitted → session cookie
    });
    return res;
  }

  // 4. Validate session cookie
  const sessionToken = req.cookies.get(COOKIE_NAME)?.value ?? "";
  if (sessionToken === expectedToken && expectedToken !== "") {
    return NextResponse.next();
  }

  // 5. Redirect to unlock page
  const unlockUrl = req.nextUrl.clone();
  unlockUrl.pathname = "/unlock";
  unlockUrl.searchParams.delete(TOKEN_PARAM);
  return NextResponse.redirect(unlockUrl);
}

export const config = {
  matcher: [
    // Match all routes except static files already served by Next.js
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
