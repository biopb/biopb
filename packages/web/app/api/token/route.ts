/**
 * /api/token — returns the API token to the authenticated browser session.
 *
 * This endpoint is protected by the middleware (requires valid session cookie).
 * The browser fetches this once after page load to get the token it needs
 * to authenticate direct calls to the FastAPI sidecar.
 */

import { cookies } from "next/headers";
import { NextResponse } from "next/server";

const COOKIE_NAME = "biopb_session";

export async function GET() {
  const cookieStore = cookies();
  const sessionToken = cookieStore.get(COOKIE_NAME)?.value ?? null;

  const devMode = process.env.NEXT_PUBLIC_DEV_MODE === "true";
  const apiBase = process.env.BIOPB_TENSOR_API ?? "http://localhost:8816";

  return NextResponse.json({
    token: devMode ? null : sessionToken,
    apiBase,
    devMode,
  });
}
