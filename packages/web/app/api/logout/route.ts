/**
 * /api/logout — clears the httpOnly session cookie server-side.
 *
 * The biopb_session cookie is httpOnly, so it cannot be deleted from
 * browser JavaScript.  This endpoint must be called instead.
 */

import { NextResponse } from "next/server";

const COOKIE_NAME = "biopb_session";

export async function POST() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set(COOKIE_NAME, "", {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 0,
  });
  return res;
}
