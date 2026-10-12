import { useEffect, useState } from "react";
import { authHeaders, getToken } from "../auth";
import { withBase } from "../base";

/** Whether the token this page carries may run code in a session or change one.
 *
 * Asked of the control (`GET /api/scope`), which is the one place that knows: a
 * `split` control gives a viewing token `"view"`, everything else `"exec"`.
 * `false` only for a definite `"view"`; with no token to ask with, an older
 * control with no such route, or a failed read, it is `true`, because the
 * control enforces the real answer and refuses with a 403 either way.
 *
 * A bare `fetch`, not `sessionFetch`: this probe only decides which buttons to
 * grey out, and a 401 to it must not send the user to /unlock. */
export async function fetchCanExec(): Promise<boolean> {
  if (!getToken()) return true;
  try {
    // eslint-disable-next-line no-restricted-globals -- a probe must not redirect
    const r = await fetch(withBase("/api/scope"), { headers: authHeaders() });
    if (!r.ok) return true;
    return (await r.json())?.scope !== "view";
  } catch {
    return true;
  }
}

export function useCanExec(): boolean {
  const [canExec, setCanExec] = useState(true);
  useEffect(() => {
    let live = true;
    void fetchCanExec().then((ok) => {
      if (live) setCanExec(ok);
    });
    return () => {
      live = false;
    };
  }, []);
  return canExec;
}

/** The tooltip for a control a viewing token cannot use. */
export const NEEDS_EXEC = "Needs the exec token — this one can only view";

/** *title* for a control that needs exec, or the reason it is off. */
export const execTitle = (canExec: boolean, title: string): string =>
  canExec ? title : NEEDS_EXEC;
