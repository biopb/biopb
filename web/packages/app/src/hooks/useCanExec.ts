import { useEffect, useState } from "react";
import { withBase } from "../base";
import { sessionFetch } from "../utils/sessionFetch";

/** Whether the token this page carries may run code in a session or change one.
 *
 * Asked of the control (`GET /api/scope`), which is the one place that knows: a
 * `split` control gives a viewing token `"view"`, everything else `"exec"`.
 * Only a definite `"view"` disables anything -- an older control with no such
 * route, or a failed read, leaves the controls alone, because the control
 * enforces the real answer and refuses with a 403 either way. */
export function useCanExec(): boolean {
  const [canExec, setCanExec] = useState(true);
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const r = await sessionFetch(withBase("/api/scope"));
        const j = await r.json();
        if (live) setCanExec(j?.scope !== "view");
      } catch {
        /* keep the default */
      }
    })();
    return () => {
      live = false;
    };
  }, []);
  return canExec;
}

/** The tooltip for a control a viewing token cannot use. */
export const NEEDS_EXEC = "Needs the exec token — this one can only view";
