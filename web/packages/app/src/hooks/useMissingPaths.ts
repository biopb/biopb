import { useEffect, useMemo, useRef, useState } from "react";
import { TensorApiError, type BrowseResponse } from "@biopb/tensor-flight-client";

/**
 * The server-side path a source `url` names, or null when it is not one worth
 * checking: blank, or a remote scheme (`grpc://`, `s3://`, `dnd://`, ...).
 */
export function checkablePath(url: unknown): string | null {
  if (typeof url !== "string") return null;
  const path = url.trim();
  return path && !path.includes("://") ? path : null;
}

/** Wait after the last keystroke before asking the server about a path. */
const SETTLE_MS = 500;

/**
 * Which of `paths` the server says do not exist.
 *
 * Asks `browse` once per distinct path, after it has stopped changing, so
 * typing a path does not ask about every prefix. A failed lookup (network, a
 * permission error, an older server) is not a verdict: the path is left
 * unflagged rather than reported missing.
 */
export function useMissingPaths(
  browse: (path?: string) => Promise<BrowseResponse>,
  paths: readonly string[],
): Set<string> {
  const [exists, setExists] = useState<Record<string, boolean>>({});
  const asked = useRef(new Set<string>());
  const key = paths.join("\n");

  useEffect(() => {
    const todo = [...new Set(key ? key.split("\n") : [])].filter(
      (p) => !asked.current.has(p),
    );
    if (todo.length === 0) return;
    const timer = setTimeout(() => {
      for (const path of todo) {
        asked.current.add(path);
        browse(path).then(
          (res) => setExists((prev) => ({ ...prev, [path]: res.exists !== false })),
          (e) => {
            // A missing child under a missing parent has nothing to list.
            const gone = e instanceof TensorApiError && e.status === 404 && /Not a directory/.test(e.message);
            if (gone) setExists((prev) => ({ ...prev, [path]: false }));
          },
        );
      }
    }, SETTLE_MS);
    return () => clearTimeout(timer);
  }, [key, browse]);

  return useMemo(
    () => new Set(Object.keys(exists).filter((p) => !exists[p])),
    [exists],
  );
}
