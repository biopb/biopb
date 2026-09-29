import { useCallback, useEffect, useRef } from "react";

/**
 * Debounces commits per key.
 *
 * `schedule(key, commit)` runs `commit` once `ms` pass with no further call *for
 * that key*. Keys are independent: moving a second slider inside the window must
 * not drop the first one's pending commit, which one timer shared by every
 * slider did. `cancelAll` drops whatever is pending.
 */
export function createDebouncer(ms: number) {
  const timers = new Map<string, ReturnType<typeof setTimeout>>();
  return {
    schedule(key: string, commit: () => void) {
      const previous = timers.get(key);
      if (previous !== undefined) clearTimeout(previous);
      timers.set(
        key,
        setTimeout(() => {
          timers.delete(key);
          commit();
        }, ms),
      );
    },
    cancelAll() {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
    },
  };
}

/**
 * {@link createDebouncer} for a component: pending commits are cancelled on
 * unmount, so a torn-down panel writes nothing.
 */
export function useDebouncedCommit(ms: number): (key: string, commit: () => void) => void {
  const debouncer = useRef<ReturnType<typeof createDebouncer> | null>(null);
  if (debouncer.current === null) debouncer.current = createDebouncer(ms);
  useEffect(() => {
    const current = debouncer.current;
    return () => current?.cancelAll();
  }, []);
  return useCallback((key, commit) => debouncer.current?.schedule(key, commit), []);
}
