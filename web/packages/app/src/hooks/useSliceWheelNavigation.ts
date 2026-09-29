import { useEffect } from "react";
import type { RefObject } from "react";
import type { TileInfo } from "@biopb/tensor-flight-client";
import { useAppStore } from "../store";
import { createDebouncer } from "./useDebouncedCommit";

/**
 * Slice navigation: hold one of these and scroll.
 *
 * Only the named axes get a key, because there is no letter to press for an
 * axis called `i` or `POS` that would not collide with something. Those are
 * navigated with their slider in {@link SliceControls}.
 */
const SLICE_KEYS = ["t", "z", "c"] as const;
const SLICE_WHEEL_QUIET_MS = 120;

/**
 * Hold t/z/c and scroll to step that axis.
 *
 * Capture phase on the pane: deck.gl listens on the canvas below, so stopping
 * the event here is what keeps a slice scroll from also zooming. The whole
 * gesture is accumulated and applied once it stops, rather than one store write
 * per wheel notch — each write costs a tile refetch.
 */
export function useSliceWheelNavigation(
  ref: RefObject<HTMLElement | null>,
  info: TileInfo | null,
) {
  useEffect(() => {
    const el = ref.current;
    if (!el || !info) return;

    const held = new Set<string>();
    const pending = { axis: null as (typeof SLICE_KEYS)[number] | null, steps: 0 };
    const debouncer = createDebouncer(SLICE_WHEEL_QUIET_MS);

    const flush = () => {
      const { axis, steps } = pending;
      pending.axis = null;
      pending.steps = 0;
      const wireAxis = axis === null ? null : info.selectable[axis];
      if (axis === null || wireAxis === null || steps === 0) return;
      const max = Math.max(0, (info.shape[wireAxis] ?? 1) - 1);
      const current = useAppStore.getState().position[axis];
      useAppStore
        .getState()
        .setPosition({ [axis]: Math.min(max, Math.max(0, current + steps)) });
    };

    const onWheel = (e: WheelEvent) => {
      const axis = SLICE_KEYS.find((k) => held.has(k));
      if (!axis || info.selectable[axis] === null) return;
      e.preventDefault();
      e.stopPropagation();
      if (pending.axis !== axis) {
        pending.axis = axis;
        pending.steps = 0;
      }
      pending.steps += e.deltaY > 0 ? -1 : 1;
      debouncer.schedule("wheel", flush);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if ((SLICE_KEYS as readonly string[]).includes(key)) held.add(key);
    };
    const onKeyUp = (e: KeyboardEvent) => held.delete(e.key.toLowerCase());
    const onBlur = () => held.clear();

    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      el.removeEventListener("wheel", onWheel, { capture: true });
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      debouncer.cancelAll();
    };
  }, [ref, info]);
}
