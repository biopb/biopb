import { create } from "zustand";

export type NoticeKind = "info" | "error";

export interface Notice {
  id: number;
  kind: NoticeKind;
  text: string;
  /** Extra lines under the text, e.g. the paths that failed. */
  details: string[];
}

/** How long an info notice stays up. An error stays until it is dismissed. */
export const INFO_MS = 6000;
/** Notices held at once; a newer one pushes the oldest out. */
const MAX_NOTICES = 4;

interface NoticeState {
  notices: Notice[];
  /** Show a notice; returns its id. */
  notify: (kind: NoticeKind, text: string, details?: string[]) => number;
  dismiss: (id: number) => void;
}

let nextId = 1;
const timers = new Map<number, ReturnType<typeof setTimeout>>();

/**
 * What happened, told once: the result of an action that has no row of its own
 * to show it on (an add that finished, a removal that failed). Not for state
 * that is still true -- that belongs where it is drawn.
 *
 * A module store rather than a slice of the app store: nothing reads it but the
 * toast stack, and an action reaches it from outside React.
 */
export const useNotices = create<NoticeState>((set, get) => ({
  notices: [],
  notify(kind, text, details = []) {
    const id = nextId++;
    set((s) => ({ notices: [...s.notices, { id, kind, text, details }].slice(-MAX_NOTICES) }));
    if (kind === "info") timers.set(id, setTimeout(() => get().dismiss(id), INFO_MS));
    return id;
  },
  dismiss(id) {
    clearTimeout(timers.get(id));
    timers.delete(id);
    set((s) => ({ notices: s.notices.filter((n) => n.id !== id) }));
  },
}));
