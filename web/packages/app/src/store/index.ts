import { create } from "zustand";
import { createConnectionSlice } from "./connection";
import { createJobsSlice } from "./jobs";
import { createPreferencesSlice } from "./preferences";
import { createRecentsSlice } from "./recents";
import { createRuntimeSlice } from "./runtime";
import { createTargetSlice } from "./target";
import type { AppState } from "./types";
import { createViewsSlice } from "./views";

export type { AppState } from "./types";
export * from "./connection";
export * from "./jobs";
export * from "./preferences";
export * from "./recents";
export * from "./runtime";
export * from "./slice";
export * from "./target";
export * from "./views";
export * from "./tensorView";
export * from "./roi";

export const useAppStore = create<AppState>()((...a) => ({
  ...createConnectionSlice(...a),
  ...createRecentsSlice(...a),
  ...createJobsSlice(...a),
  ...createPreferencesSlice(...a),
  ...createTargetSlice(...a),
  ...createViewsSlice(...a),
  ...createRuntimeSlice(...a),
}));
