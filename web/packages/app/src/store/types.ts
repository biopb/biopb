import type { ConnectionSlice } from "./connection";
import type { JobsSlice } from "./jobs";
import type { PreferencesSlice } from "./preferences";
import type { RecentsSlice } from "./recents";
import type { RuntimeSlice } from "./runtime";
import type { TargetSlice } from "./target";
import type { ViewsSlice } from "./views";

/** The whole store: one slice per lifetime, composed in `index.ts`. */
export type AppState = ConnectionSlice &
  RecentsSlice &
  JobsSlice &
  PreferencesSlice &
  TargetSlice &
  ViewsSlice &
  RuntimeSlice;

export type Get = () => AppState;
export type Set = (partial: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void;
