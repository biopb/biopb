import type { CSSProperties } from "react";

/** Centred status text over the canvas: "Loading tiles…" and its kin. */
export const OVERLAY_TEXT: CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "#94a3b8",
  fontSize: 13,
};

/** A small readout laid over the canvas. Callers place it. */
export const BADGE: CSSProperties = {
  position: "absolute",
  padding: "4px 8px",
  borderRadius: 4,
  background: "rgba(0, 0, 0, 0.65)",
  color: "#cbd5e1",
  fontSize: 11,
  pointerEvents: "none",
  zIndex: 2,
};

/** A grey level as the badges print it: whole, or four significant digits. */
export function greyLevel(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}
