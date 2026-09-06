"use client";

/**
 * The drawing tools, over the canvas.
 *
 * Here rather than in the control column because switching tool is a per-gesture
 * action taken with the eyes on the image; the things you set once per session
 * (label, set, which axes to pin) live in the panel.
 *
 * Split into a props-taking view and a store container, as RoiPanel is: zustand's
 * server snapshot is `getInitialState()`, so only the view is testable in this
 * workspace's node-only environment.
 */

import { useAppStore } from "../store";
import { isCompletable, minimumPoints, type RoiTool } from "../utils/roiDraft";
import type { RoiDraft } from "../utils/roiDraft";

const TOOLS: Array<{ tool: RoiTool; glyph: string; title: string }> = [
  { tool: "select", glyph: "↖", title: "Select (click an annotation)" },
  { tool: "point", glyph: "•", title: "Point (one click)" },
  { tool: "rectangle", glyph: "▭", title: "Rectangle (two clicks)" },
  { tool: "polygon", glyph: "⬠", title: "Polygon (click vertices, Enter to close)" },
  { tool: "polyline", glyph: "〜", title: "Freehand path (click vertices, Enter to end)" },
];

export interface RoiToolStripViewProps {
  tool: RoiTool;
  draft: RoiDraft | null;
  onSetTool: (tool: RoiTool) => void;
  onFinish: () => void;
  onCancel: () => void;
}

export function RoiToolStripView({
  tool,
  draft,
  onSetTool,
  onFinish,
  onCancel,
}: RoiToolStripViewProps) {
  const placed = draft?.points.length ?? 0;
  const needed = draft ? minimumPoints(draft.tool) : 0;
  const canFinish = isCompletable(draft);

  return (
    <div className="roi-tools">
      <div className="roi-tool-row">
        {TOOLS.map((entry) => (
          <button
            key={entry.tool}
            type="button"
            className={`roi-tool${tool === entry.tool ? " active" : ""}`}
            title={entry.title}
            aria-pressed={tool === entry.tool}
            onClick={() => onSetTool(entry.tool)}
          >
            {entry.glyph}
          </button>
        ))}
      </div>
      {draft && (
        <div className="roi-draft-status">
          <span>
            {canFinish
              ? `${placed} points`
              : `${placed} of ${needed} points`}
          </span>
          <button type="button" className="roi-tool-text" onClick={onFinish} disabled={!canFinish}>
            Finish
          </button>
          <button type="button" className="roi-tool-text" onClick={onCancel}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * The draft comes from the caller, not from the store.
 *
 * TileViewer already holds it through `selectDraft`, which hides a draft whose
 * plane, tensor or render mode has moved on. Reading `s.draft` here instead
 * would show a status line and an enabled Finish for a draft the viewer
 * considers gone -- and Finish would then no-op, because it closes over the
 * guarded value. Taking it as a prop makes the two impossible to disagree.
 */
export function RoiToolStrip({
  draft,
  onFinish,
}: {
  draft: RoiDraft | null;
  onFinish: () => void;
}) {
  const tool = useAppStore((s) => s.tool);
  const onSetTool = useAppStore((s) => s.setTool);
  const setDraft = useAppStore((s) => s.setDraft);
  const unavailable = useAppStore((s) => s.roisUnavailable);
  if (unavailable) return null;
  return (
    <RoiToolStripView
      tool={tool}
      draft={draft}
      onSetTool={onSetTool}
      onFinish={onFinish}
      onCancel={() => setDraft(null)}
    />
  );
}
