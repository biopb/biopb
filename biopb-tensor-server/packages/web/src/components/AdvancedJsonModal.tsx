import { useMemo, useState } from "react";
import { Modal } from "./Modal";

/**
 * Modal editor for the whole config as raw JSON (the "advanced" surface).
 *
 * Owns a local text draft seeded from the current config and only commits on
 * Apply, so a malformed edit never touches the canonical config and the
 * structured Sources editor never sees a half-typed object. Validity is derived
 * live from the draft: invalid JSON is shown in-dialog and disables Apply.
 */

interface AdvancedJsonModalProps {
  config: Record<string, unknown>;
  onApply: (next: Record<string, unknown>) => void;
  onClose: () => void;
}

function parseConfig(text: string): {
  value?: Record<string, unknown>;
  error: string | null;
} {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return { error: e instanceof Error ? e.message : "Invalid JSON" };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { error: "Config must be a JSON object." };
  }
  return { value: parsed as Record<string, unknown>, error: null };
}

export function AdvancedJsonModal({ config, onApply, onClose }: AdvancedJsonModalProps) {
  const [text, setText] = useState(() => JSON.stringify(config, null, 2));
  const { value, error } = useMemo(() => parseConfig(text), [text]);

  function apply() {
    if (!value) return;
    onApply(value);
    onClose();
  }

  return (
    <Modal
      title="Advanced — full config (raw JSON)"
      onClose={onClose}
      className="wide"
      labelId="admin-advanced-title"
    >
      {error && (
        <div className="admin-banner error" role="alert">
          <strong>Invalid JSON</strong> — {error}
        </div>
      )}
      <textarea
        className={`admin-raw${error ? " invalid" : ""}`}
        spellCheck={false}
        value={text}
        autoFocus
        onChange={(e) => setText(e.target.value)}
      />
      <div className="admin-modal-actions">
        <button type="button" className="icon-btn" onClick={onClose}>
          Cancel
        </button>
        <button
          type="button"
          className="submit-btn"
          onClick={apply}
          disabled={!!error}
        >
          Apply
        </button>
      </div>
    </Modal>
  );
}
