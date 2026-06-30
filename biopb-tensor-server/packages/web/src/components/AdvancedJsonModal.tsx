import { useState } from "react";
import { Modal } from "./Modal";

/**
 * Modal editor for the whole config as raw JSON (the "advanced" surface).
 *
 * Owns a local text draft seeded from the current config and only commits on
 * Apply, so a malformed edit never touches the canonical config and the
 * structured Sources editor never sees a half-typed object. Invalid JSON is
 * shown prominently inside the dialog and blocks Apply.
 */

interface AdvancedJsonModalProps {
  config: Record<string, unknown>;
  onApply: (next: Record<string, unknown>) => void;
  onClose: () => void;
}

export function AdvancedJsonModal({ config, onApply, onClose }: AdvancedJsonModalProps) {
  const [text, setText] = useState(() => JSON.stringify(config, null, 2));
  const [error, setError] = useState<string | null>(null);

  function apply() {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid JSON");
      return;
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      setError("Config must be a JSON object.");
      return;
    }
    onApply(parsed as Record<string, unknown>);
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
        onChange={(e) => {
          setText(e.target.value);
          if (error) setError(null);
        }}
      />
      <div className="admin-modal-actions">
        <button type="button" className="icon-btn" onClick={onClose}>
          Cancel
        </button>
        <button type="button" className="submit-btn" onClick={apply}>
          Apply
        </button>
      </div>
    </Modal>
  );
}
