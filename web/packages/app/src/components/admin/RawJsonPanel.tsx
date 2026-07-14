import { useEffect, useMemo, useState } from "react";

/**
 * Inline raw-JSON editor panel (the "advanced" escape hatch), for keys the
 * structured sections don't surface or for bulk edits.
 *
 * Owns a local text draft seeded from the current config; it only commits to the
 * canonical config on "Apply to form", so a half-typed/invalid edit never reaches
 * the structured editors. Invalid JSON is shown in-panel and disables Apply. When
 * the config changes underneath it (another section edited, or a reload after
 * restart) the draft re-seeds — but only while it is unedited, so an in-progress
 * raw edit is never clobbered.
 */

interface RawJsonPanelProps {
  config: Record<string, unknown>;
  disabled?: boolean;
  onApply: (next: Record<string, unknown>) => void;
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

export function RawJsonPanel({ config, disabled, onApply }: RawJsonPanelProps) {
  const serialized = useMemo(() => JSON.stringify(config, null, 2), [config]);
  const [text, setText] = useState(serialized);
  const [dirty, setDirty] = useState(false);

  // Re-seed from the canonical config when it changes underneath us, but only
  // while the draft is untouched — never discard an in-progress raw edit.
  useEffect(() => {
    if (!dirty) setText(serialized);
  }, [serialized, dirty]);

  const { value, error } = useMemo(() => parseConfig(text), [text]);
  const changed = text !== serialized;

  function apply() {
    if (!value) return;
    onApply(value);
    setDirty(false);
  }

  return (
    <div className="raw-json-panel">
      {error && (
        <div className="admin-banner error" role="alert">
          <strong>Invalid JSON</strong> — {error}
        </div>
      )}
      <textarea
        className={`admin-raw${error ? " invalid" : ""}`}
        spellCheck={false}
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value);
          setDirty(true);
        }}
      />
      <div className="raw-json-actions">
        <button
          type="button"
          className="submit-btn"
          disabled={disabled || !!error || !changed}
          onClick={apply}
        >
          Apply to form
        </button>
        <button
          type="button"
          className="icon-btn"
          disabled={disabled || !changed}
          onClick={() => {
            setText(serialized);
            setDirty(false);
          }}
        >
          Revert
        </button>
        <span className="admin-hint">
          Applies into the form above; use <strong>Save</strong> to write it to
          disk.
        </span>
      </div>
    </div>
  );
}
