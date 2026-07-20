import type { ReactNode } from "react";
import {
  enumValues,
  isDeprecated,
  primaryType,
  type SchemaProp,
} from "@biopb/tensor-flight-client";
import { humanizeKey } from "./adminSections";

/**
 * The one schema-driven field renderer shared by every admin editor
 * (SectionFields, CredentialsEditor, SourcesEditor). Given a field key + its
 * JSON-Schema `prop`, it picks the control (boolean checkbox / tri-state,
 * enum select, number, text, or password for a secret), a humanized label with
 * the raw key beside it, the prose `description`, the `constraint` hint, and any
 * inline errors — so the three editors stay consistent and none re-implements
 * control logic.
 *
 * Pure-controlled: it holds no state; `onChange(value)` receives the next value
 * (or `undefined` to unset the key).
 */

interface SchemaFieldProps {
  fieldKey: string;
  prop: SchemaProp;
  value: unknown;
  errs: string[];
  disabled?: boolean;
  onChange: (value: unknown) => void;
  /** Render this field's text/password control as a secret (masked) input. */
  secret?: boolean;
  /** Override the humanized label (e.g. a curated caption). */
  labelText?: string;
  /** Placeholder for text/number inputs. */
  placeholder?: string;
  /** Caption for the "empty" option of an enum/deprecated-bool select. */
  unsetLabel?: string;
  /** Node rendered next to the control (e.g. a "Browse…" button on a path). */
  append?: ReactNode;
  /** Extra help line under the control (e.g. a masked-secret note). */
  note?: ReactNode;
}

export function SchemaField({
  fieldKey,
  prop,
  value,
  errs,
  disabled,
  onChange,
  secret,
  labelText,
  placeholder,
  unsetLabel = "(unset)",
  append,
  note,
}: SchemaFieldProps) {
  const deprecated = isDeprecated(prop);
  const type = primaryType(prop);
  const hardEnum = enumValues(prop);
  const invalid = errs.length > 0;
  const cls = invalid ? "invalid" : undefined;

  let control: ReactNode;
  let isCheckbox = false;

  if (secret) {
    control = (
      <input
        type="password"
        value={value == null ? "" : String(value)}
        disabled={disabled}
        autoComplete="new-password"
        placeholder={placeholder}
        className={cls}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      />
    );
  } else if (type === "boolean" && deprecated) {
    // A deprecated boolean must be *removable* — an unset-able tri-state.
    control = (
      <select
        value={value === true ? "true" : value === false ? "false" : ""}
        disabled={disabled}
        className={cls}
        onChange={(e) =>
          onChange(e.target.value === "" ? undefined : e.target.value === "true")
        }
      >
        <option value="">{unsetLabel}</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  } else if (type === "boolean") {
    // Render the *effective* value (schema default when the key is omitted);
    // toggling back to the default drops the key rather than writing an override.
    const def = prop.default === true;
    const effective = value === undefined ? def : value === true;
    isCheckbox = true;
    control = (
      <input
        type="checkbox"
        checked={effective}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked === def ? undefined : e.target.checked)}
      />
    );
  } else if (hardEnum) {
    control = (
      <select
        value={value == null ? "" : String(value)}
        disabled={disabled}
        className={cls}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      >
        <option value="">{unsetLabel}</option>
        {hardEnum
          .filter((v): v is string => typeof v === "string")
          .map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
      </select>
    );
  } else if (type === "integer" || type === "number") {
    control = (
      <input
        type="number"
        value={value == null ? "" : String(value)}
        disabled={disabled}
        min={prop.minimum}
        max={prop.maximum}
        step={type === "integer" ? 1 : "any"}
        placeholder={placeholder}
        className={cls}
        onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
      />
    );
  } else if (type === "array") {
    // Two array shapes: a fixed numeric vector (grid tile sizes -> one number
    // input per element) and a variable string list (server / origin lists ->
    // editable rows with add/remove). The item type comes from the schema; the
    // starting value falls back to the schema default so an omitted key still
    // renders its effective vector/list.
    const itemType = prop.items ? primaryType(prop.items) : "string";
    const arr: unknown[] = Array.isArray(value)
      ? value
      : Array.isArray(prop.default)
        ? (prop.default as unknown[])
        : [];
    if (itemType === "integer" || itemType === "number") {
      control = (
        <span className="setting-field-vector">
          {arr.map((el, i) => (
            <input
              key={i}
              type="number"
              value={el == null ? "" : String(el)}
              disabled={disabled}
              step={itemType === "integer" ? 1 : "any"}
              className={cls}
              onChange={(e) => {
                const next = arr.slice();
                next[i] = e.target.value === "" ? 0 : Number(e.target.value);
                onChange(next);
              }}
            />
          ))}
        </span>
      );
    } else {
      const list = arr.map((v) => (v == null ? "" : String(v)));
      control = (
        <span className="setting-field-list">
          {list.map((el, i) => (
            <span key={i} className="setting-field-list-row">
              <input
                type="text"
                value={el}
                disabled={disabled}
                placeholder={placeholder}
                onChange={(e) => {
                  const next = list.slice();
                  next[i] = e.target.value;
                  onChange(next);
                }}
              />
              <button
                type="button"
                className="icon-btn"
                disabled={disabled}
                aria-label="Remove"
                onClick={() => onChange(list.filter((_, j) => j !== i))}
              >
                ✕
              </button>
            </span>
          ))}
          <button
            type="button"
            className="icon-btn setting-field-list-add"
            disabled={disabled}
            onClick={() => onChange([...list, ""])}
          >
            + Add
          </button>
        </span>
      );
    }
  } else {
    control = (
      <input
        type="text"
        value={value == null ? "" : String(value)}
        disabled={disabled}
        placeholder={placeholder}
        className={cls}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      />
    );
  }

  return (
    <div
      className={`setting-field${isCheckbox ? " setting-field-bool" : ""}${
        invalid ? " has-error" : ""
      }`}
    >
      <label className="setting-field-label">
        <span className="setting-field-name">
          <span className="setting-field-title">{labelText ?? humanizeKey(fieldKey)}</span>
          <code>{fieldKey}</code>
          {secret && <span className="secret-tag">secret</span>}
          {deprecated && <span className="deprecated-tag">deprecated</span>}
        </span>
        {append ? (
          <span className="setting-field-control-row">
            {control}
            {append}
          </span>
        ) : (
          control
        )}
      </label>
      {prop.description && <p className="setting-field-help">{prop.description}</p>}
      {prop.constraint && <p className="setting-field-constraint">{prop.constraint}</p>}
      {note && <p className="setting-field-help">{note}</p>}
      {errs.map((m, i) => (
        <p key={i} className="setting-field-error">
          {m}
        </p>
      ))}
    </div>
  );
}
