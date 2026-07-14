import { useState } from "react";
import {
  enumValues,
  fieldErrors,
  isDeprecated,
  primaryType,
  sectionProperties,
  type ConfigError,
  type ConfigSchema,
  type SchemaProp,
} from "@biopb/tensor-flight-client";
import { humanizeKey } from "./adminSections";

/**
 * Schema-driven field editor for one config section (server / cache / …), laid
 * out as a fixed settings panel with a **common / advanced split**.
 *
 * The section's curated `commonFields` render directly; every other visible
 * schema field (all non-deprecated fields, plus any deprecated key already on
 * disk) falls under a collapsible "Advanced options" disclosure — so nothing is
 * hidden from a power user, but the default view stays approachable. Field types,
 * bounds, enums, defaults, and help text all come from the schema; this component
 * owns only the grouping.
 *
 * Pure-controlled: every edit calls `onChange` with the next whole config so the
 * one canonical object stays authoritative (the raw-JSON panel stays in sync).
 */

type Config = Record<string, unknown>;

interface SectionFieldsProps {
  section: string;
  commonFields: string[];
  config: Config;
  schema: ConfigSchema | null;
  /** Combined client + server field errors (path-addressed). */
  errors: ConfigError[];
  disabled?: boolean;
  onChange: (next: Config) => void;
}

function sectionObject(config: Config, section: string): Config {
  const v = config[section];
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Config) : {};
}

/** Fields worth rendering: all non-deprecated, plus deprecated ones present. */
function visibleFields(
  props: Record<string, SchemaProp>,
  sectionCfg: Config,
): [string, SchemaProp][] {
  return Object.entries(props).filter(
    ([key, prop]) => !isDeprecated(prop) || key in sectionCfg,
  );
}

export function SectionFields({
  section,
  commonFields,
  config,
  schema,
  errors,
  disabled,
  onChange,
}: SectionFieldsProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  if (!schema) return null;

  const props = sectionProperties(schema, section);
  const sectionCfg = sectionObject(config, section);
  const visible = visibleFields(props, sectionCfg);

  // Split into common (curated order) and advanced (everything else). A field
  // listed as common but absent from the schema is skipped; a common field that
  // is deprecated-and-absent won't be in `visible`, so it won't render either.
  const visibleByKey = new Map(visible);
  const common = commonFields
    .filter((k) => visibleByKey.has(k))
    .map((k) => [k, visibleByKey.get(k)!] as [string, SchemaProp]);
  const commonSet = new Set(common.map(([k]) => k));
  const advanced = visible.filter(([k]) => !commonSet.has(k));

  // An error on an advanced field must force the disclosure open so the message
  // is never hidden (mirrors the old auto-expand behavior).
  const advancedHasError = advanced.some(([key]) =>
    errors.some(
      (e) => e.path.length >= 2 && e.path[0] === section && e.path[1] === key,
    ),
  );
  const advOpen = advancedOpen || advancedHasError;

  function setField(key: string, value: unknown) {
    const sect = { ...sectionObject(config, section) };
    if (value === undefined) delete sect[key];
    else sect[key] = value;
    onChange({ ...config, [section]: sect });
  }

  const renderField = ([key, prop]: [string, SchemaProp]) => (
    <Field
      key={key}
      section={section}
      fieldKey={key}
      prop={prop}
      value={sectionCfg[key]}
      errs={fieldErrors(errors, [section, key])}
      disabled={disabled}
      onChange={(v) => setField(key, v)}
    />
  );

  return (
    <div className="section-fields">
      {common.length > 0 && (
        <div className="section-field-grid">{common.map(renderField)}</div>
      )}

      {advanced.length > 0 && (
        <div className={`section-advanced${advancedHasError ? " has-error" : ""}`}>
          <button
            type="button"
            className="section-advanced-toggle"
            aria-expanded={advOpen}
            onClick={() => setAdvancedOpen(!advOpen)}
          >
            <span className="adv-caret">{advOpen ? "▾" : "▸"}</span>
            <span>Advanced options</span>
            <span className="section-advanced-hint">
              {advanced
                .slice(0, 4)
                .map(([k]) => k)
                .join(", ")}
              {advanced.length > 4 ? ", …" : ""}
            </span>
            {advancedHasError && (
              <span className="adv-section-errdot" title="Has errors" />
            )}
          </button>
          {advOpen && (
            <div className="section-field-grid section-advanced-body">
              {advanced.map(renderField)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface FieldProps {
  section: string;
  fieldKey: string;
  prop: SchemaProp;
  value: unknown;
  errs: string[];
  disabled?: boolean;
  onChange: (value: unknown) => void;
}

function Field({ fieldKey, prop, value, errs, disabled, onChange }: FieldProps) {
  const deprecated = isDeprecated(prop);
  const type = primaryType(prop);
  const hardEnum = enumValues(prop);
  const invalid = errs.length > 0;

  let control;
  if (type === "boolean" && deprecated) {
    // A deprecated boolean (e.g. metadata_db.enabled) must be *removable* — the
    // hint says to drop it — so it gets an unset-able tri-state.
    control = (
      <select
        value={value === true ? "true" : value === false ? "false" : ""}
        disabled={disabled}
        className={invalid ? "invalid" : undefined}
        onChange={(e) =>
          onChange(e.target.value === "" ? undefined : e.target.value === "true")
        }
      >
        <option value="">(unset)</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  } else if (type === "boolean") {
    // Render the *effective* value (schema default when the key is omitted);
    // toggling back to the default drops the key rather than writing an override.
    const def = prop.default === true;
    const effective = value === undefined ? def : value === true;
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
        className={invalid ? "invalid" : undefined}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      >
        <option value="">(unset)</option>
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
        className={invalid ? "invalid" : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(raw === "" ? undefined : Number(raw));
        }}
      />
    );
  } else {
    control = (
      <input
        type="text"
        value={value == null ? "" : String(value)}
        disabled={disabled}
        className={invalid ? "invalid" : undefined}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      />
    );
  }

  const isCheckbox = type === "boolean" && !deprecated;

  return (
    <div
      className={`setting-field${isCheckbox ? " setting-field-bool" : ""}${
        invalid ? " has-error" : ""
      }`}
    >
      <label className="setting-field-label">
        <span className="setting-field-name">
          <span className="setting-field-title">{humanizeKey(fieldKey)}</span>
          <code>{fieldKey}</code>
          {deprecated && <span className="deprecated-tag">deprecated</span>}
        </span>
        {control}
      </label>
      {prop.description && <p className="setting-field-help">{prop.description}</p>}
      {errs.map((m, i) => (
        <p key={i} className="setting-field-error">
          {m}
        </p>
      ))}
    </div>
  );
}
