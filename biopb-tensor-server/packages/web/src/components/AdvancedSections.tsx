import { useEffect, useState } from "react";
import {
  ADVANCED_SECTIONS,
  enumValues,
  fieldErrors,
  isDeprecated,
  primaryType,
  sectionProperties,
  type ConfigError,
  type ConfigSchema,
  type SchemaProp,
} from "@biopb/tensor-flight-client";

/**
 * Structured "advanced" editor: collapsible per-field sections for
 * server / cache / pyramid / precache (…), driven entirely by the config JSON
 * Schema from `GET /api/config`. Replaces the raw-JSON blob as the primary
 * surface; the raw editor stays reachable as an escape hatch (`onEditRaw`).
 *
 * Pure-controlled, like SourcesEditor: every edit calls `onChange` with the
 * next whole config so the one canonical object stays authoritative. Deprecated
 * keys are only rendered when already present on disk (never offered fresh),
 * tagged and hinted toward their canonical key; each field validates inline
 * against `errors` and surfaces its schema `description` as helper text.
 */

type Config = Record<string, unknown>;

interface AdvancedSectionsProps {
  config: Config;
  schema: ConfigSchema | null;
  /** Combined client + server field errors (path-addressed). */
  errors: ConfigError[];
  disabled?: boolean;
  onChange: (next: Config) => void;
  onEditRaw: () => void;
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

function title(section: string): string {
  return section.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function AdvancedSections({
  config,
  schema,
  errors,
  disabled,
  onChange,
  onEditRaw,
}: AdvancedSectionsProps) {
  const [userOpen, setUserOpen] = useState<Record<string, boolean>>({});

  // Latch a section open the moment it acquires an error, so that *fixing* the
  // error (which clears `hasError`) doesn't collapse the section and unmount the
  // input being edited mid-keystroke. `open` still ORs `hasError`, so an errored
  // section can't be collapsed until fixed.
  useEffect(() => {
    const errored = new Set(errors.map((e) => String(e.path[0])));
    if (errored.size === 0) return;
    setUserOpen((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const s of errored) {
        if ((ADVANCED_SECTIONS as readonly string[]).includes(s) && !next[s]) {
          next[s] = true;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [errors]);

  if (!schema) return null;

  function setField(section: string, key: string, value: unknown) {
    const sect = { ...sectionObject(config, section) };
    if (value === undefined) delete sect[key];
    else sect[key] = value;
    onChange({ ...config, [section]: sect });
  }

  const sections = ADVANCED_SECTIONS.map((section) => {
    const props = sectionProperties(schema, section);
    const sectionCfg = sectionObject(config, section);
    const fields = visibleFields(props, sectionCfg);
    return { section, fields, sectionCfg };
  }).filter((s) => s.fields.length > 0);

  return (
    <section className="advanced-sections">
      <div className="advanced-sections-head">
        <h2>Advanced settings</h2>
        <button type="button" className="icon-btn" disabled={disabled} onClick={onEditRaw}>
          Edit raw JSON…
        </button>
      </div>

      {sections.map(({ section, fields, sectionCfg }) => {
        const hasError = errors.some((e) => e.path[0] === section);
        // An error always forces its section open so the message is never hidden.
        const open = (userOpen[section] ?? false) || hasError;
        const hint = fields
          .slice(0, 4)
          .map(([k]) => k)
          .join(", ");
        return (
          <div key={section} className={`adv-section${hasError ? " has-error" : ""}`}>
            <button
              type="button"
              className="adv-section-header"
              aria-expanded={open}
              onClick={() => setUserOpen((s) => ({ ...s, [section]: !open }))}
            >
              <span className="adv-caret">{open ? "▾" : "▸"}</span>
              <span className="adv-section-title">{title(section)}</span>
              <span className="adv-section-hint">
                {hint}
                {fields.length > 4 ? ", …" : ""}
              </span>
              {hasError && <span className="adv-section-errdot" title="Has errors" />}
            </button>

            {open && (
              <div className="adv-section-body">
                {fields.map(([key, prop]) => (
                  <Field
                    key={key}
                    section={section}
                    fieldKey={key}
                    prop={prop}
                    value={sectionCfg[key]}
                    errs={fieldErrors(errors, [section, key])}
                    disabled={disabled}
                    onChange={(v) => setField(section, key, v)}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </section>
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
    // hint says to drop it — so it gets an unset-able tri-state rather than a
    // checkbox that can only write true/false.
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
    control = (
      <input
        type="checkbox"
        checked={value === true}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  } else if (hardEnum) {
    control = (
      <select
        value={value == null ? "" : String(value)}
        disabled={disabled}
        className={invalid ? "invalid" : undefined}
        onChange={(e) =>
          onChange(e.target.value === "" ? undefined : e.target.value)
        }
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

  return (
    <div className={`adv-field${invalid ? " has-error" : ""}`}>
      <label className="adv-field-label">
        <span className="adv-field-name">
          <code>{fieldKey}</code>
          {deprecated && <span className="deprecated-tag">deprecated</span>}
        </span>
        {control}
      </label>
      {prop.description && <p className="adv-field-help">{prop.description}</p>}
      {errs.map((m, i) => (
        <p key={i} className="adv-field-error">
          {m}
        </p>
      ))}
    </div>
  );
}
