import { describe, it, expect } from "vitest";
import {
  ADVANCED_SECTIONS,
  credentialProfileProperties,
  credentialProfileRequired,
  enumValues,
  fieldErrors,
  isDeprecated,
  isSecretProfileKey,
  primaryType,
  REDACTED_SENTINEL,
  sectionProperties,
  sourceItemProperties,
  sourceRequiredKeys,
  validateConfig,
  type ConfigSchema,
} from "./config-schema.js";

// A trimmed schema shaped exactly like build_config_schema() output: sections
// keep additionalProperties:true, case-sensitive enums carry `enum`, and
// case-insensitive ones (reduction_method) carry only a `description`.
const SCHEMA: ConfigSchema = {
  type: "object",
  additionalProperties: true,
  properties: {
    server: {
      type: "object",
      additionalProperties: true,
      properties: {
        host: { type: "string" },
        port: { type: "integer", minimum: 1, maximum: 65535 },
        log_level: {
          type: "string",
          description: "one of: debug, error, info, warning",
        },
        watcher_type: {
          type: "string",
          description: "Deprecated alias for monitor_mode.",
          deprecated: true,
        },
      },
    },
    pyramid: {
      type: "object",
      additionalProperties: true,
      properties: {
        downscale_factor: { type: "integer", minimum: 2 },
        reduction_method: {
          type: "string",
          description: "one of: area, linear, nearest",
        },
      },
    },
    cache: {
      type: "object",
      additionalProperties: true,
      properties: {
        backend: { type: "string", enum: ["memory", "file"] },
        max_bytes: { type: "integer", minimum: 1 },
      },
    },
    sources: {
      type: "array",
      items: {
        type: "object",
        required: ["url"],
        additionalProperties: true,
        properties: {
          url: { type: "string" },
          monitor: { type: "boolean" },
          type: { type: ["string", "null"], enum: ["zarr", "ome-zarr", null] },
          path: { type: "string", deprecated: true },
        },
      },
    },
    credentials: {
      type: "object",
      properties: {
        default_profile: { type: "string" },
        profiles: {
          type: "array",
          items: {
            type: "object",
            required: ["name"],
            properties: {
              name: { type: "string" },
              storage_type: { type: "string" },
              key: { type: "string" },
              secret: { type: "string" },
              token: { type: "string" },
            },
          },
        },
      },
    },
  },
};

describe("schema accessors", () => {
  it("lists section properties and skips non-object sections", () => {
    expect(Object.keys(sectionProperties(SCHEMA, "server"))).toContain("port");
    expect(sectionProperties(SCHEMA, "sources")).toEqual({});
    expect(sectionProperties(null, "server")).toEqual({});
  });

  it("reads source item props and required keys", () => {
    expect(Object.keys(sourceItemProperties(SCHEMA))).toContain("url");
    expect(sourceRequiredKeys(SCHEMA)).toEqual(["url"]);
  });

  it("detects deprecated / enum / primary type", () => {
    const server = sectionProperties(SCHEMA, "server");
    expect(isDeprecated(server.watcher_type)).toBe(true);
    expect(isDeprecated(server.host)).toBe(false);
    expect(enumValues(sectionProperties(SCHEMA, "cache").backend)).toEqual([
      "memory",
      "file",
    ]);
    expect(enumValues(server.host)).toBeNull();
    // A union ["string","null"] picks the non-null member for the control.
    expect(primaryType(sourceItemProperties(SCHEMA).type)).toBe("string");
    expect(primaryType(server.port)).toBe("integer");
  });

  it("exposes the ordered advanced sections", () => {
    expect(ADVANCED_SECTIONS).toContain("server");
    expect(ADVANCED_SECTIONS).toContain("precache");
  });
});

describe("validateConfig", () => {
  it("passes a valid config", () => {
    const cfg = {
      server: { host: "0.0.0.0", port: 8815 },
      cache: { backend: "memory", max_bytes: 1000 },
      sources: [{ url: "/data", monitor: true, type: "zarr" }],
    };
    expect(validateConfig(cfg, SCHEMA)).toEqual([]);
  });

  it("flags an out-of-range integer with a bound message", () => {
    const errs = validateConfig({ server: { port: 70000 } }, SCHEMA);
    expect(errs).toEqual([{ path: ["server", "port"], message: "must be ≤ 65535" }]);
  });

  it("flags a below-minimum integer", () => {
    const errs = validateConfig({ pyramid: { downscale_factor: 1 } }, SCHEMA);
    expect(errs).toEqual([
      { path: ["pyramid", "downscale_factor"], message: "must be ≥ 2" },
    ]);
  });

  it("flags a bad hard enum", () => {
    const errs = validateConfig({ cache: { backend: "bogus" } }, SCHEMA);
    expect(errs).toHaveLength(1);
    expect(errs[0]!.path).toEqual(["cache", "backend"]);
    expect(errs[0]!.message).toContain("memory");
    expect(errs[0]!.message).toContain("file");
  });

  it("does NOT hard-fail case-insensitive enums (no schema enum)", () => {
    // reduction_method / log_level carry only a description, so the server is
    // lenient and so are we.
    expect(
      validateConfig({ pyramid: { reduction_method: "AREA" } }, SCHEMA),
    ).toEqual([]);
    expect(validateConfig({ server: { log_level: "DEBUG" } }, SCHEMA)).toEqual([]);
  });

  it("requires a non-empty source url", () => {
    const errs = validateConfig({ sources: [{ url: "" }, {}] }, SCHEMA);
    expect(errs).toEqual([
      { path: ["sources", 0, "url"], message: "url is required" },
      { path: ["sources", 1, "url"], message: "url is required" },
    ]);
  });

  it("accepts a legacy `path`-only source (deprecated alias for url)", () => {
    // The server reads `url or path`, so a path-only source is valid and must
    // not trip the required-url check (regression: greyed-out Save on load).
    expect(validateConfig({ sources: [{ path: "/legacy" }] }, SCHEMA)).toEqual([]);
  });

  it("accepts null for a nullable source type enum", () => {
    expect(
      validateConfig({ sources: [{ url: "/d", type: null }] }, SCHEMA),
    ).toEqual([]);
  });

  it("flags a bad source type enum on the right row", () => {
    const errs = validateConfig(
      { sources: [{ url: "/d", type: "zarr" }, { url: "/e", type: "nope" }] },
      SCHEMA,
    );
    expect(errs).toHaveLength(1);
    expect(errs[0]!.path).toEqual(["sources", 1, "type"]);
  });

  it("flags a gross type mismatch", () => {
    const errs = validateConfig({ server: { port: "8815" } }, SCHEMA);
    expect(errs).toEqual([
      { path: ["server", "port"], message: "expected integer" },
    ]);
  });

  it("ignores unknown keys (additionalProperties) and blank optionals", () => {
    const cfg = {
      server: { host: "", made_up_key: 1 },
      sources: [{ url: "/d", type: "" }],
    };
    expect(validateConfig(cfg, SCHEMA)).toEqual([]);
  });

  it("is null-safe", () => {
    expect(validateConfig(null, SCHEMA)).toEqual([]);
    expect(validateConfig({}, null)).toEqual([]);
  });

  it("requires a credential profile name but ignores masked secrets", () => {
    const cfg = {
      credentials: {
        default_profile: "s3",
        profiles: [
          // valid: name present, secret is the round-tripping mask
          { name: "s3", storage_type: "s3", secret: REDACTED_SENTINEL },
          // invalid: no name
          { storage_type: "gs" },
        ],
      },
    };
    const errs = validateConfig(cfg, SCHEMA);
    expect(errs).toEqual([
      { path: ["credentials", "profiles", 1, "name"], message: "name is required" },
    ]);
  });
});

describe("credential helpers", () => {
  it("exposes profile props / required / secret keys", () => {
    expect(Object.keys(credentialProfileProperties(SCHEMA))).toContain("secret");
    expect(credentialProfileRequired(SCHEMA)).toEqual(["name"]);
    expect(isSecretProfileKey("secret")).toBe(true);
    expect(isSecretProfileKey("key")).toBe(true);
    expect(isSecretProfileKey("token")).toBe(true);
    expect(isSecretProfileKey("region")).toBe(false);
    expect(isSecretProfileKey("name")).toBe(false);
  });
});

describe("fieldErrors", () => {
  const errs = [
    { path: ["server", "port"], message: "must be ≤ 65535" },
    { path: ["sources", 0, "url"], message: "url is required" },
  ];
  it("matches an exact path element-wise (string-compared)", () => {
    expect(fieldErrors(errs, ["server", "port"])).toEqual(["must be ≤ 65535"]);
    // numeric vs string index still matches
    expect(fieldErrors(errs, ["sources", "0", "url"])).toEqual([
      "url is required",
    ]);
  });
  it("returns [] for a non-match / prefix", () => {
    expect(fieldErrors(errs, ["server"])).toEqual([]);
    expect(fieldErrors(errs, ["cache", "backend"])).toEqual([]);
  });
});
