import { describe, expect, it } from "vitest";
import { sqlLiteral } from "./sql";

describe("sqlLiteral", () => {
  it("quotes, and doubles an embedded quote", () => {
    expect(sqlLiteral("src0")).toBe("'src0'");
    expect(sqlLiteral("it's")).toBe("'it''s'");
  });

  it("leaves a backslash and a LIKE wildcard alone (it is an equality literal)", () => {
    expect(sqlLiteral("a\\_b%")).toBe("'a\\_b%'");
  });
});
