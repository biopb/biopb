import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SourcesEditor } from "./SourcesEditor";

const schema = {
  properties: {
    sources: {
      items: {
        properties: {
          url: { type: "string" },
          alias: { type: "string", description: "(experimental) Name it appears under" },
          cloud: { type: "boolean", description: "(experimental) Cloud root" },
        },
      },
    },
  },
};

const render = (sources: Record<string, unknown>[]) =>
  renderToStaticMarkup(
    <SourcesEditor
      sources={sources}
      onChange={() => {}}
      schema={schema as never}
      errorsByIndex={{}}
    />,
  );

describe("SourcesEditor", () => {
  it("shows alias and cloud on a freshly added local source, flagged experimental", () => {
    const html = render([{ url: "", monitor: false }]);
    expect(html).toContain("alias");
    expect(html).toContain("cloud");
    expect(html).toContain("(experimental) Cloud root");
  });

  it("offers no monitor option on a remote source, which is always re-listed", () => {
    const html = render([{ url: "grpc://lab:8815", alias: "" }]);
    expect(html).not.toContain("monitor");
  });
});
