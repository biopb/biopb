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

  it("shows a row's warning apart from its errors", () => {
    const html = renderToStaticMarkup(
      <SourcesEditor
        sources={[{ url: "/data/typo" }, { url: "/data/ok" }]}
        onChange={() => {}}
        schema={schema as never}
        errorsByIndex={{}}
        warningsByIndex={{ 0: ["No such file or folder on the server."] }}
      />,
    );
    expect(html.match(/source-row-warnings/g)).toHaveLength(1);
    expect(html).toContain("No such file or folder on the server.");
    expect(html).not.toContain("source-row-errors");
  });

  it("offers Browse… on a local row whenever the page gives it a handler", () => {
    const html = renderToStaticMarkup(
      <SourcesEditor
        sources={[{ url: "/data" }, { url: "grpc://lab:8815" }]}
        onChange={() => {}}
        schema={schema as never}
        errorsByIndex={{}}
        onBrowse={() => {}}
      />,
    );
    expect(html.match(/Browse…/g)).toHaveLength(1);
  });
});
