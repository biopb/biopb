import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // .tsx too, for server-rendered smoke tests of a component. Still no DOM:
    // renderToStaticMarkup runs in node, which is enough to catch a component
    // that throws on mount — the failure tsc cannot see.
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    environment: "node",
    globals: false,
  },
});
