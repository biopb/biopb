import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

/**
 * Store fields that are only correct through their selector.
 *
 * `views` holds one record per tensor, and which record is "the one in view"
 * lives in the selectors (`selectView`, `selectRois`, `selectDraft`, ...).
 * Reading it directly gets some other tensor's annotations. `selectedRoiId` is
 * global but only meaningful once resolved against the rows in view.
 *
 * `getState()` and `setState()` are deliberately not matched: tests seed and
 * assert on raw state, which is the point of them.
 */
const SELECTOR_ONLY_FIELDS = ["views", "selectedRoiId"];

export default tseslint.config(
  { ignores: ["dist", "node_modules", "app"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "no-restricted-syntax": [
        "error",
        {
          selector: `CallExpression[callee.name="useAppStore"] MemberExpression[property.name=/^(${SELECTOR_ONLY_FIELDS.join(
            "|",
          )})$/]`,
          message:
            "This store field is only correct through its selector (selectRois, selectDraft, selectView, ...): the raw field is not scoped to the tensor in view. See src/store/.",
        },
        {
          // zustand v5 reads the snapshot on every render and compares by
          // identity, so a selector that computes an argument computes a fresh
          // array or object each time and React never settles. Cost of getting
          // it wrong is the whole viewer, and no test here can see it: this
          // workspace renders to static markup, which renders once.
          selector:
            'CallExpression[callee.name="useAppStore"] > ArrowFunctionExpression > CallExpression > CallExpression',
          message:
            "Compute this outside the selector (useMemo) and pass the value in: a fresh array or object per snapshot read loops React until it gives up.",
        },
      ],
    },
  },
  {
    // A bare `fetch` carries no token, and a 401 body is valid JSON that
    // callers read straight through (biopb/biopb#730). Everything past the
    // unlock gate goes through `sessionFetch`; the files below are the calls
    // that must stay bare.
    files: ["src/**/*.{ts,tsx}"],
    ignores: [
      // Decide whether a token is needed at all.
      "src/auth.ts",
      // Waits for the sidecar's unauthenticated /readyz.
      "src/ClientBootstrap.tsx",
      // The wrapper itself.
      "src/utils/sessionFetch.ts",
      "src/**/*.test.{ts,tsx}",
    ],
    rules: {
      "no-restricted-globals": [
        "error",
        {
          name: "fetch",
          message: "Use sessionFetch (utils/sessionFetch.ts): a bare fetch sends no token.",
        },
      ],
    },
  },
);
