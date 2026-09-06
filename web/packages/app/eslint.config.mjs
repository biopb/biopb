import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

/**
 * Store fields that are only correct through their selector.
 *
 * Each is scoped to something -- the tensor in view, the plane on screen, the
 * render mode -- and the scope lives in the selector, not the field. Reading the
 * field directly gets the previous tensor's annotations, a warning about a
 * tensor no longer on screen, or a draft the viewer has already dropped. Every
 * one of those has actually shipped and been fixed once.
 *
 * `getState()` and `setState()` are deliberately not matched: tests seed and
 * assert on raw state, which is the point of them.
 */
const SELECTOR_ONLY_FIELDS = [
  "rois",
  "roisPending",
  "roisTruncated",
  "roisSkipped",
  "roisError",
  "hiddenSets",
  "draft",
  "selectedRoiId",
  "broadcastAxes",
  "tileInfo",
];

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
            "This store field is only correct through its selector (selectRois, selectDraft, selectTileInfo, ...): the raw field is not scoped to the tensor, plane or render mode in view. See src/store.ts.",
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
);
