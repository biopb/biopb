import { useEffect } from "react";

// Sets the browser-tab title for the current route. The SPA ships one static
// <title> in index.html, so without this every route (dashboard, viewer, admin,
// observe, …) would read the same thing; each page sets its own.
export function useDocumentTitle(title: string): void {
  useEffect(() => {
    document.title = title;
  }, [title]);
}
