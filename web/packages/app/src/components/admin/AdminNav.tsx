import { ADMIN_NAV } from "./adminSections";

/**
 * The admin settings left navigation: a fixed, flat list of section links
 * (chrome://settings style). The active section is highlighted; a section that
 * currently holds a validation error shows a red dot so the user can find it
 * without hunting, since Save is disabled while any error exists.
 */

interface AdminNavProps {
  active: string;
  /** Nav ids that currently carry a validation error. */
  erroredIds: Set<string>;
  onSelect: (id: string) => void;
}

export function AdminNav({ active, erroredIds, onSelect }: AdminNavProps) {
  return (
    <nav className="admin-nav" aria-label="Settings sections">
      {ADMIN_NAV.map((item) => (
        <button
          key={item.id}
          type="button"
          className={`admin-nav-item${item.id === active ? " active" : ""}`}
          aria-current={item.id === active ? "page" : undefined}
          onClick={() => onSelect(item.id)}
        >
          <span className="admin-nav-label">{item.label}</span>
          {erroredIds.has(item.id) && (
            <span className="admin-nav-errdot" title="Has errors" />
          )}
        </button>
      ))}
    </nav>
  );
}
