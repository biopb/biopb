import { useNotices } from "../utils/notices";

/** The viewer's notices, bottom-right: info ones fade on their own, errors wait. */
export function NoticeToasts() {
  const notices = useNotices((s) => s.notices);
  const dismiss = useNotices((s) => s.dismiss);
  if (notices.length === 0) return null;
  return (
    <div className="notice-stack">
      {notices.map((n) => (
        <div key={n.id} className={`notice ${n.kind}`} role={n.kind === "error" ? "alert" : "status"}>
          <div className="notice-body">
            <div>{n.text}</div>
            {n.details.map((d) => (
              <div key={d} className="notice-detail" title={d}>
                {d}
              </div>
            ))}
          </div>
          <button
            type="button"
            className="notice-close"
            aria-label="Dismiss"
            onClick={() => dismiss(n.id)}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
