import { useEffect, useRef, type ReactNode } from "react";

/**
 * Accessible modal dialog: backdrop-click / Esc to close, Tab focus-trap, and
 * focus restored to the trigger on unmount. Mounted conditionally by the caller
 * (so the open/close effect runs once per appearance). `onClose` is read through
 * a ref, so the caller need not memoize it to avoid re-running the trap.
 */

interface ModalProps {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  /** Extra class on the dialog box, e.g. "wide" for the JSON editor. */
  className?: string;
  labelId?: string;
}

export function Modal({ title, onClose, children, className, labelId = "modal-title" }: ModalProps) {
  const ref = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null;
    const focusable = (): HTMLElement[] =>
      Array.from(
        ref.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ) ?? [],
      ).filter((el) => !el.hasAttribute("disabled"));
    focusable()[0]?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const items = focusable();
      const first = items[0];
      const last = items[items.length - 1];
      if (!first || !last) return;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      prevFocus?.focus?.();
    };
  }, []);

  return (
    <div className="admin-modal-backdrop" onClick={() => onCloseRef.current()}>
      <div
        className={`admin-modal${className ? ` ${className}` : ""}`}
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelId}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id={labelId}>{title}</h2>
        {children}
      </div>
    </div>
  );
}
