"use client";

/**
 * The footer strip: one non-obvious thing about the viewer at a time.
 *
 * Rotating text at the bottom of the screen is easy to make into wallpaper, so
 * three things keep it honest: only tips whose context is on screen are in the
 * rotation (see `TIPS`), hovering stops the clock so a tip cannot vanish
 * mid-sentence, and dismissing is permanent rather than per-session — a user who
 * has read them should not have to dismiss them again tomorrow.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useAppStore } from "../store";
import { eligibleTips, nextTip } from "../utils/tips";

const ROTATE_MS = 20_000;
const DISMISSED_KEY = "biopb_tips_dismissed";
/** Where the rotation had got to, so a reload does not restart at tip one. */
const CURSOR_KEY = "biopb_tips_cursor";

function readFlag(key: string): boolean {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // A viewer that cannot persist a tip cursor still shows tips.
  }
}

export function TipBar() {
  const sourceCount = useAppStore((s) => s.sources.length);
  const scanning = useAppStore((s) => s.scanning);
  const hasSelection = useAppStore((s) => !!s.activeSourceId && !!s.activeTensorId);

  const [dismissed, setDismissed] = useState(() => readFlag(DISMISSED_KEY));
  const [paused, setPaused] = useState(false);
  const [currentId, setCurrentId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(CURSOR_KEY);
    } catch {
      return null;
    }
  });

  const tips = useMemo(
    () => eligibleTips({ sourceCount, hasSelection, scanning }),
    [sourceCount, hasSelection, scanning],
  );

  // The shown tip is derived, not stored: the eligible list changes as soon as a
  // source is selected, and a tip that has dropped out of it is describing
  // something no longer on screen.
  const current = useMemo(
    () => tips.find((tip) => tip.id === currentId) ?? tips[0] ?? null,
    [tips, currentId],
  );

  const advance = useCallback(() => {
    const next = nextTip(current?.id ?? null, tips);
    if (!next) return;
    setCurrentId(next.id);
    write(CURSOR_KEY, next.id);
  }, [current, tips]);

  useEffect(() => {
    if (dismissed || paused || tips.length < 2) return;
    const timer = setInterval(advance, ROTATE_MS);
    return () => clearInterval(timer);
  }, [dismissed, paused, tips.length, advance]);

  if (dismissed) {
    return (
      <div className="tip-bar">
        <div className="tip-bar-text" />
        <button
          type="button"
          className="tip-bar-btn"
          title="Show tips again"
          onClick={() => {
            setDismissed(false);
            write(DISMISSED_KEY, "0");
          }}
        >
          Tips
        </button>
      </div>
    );
  }

  if (!current) return <div className="tip-bar" />;

  return (
    <div
      className="tip-bar"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <span className="tip-bar-icon" aria-hidden="true">
        💡
      </span>
      {/* Clickable so an interesting tip can be held, and a dull one skipped,
          without waiting out the interval. */}
      <button
        type="button"
        className="tip-bar-text"
        title="Next tip"
        onClick={advance}
        disabled={tips.length < 2}
      >
        {current.text}
      </button>
      <button
        type="button"
        className="tip-bar-btn"
        title="Stop showing tips"
        aria-label="Stop showing tips"
        onClick={() => {
          setDismissed(true);
          write(DISMISSED_KEY, "1");
        }}
      >
        ✕
      </button>
    </div>
  );
}
