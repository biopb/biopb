"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DataSourceDescriptor, QuerySourcesResult } from "@biopb/tensor-flight-client";
import { toggled, useAppStore } from "../store";
import { AddSourceControl } from "./AddSourceControl";
import { Modal } from "./Modal";
import { REASON_COLUMN, SOURCE_COLUMNS, descriptorFromRow } from "../utils/catalogRow";
import { readRecents, subscribeRecents } from "../utils/recentSources";
import {
  type TreeNode,
  FAILED_TOOLTIP,
  PENDING_TOOLTIP,
  UNRESOLVED_GLYPH,
  UNRESOLVED_TOOLTIP,
  buildTree,
  droppedRootUrl,
  groupTensors,
  isEmptySource,
  isUnresolved,
  lazyTreeNodes,
  levelNote,
  matchesQuery,
  recentNode,
  unresolvedKind,
} from "../utils/sourceTree";

// Most matches a search shows. The server's own cap (100k rows by default) is a
// safety valve, not a page size: every row is an id to ship, a tree node to build
// and a row to render, and nobody reads a hundred thousand results. One more than
// this is asked for, so "there were more" is known without a count.
const SERVER_QUERY_LIMIT = 2000;

/** The server-side search: a substring match over id, url and type, returning
 *  `columns` of each match. DuckDB's LIKE has no default escape character, so
 *  `\` escapes only with an explicit ESCAPE clause, and a literal `\` must be
 *  escaped first. */
export function sourceSearchSql(q: string, columns = SOURCE_COLUMNS): string {
  const escaped = q
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "''")
    .replace(/%/g, "\\%")
    .replace(/_/g, "\\_");
  return `SELECT ${columns} FROM sources WHERE
      LOWER(source_id) LIKE '%${escaped}%' ESCAPE '\\' OR
      LOWER(source_url) LIKE '%${escaped}%' ESCAPE '\\' OR
      LOWER(source_type) LIKE '%${escaped}%' ESCAPE '\\'
      ORDER BY source_url LIMIT ${SERVER_QUERY_LIMIT + 1}`;
}

/**
 * The search, as a function of the query runner. It asks for
 * `unresolved_reason` too, which a server older than the column refuses: that
 * first search asks again without, and the answer is remembered.
 */
export function createSourceSearch() {
  let hasReason: boolean | undefined;
  return async function search(
    query: (sql: string) => Promise<QuerySourcesResult>,
    q: string,
  ): Promise<QuerySourcesResult> {
    if (hasReason === false) return query(sourceSearchSql(q));
    try {
      const result = await query(sourceSearchSql(q, `${SOURCE_COLUMNS}, ${REASON_COLUMN}`));
      hasReason = true;
      return result;
    } catch (err) {
      if (hasReason) throw err;
      // Only an answer without the column says the column was the trouble; a
      // dropped connection fails this retry too, and teaches nothing.
      const result = await query(sourceSearchSql(q)).catch(() => {
        throw err;
      });
      hasReason = false;
      return result;
    }
  };
}

const searchSources = createSourceSearch();

function tensorShortName(arrayId: string): string {
  const parts = arrayId.split("/").filter(Boolean);
  return parts[parts.length - 1] || arrayId;
}

/** On, off. A radio rather than a checkbox: only one set is drawn at a time. */
const LABEL_ON_GLYPH = "\u25c9";
const LABEL_OFF_GLYPH = "\u25cb";

function formatShape(shape: number[]): string {
  return shape.join("×");
}

// The badge's shape without its leading singleton axes (1×1×343×278×300 reads
// as 343×278×300): they carry nothing a glance needs and crowd the row. The
// tooltip keeps the full shape. An all-ones shape stays "1".
function compactShape(shape: number[]): string {
  const first = shape.findIndex((n) => n !== 1);
  return formatShape(first < 0 ? shape.slice(-1) : shape.slice(first));
}

/** Indent per tree level, px: half the Chevron slot. */
const INDENT_STEP = 8;

function Chevron({ expanded }: { expanded: boolean }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 16,
        flexShrink: 0,
        // Centred so the rotation (about the box centre) turns the glyph in
        // place; left-aligned, the expanded triangle lands right of the collapsed one.
        textAlign: "center",
        fontSize: 10,
        transition: "transform 0.15s",
        transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
        opacity: 0.6,
      }}
    >
      ▶
    </span>
  );
}

// Empty gutter the width of a Chevron, so leaf (source/tensor) rows keep their
// labels in the same column as folder labels — otherwise a folder's children
// align left of the folder's own label and the indent steps unevenly.
function ChevronSlot() {
  return (
    <span
      aria-hidden="true"
      style={{ display: "inline-block", width: 16, flexShrink: 0 }}
    />
  );
}

/** A line under an open folder: loading, failed, or cut short. */
function TreeNote({ text, indent }: { text: string; indent: number }) {
  return (
    <div style={{ fontSize: 11, color: "#64748b", padding: "2px 0" }}>
      <span style={{ paddingLeft: indent + 20 }}>{text}</span>
    </div>
  );
}

/**
 * Marks a top-level row that came from a drag-and-drop rather than from one of
 * the server's configured roots: the tree shows both as plain paths, so without
 * this a dropped folder is indistinguishable from a root that is always there.
 */
function DroppedBadge({ onRemove }: { onRemove?: () => void }) {
  const why = "Added by drag-and-drop: not one of the server's configured roots, and not kept across a restart";
  if (!onRemove) {
    return (
      <span className="tree-dropped" title={why}>
        dropped
      </span>
    );
  }
  // A span, not a button: the row it sits in is one, and buttons do not nest.
  return (
    <span
      className="tree-dropped removable"
      role="button"
      tabIndex={0}
      title={`${why}.\nClick to stop serving it.`}
      onClick={(e) => {
        e.stopPropagation();
        onRemove();
      }}
      onKeyDown={(e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault();
        e.stopPropagation();
        onRemove();
      }}
    >
      dropped ✕
    </span>
  );
}

export interface TreeRowProps {
  node: TreeNode;
  activeSourceId: string | null;
  activeTensorId: string | null;
  expandedFolders: Set<string>;
  toggleFolder: (id: string) => void;
  selectSource: (sourceId: string, tensorId?: string) => void;
  /**
   * Begin resolving an unresolved source. Optional so `TreeRow` stays a pure
   * props component (it is rendered standalone in tests); a row without it
   * simply shows no resolve button.
   */
  startResolve?: (sourceId: string) => void;
  /** Source ids with a resolve already under way, so the button can say so. */
  resolving?: ReadonlySet<string>;
  /**
   * The label set currently drawn, as its whole `array_id`, or null. Unscoped
   * on purpose: the rows this marks belong to one image, and a set of another
   * image can never be one of them.
   */
  labelOverlay?: string | null;
  /** Draw this set, or nothing. Optional, so a row renders standalone. */
  setLabelOverlay?: (arrayId: string | null) => void;
  /**
   * Stop serving a dropped top-level root: its `dnd://` url and the name to
   * show. Only the catalog's own rows are given it; a row without it shows the
   * "dropped" tag but cannot remove.
   */
  onRemoveDropped?: (rootUrl: string, name: string) => void;
}

export function TreeRow({
  node,
  activeSourceId,
  activeTensorId,
  expandedFolders,
  toggleFolder,
  selectSource,
  startResolve,
  resolving,
  labelOverlay,
  setLabelOverlay,
  onRemoveDropped,
}: TreeRowProps) {
  // Half a Chevron's width per level, so a deep tree keeps room for its labels.
  const indent = node.depth * INDENT_STEP + 12;
  const droppedRoot = onRemoveDropped ? droppedRootUrl(node) : null;
  const droppedBadge = node.dropped && (
    <DroppedBadge
      onRemove={
        droppedRoot && onRemoveDropped ? () => onRemoveDropped(droppedRoot, node.name) : undefined
      }
    />
  );
  // Label sets filed under the image they annotate, rather than listed beside
  // it: a set is a tensor of the source, but it is *about* one of the others.
  //
  // Memoized: an unrelated store change (e.g. toggling the overlay) re-renders
  // every row, and re-sorting every source's tensors on each one adds up.
  //
  // Hoisted above the folder branch because that branch returns: a hook after
  // it runs only for source rows, which is the hook-order rule React enforces
  // and eslint refuses to build. A folder has no tensors, so it memoizes an
  // empty list and pays nothing.
  const groups = useMemo(() => groupTensors(node.source?.tensors ?? []), [node.source]);

  if (node.type === "folder") {
    const expanded = expandedFolders.has(node.id);
    return (
      <>
        <button
          className="tree-item tree-folder"
          style={{
            width: "100%",
            textAlign: "left",
            display: "flex",
            alignItems: "center",
            paddingLeft: indent,
          }}
          onClick={() => toggleFolder(node.id)}
        >
          <Chevron expanded={expanded} />
          <span className="tree-name" style={{ marginLeft: 4 }} title={node.name}>
            {node.name}
          </span>
          {droppedBadge}
        </button>
        {expanded &&
          node.children.map((child) => (
            <TreeRow
              key={child.id}
              node={child}
              activeSourceId={activeSourceId}
              activeTensorId={activeTensorId}
              expandedFolders={expandedFolders}
              toggleFolder={toggleFolder}
              selectSource={selectSource}
              startResolve={startResolve}
              resolving={resolving}
              labelOverlay={labelOverlay}
              setLabelOverlay={setLabelOverlay}
            />
          ))}
        {expanded && node.note && <TreeNote text={node.note} indent={indent + INDENT_STEP} />}
      </>
    );
  }

  // Source node
  const src = node.source!;
  const isActive = src.source_id === activeSourceId;
  // Grouped, not the raw tensor count: a label set files under the image it
  // annotates (see `groups` above), so one image plus its label set is still
  // one tensor as far as expanding into a per-image list is concerned. Label
  // toggle rows render regardless (below) -- they're the only place to reach
  // the overlay -- but the redundant image row is skipped when there's only
  // one to pick from.
  const showImageRows = groups.length > 1;
  const firstTensor = src.tensors[0];
  // An unresolved source has no tensor to read, so selecting it would send the
  // viewer after a tile that cannot exist.
  const unresolved = isUnresolved(src);
  const inFlight = resolving?.has(src.source_id) ?? false;

  // An unresolved row is a plain div, not a disabled button: it carries a real
  // Resolve button, and interactive content cannot nest inside a button. The
  // row itself has nothing to activate -- there is no tensor to open until the
  // server hydrates it -- so dropping it from the tab order costs nothing and
  // leaves exactly one focusable control, the one that does something.
  if (unresolved) {
    // Only a cloud placeholder is a download; a pending or failed local source
    // is read from disk, so its copy and its button say so.
    const kind = unresolvedKind(src);
    const tooltip =
      kind === "pending"
        ? PENDING_TOOLTIP
        : kind === "failed"
          ? FAILED_TOOLTIP
          : UNRESOLVED_TOOLTIP;
    const idleLabel = kind === "recall" ? "Resolve" : kind === "pending" ? "Load" : "Retry";
    const busyLabel = kind === "recall" ? "Resolving\u2026" : "Loading\u2026";
    // A cloud resolve downloads the content, which can take minutes, so it takes
    // a deliberate double-click rather than a stray single click on a button.
    // A pending or failed row takes the same gesture in addition to its button,
    // which a narrow pane can push off the edge.
    const doubleClickToResolve = startResolve && !inFlight;
    return (
      <div
        className={`tree-item unresolved${inFlight ? " resolving" : ""}`}
        // Focusable so a click (or Tab) highlights the row like a selection
        // without opening anything: there is no tensor to open yet.
        tabIndex={0}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          paddingLeft: indent,
        }}
        data-source-id={node.id === src.source_id ? src.source_id : undefined}
        title={`${src.source_url}\n${tooltip}`}
        onDoubleClick={
          doubleClickToResolve ? () => startResolve(src.source_id) : undefined
        }
      >
        {/* The glyph is this row's chevron slot, not a second element after it,
            so the name stays in the label column of its resolved siblings. */}
        <span className="unresolved-glyph" role="img" aria-label="Not resolved">
          {kind === "recall" ? UNRESOLVED_GLYPH : "\u2026"}
        </span>
        <span className="tree-name" style={{ flex: 1, marginLeft: 4 }}>
          {node.name}
        </span>
        {droppedBadge}
        {!startResolve ? null : kind === "recall" ? (
          <span
            className="resolve-hint"
            title="Double-click this row to download and resolve it \u2014 can take minutes"
          >
            {inFlight ? busyLabel : "Double-click to resolve"}
          </span>
        ) : (
          <button
            className="resolve-btn"
            disabled={inFlight}
            onClick={() => startResolve(src.source_id)}
            title={
              inFlight
                ? "Already resolving this source"
                : "Read this source's metadata now"
            }
          >
            {inFlight ? busyLabel : idleLabel}
          </button>
        )}
      </div>
    );
  }

  return (
    <>
      <button
        className={`tree-item ${isActive ? "active" : ""}`}
        style={{
          width: "100%",
          textAlign: "left",
          display: "flex",
          alignItems: "center",
          paddingLeft: indent,
        }}
        // Only the catalog row carries this: it is what the reveal effect
        // scrolls to, and the "Recent" copy of the same source -- same
        // descriptor, different node -- comes first in document order and would
        // otherwise shadow it.
        data-source-id={node.id === src.source_id ? src.source_id : undefined}
        onClick={() => {
          // One image is unambiguous whether or not it carries label sets; two
          // are a choice, and guessing it here is what biopb/biopb#75 was about.
          if (groups.length === 1) {
            selectSource(src.source_id, groups[0]?.image.array_id);
          } else {
            selectSource(src.source_id);
          }
        }}
        title={src.source_url}
      >
        <ChevronSlot />
        <span className="tree-name" style={{ flex: 1, marginLeft: 4 }}>
          {node.name}
        </span>
        {droppedBadge}
        {groups.length > 1 ? (
          <span className="tensor-pill" style={{ marginLeft: 8 }}>
            {groups.length}
          </span>
        ) : firstTensor ? (
          <span
            className="dim-badge"
            style={{ marginLeft: 8 }}
            title={formatShape(firstTensor.shape)}
          >
            {compactShape(firstTensor.shape)}
          </span>
        ) : null}
      </button>

      {/* Nested rows when the source is active. The per-image row only earns
          its place when there's an actual choice between images; a label set's
          toggle row renders regardless, since it's the only place to reach it. */}
      {isActive &&
        groups.map(({ image, labelSets }) => {
          const tActive = image.array_id === activeTensorId;
          return (
            <Fragment key={`tensor:${src.source_id}:${image.array_id}`}>
              {showImageRows && (
                <button
                  className={`tree-item tensor-item ${tActive ? "active" : ""}`}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    paddingLeft: indent + INDENT_STEP,
                    display: "flex",
                    alignItems: "center",
                    fontSize: 12,
                  }}
                  onClick={() => selectSource(src.source_id, image.array_id)}
                  title={`${image.array_id}\nShape: ${formatShape(image.shape)}\nDtype: ${image.dtype}`}
                >
                  <ChevronSlot />
                  <span className="tree-name" style={{ flex: 1, marginLeft: 4 }}>
                    {tensorShortName(image.array_id)}
                  </span>
                </button>
              )}
              {labelSets.map((set) => {
                const on = set.array_id === labelOverlay;
                return (
                  <button
                    key={`labels:${src.source_id}:${set.array_id}`}
                    className={`tree-item label-item ${on ? "active" : ""}`}
                    style={{
                      width: "100%",
                      textAlign: "left",
                      paddingLeft: indent + (showImageRows ? 2 : 1) * INDENT_STEP,
                      display: "flex",
                      alignItems: "center",
                      fontSize: 12,
                    }}
                    aria-pressed={on}
                    // Selecting the image as well, because the overlay is drawn
                    // over whatever the viewer has: switching it on from a row
                    // whose image is not open would otherwise be a control with
                    // nothing to act on.
                    onClick={() => {
                      if (!tActive) selectSource(src.source_id, image.array_id);
                      setLabelOverlay?.(on ? null : set.array_id);
                    }}
                    title={
                      `${set.array_id}\nLabel set over ${image.array_id}` +
                      `\nShape: ${formatShape(set.shape)}\nDtype: ${set.dtype}` +
                      `\n${on ? "Drawn — click to hide" : "Click to draw it over the image"}`
                    }
                  >
                    <ChevronSlot />
                    <span aria-hidden="true">{on ? LABEL_ON_GLYPH : LABEL_OFF_GLYPH}</span>
                    <span className="tree-name" style={{ flex: 1, marginLeft: 4 }}>
                      {tensorShortName(set.array_id)}
                    </span>
                  </button>
                );
              })}
            </Fragment>
          );
        })}
    </>
  );
}

export function SourceTree() {
  const treeLevels = useAppStore((s) => s.treeLevels);
  const openFolders = useAppStore((s) => s.openFolders);
  const levelStatus = useAppStore((s) => s.levelStatus);
  const toggleOpenFolder = useAppStore((s) => s.toggleFolder);
  const revealSource = useAppStore((s) => s.revealSource);
  const sourcesLoading = useAppStore((s) => s.sourcesLoading);
  const scanning = useAppStore((s) => s.scanning);
  const activeSourceId = useAppStore((s) => s.activeSourceId);
  // Which tensor row to mark, in the catalog's own spelling: the target's
  // resolved, token-free key, and no row carries a token. Until it resolves,
  // the address that was asked for.
  const activeTensorId = useAppStore((s) => s.target.key ?? s.activeTensorId);
  const openTensor = useAppStore((s) => s.openTensor);
  // A source row opens the bare source, which the server resolves to its
  // default tensor.
  const selectSource = useCallback(
    (sourceId: string, tensorId?: string) => openTensor(tensorId ?? sourceId),
    [openTensor],
  );
  const querySources = useAppStore((s) => s.querySources);
  const recentIds = useAppStore((s) => s.recentIds);
  const recentSources = useAppStore((s) => s.recentSources);
  const syncRecents = useAppStore((s) => s.syncRecents);
  const hydrateRecents = useAppStore((s) => s.hydrateRecents);
  const startResolve = useAppStore((s) => s.startResolve);
  const sourceJobs = useAppStore((s) => s.sourceJobs);
  // Raw, not through `selectLabelOverlay`: a row marks the set it names, and a
  // set of another image is never one of the rows shown here. Reading the
  // scoped value would leave the mark off for the moment between selecting an
  // image and its grid landing.
  const labelOverlay = useAppStore((s) => s.labelOverlay);
  const setLabelOverlay = useAppStore((s) => s.setLabelOverlay);

  // Ids whose resolve is under way, so a row can say so rather than offering a
  // button that would only join the job it already started.
  const resolving = useMemo(
    () =>
      new Set(
        Object.values(sourceJobs)
          .filter((j) => j.kind === "resolve" && j.state === "running")
          .map((j) => j.source_id),
      ),
    [sourceJobs],
  );

  const client = useAppStore((s) => s.client);
  const refreshTree = useAppStore((s) => s.refreshTree);
  // A dropped root the user asked to remove, awaiting their confirmation.
  const [pendingRemove, setPendingRemove] = useState<{ rootUrl: string; name: string } | null>(
    null,
  );
  const [removeError, setRemoveError] = useState<string | null>(null);
  const removeDropped = async () => {
    const target = pendingRemove;
    setPendingRemove(null);
    if (!client || !target) return;
    setRemoveError(null);
    try {
      const { failed } = await client.http.removeSource(target.rootUrl);
      if (failed.length > 0) {
        setRemoveError(`Could not remove ${failed.length} of ${target.name}: ${failed[0]!.reason}`);
      }
    } catch (e) {
      setRemoveError(`Could not remove ${target.name}: ${e instanceof Error ? e.message : String(e)}`);
    }
    await refreshTree().catch(() => {});
  };

  const [query, setQuery] = useState("");
  // The sources a server-side search matched, whole: the catalog is searched,
  // not the copy of it held here, so a match is shown whether or not it is loaded.
  const [serverMatches, setServerMatches] = useState<DataSourceDescriptor[] | null>(null);
  const [serverQueryLoading, setServerQueryLoading] = useState(false);
  // More sources matched than `SERVER_QUERY_LIMIT` lets through.
  const [serverMoreMatches, setServerMoreMatches] = useState(false);
  // Open folders of the trees built here -- "Recent" and a search's matches. The
  // catalog's own open folders are in the store, because reading a folder is
  // what opening it does.
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    () => new Set(),
  );

  // Another tab's writes. localStorage is shared but does not re-render, so
  // without this two tabs of one link would drift apart -- which is the reason
  // the list is not in sessionStorage.
  useEffect(() => {
    syncRecents(readRecents());
    return subscribeRecents(syncRecents);
  }, [syncRecents]);

  // Re-resolved when the list changes or the tree does: a recent id that the
  // tree has just shown should stop costing a `tile_info` round trip.
  useEffect(() => {
    void hydrateRecents();
  }, [hydrateRecents, recentIds, treeLevels]);

  // The box itself stays on `query` so typing is never held up; the server
  // query and everything derived from it follow the debounced value, so a burst
  // of keystrokes costs one query instead of one per key.
  const [debouncedQuery, setDebouncedQuery] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300);
    return () => clearTimeout(timer);
  }, [query]);

  // Server-side search
  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setServerMatches(null);
      setServerMoreMatches(false);
      return;
    }

    // A query superseded while in flight must not land over the newer one: the
    // answers can arrive out of order.
    let stale = false;
    setServerQueryLoading(true);
    searchSources(querySources, debouncedQuery.trim().toLowerCase())
      .then((result) => {
        if (stale) return;
        setServerMatches(result.rows.slice(0, SERVER_QUERY_LIMIT).map(descriptorFromRow));
        setServerMoreMatches(result.rows.length > SERVER_QUERY_LIMIT);
        setServerQueryLoading(false);
      })
      .catch((err) => {
        if (stale) return;
        console.warn("Server query failed:", err);
        setServerMatches(null);
        setServerMoreMatches(false);
        setServerQueryLoading(false);
      });
    return () => {
      stale = true;
    };
  }, [debouncedQuery, querySources]);

  // A search shows its matches as a tree of their own, built here from at most
  // `SERVER_QUERY_LIMIT` rows. Until the first answer lands the catalog stays up
  // (the header says the search is under way): swapping it for a notice would
  // unmount every row and mount them again when the answer came.
  const searching = debouncedQuery.trim() !== "" && serverMatches !== null;
  // Empty sources are dropped, as the server drops them from the catalog's own
  // levels: a source with nothing on it is never worth a row.
  const matches = useMemo(
    () => (searching ? serverMatches.filter((s) => !isEmptySource(s)) : []),
    [searching, serverMatches],
  );
  const searchTree = useMemo(() => buildTree(matches), [matches]);

  const top = treeLevels[""]?.level;
  const catalogNodes = useMemo(
    () => lazyTreeNodes(treeLevels, levelStatus, openFolders),
    [treeLevels, levelStatus, openFolders],
  );
  const topNote = levelNote(top, levelStatus[""]);

  // Recents filter client-side: that query is SQL against the catalog, which by
  // construction does not hold the uploads this node exists to show.
  const recent = useMemo(() => {
    const q = debouncedQuery.trim().toLowerCase();
    const visible = recentSources.filter((s) => !isEmptySource(s));
    const matching = q ? visible.filter((s) => matchesQuery(s, q)) : visible;
    return recentNode(matching);
  }, [recentSources, debouncedQuery]);

  const listRef = useRef<HTMLDivElement | null>(null);
  // The selection this has already revealed. Latched so a later poll cannot
  // re-open a folder the user just collapsed; a selection the server does not
  // show *yet* is unlatched, and revealed once a level that holds it lands.
  const revealed = useRef<string | null>(null);
  const pendingScroll = useRef<string | null>(null);

  useEffect(() => {
    if (!activeSourceId) {
      revealed.current = null;
      return;
    }
    if (revealed.current === activeSourceId) return;
    revealed.current = activeSourceId;
    // A row the user can already see needs no reveal -- the common case, a click
    // in the tree. The rest is a link or a recent: the server walks down to it
    // in one request and the folders open on the way.
    const { treeLevels: levels, openFolders: open } = useAppStore.getState();
    const visible = Object.entries(levels).some(
      ([key, l]) =>
        (key === "" || open.has(key)) &&
        l.level.sources.some((x) => x.descriptor.source_id === activeSourceId),
    );
    if (visible) {
      pendingScroll.current = activeSourceId;
      return;
    }
    revealSource(activeSourceId)
      .then((ok) => {
        if (ok) pendingScroll.current = activeSourceId;
        else if (revealed.current === activeSourceId) revealed.current = null;
      })
      .catch(() => {
        if (revealed.current === activeSourceId) revealed.current = null;
      });
  }, [activeSourceId, treeLevels, revealSource]);

  // Deliberately every commit: the row this wants may not exist until the
  // expansion above has rendered, and it is one ref read until it does.
  useEffect(() => {
    const id = pendingScroll.current;
    if (id === null) return;
    const row = listRef.current?.querySelector(`[data-source-id="${CSS.escape(id)}"]`);
    if (!row) return;
    pendingScroll.current = null;
    row.scrollIntoView({ block: "nearest" });
  });

  const toggleFolder = useCallback(
    (id: string) => setExpandedFolders((prev) => toggled(prev, id)),
    [],
  );

  /**
   * Selecting from "Recent" or from a search's matches, which must leave the
   * catalog exactly where it is.
   *
   * The reveal effect above keys on `activeSourceId` alone and so cannot tell a
   * catalog click from a shortcut; without this, picking a row near the top of
   * the pane opens a folder chain further down and scrolls the tree to it -- a
   * jump to somewhere the reader did not click. Recording the origin here is
   * what the effect cannot work out for itself.
   *
   * Marking it *revealed* rather than adding a flag to skip: the two say the
   * same thing (this selection needs no reveal) and the latch already means
   * "shown", so a later poll does not reopen the question either. A catalog
   * click and a shared link leave it unlatched and still reveal.
   */
  const selectInPlace = useCallback(
    (sourceId: string, tensorId?: string) => {
      revealed.current = sourceId;
      selectSource(sourceId, tensorId);
    },
    [selectSource],
  );

  const rowProps = {
    activeSourceId,
    activeTensorId,
    selectSource,
    startResolve,
    resolving,
    labelOverlay,
    setLabelOverlay,
  };
  const topEmpty = !top || top.folders.length + top.sources.length === 0;

  return (
    <section style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100%" }}>
      <div style={{ padding: "0.5rem 1rem" }}>
        <div
          style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 6, alignItems: "center" }}
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search (SQL filter)"
            aria-label="Search sources"
            style={{ width: "100%" }}
          />
          <AddSourceControl />
        </div>
        {removeError && (
          <div style={{ fontSize: 11, color: "#fca5a5", marginTop: 4 }}>{removeError}</div>
        )}
        {scanning && !topEmpty && (
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            Still indexing the data folder; more sources may appear
          </div>
        )}
        {(searching || serverQueryLoading) && (
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>
            {searching &&
              `${serverMatches.length.toLocaleString()} matches${
                serverMoreMatches && !serverQueryLoading ? " (truncated)" : ""
              } - server-side filter`}
            {serverQueryLoading && `${searching ? " • " : ""}Searching…`}
          </div>
        )}
      </div>

      <div ref={listRef} style={{ overflow: "auto" }}>
        {sourcesLoading ? (
          <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>Loading sources...</div>
        ) : (
          <>
            {recent && (
              <TreeRow
                node={recent}
                {...rowProps}
                expandedFolders={expandedFolders}
                toggleFolder={toggleFolder}
                selectSource={selectInPlace}
                startResolve={undefined}
                resolving={undefined}
              />
            )}
            {/* The empty notice is about the catalog, so it is suppressed while
                "Recent" has rows -- "No sources" above a list of sources reads
                as a bug. */}
            {searching ? (
              matches.length === 0 && !recent ? (
                <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>No sources</div>
              ) : (
                searchTree.children.map((child) => (
                  <TreeRow
                    key={child.id}
                    node={child}
                    {...rowProps}
                    selectSource={selectInPlace}
                    expandedFolders={expandedFolders}
                    toggleFolder={toggleFolder}
                  />
                ))
              )
            ) : topEmpty && !recent ? (
              <div style={{ padding: "0.5rem 1rem", opacity: 0.8 }}>
                {scanning
                  ? "Indexing data folder… (sources will appear as they are found)"
                  : "No sources"}
              </div>
            ) : (
              <>
                {catalogNodes.map((child) => (
                  <TreeRow
                    key={child.id}
                    node={child}
                    {...rowProps}
                    expandedFolders={openFolders}
                    toggleFolder={toggleOpenFolder}
                    onRemoveDropped={(rootUrl, name) => setPendingRemove({ rootUrl, name })}
                  />
                ))}
                {topNote && <TreeNote text={topNote} indent={0} />}
              </>
            )}
          </>
        )}
      </div>
      {pendingRemove && (
        <Modal
          title="Stop serving this?"
          onClose={() => setPendingRemove(null)}
          labelId="remove-dropped-title"
        >
          <p>
            <strong>{pendingRemove.name}</strong> and the sources under it are taken off this
            server until it is added again. The files on disk are not touched.
          </p>
          <div className="admin-modal-actions">
            <button type="button" className="icon-btn" onClick={() => setPendingRemove(null)}>
              Cancel
            </button>
            <button type="button" className="submit-btn" onClick={removeDropped}>
              Remove
            </button>
          </div>
        </Modal>
      )}
    </section>
  );
}
