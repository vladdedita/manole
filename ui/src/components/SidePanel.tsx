import { useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { DirectoryStats } from "../lib/protocol";

export interface DirectoryEntry {
  id: string;
  path: string;
  state: "indexing" | "ready" | "error";
  stats?: DirectoryStats;
  summary?: string;
  error?: string;
  captioningProgress?: { done: number; total: number };
}

interface SidePanelProps {
  open: boolean;
  directories: DirectoryEntry[];
  activeDirectoryId: string | null;
  searchAll: boolean;
  onSelectDirectory: (id: string) => void;
  onAddFolder: () => void;
  onReindex: (id: string) => void;
  onRemove: (id: string) => void;
  onToggleSearchAll: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatPercent(part: number, total: number): string {
  if (total === 0) return "0%";
  const pct = Math.round((part / total) * 100);
  return pct < 1 ? "<1%" : `${pct}%`;
}

function DirectoryCard({
  entry,
  isActive,
  onSelect,
  onReindex,
  onRemove,
}: {
  entry: DirectoryEntry;
  isActive: boolean;
  onSelect: () => void;
  onReindex: () => void;
  onRemove: () => void;
}) {
  const folderName = entry.path.split("/").pop() || entry.path;
  const parentPath = entry.path.split("/").slice(0, -1).join("/");
  const truncatedParent =
    parentPath.length > 30 ? "\u2026" + parentPath.slice(-29) : parentPath;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20, height: 0 }}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      onClick={onSelect}
      className={`group relative cursor-pointer rounded-lg border transition-colors duration-200 ${
        isActive
          ? "border-accent/30 bg-accent/[0.06]"
          : "border-transparent hover:border-border hover:bg-bg-tertiary/50"
      }`}
    >
      {/* Active indicator — thin left accent bar */}
      {isActive && (
        <motion.div
          layoutId="active-bar"
          className="absolute left-0 top-2 bottom-2 w-[2px] rounded-full bg-accent"
          transition={{ type: "spring", stiffness: 400, damping: 30 }}
        />
      )}

      <div className="px-3.5 py-3">
        {/* Header row: name + state dot */}
        <div className="flex items-center justify-between gap-2">
          <span className="font-sans text-sm font-medium text-text-primary truncate">
            {folderName}
          </span>
          <div className="flex items-center gap-1.5 shrink-0">
            {/* State indicator */}
            {entry.state === "indexing" ? (
              <motion.span
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="inline-block h-1.5 w-1.5 rounded-full bg-warning"
              />
            ) : entry.state === "ready" ? (
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-success" />
            ) : (
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-error" />
            )}

            {/* Hover actions */}
            <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onReindex();
                }}
                className="flex items-center justify-center h-5 w-5 rounded text-text-tertiary hover:text-accent hover:bg-accent/10 transition-colors"
                title="Reindex"
              >
                <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
                  <path
                    d="M1.5 8a6.5 6.5 0 0 1 11.25-4.5M14.5 8a6.5 6.5 0 0 1-11.25 4.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                  <path
                    d="M13.5 1v3h-3M2.5 15v-3h3"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove();
                }}
                className="flex items-center justify-center h-5 w-5 rounded text-text-tertiary hover:text-error hover:bg-error/10 transition-colors"
                title="Remove"
              >
                <svg width="10" height="10" viewBox="0 0 14 14" fill="none">
                  <path
                    d="M2 2l10 10M12 2L2 12"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Path */}
        <span className="block mt-0.5 font-mono text-[10px] text-text-tertiary truncate">
          {truncatedParent}
        </span>

        {/* Stats (when ready) */}
        {entry.state === "ready" && entry.stats && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.3 }}
            className="mt-2 space-y-1.5"
          >
            {/* Type badges */}
            <div className="flex flex-wrap items-center gap-1.5">
              {Object.entries(entry.stats.types).map(([ext, count]) => (
                <span
                  key={ext}
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-bg-elevated font-mono text-[10px] text-text-secondary"
                  title={`${formatSize(entry.stats!.sizeByType[ext] ?? 0)} (${formatPercent(entry.stats!.sizeByType[ext] ?? 0, entry.stats!.totalSize)})`}
                >
                  <span className="text-accent">{count}</span>
                  <span className="uppercase">{ext}</span>
                </span>
              ))}
            </div>

            {/* Size + structure row */}
            <div className="flex items-center gap-2 font-mono text-[10px] text-text-tertiary">
              <span>{formatSize(entry.stats.totalSize)}</span>
              <span className="text-border">|</span>
              <span>{entry.stats.dirs.count} {entry.stats.dirs.count === 1 ? "folder" : "folders"}</span>
              {entry.stats.dirs.maxDepth > 0 && (
                <>
                  <span className="text-border">|</span>
                  <span>{entry.stats.dirs.maxDepth} deep</span>
                </>
              )}
            </div>
          </motion.div>
        )}

        {/* Captioning progress */}
        {entry.captioningProgress && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 font-mono text-[10px] text-warning"
          >
            Captioning images ({entry.captioningProgress.done}/{entry.captioningProgress.total})...
          </motion.div>
        )}

        {/* Indexing state */}
        {entry.state === "indexing" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 font-mono text-[10px] text-warning"
          >
            {"\u2026indexing"}
          </motion.div>
        )}

        {/* Error state */}
        {entry.state === "error" && entry.error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 font-mono text-[10px] text-error truncate"
            title={entry.error}
          >
            {entry.error}
          </motion.div>
        )}

        {/* Summary */}
        {entry.summary && (
          <div className="relative group/summary">
            <motion.p
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.4 }}
              className="mt-2 font-sans text-[11px] leading-relaxed text-text-tertiary line-clamp-3 cursor-default"
            >
              {entry.summary}
            </motion.p>
            {/* Tooltip — full summary on hover */}
            <div className="absolute left-0 bottom-full mb-1 w-64 p-3 rounded-lg border border-border bg-bg-elevated shadow-lg font-sans text-[11px] leading-relaxed text-text-secondary opacity-0 pointer-events-none group-hover/summary:opacity-100 group-hover/summary:pointer-events-auto transition-opacity duration-200 z-50">
              {entry.summary}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

export function SidePanel({
  open,
  directories,
  activeDirectoryId,
  searchAll,
  onSelectDirectory,
  onAddFolder,
  onReindex,
  onRemove,
  onToggleSearchAll,
}: SidePanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  return (
    <AnimatePresence initial={false}>
      {open && (
        <motion.aside
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 260, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
          className="shrink-0 flex flex-col border-r border-border bg-bg-secondary overflow-hidden"
        >
          {/* Panel header */}
          <div className="flex items-center justify-between h-10 px-4 border-b border-border shrink-0">
            <span className="font-mono text-[10px] uppercase tracking-widest text-text-tertiary">
              Indexes
            </span>
          </div>

          {/* Search All toggle (shown when 2+ directories) */}
          {directories.length >= 2 && (
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-border">
              <span className="font-sans text-xs text-text-secondary">
                Search all
              </span>
              <button
                onClick={onToggleSearchAll}
                className={`relative h-5 w-9 rounded-full transition-colors duration-200 ${
                  searchAll ? "bg-accent" : "bg-bg-elevated"
                }`}
              >
                <motion.div
                  animate={{ x: searchAll ? 16 : 2 }}
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  className="absolute top-0.5 h-4 w-4 rounded-full bg-text-primary"
                />
              </button>
            </div>
          )}

          {/* Directory list */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto p-2.5 space-y-1"
          >
            {directories.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center px-4">
                <div className="font-display text-lg text-text-tertiary italic">
                  No indexes yet
                </div>
                <p className="mt-1 font-sans text-xs text-text-tertiary/60">
                  Add a folder to begin indexing
                </p>
              </div>
            ) : (
              <AnimatePresence initial={false}>
                {directories.map((entry) => (
                  <DirectoryCard
                    key={entry.id}
                    entry={entry}
                    isActive={activeDirectoryId === entry.id}
                    onSelect={() => onSelectDirectory(entry.id)}
                    onReindex={() => onReindex(entry.id)}
                    onRemove={() => onRemove(entry.id)}
                  />
                ))}
              </AnimatePresence>
            )}
          </div>

          {/* Add folder button — pinned bottom */}
          <div className="px-3 py-3 border-t border-border shrink-0">
            <button
              onClick={onAddFolder}
              className="flex items-center justify-center gap-2 w-full py-2.5 rounded-lg border border-dashed border-border text-text-secondary font-sans text-xs hover:border-accent/40 hover:text-accent hover:bg-accent/[0.04] transition-colors"
            >
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
                <path
                  d="M7 2v10M2 7h10"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinecap="round"
                />
              </svg>
              Add Folder
            </button>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
