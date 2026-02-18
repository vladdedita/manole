import { useCallback, useState, useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useChat } from "./hooks/useChat";
import type { Response, DirectoryStats } from "./lib/protocol";
import { ChatPanel } from "./components/ChatPanel";
import { FileGraphPanel } from "./components/FileGraphPanel";
import { StatusBar } from "./components/StatusBar";
import { DevPanel } from "./components/DevPanel";
import { SidePanel, type DirectoryEntry } from "./components/SidePanel";
import { LoadingScreen } from "./components/LoadingScreen";
import { useFileGraph } from "./hooks/useFileGraph";

export default function App() {
  const { messages, isLoading, error, backendState, logs, sendMessage, initBackend, resetBackendState, clearChat, subscribe, send } =
    useChat();

  const [directories, setDirectories] = useState<DirectoryEntry[]>([]);
  const [activeDirectoryId, setActiveDirectoryId] = useState<string | null>(null);
  const [searchAll, setSearchAll] = useState(false);
  const [mode, setMode] = useState<"chat" | "map">("chat");
  const [sidePanelOpen, setSidePanelOpen] = useState(true);
  const [devPanelOpen, setDevPanelOpen] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const { graph, isLoading: graphLoading, error: graphError, fetchGraph, clearGraph } = useFileGraph();

  const activeDirectory = directories.find((d) => d.id === activeDirectoryId);

  useEffect(() => {
    return subscribe((response: Response) => {
      if (response.type === "directory_update") {
        const data = response.data as {
          directoryId: string;
          state: "indexing" | "ready" | "error";
          stats?: DirectoryStats;
          summary?: string;
          error?: string;
        };
        setDirectories((prev) =>
          prev.map((d) =>
            d.id === data.directoryId
              ? {
                  ...d,
                  state: data.state,
                  ...(data.stats ? { stats: data.stats } : {}),
                  ...(data.summary ? { summary: data.summary } : {}),
                  ...(data.error ? { error: data.error } : {}),
                }
              : d
          )
        );
      } else if (response.type === "captioning_progress") {
        const data = response.data as {
          directoryId: string;
          done: number;
          total: number;
          state?: "complete";
        };
        setDirectories((prev) =>
          prev.map((d) =>
            d.id === data.directoryId
              ? {
                  ...d,
                  captioningProgress: data.state === "complete"
                    ? undefined
                    : { done: data.done, total: data.total },
                }
              : d
          )
        );
      }
    });
  }, [subscribe]);

  const handleOpenFolder = useCallback(async () => {
    const result = await window.api.selectDirectory();
    if (result) {
      setInitError(null);

      // Create a directory entry
      const dirId = result.split("/").pop()?.replace(/\s/g, "_").replace(/\//g, "_") || result;

      // Check if already indexed
      if (directories.some((d) => d.id === dirId)) {
        setActiveDirectoryId(dirId);
        return;
      }

      const newEntry: DirectoryEntry = {
        id: dirId,
        path: result,
        state: "indexing",
      };
      setDirectories((prev) => [...prev, newEntry]);
      setActiveDirectoryId(dirId);

      try {
        const response = await initBackend(result);
        if (response.type === "error") {
          const msg = (response.data as { message: string }).message;
          setDirectories((prev) =>
            prev.map((d) =>
              d.id === dirId ? { ...d, state: "error" as const, error: msg } : d
            )
          );
          setInitError(msg);
        } else {
          // Backend may return a different directoryId than our guess — reconcile
          const realDirId = (response.data as { directoryId?: string }).directoryId;
          if (realDirId && realDirId !== dirId) {
            setDirectories((prev) =>
              prev.map((d) => (d.id === dirId ? { ...d, id: realDirId } : d))
            );
            setActiveDirectoryId(realDirId);
          }
        }
        // Don't set "ready" here — let directory_update from backend handle it
        // (it includes stats and summary)
      } catch (err) {
        setDirectories((prev) =>
          prev.map((d) =>
            d.id === dirId
              ? { ...d, state: "error" as const, error: String(err) }
              : d
          )
        );
        setInitError(String(err));
      }
    }
  }, [initBackend, directories]);

  const handleReindex = useCallback(
    async (dirId: string) => {
      setDirectories((prev) =>
        prev.map((d) =>
          d.id === dirId ? { ...d, state: "indexing" as const } : d
        )
      );
      try {
        await send("reindex", { directoryId: dirId });
      } catch (err) {
        setDirectories((prev) =>
          prev.map((d) =>
            d.id === dirId
              ? { ...d, state: "error" as const, error: String(err) }
              : d
          )
        );
      }
    },
    [send]
  );

  const handleRemove = useCallback(async (dirId: string) => {
    setDirectories((prev) => prev.filter((d) => d.id !== dirId));
    setActiveDirectoryId((prev) => (prev === dirId ? null : prev));
    try {
      await send("remove_directory", { directoryId: dirId });
    } catch {
      // Directory already removed from UI
    }
  }, [send]);

  const handleSend = useCallback(
    (text: string) => {
      sendMessage(text, activeDirectoryId, searchAll);
    },
    [sendMessage, activeDirectoryId, searchAll]
  );

  const handleFetchGraph = useCallback(() => {
    if (activeDirectoryId) {
      fetchGraph(activeDirectoryId);
    }
  }, [activeDirectoryId, fetchGraph]);

  useEffect(() => {
    clearGraph();
  }, [activeDirectoryId, clearGraph]);

  const hasDirectories = directories.length > 0;
  const isInitializing = activeDirectory?.state === "indexing";
  const isReady = activeDirectory?.state === "ready";

  return (
    <div className="app-root flex flex-col h-screen bg-bg-primary">
      {/* Header */}
      <header className="flex items-center justify-between h-12 pr-5 pl-[78px] border-b border-border bg-bg-secondary shrink-0 z-30" style={{ WebkitAppRegion: "drag" } as React.CSSProperties}>
        <div className="flex items-center gap-3" style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}>
          {/* Side panel toggle */}
          <button
            onClick={() => setSidePanelOpen((v) => !v)}
            className={`flex items-center justify-center h-7 w-7 rounded-md transition-colors ${
              sidePanelOpen
                ? "bg-accent/20 text-accent"
                : "text-text-tertiary hover:text-text-secondary hover:bg-bg-elevated"
            }`}
            aria-label="Toggle index panel"
            title="Indexes"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M2 3.5h10M2 7h10M2 10.5h10"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
              />
            </svg>
          </button>

          <span className="font-display text-xl font-semibold text-text-primary tracking-tight">
            Manole
          </span>
        </div>

        <div className="flex items-center gap-3" style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}>
          {hasDirectories && isReady && (
            <div className="flex items-center h-7 rounded-full bg-bg-elevated border border-border p-0.5">
              {(["chat", "map"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`relative px-3 py-0.5 font-display text-xs rounded-full transition-colors ${
                    mode === m ? "text-accent" : "text-text-tertiary hover:text-text-secondary"
                  }`}
                >
                  {mode === m && (
                    <motion.div
                      layoutId="mode-indicator"
                      className="absolute inset-0 rounded-full bg-accent-muted"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                  <span className="relative capitalize">{m === "chat" ? "Chat" : "Map"}</span>
                </button>
              ))}
            </div>
          )}
          {hasDirectories && (
            <button
              onClick={() => setSidePanelOpen((v) => !v)}
              className="font-mono text-xs text-text-tertiary hover:text-text-secondary transition-colors truncate max-w-[300px]"
              title={searchAll ? "Searching all indexes" : activeDirectory?.path}
            >
              {searchAll ? (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
                  All indexes
                </span>
              ) : (
                activeDirectory?.path
              )}
            </button>
          )}
          {/* Dev panel toggle */}
          <button
            onClick={() => setDevPanelOpen((v) => !v)}
            className={`flex items-center justify-center h-7 w-7 rounded-md transition-colors ${
              devPanelOpen
                ? "bg-accent/20 text-accent"
                : "text-text-tertiary hover:text-text-secondary hover:bg-bg-elevated"
            }`}
            aria-label="Toggle developer panel"
            title="Python output"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M4.5 4L2 7l2.5 3M9.5 4L12 7l-2.5 3M8 2.5L6 11.5"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </header>

      {/* Body: sidebar + main content side by side */}
      <div className="flex flex-1 min-h-0">
        {/* Side panel — inline, takes real space */}
        <SidePanel
          open={sidePanelOpen}
          directories={directories}
          activeDirectoryId={activeDirectoryId}
          searchAll={searchAll}
          onSelectDirectory={setActiveDirectoryId}
          onAddFolder={handleOpenFolder}
          onReindex={handleReindex}
          onRemove={handleRemove}
          onToggleSearchAll={() => setSearchAll((v) => !v)}
        />

        {/* Main content */}
        <div className="flex-1 flex flex-col min-w-0">
          <AnimatePresence mode="wait">
            {!hasDirectories ? (
              <ChatPanel
                key="welcome"
                messages={[]}
                isLoading={false}
                error={initError}
                onSend={handleSend}
                onOpenFolder={handleOpenFolder}
              />
            ) : isInitializing ? (
              <LoadingScreen key="loading" backendState={backendState} captioningProgress={activeDirectory?.captioningProgress} />
            ) : isReady && mode === "map" ? (
              <motion.div
                key="map"
                initial={{ opacity: 0, filter: "blur(4px)" }}
                animate={{ opacity: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, filter: "blur(4px)" }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                className="flex flex-col flex-1 min-h-0"
              >
                <FileGraphPanel
                  graph={graph}
                  isLoading={graphLoading}
                  error={graphError}
                  onFetchGraph={handleFetchGraph}
                />
              </motion.div>
            ) : isReady ? (
              <ChatPanel
                key={`chat-${activeDirectoryId}`}
                messages={messages}
                isLoading={isLoading}
                error={error}
                onSend={handleSend}
                onOpenFolder={handleOpenFolder}
              />
            ) : (
              <ChatPanel
                key="no-selection"
                messages={[]}
                isLoading={false}
                error={activeDirectory?.error ?? null}
                onSend={handleSend}
                onOpenFolder={handleOpenFolder}
              />
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Dev panel */}
      <DevPanel
        open={devPanelOpen}
        onClose={() => setDevPanelOpen(false)}
        logs={logs}
      />

      {/* Status bar */}
      <StatusBar backendState={backendState} directory={activeDirectory?.path} />
    </div>
  );
}
