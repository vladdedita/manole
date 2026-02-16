import { useCallback, useState } from "react";
import { AnimatePresence } from "motion/react";
import { useChat } from "./hooks/useChat";
import { ChatPanel } from "./components/ChatPanel";
import { StatusBar } from "./components/StatusBar";
import { LoadingScreen } from "./components/LoadingScreen";
import { DevPanel } from "./components/DevPanel";

export default function App() {
  const { messages, isLoading, error, backendState, logs, sendMessage, initBackend } =
    useChat();
  const [directory, setDirectory] = useState<string | undefined>();
  const [devPanelOpen, setDevPanelOpen] = useState(false);

  const handleOpenFolder = useCallback(async () => {
    const result = await window.api.selectDirectory();
    if (result) {
      setDirectory(result);
      await initBackend(result);
    }
  }, [initBackend]);

  const isInitializing =
    backendState === "loading_model" || backendState === "indexing";
  const showChat = backendState === "ready" || (directory && !isInitializing);

  return (
    <div className="app-root flex flex-col h-screen bg-bg-primary">
      {/* Header */}
      <header className="flex items-center justify-between h-12 px-5 border-b border-border bg-bg-secondary shrink-0">
        <span className="font-display text-xl font-semibold text-text-primary tracking-tight">
          NeuroFind
        </span>
        <div className="flex items-center gap-3">
          {directory && (
            <span className="font-mono text-xs text-text-tertiary">
              {directory}
            </span>
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

      {/* Main content */}
      <AnimatePresence mode="wait">
        {!directory ? (
          <ChatPanel
            key="welcome"
            messages={[]}
            isLoading={false}
            error={null}
            onSend={sendMessage}
            onOpenFolder={handleOpenFolder}
          />
        ) : isInitializing ? (
          <LoadingScreen key="loading" backendState={backendState} />
        ) : (
          <ChatPanel
            key="chat"
            messages={messages}
            isLoading={isLoading}
            error={error}
            onSend={sendMessage}
            onOpenFolder={handleOpenFolder}
          />
        )}
      </AnimatePresence>

      {/* Dev panel */}
      <DevPanel
        open={devPanelOpen}
        onClose={() => setDevPanelOpen(false)}
        logs={logs}
      />

      {/* Status bar */}
      <StatusBar backendState={backendState} directory={directory} />
    </div>
  );
}
