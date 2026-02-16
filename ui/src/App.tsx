import { useCallback, useState } from "react";
import { useChat } from "./hooks/useChat";
import { ChatPanel } from "./components/ChatPanel";
import { StatusBar } from "./components/StatusBar";

export default function App() {
  const { messages, isLoading, error, backendState, sendMessage, initBackend } =
    useChat();
  const [directory, setDirectory] = useState<string | undefined>();

  const handleOpenFolder = useCallback(async () => {
    // selectDirectory will be wired in Task 9 (FileBrowser).
    // For now, use the Electron dialog if available.
    if (!window.api || typeof (window.api as any).selectDirectory !== "function") {
      return;
    }
    const result = await (window.api as any).selectDirectory();
    if (result) {
      setDirectory(result);
      await initBackend(result);
    }
  }, [initBackend]);

  return (
    <div className="app-root flex flex-col h-screen bg-bg-primary">
      {/* Header */}
      <header className="flex items-center justify-between h-12 px-5 border-b border-border bg-bg-secondary shrink-0">
        <span className="font-display text-xl font-semibold text-text-primary tracking-tight">
          NeuroFind
        </span>
        <span className="font-mono text-xs text-text-tertiary hover:text-accent transition-colors cursor-default">
          {directory ?? ""}
        </span>
      </header>

      {/* Chat area */}
      <ChatPanel
        messages={messages}
        isLoading={isLoading}
        error={error}
        onSend={sendMessage}
        onOpenFolder={handleOpenFolder}
      />

      {/* Status bar */}
      <StatusBar backendState={backendState} directory={directory} />
    </div>
  );
}
