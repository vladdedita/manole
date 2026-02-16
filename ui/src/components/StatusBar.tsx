interface StatusBarProps {
  backendState: string;
  directory?: string;
}

function statusConfig(state: string): { label: string; color: string } {
  switch (state) {
    case "ready":
      return { label: "Ready", color: "bg-success" };
    case "loading_model":
      return { label: "Loading model", color: "bg-warning" };
    case "indexing":
      return { label: "Indexing", color: "bg-warning" };
    case "not_initialized":
      return { label: "Not initialized", color: "bg-text-tertiary" };
    default:
      if (state.startsWith("error")) {
        return { label: "Error", color: "bg-error" };
      }
      return { label: state, color: "bg-text-tertiary" };
  }
}

export function StatusBar({ backendState, directory }: StatusBarProps) {
  const { label, color } = statusConfig(backendState);

  const truncatedDir = directory
    ? directory.length > 40
      ? "\u2026" + directory.slice(-39)
      : directory
    : "\u2014";

  return (
    <div className="flex items-center h-8 px-5 border-t border-border bg-bg-primary font-mono text-[11px] text-text-tertiary shrink-0">
      {/* Connection status */}
      <div className="flex items-center gap-2 pr-4 border-r border-border">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${color} transition-colors duration-500`}
        />
        <span>{label}</span>
      </div>

      {/* Model name */}
      <div className="px-4 border-r border-border">LFM2.5-1.2B</div>

      {/* Directory */}
      <div className="px-4 truncate">{truncatedDir}</div>
    </div>
  );
}
