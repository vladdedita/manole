import { motion } from "motion/react";

export interface SetupModelState {
  id: string;
  name: string;
  filename: string;
  downloadedBytes: number;
  totalBytes: number;
  status: "pending" | "downloading" | "complete" | "error";
  error?: string;
}

interface SetupScreenProps {
  models: SetupModelState[];
  onComplete: () => void;
}

/** Format byte count as human-readable string (e.g. "1.0 MB") */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = bytes / Math.pow(k, i);
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** Calculate overall download progress as integer percentage (0-100) */
export function overallProgress(models: SetupModelState[]): number {
  const totalBytes = models.reduce((sum, m) => sum + m.totalBytes, 0);
  if (totalBytes === 0) return 0;
  const downloaded = models.reduce((sum, m) => sum + m.downloadedBytes, 0);
  return Math.round((downloaded / totalBytes) * 100);
}

/** Check whether all models have finished downloading */
export function isSetupComplete(models: SetupModelState[]): boolean {
  return models.length > 0 && models.every((m) => m.status === "complete");
}

export function SetupScreen({ models, onComplete }: SetupScreenProps) {
  const complete = isSetupComplete(models);
  const progress = overallProgress(models);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="flex flex-1 flex-col items-center justify-center gap-8 px-8"
    >
      <div className="flex flex-col items-center gap-2">
        <h1 className="font-display text-4xl font-bold text-text-primary tracking-tight">
          Welcome to Manole
        </h1>
        <p className="font-mono text-sm text-text-tertiary">
          {complete ? "Setup complete" : "Setting up for first use..."}
        </p>
      </div>

      {/* Model list */}
      <div className="flex flex-col gap-3 w-full max-w-md">
        {models.map((model) => {
          const pct =
            model.totalBytes > 0
              ? Math.round(
                  (model.downloadedBytes / model.totalBytes) * 100
                )
              : 0;

          return (
            <div key={model.id} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-text-primary">
                  {model.status === "complete" && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="mr-2 text-accent"
                    >
                      ✓
                    </motion.span>
                  )}
                  {model.name}
                </span>
                <span className="font-mono text-xs text-text-tertiary">
                  {formatBytes(model.downloadedBytes)} / {formatBytes(model.totalBytes)}
                </span>
              </div>

              {/* Progress bar */}
              <div className="h-2 w-full rounded-full bg-bg-elevated overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className={`h-full rounded-full ${
                    model.status === "error" ? "bg-red-500" : "bg-accent"
                  }`}
                />
              </div>

              {model.status === "error" && model.error && (
                <span className="font-mono text-xs text-red-400">
                  {model.error}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Overall progress */}
      {!complete && (
        <p className="font-mono text-sm text-text-tertiary">
          Overall: {progress}%
        </p>
      )}

      {/* Footer */}
      <p className="font-mono text-xs text-text-tertiary text-center max-w-sm">
        This only happens once. After setup, Manole works completely offline.
      </p>

      {/* Get Started button */}
      {complete && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          onClick={onComplete}
          className="px-6 py-2 rounded-lg bg-accent text-text-primary font-mono text-sm font-semibold hover:opacity-90 transition-opacity"
        >
          Get Started
        </motion.button>
      )}
    </motion.div>
  );
}
