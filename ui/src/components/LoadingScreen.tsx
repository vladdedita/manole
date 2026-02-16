import { motion } from "motion/react";

const STEPS = [
  { key: "loading_model", label: "Loading model" },
  { key: "indexing", label: "Indexing files" },
  { key: "ready", label: "Ready" },
];

function stepIndex(state: string): number {
  const idx = STEPS.findIndex((s) => s.key === state);
  return idx === -1 ? 0 : idx;
}

interface LoadingScreenProps {
  backendState: string;
}

export function LoadingScreen({ backendState }: LoadingScreenProps) {
  const current = stepIndex(backendState);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="flex flex-1 flex-col items-center justify-center gap-10"
    >
      <motion.h1
        animate={{ opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
        className="font-display text-4xl font-bold text-text-primary tracking-tight"
      >
        Manole
      </motion.h1>

      {/* Step indicator */}
      <div className="flex flex-col gap-0">
        {STEPS.map((step, i) => {
          const isDone = i < current;
          const isActive = i === current;
          const isLast = i === STEPS.length - 1;

          return (
            <div key={step.key} className="flex items-stretch">
              {/* Left column: dot + connector */}
              <div className="flex flex-col items-center w-6">
                {/* Dot */}
                <div className="flex items-center justify-center h-6 w-6 shrink-0">
                  {isActive ? (
                    <motion.div
                      animate={{ scale: [1, 1.4, 1] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                      className="h-2.5 w-2.5 rounded-full bg-accent"
                    />
                  ) : (
                    <div
                      className={`h-2.5 w-2.5 rounded-full transition-colors duration-500 ${
                        isDone ? "bg-accent" : "bg-bg-elevated"
                      }`}
                    />
                  )}
                </div>
                {/* Connector line */}
                {!isLast && (
                  <div
                    className={`w-px flex-1 min-h-4 transition-colors duration-500 ${
                      isDone ? "bg-accent/40" : "bg-border"
                    }`}
                  />
                )}
              </div>

              {/* Label */}
              <span
                className={`font-mono text-sm pl-3 pt-0.5 pb-3 transition-colors duration-500 ${
                  isActive
                    ? "text-text-primary"
                    : isDone
                      ? "text-accent"
                      : "text-text-tertiary"
                }`}
              >
                {step.label}
                {isDone && (
                  <motion.span
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="ml-2"
                  >
                    ✓
                  </motion.span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
