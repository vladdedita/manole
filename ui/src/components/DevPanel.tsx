import { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";

interface DevPanelProps {
  open: boolean;
  onClose: () => void;
  logs: string[];
}

export function DevPanel({ open, onClose, logs }: DevPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 200, opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="border-t border-border bg-bg-primary overflow-hidden shrink-0"
        >
          <div className="flex items-center justify-between h-7 px-4 border-b border-border bg-bg-secondary">
            <span className="font-mono text-[10px] uppercase tracking-widest text-text-tertiary">
              Python Output
            </span>
            <button
              onClick={onClose}
              className="font-mono text-[10px] text-text-tertiary hover:text-text-secondary transition-colors"
            >
              ✕
            </button>
          </div>
          <div
            ref={scrollRef}
            className="h-[calc(200px-28px)] overflow-y-auto p-3 font-mono text-[11px] leading-relaxed text-text-secondary"
          >
            {logs.length === 0 ? (
              <span className="text-text-tertiary">No output yet</span>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all">
                  {line}
                </div>
              ))
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
