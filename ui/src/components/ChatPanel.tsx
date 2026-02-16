import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { ChatMessage } from "../hooks/useChat";
import { MessageBubble } from "./MessageBubble";

interface ChatPanelProps {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  onSend: (text: string) => void;
  onOpenFolder: () => void;
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1.5 px-4 py-3 rounded-2xl rounded-bl-md bg-bg-tertiary border border-border">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="inline-block h-2 w-2 rounded-full bg-accent"
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
          />
        ))}
      </div>
    </div>
  );
}

function WelcomeScreen({ onOpenFolder }: { onOpenFolder: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-1 items-center justify-center"
    >
      <div className="text-center max-w-md">
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="font-display text-5xl font-bold text-text-primary tracking-tight"
        >
          Manole
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="mt-2 font-display text-lg italic text-text-tertiary"
        >
          the obsessive searcher
        </motion.p>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="mt-6 font-sans text-text-secondary text-sm leading-relaxed"
        >
          Select a folder. Manole will read every document,<br />
          build an index, and answer your questions.
        </motion.p>
        <motion.button
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.4 }}
          onClick={onOpenFolder}
          className="mt-8 px-8 py-3 rounded-lg bg-accent text-bg-primary font-sans font-medium text-sm hover:bg-accent-hover transition-colors active:scale-95"
        >
          Open Folder
        </motion.button>
      </div>
    </motion.div>
  );
}

export function ChatPanel({
  messages,
  isLoading,
  error,
  onSend,
  onOpenFolder,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [showNewMessage, setShowNewMessage] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);

  const checkNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const threshold = 100;
    isNearBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    if (isNearBottomRef.current) {
      setShowNewMessage(false);
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, []);

  // Auto-scroll when new messages/tokens arrive
  useEffect(() => {
    if (isNearBottomRef.current) {
      scrollToBottom();
    } else if (messages.length > 0) {
      setShowNewMessage(true);
    }
  }, [messages, scrollToBottom]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setInput("");
  };

  const showTypingIndicator =
    isLoading &&
    messages.length > 0 &&
    messages[messages.length - 1]?.role === "assistant" &&
    messages[messages.length - 1]?.text === "";

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Error banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="mx-4 mt-2 px-4 py-2 rounded-lg border border-accent/40 bg-accent/10 text-accent text-sm font-sans"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Message area or welcome */}
      {!hasMessages ? (
        <WelcomeScreen onOpenFolder={onOpenFolder} />
      ) : (
        <div className="relative flex-1 min-h-0">
          <div
            ref={scrollRef}
            onScroll={checkNearBottom}
            role="log"
            aria-live="polite"
            className="flex flex-col gap-3 p-5 h-full overflow-y-auto"
          >
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
            </AnimatePresence>

            {showTypingIndicator && <TypingIndicator />}
          </div>

          {/* New message pill */}
          <AnimatePresence>
            {showNewMessage && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.2 }}
                onClick={() => {
                  scrollToBottom();
                  setShowNewMessage(false);
                }}
                className="absolute bottom-3 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-accent text-bg-primary font-sans text-xs font-medium hover:bg-accent-hover transition-colors"
              >
                New message
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Input bar */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-3 px-5 py-3 border-t border-border bg-bg-secondary shrink-0"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your files\u2026"
          className="flex-1 h-11 px-4 rounded-xl bg-bg-primary border border-border text-text-primary font-sans text-sm placeholder:text-text-tertiary shadow-[inset_0_1px_3px_rgba(0,0,0,0.3)] focus:outline-none focus:border-accent/50 transition-colors"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="flex items-center justify-center h-10 w-10 rounded-full bg-accent text-bg-primary disabled:opacity-30 hover:bg-accent-hover transition-colors active:scale-95 shrink-0"
          aria-label="Send message"
        >
          {/* Arrow-up SVG icon */}
          <svg
            width="18"
            height="18"
            viewBox="0 0 18 18"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M9 14V4M9 4L4.5 8.5M9 4L13.5 8.5"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </form>
    </div>
  );
}
