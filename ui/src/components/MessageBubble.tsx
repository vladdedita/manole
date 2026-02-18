import { motion } from "motion/react";
import type { ChatMessage } from "../hooks/useChat";
import { AgentSteps } from "./AgentSteps";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, x: isUser ? 12 : -12, y: 8 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[75%] px-4 py-3 ${
          isUser
            ? "rounded-2xl rounded-br-md bg-accent-muted border border-accent/20 text-text-primary"
            : "rounded-2xl rounded-bl-md bg-bg-tertiary border border-border text-text-primary"
        }`}
      >
        {message.isStreaming && message.text === "" ? (
          <div className="flex items-center gap-1.5 py-0.5">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="inline-block h-2 w-2 rounded-full bg-accent animate-pulse"
                style={{ animationDelay: `${i * 150}ms` }}
              />
            ))}
          </div>
        ) : (
          <p className="font-sans text-sm leading-relaxed whitespace-pre-wrap break-words">
            {message.text}
            {message.isStreaming && (
              <span className="inline-block w-0.5 h-4 bg-accent animate-pulse ml-0.5 align-text-bottom" />
            )}
          </p>
        )}

        {!isUser && !message.isStreaming && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-border/50">
            {message.sources.map((source) => {
              const basename = source.split("/").pop() || source;
              return (
                <button
                  key={source}
                  onClick={() => window.api.openFile(source)}
                  title={source}
                  className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium
                             bg-bg-secondary border border-border rounded-full
                             hover:bg-accent-muted hover:border-accent/30
                             transition-colors cursor-pointer text-text-secondary hover:text-text-primary"
                >
                  <svg className="w-3 h-3 shrink-0" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M3.5 2A1.5 1.5 0 002 3.5v9A1.5 1.5 0 003.5 14h9a1.5 1.5 0 001.5-1.5V6.621a1.5 1.5 0 00-.44-1.06l-3.12-3.122A1.5 1.5 0 009.378 2H3.5z"/>
                  </svg>
                  {basename}
                </button>
              );
            })}
          </div>
        )}

        {!isUser && message.agentSteps.length > 0 && (
          <AgentSteps steps={message.agentSteps} />
        )}
      </div>
    </motion.div>
  );
}
