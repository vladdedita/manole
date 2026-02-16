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
        <p className="font-sans text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.text}
          {message.isStreaming && (
            <span className="inline-block w-0.5 h-4 bg-accent animate-pulse ml-0.5 align-text-bottom" />
          )}
        </p>

        {!isUser && message.agentSteps.length > 0 && (
          <AgentSteps steps={message.agentSteps} />
        )}
      </div>
    </motion.div>
  );
}
