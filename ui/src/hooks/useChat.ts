import { useReducer, useCallback, useEffect } from "react";
import { usePython } from "./usePython";
import type { Response } from "../lib/protocol";

export interface AgentStep {
  step: number;
  tool: string;
  params: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  isStreaming: boolean;
  agentSteps: AgentStep[];
}

interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
}

type ChatAction =
  | { type: "user_message"; text: string }
  | { type: "stream_token"; text: string }
  | { type: "agent_step"; step: AgentStep }
  | { type: "response_complete"; text: string }
  | { type: "error"; message: string }
  | { type: "clear" };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "user_message": {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        text: action.text,
        isStreaming: false,
        agentSteps: [],
      };
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        text: "",
        isStreaming: true,
        agentSteps: [],
      };
      return {
        ...state,
        messages: [...state.messages, userMsg, assistantMsg],
        isLoading: true,
        error: null,
      };
    }
    case "stream_token": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.isStreaming) {
        messages[messages.length - 1] = { ...last, text: last.text + action.text };
      }
      return { ...state, messages };
    }
    case "agent_step": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.isStreaming) {
        messages[messages.length - 1] = {
          ...last,
          agentSteps: [...last.agentSteps, action.step],
        };
      }
      return { ...state, messages };
    }
    case "response_complete": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last?.role === "assistant") {
        messages[messages.length - 1] = { ...last, text: action.text, isStreaming: false };
      }
      return { ...state, messages, isLoading: false };
    }
    case "error":
      return { ...state, isLoading: false, error: action.message };
    case "clear":
      return { messages: [], isLoading: false, error: null };
    default:
      return state;
  }
}

export function useChat() {
  const { send, subscribe, backendState } = usePython();
  const [state, dispatch] = useReducer(chatReducer, {
    messages: [],
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    return subscribe((response: Response) => {
      switch (response.type) {
        case "token":
          dispatch({ type: "stream_token", text: (response.data as { text: string }).text });
          break;
        case "agent_step":
          dispatch({ type: "agent_step", step: response.data as unknown as AgentStep });
          break;
        case "error":
          if (response.id !== null) {
            dispatch({ type: "error", message: (response.data as { message: string }).message });
          }
          break;
      }
    });
  }, [subscribe]);

  const sendMessage = useCallback(
    async (text: string) => {
      dispatch({ type: "user_message", text });
      const result = await send("query", { text });
      if (result.type === "result") {
        dispatch({ type: "response_complete", text: (result.data as { text: string }).text });
      } else if (result.type === "error") {
        dispatch({ type: "error", message: (result.data as { message: string }).message });
      }
    },
    [send],
  );

  const initBackend = useCallback(
    (dataDir: string, reuse?: string) =>
      send("init", { dataDir, ...(reuse ? { reuse } : {}) }),
    [send],
  );

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    error: state.error,
    backendState,
    sendMessage,
    initBackend,
    clearChat: () => dispatch({ type: "clear" }),
  };
}
