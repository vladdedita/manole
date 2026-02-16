import { useEffect, useCallback, useState, useRef } from "react";
import type { Response, StatusData } from "../lib/protocol";

export type MessageHandler = (response: Response) => void;

export function usePython() {
  const [backendState, setBackendState] = useState<string>("not_initialized");
  const handlersRef = useRef<Set<MessageHandler>>(new Set());

  useEffect(() => {
    const cleanup = window.api.onMessage((response: Response) => {
      if (response.type === "status") {
        const data = response.data as unknown as StatusData;
        setBackendState(data.state);
      }
      for (const handler of handlersRef.current) {
        handler(response);
      }
    });
    return cleanup;
  }, []);

  const send = useCallback(
    (method: string, params?: Record<string, unknown>) =>
      window.api.send(method, params),
    [],
  );

  const subscribe = useCallback((handler: MessageHandler) => {
    handlersRef.current.add(handler);
    return () => { handlersRef.current.delete(handler); };
  }, []);

  return { send, subscribe, backendState };
}
