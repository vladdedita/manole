import { useState, useCallback } from "react";
import type { FileGraphData } from "../lib/protocol";
import { usePython } from "./usePython";

interface UseFileGraphReturn {
  graph: FileGraphData | null;
  isLoading: boolean;
  error: string | null;
  fetchGraph: (directoryId: string) => Promise<void>;
  clearGraph: () => void;
}

export function useFileGraph(): UseFileGraphReturn {
  const { send } = usePython();
  const [graph, setGraph] = useState<FileGraphData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = useCallback(
    async (directoryId: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await send("getFileGraph", { directoryId });
        if (result.type === "result") {
          setGraph(result.data as unknown as FileGraphData);
        } else if (result.type === "error") {
          setError((result.data as { message: string }).message);
        }
      } catch (err) {
        setError(String(err));
      } finally {
        setIsLoading(false);
      }
    },
    [send]
  );

  const clearGraph = useCallback(() => {
    setGraph(null);
    setError(null);
  }, []);

  return { graph, isLoading, error, fetchGraph, clearGraph };
}
