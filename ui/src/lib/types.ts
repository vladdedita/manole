import type { Response } from "./protocol";

export interface PythonAPI {
  send: (method: string, params?: Record<string, unknown>) => Promise<Response>;
  onMessage: (callback: (response: Response) => void) => () => void;
  selectDirectory: () => Promise<string | null>;
}

declare global {
  interface Window {
    api: PythonAPI;
  }
}
