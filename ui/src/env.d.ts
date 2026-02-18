/// <reference types="electron-vite/node" />

interface Window {
  api: {
    send: (method: string, params?: Record<string, unknown>) => Promise<unknown>;
    onMessage: (callback: (response: unknown) => void) => () => void;
    selectDirectory: () => Promise<string | null>;
    openFile: (filePath: string) => Promise<string>;
  };
}
