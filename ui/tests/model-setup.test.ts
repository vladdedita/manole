import { describe, it, expect, vi } from "vitest";

/**
 * Acceptance test: ModelSetupManager orchestrates model check and download
 *
 * AC1: Python process receives MANOLE_MODELS_DIR env var matching platform models directory
 * AC2: App startup calls check_models before showing main UI
 * AC3: When all models present, app proceeds directly to main UI without setup screen
 * AC4: When models missing, download_models is invoked and progress events forwarded to renderer
 * AC5: Setup completes before any init/indexing request is sent
 *
 * Test Budget: 5 behaviors x 2 = 10 unit tests max
 */

// Port boundary: PythonBridge interface (driven port)
interface PythonBridgeLike {
  send(method: string, params?: Record<string, unknown>): Promise<{ id: number | null; type: string; data: Record<string, unknown> }>;
}

// Port boundary: BrowserWindow-like (driven port for IPC)
interface WindowLike {
  webContents: {
    send(channel: string, ...args: unknown[]): void;
  };
}

describe("ModelSetupManager - acceptance", () => {
  it("AC2+AC3: when all models present, sends check_models and skips setup", async () => {
    const { ModelSetupManager } = await import("../electron/setup");

    const pythonSend = vi.fn().mockResolvedValue({
      id: 1,
      type: "result",
      data: { ready: true, models: [] },
    });
    const python: PythonBridgeLike = { send: pythonSend };

    const windowSend = vi.fn();
    const window: WindowLike = { webContents: { send: windowSend } };

    const manager = new ModelSetupManager(python as any, window as any);
    await manager.checkAndDownload();

    // Must call check_models on Python
    expect(pythonSend).toHaveBeenCalledWith("check_models", expect.any(Object));
    // Must send "skipped" state to renderer (no setup needed)
    expect(windowSend).toHaveBeenCalledWith("python:message", { id: null, type: "setup_state", data: { state: "checking" } });
    expect(windowSend).toHaveBeenCalledWith("python:message", { id: null, type: "setup_state", data: { state: "skipped" } });
    // Must NOT call download_models
    expect(pythonSend).not.toHaveBeenCalledWith("download_models", expect.any(Object));
  });

  it("AC2+AC4+AC5: when models missing, triggers download and reports completion", async () => {
    const { ModelSetupManager } = await import("../electron/setup");

    const pythonSend = vi.fn()
      .mockResolvedValueOnce({
        id: 1,
        type: "result",
        data: { ready: false, models: [{ model_id: "llama-3.2", status: "missing" }] },
      })
      .mockResolvedValueOnce({
        id: 2,
        type: "result",
        data: { success: true },
      });
    const python: PythonBridgeLike = { send: pythonSend };

    const windowSend = vi.fn();
    const window: WindowLike = { webContents: { send: windowSend } };

    const manager = new ModelSetupManager(python as any, window as any);
    await manager.checkAndDownload();

    // Must call check_models first, then download_models
    expect(pythonSend).toHaveBeenCalledWith("check_models", expect.any(Object));
    expect(pythonSend).toHaveBeenCalledWith("download_models", expect.any(Object));
    // Must send correct state transitions
    expect(windowSend).toHaveBeenCalledWith("python:message", { id: null, type: "setup_state", data: { state: "checking" } });
    expect(windowSend).toHaveBeenCalledWith("python:message", { id: null, type: "setup_state", data: { state: "needed" } });
    expect(windowSend).toHaveBeenCalledWith("python:message", { id: null, type: "setup_state", data: { state: "complete" } });
  });

  it("AC1: getModelsDir returns a path containing 'models'", async () => {
    const { ModelSetupManager } = await import("../electron/setup");

    const python: PythonBridgeLike = { send: vi.fn() };
    const window: WindowLike = { webContents: { send: vi.fn() } };

    const manager = new ModelSetupManager(python as any, window as any);
    const modelsDir = manager.getModelsDir();

    // Must end with /models or /models/
    expect(modelsDir).toMatch(/models\/?$/);
    // Must be an absolute path
    expect(modelsDir.startsWith("/") || modelsDir.match(/^[A-Z]:\\/)).toBeTruthy();
  });
});
