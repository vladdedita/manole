import { describe, it, expect } from "vitest";

/**
 * Acceptance test: App.tsx setup phase state management
 *
 * AC1: App shows SetupScreen when backend reports models missing (setup_state "needed")
 * AC2: App transitions from SetupScreen to normal flow after downloads complete
 * AC3: Setup progress data (per-model bytes, overall percent) flows to SetupScreen component
 * AC4: First launch shows setup screen; subsequent launches skip directly to main UI
 *
 * Test Budget: 4 behaviors x 2 = 8 unit tests max
 *
 * Testing approach: extract setup phase reducer logic from App.tsx into a pure function
 * (appSetupReducer) that can be tested through its public interface without React rendering.
 * This is the driving port — the reducer processes events and produces state.
 */

describe("App setup phase - acceptance", () => {
  it("AC1+AC2+AC4: setup_state events drive phase transitions correctly", async () => {
    const { appSetupReducer } = await import("../src/lib/appSetupReducer");

    // Initial state is "checking"
    const initial = appSetupReducer(undefined, { type: "init" });
    expect(initial.appPhase).toBe("checking");

    // "needed" transitions to "setup" with model list
    const models = [
      { model_id: "llama-3.2", name: "Llama 3.2", filename: "llama.gguf", total_bytes: 4000, status: "missing" as const },
    ];
    const afterNeeded = appSetupReducer(initial, {
      type: "setup_state",
      data: { state: "needed", models },
    });
    expect(afterNeeded.appPhase).toBe("setup");
    expect(afterNeeded.setupModels).toHaveLength(1);
    expect(afterNeeded.setupModels[0].id).toBe("llama-3.2");
    expect(afterNeeded.setupModels[0].status).toBe("pending");

    // "complete" transitions to "ready"
    const afterComplete = appSetupReducer(afterNeeded, {
      type: "setup_state",
      data: { state: "complete" },
    });
    expect(afterComplete.appPhase).toBe("ready");

    // "skipped" also transitions directly to "ready" (AC4: subsequent launches)
    const afterSkipped = appSetupReducer(initial, {
      type: "setup_state",
      data: { state: "skipped" },
    });
    expect(afterSkipped.appPhase).toBe("ready");
  });

  it("AC3: setup_progress events update per-model download bytes", async () => {
    const { appSetupReducer } = await import("../src/lib/appSetupReducer");

    // Start in setup phase with models
    const setupState = appSetupReducer(undefined, {
      type: "setup_state",
      data: {
        state: "needed",
        models: [
          { model_id: "model-a", name: "Model A", filename: "a.gguf", total_bytes: 2000, status: "missing" },
          { model_id: "model-b", name: "Model B", filename: "b.gguf", total_bytes: 3000, status: "missing" },
        ],
      },
    });

    // Progress event updates specific model
    const afterProgress = appSetupReducer(setupState, {
      type: "setup_progress",
      data: {
        model_id: "model-a",
        downloaded_bytes: 500,
        total_bytes: 2000,
        status: "downloading",
      },
    });
    expect(afterProgress.setupModels[0].downloadedBytes).toBe(500);
    expect(afterProgress.setupModels[0].status).toBe("downloading");
    // Other model unchanged
    expect(afterProgress.setupModels[1].downloadedBytes).toBe(0);

    // Model completion
    const afterModelDone = appSetupReducer(afterProgress, {
      type: "setup_progress",
      data: {
        model_id: "model-a",
        downloaded_bytes: 2000,
        total_bytes: 2000,
        status: "complete",
      },
    });
    expect(afterModelDone.setupModels[0].status).toBe("complete");
    expect(afterModelDone.setupModels[0].downloadedBytes).toBe(2000);
  });
});
