import { describe, it, expect } from "vitest";
import type { SetupProgressData } from "../src/lib/protocol";
import type { ResponseType } from "../src/lib/protocol";

/**
 * Acceptance test: SetupScreen protocol types and progress logic
 *
 * AC1: SetupProgressData type includes model name, bytes downloaded, total bytes, overall percent
 * AC2: setup_progress added to ResponseType union
 * AC3: SetupScreen renders a progress bar per model with download size
 * AC4: SetupScreen shows overall progress across all models
 * AC5: SetupScreen displays completion state when all models finish
 *
 * Test Budget: 5 behaviors x 2 = 10 unit tests max
 */

describe("SetupScreen - acceptance", () => {
  it("AC1+AC2: SetupProgressData type is available and setup_progress is a valid ResponseType", () => {
    // Type-level: SetupProgressData must have required fields
    const progress: SetupProgressData = {
      model_id: "llama-3.2",
      status: "downloading",
      downloaded_bytes: 1024,
      total_bytes: 4096,
    };
    expect(progress.model_id).toBe("llama-3.2");
    expect(progress.status).toBe("downloading");

    // setup_progress must be assignable to ResponseType
    const rt: ResponseType = "setup_progress";
    expect(rt).toBe("setup_progress");
  });

  it("AC3+AC4: per-model progress and overall progress can be computed", async () => {
    const { formatBytes, overallProgress } = await import(
      "../src/components/SetupScreen"
    );

    // Per-model: format bytes for display
    expect(formatBytes(1048576)).toBe("1.0 MB");

    // Overall progress across models
    const models = [
      { id: "a", name: "Model A", filename: "a.bin", downloadedBytes: 500, totalBytes: 1000, status: "downloading" as const },
      { id: "b", name: "Model B", filename: "b.bin", downloadedBytes: 2000, totalBytes: 2000, status: "complete" as const },
    ];
    expect(overallProgress(models)).toBe(83); // (500+2000)/(1000+2000) = 83%
  });

  it("AC5: completion state detected when all models finish", async () => {
    const { isSetupComplete } = await import("../src/components/SetupScreen");

    const allDone = [
      { id: "a", name: "A", filename: "a.bin", downloadedBytes: 1000, totalBytes: 1000, status: "complete" as const },
      { id: "b", name: "B", filename: "b.bin", downloadedBytes: 2000, totalBytes: 2000, status: "complete" as const },
    ];
    expect(isSetupComplete(allDone)).toBe(true);

    const notDone = [
      { id: "a", name: "A", filename: "a.bin", downloadedBytes: 500, totalBytes: 1000, status: "downloading" as const },
    ];
    expect(isSetupComplete(notDone)).toBe(false);
  });
});
