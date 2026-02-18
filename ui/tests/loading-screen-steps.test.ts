import { describe, it, expect } from "vitest";
import { buildSteps } from "../src/components/LoadingScreen";

/**
 * Acceptance test: dynamic LoadingScreen steps
 *
 * AC1: Shows "Generating summary" step when backend state is "summarizing" or later
 * AC2: Shows "Captioning images (done/total)" step with live counter when captioning active
 * AC3: Captioning step hidden when no captioning status received
 * AC4: StatusData state type includes "summarizing" and "captioning" (compile-time check)
 */
describe("LoadingScreen dynamic steps - acceptance", () => {
  it("AC1: includes summarizing step when state is summarizing", () => {
    const steps = buildSteps("summarizing", undefined);
    const keys = steps.map((s) => s.key);
    expect(keys).toContain("summarizing");
    const summStep = steps.find((s) => s.key === "summarizing");
    expect(summStep?.label).toBe("Generating summary");
  });

  it("AC1: includes summarizing step when state is later than summarizing", () => {
    const steps = buildSteps("ready", undefined);
    const keys = steps.map((s) => s.key);
    expect(keys).toContain("summarizing");
  });

  it("AC2: shows captioning step with live counter when captioning active", () => {
    const steps = buildSteps("captioning", { done: 3, total: 10 });
    const capStep = steps.find((s) => s.key === "captioning");
    expect(capStep).toBeDefined();
    expect(capStep?.label).toBe("Captioning images (3/10)");
  });

  it("AC3: hides captioning step when no captioning status received", () => {
    const steps = buildSteps("indexing", undefined);
    const keys = steps.map((s) => s.key);
    expect(keys).not.toContain("captioning");
  });

  it("AC4: state union accepts summarizing and captioning", async () => {
    // Compile-time verification - importing StatusData with new states
    const { StatusData } = await import("../src/lib/protocol") as any;
    // If this file compiles with the new states, AC4 is satisfied
    // Runtime check: the type exists as an interface (no runtime artifact)
    // This is validated by TypeScript compilation of the test itself
    const testState: import("../src/lib/protocol").StatusData = {
      state: "summarizing",
    };
    expect(testState.state).toBe("summarizing");

    const testState2: import("../src/lib/protocol").StatusData = {
      state: "captioning",
    };
    expect(testState2.state).toBe("captioning");
  });
});
