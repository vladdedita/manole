import type { SetupModelState } from "../components/SetupScreen";

export type AppPhase = "checking" | "setup" | "ready";

export interface AppSetupState {
  appPhase: AppPhase;
  setupModels: SetupModelState[];
}

interface InitAction {
  type: "init";
}

interface SetupStateAction {
  type: "setup_state";
  data: {
    state: "checking" | "needed" | "complete" | "skipped";
    models?: Array<{
      model_id: string;
      name?: string;
      filename?: string;
      total_bytes?: number;
      status?: string;
    }>;
  };
}

interface SetupProgressAction {
  type: "setup_progress";
  data: {
    model_id: string;
    downloaded_bytes?: number;
    total_bytes?: number;
    status: "downloading" | "verifying" | "complete" | "error" | "skipped";
    error?: string;
  };
}

interface SetupCompleteAction {
  type: "setup_complete";
}

export type AppSetupAction =
  | InitAction
  | SetupStateAction
  | SetupProgressAction
  | SetupCompleteAction;

const initialState: AppSetupState = {
  appPhase: "checking",
  setupModels: [],
};

export function appSetupReducer(
  state: AppSetupState | undefined,
  action: AppSetupAction,
): AppSetupState {
  const current = state ?? initialState;

  switch (action.type) {
    case "init":
      return initialState;

    case "setup_state": {
      const { state: setupState, models } = action.data;

      if (setupState === "needed") {
        const setupModels: SetupModelState[] = (models ?? []).map((m) => ({
          id: m.model_id,
          name: m.name ?? m.model_id,
          filename: m.filename ?? "",
          downloadedBytes: 0,
          totalBytes: m.total_bytes ?? 0,
          status: "pending" as const,
        }));
        return { appPhase: "setup", setupModels };
      }

      if (setupState === "complete" || setupState === "skipped") {
        return { ...current, appPhase: "ready" };
      }

      // "checking" — stay in current state
      return current;
    }

    case "setup_progress": {
      const { model_id, downloaded_bytes, total_bytes, status, error } = action.data;
      const setupModels = current.setupModels.map((m) =>
        m.id === model_id
          ? {
              ...m,
              downloadedBytes: downloaded_bytes ?? m.downloadedBytes,
              totalBytes: total_bytes ?? m.totalBytes,
              status: (status === "verifying" ? "downloading" : status === "skipped" ? "complete" : status) as SetupModelState["status"],
              ...(error ? { error } : {}),
            }
          : m,
      );
      return { ...current, setupModels };
    }

    case "setup_complete":
      return { ...current, appPhase: "ready" };

    default:
      return current;
  }
}
